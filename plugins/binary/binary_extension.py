import os
import random
import shutil
import cv2

import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

from core.extensions.callbacks import Callback
from core.extensions.metric import Metric
from core.logger import ChronosLogger
from plugins.base.base_extension import BaseExtension
from utils.directory_ops import make_directory
from utils.pt_tensor import make_cuda

EPSILON = 1e-11

logger = ChronosLogger.get_logger()


class BinaryExtension(BaseExtension):
    def __init__(self, config):
        super().__init__(config)

    def callbacks(self) -> list:
        return [TestCallback(self.pth)]

    def metrics(self) -> list:
        return [Accuracy(), Precision(), Recall(), F1(), IOU()]


class Accuracy(Metric):
    def compute_metric(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]
        prediction = self.get_numpy(prediction).flatten()
        ground_truth = self.get_numpy(ground_truth).flatten()
        prediction = to_binary(prediction)
        tn, fp, fn, tp = confusion_matrix(
            ground_truth, prediction, labels=[0, 1]
        ).ravel()
        num = tp + tn
        den = tp + tn + fp + fn
        return num / (den + EPSILON)


class F1(Metric):
    def compute_metric(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]
        prediction = self.get_numpy(prediction).flatten()
        ground_truth = self.get_numpy(ground_truth).flatten()
        prediction = to_binary(prediction)
        tn, fp, fn, tp = confusion_matrix(
            ground_truth, prediction, labels=[0, 1]
        ).ravel()
        num = 2 * tp
        den = (2 * tp) + fp + fn
        return num / (den + EPSILON)


class Recall(Metric):
    def compute_metric(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]
        prediction = self.get_numpy(prediction).flatten()
        ground_truth = self.get_numpy(ground_truth).flatten()
        prediction = to_binary(prediction)
        tn, fp, fn, tp = confusion_matrix(
            ground_truth, prediction, labels=[0, 1]
        ).ravel()
        return tp / (tp + fn + EPSILON)


class Precision(Metric):
    def compute_metric(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]
        prediction = self.get_numpy(prediction).flatten()
        ground_truth = self.get_numpy(ground_truth).flatten()
        prediction = to_binary(prediction)
        tn, fp, fn, tp = confusion_matrix(
            ground_truth, prediction, labels=[0, 1]
        ).ravel()
        return tp / (tp + fp + EPSILON)


class IOU(Metric):
    def compute_metric(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]
        prediction = self.get_numpy(prediction).flatten()
        ground_truth = self.get_numpy(ground_truth).flatten()
        prediction = to_binary(prediction)
        tn, fp, fn, tp = confusion_matrix(
            ground_truth, prediction, labels=[0, 1]
        ).ravel()
        denominator = tp + fp + fn
        if denominator == 0:
            value = 0
        else:
            value = float(tp) / (denominator + EPSILON)
        return value


class TestCallback(Callback):
    def __init__(self, pth):
        super().__init__()
        self.pth = pth

    def on_batch_end(self, batch, logs=None):
        model = logs["model"]
        test_loader = logs["test_loader"]
        model.eval()
        if random.random() < 0.10:
            try:
                for i, (inputs, file_path) in enumerate(test_loader):

                    image = make_cuda(inputs)
                    prediction = model(image)
                    prediction = prediction["output"]
                    prediction = prediction.sigmoid()
                    prediction = to_binary(prediction)
                    stacked_image = create_prediction_grid(inputs["image"], prediction)
                    save_path = make_directory(self.pth, "test_prediction")

                    shutil.rmtree(save_path)
                    os.makedirs(save_path)

                    save_image_path = os.path.join(save_path, "{}.png".format(batch))
                    cv2.imwrite(
                        save_image_path, cv2.cvtColor(stacked_image, cv2.COLOR_RGB2BGR)
                    )
                    break
            except Exception as ex:
                logger.exception(
                    "Skipped Exception in {}".format(self.__class__.__name__)
                )
                logger.exception("Exception {}".format(ex))
                pass


def to_binary(prediction, cutoff=0.40):
    prediction[prediction >= cutoff] = 1
    prediction[prediction < cutoff] = 0
    return prediction


def create_prediction_grid(image, prediction):
    display_image = make_cuda(image)
    display_image = display_image.cpu()
    grid = make_grid(prediction, nrow=2, normalize=True)
    nda = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    grid_sat = make_grid(display_image, nrow=2, normalize=True)
    grid_sat_nda = grid_sat.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    return np.vstack((nda, grid_sat_nda))
