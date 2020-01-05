import numpy as np
import torch
from torchvision.utils import make_grid

from ml.ml_type.sem_seg.metrics import iou, precision, recall, f_score
from ml.ml_type.Base.base_validation import BaseValidation
from utils.torch_tensor_conversion import cuda_variable


class BinaryValidation(BaseValidation):
    def __init__(self, problem_type):
        super().__init__(problem_type)
        self.jaccard = []
        self.precision_metric = []
        self.recall_metric = []
        self.f_score_metric = []

    def compute_metric(self, targets, outputs):
        self.jaccard += [iou(targets, (outputs > 0).float()).item()]
        self.precision_metric += [precision(targets, (outputs > 0).float()).item()]
        self.recall_metric += [recall(targets, (outputs > 0).float()).item()]
        self.f_score_metric += [f_score(targets, (outputs > 0).float()).item()]

    def get_computed_mean_metric(self, **kwargs):
        self.jaccard = np.mean(self.jaccard)
        self.precision_metric = np.mean(self.precision_metric)
        self.recall_metric = np.mean(self.recall_metric)
        self.f_score_metric = np.mean(self.f_score_metric)
        metrics = self.generate_dictionary()
        self._re_initialize_metric_variables()
        return metrics

    def generate_dictionary(self):
        metrics = {
            "Jaccard": self.jaccard,
            "Precision": self.precision_metric,
            "Recall": self.recall_metric,
            "F_Score": self.f_score_metric,
        }
        return metrics

    @staticmethod
    def generate_inference_output(img):
        img = img.sigmoid().data.cpu().numpy()

        _, w, h = img.shape
        img = img.reshape((w, h))
        return img

    def get_accuracy(self, true_values, predicted_values):
        raise NotImplementedError

    def _re_initialize_metric_variables(self):
        self.jaccard = []
        self.precision_metric = []
        self.recall_metric = []
        self.f_score_metric = []

    @staticmethod
    def create_prediction_grid(inputs, prediction):
        prediction = prediction.sigmoid()
        display_image = cuda_variable(inputs)
        display_image = torch.FloatTensor(display_image["image"].cpu())
        grid = make_grid(prediction, nrow=2, normalize=True)
        nda = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        grid_sat = make_grid(display_image, nrow=2, normalize=True)
        grid_sat_nda = (
            grid_sat.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )

        return np.vstack((nda, grid_sat_nda))
