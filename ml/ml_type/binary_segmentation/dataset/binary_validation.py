import numpy as np
import torch
from torchvision.utils import make_grid

from ml.commons.metrics import iou, precision, recall, f_score
from ml.base.base_dataset.base_pt_validation import BaseValidationPt
from ml.commons.utils.torch_tensor_conversion import cuda_variable


class BinaryValidation(BaseValidationPt):
    def compute_metric(self, targets, outputs):
        jaccard = iou(targets, (outputs > 0).float()).item()
        precision_metric = precision(targets, (outputs > 0).float()).item()
        recall_metric = recall(targets, (outputs > 0).float()).item()
        f_score_metric = f_score(targets, (outputs > 0).float()).item()
        return {
            "Jaccard": jaccard,
            "Precision": precision_metric,
            "Recall": recall_metric,
            "F_Score": f_score_metric,
        }

    @staticmethod
    def generate_inference_output(img):
        img = img.sigmoid().data.cpu().numpy()

        _, w, h = img.shape
        img = img.reshape((w, h))
        return img

    def get_accuracy(self, true_values, predicted_values):
        raise NotImplementedError

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
