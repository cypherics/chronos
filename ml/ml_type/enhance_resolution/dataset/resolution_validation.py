import numpy as np
from torchvision.utils import make_grid

from ml.ml_type.base import BaseValidationPt


class ResolutionValidation(BaseValidationPt):
    def compute_metric(self, **kwargs):
        return None

    def generate_inference_output(self, img):
        img = img.data.cpu().numpy()
        img = np.moveaxis(img, 0, -1)
        return img * 255

    def create_prediction_grid(self, inputs, prediction):
        grid = make_grid(prediction, nrow=2, normalize=True)
        nda = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

        return np.vstack((nda, nda))
