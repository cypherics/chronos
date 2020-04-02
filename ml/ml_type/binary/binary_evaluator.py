import numpy as np
from torchvision.utils import make_grid

from ..base.base_evaluator import BaseEvaluator
from ...commons.utils.tensor_util import cuda_variable


class BinaryEvaluator(BaseEvaluator):
    def create_prediction_grid(self, inputs, prediction):
        display_image = cuda_variable(inputs)
        display_image = display_image["image"].cpu()
        grid = make_grid(prediction, nrow=2, normalize=True)
        nda = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        grid_sat = make_grid(display_image, nrow=2, normalize=True)
        grid_sat_nda = (
            grid_sat.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )

        return np.vstack((nda, grid_sat_nda))

    def classifier_activation(self, prediction):
        prediction = prediction.sigmoid()
        return prediction

    def generate_image(self, prediction):
        prediction[prediction >= 0.40] = 1
        prediction[prediction < 0.40] = 0
        return prediction
