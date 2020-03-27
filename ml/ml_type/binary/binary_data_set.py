import cv2
import numpy as np

from ml.commons.utils.torch_tensor_conversion import (
    to_input_image_tensor,
    to_label_image_tensor,
)
from ..base.base_data_set import BaseDataSetPt


class BinaryDataSet(BaseDataSetPt):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def perform_image_operation_train_and_val(self, img, mask) -> dict:
        img, mask = super().handle_image_size(img, mask, self.model_input_dimension)
        img, mask = super().perform_transformation(img, mask)
        img = super().perform_normalization(img)

        mask = self.get_label_normalization(mask)
        return {
            "image": to_input_image_tensor(img),
            "label": to_label_image_tensor(mask),
        }

    def perform_image_operation_test(self, img) -> dict:
        img, _ = super().handle_image_size(img, None, self.model_input_dimension)
        img = super().perform_normalization(img)

        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    @staticmethod
    def get_label_normalization(mask) -> np.ndarray:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        normalized_mask = mask / 255
        return normalized_mask
