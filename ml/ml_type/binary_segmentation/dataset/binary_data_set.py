import numpy as np
from ml.base.base_dataset.base_pt_dataset import BaseDataSetPt
from ml.commons.utils.torch_tensor_conversion import (
    to_input_image_tensor,
    to_label_image_tensor,
)


class BinaryDataSet(BaseDataSetPt):
    def __init__(
        self,
        root,
        model_input_dim=None,
        mode="train",
        transform=None,
        normalization=None,
    ):
        super().__init__(root, model_input_dim, mode, transform, normalization)

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
        normalized_mask = mask / 255
        return normalized_mask