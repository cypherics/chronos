import numpy as np

from ml.ml_type.base import BaseDataSetPt
from ml.commons.utils.torch_tensor_conversion import to_input_image_tensor, to_tensor


class BinarySoftmaxDataSet(BaseDataSetPt):
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
        # https://discuss.pytorch.org/t/multiclass-segmentation/54065/5
        return {"image": to_input_image_tensor(img), "label": to_tensor(mask).long()}

    def perform_image_operation_test(self, img) -> dict:
        img, _ = super().handle_image_size(img, None, self.model_input_dimension)
        img = super().perform_normalization(img)

        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    def get_label_normalization(self, mask) -> np.ndarray:
        normalized_mask = mask / 255
        return normalized_mask
