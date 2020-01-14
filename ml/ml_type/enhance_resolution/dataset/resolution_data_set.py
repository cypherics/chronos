import numpy as np

from ml.base import BaseDataSetPt
from ml.commons.utils.image import perform_scale
from ml.commons.utils.torch_tensor_conversion import to_input_image_tensor


class ResolutionDataSet(BaseDataSetPt):
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
        w, h, _ = mask.shape
        img = perform_scale(mask, dimension=(w // 4, h // 4))

        img, mask = super().perform_transformation(img, mask)

        img = super().perform_normalization(img)
        mask = self.get_label_normalization(mask)

        return {
            "image": to_input_image_tensor(img),
            "label": to_input_image_tensor(mask),
        }

    def perform_image_operation_test(self, img) -> dict:
        w, h, _ = img.shape
        img = perform_scale(img, dimension=(w // 4, h // 4))
        img = super().perform_normalization(img)
        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    def get_label_normalization(self, mask) -> np.ndarray:
        return mask / 255
