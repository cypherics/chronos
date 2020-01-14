import cv2
import numpy as np

from ml.base import BaseDataSetPt
from ml.commons.utils.torch_tensor_conversion import (
    to_input_image_tensor,
    to_multi_output_label_image_tensor,
)


class BuildingBoundaryDataSet(BaseDataSetPt):
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

        boundary_mask = self.get_label_normalization(self.get_boundary_building(mask))
        building_mask = self.get_label_normalization(mask)

        mask = self.combine_building_mask_boundary_mask(building_mask, boundary_mask)

        return {
            "image": to_input_image_tensor(img),
            "label": to_multi_output_label_image_tensor(mask),
        }

    def perform_image_operation_test(self, img) -> dict:
        img, _ = super().handle_image_size(img, None, self.model_input_dimension)
        img = super().perform_normalization(img)

        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    def get_label_normalization(self, mask) -> np.ndarray:
        normalized_mask = mask / 255

        return normalized_mask

    @staticmethod
    def combine_building_mask_boundary_mask(building_mask, boundary_mask):
        building_mask_height, building_mask_width = building_mask.shape
        boundary_mask_height, boundary_mask_width = boundary_mask.shape

        assert (
            boundary_mask_height == building_mask_height
            and boundary_mask_width == building_mask_width
        ), "Dimension mismatch"
        return np.concatenate(
            (
                building_mask.reshape((building_mask_height, building_mask_width, 1)),
                boundary_mask.reshape(boundary_mask_height, boundary_mask_width, 1),
            ),
            axis=2,
        )

    @staticmethod
    def get_boundary_building(image):
        boundary_image = np.zeros(image.shape)
        (_, counts, _) = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # loop over the contours
        for c in counts:
            cv2.drawContours(boundary_image, [c], -1, 255, 1)
        return boundary_image
