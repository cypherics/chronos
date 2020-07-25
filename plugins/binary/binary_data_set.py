import cv2
import numpy as np

from utils.image_ops import handle_image_size
from ..base.base_data_set import BaseDataSetPt


class BinaryDataSet(BaseDataSetPt):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    @staticmethod
    def adjust_learner_data(img, mask, dimension) -> [dict]:
        img, mask = handle_image_size(img, mask, dimension)
        return [{"image": img, "label": mask}]

    @staticmethod
    def adjust_evaluator_data(img, dimension) -> [dict]:
        img, _ = handle_image_size(img, None, dimension)
        return [{"image": img}]

    @staticmethod
    def normalize_label(mask) -> np.ndarray:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        normalized_mask = mask / 255
        return np.expand_dims(normalized_mask, -1)

    def normalize_image(self, img) -> np.ndarray:
        return getattr(self, self.config.normalization)(img)

    @staticmethod
    def inria_data(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255
        img -= np.ones(img.shape) * (
            0.42068335885143315,
            0.43821200008781647,
            0.4023395608370018,
        )
        img /= np.ones(img.shape) * (
            0.03871459540580076,
            0.039615887087616986,
            0.04203108867447648,
        )
        return img

    @staticmethod
    def divide_by_255(img: np.ndarray) -> np.ndarray:
        return img / 255
