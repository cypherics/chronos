import sys
import cv2
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from utils.print_format import print_exception
from data_processing import data_normalization, data_transformation

from abc import ABCMeta


class BaseDataLoader(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        root,
        model_input_dim=None,
        mode="train",
        transform=None,
        normalization=None,
    ):
        if mode == "train":
            self.transform = self.load_transformation(transform)
        else:
            self.transform = None

        self.mode = mode
        self.normalization = self.load_normalization(normalization)
        self.model_input_dimension = tuple(model_input_dim)

        self.root = Path(root)

        self.images = sorted(list((self.root / self.mode / "images").glob("*")))
        self.labels = sorted(list((self.root / self.mode / "labels").glob("*")))

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def perform_image_operation_train_and_val(self, **kwargs):
        raise NotImplementedError

    def perform_image_operation_test(self, **kwargs):
        raise NotImplementedError

    def get_label_normalization(self, **kwargs):
        raise NotImplementedError

    def load_transformation(self, transformation_param):
        try:
            transform_type = list(transformation_param.keys())[0]
            transformation_to_perform = list(transformation_param.values())[0]
            number_of_transformation = len(list(transformation_param.values())[0])

            transformation_to_applied = list()
            for i in range(number_of_transformation):
                for _, transform_param in transformation_to_perform.items():
                    transformation_to_applied.append(
                        self._get_train_transformation(**transform_param)
                    )
            if number_of_transformation == 1:
                transformation = transformation_to_applied[0]
            else:
                transformation = getattr(data_transformation, transform_type)(
                    transformation_to_applied, prob=0.5
                )
            return transformation

        except Exception as ex:
            print_exception(
                exception=str(ex),
                error_name="Configuration",
                error_message="Configuring Transformation failed",
            )
            sys.exit(1)

    @staticmethod
    def _get_train_transformation(to_perform, transform_type, augment_prob):
        transformation = []
        transforms_type = getattr(data_transformation, transform_type)

        for trans in to_perform:
            transformation.append(
                getattr(data_transformation, trans)(prob=augment_prob)
            )

        train_transformation = transforms_type(transformation)
        return train_transformation

    @staticmethod
    def load_normalization(normalization_name):
        try:
            normalization = getattr(data_normalization, normalization_name)()
            return normalization

        except Exception as ex:
            print_exception(
                exception=str(ex),
                error_name="Configuration",
                error_message="Configuring normalization failed",
            )
            sys.exit(1)

    @staticmethod
    def crop_image(
        img: np.ndarray, model_input_dimension: tuple, random_crop_coord: tuple
    ):
        model_height, model_width = model_input_dimension
        height, width = random_crop_coord

        img = img[height : height + model_height, width : width + model_width]

        return img

    @staticmethod
    def get_random_crop_x_and_y(
        model_input_dimension: tuple, image_input_dimension: tuple
    ):
        model_height, model_width = model_input_dimension
        image_height, image_width, _ = image_input_dimension
        h_start = np.random.randint(0, image_height - model_height)
        w_start = np.random.randint(0, image_width - model_height)

        return h_start, w_start

    @staticmethod
    def get_pad_limit(model_input_dimension: tuple, image_input_dimension: tuple):
        model_height, model_width = model_input_dimension
        image_height, image_width, _ = image_input_dimension

        limit = (model_height - image_height) // 2
        return limit

    @staticmethod
    def pad_image(img: np.ndarray, limit: int):
        img = cv2.copyMakeBorder(
            img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101
        )
        return img

    def handle_image_size(self, img, mask, model_input_dimension):
        if model_input_dimension < (img.shape[0], img.shape[1]):
            height, width = self.get_random_crop_x_and_y(
                model_input_dimension, img.shape
            )
            img = self.crop_image(img, model_input_dimension, (height, width))
            mask = self.crop_image(mask, model_input_dimension, (height, width))
            return img, mask

        elif model_input_dimension > (img.shape[0], img.shape[1]):
            limit = self.get_pad_limit(model_input_dimension, img.shape)
            img = self.pad_image(img, limit)
            mask = self.pad_image(mask, limit)
            return img, mask
        else:
            return img, mask

    def perform_normalization(self, img):
        img = self.normalization(img)
        return img

    def perform_transformation(self, img, mask):
        if self.mode == "train":
            img, mask = self.transform(img, mask)
        return img, mask
