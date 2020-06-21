import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from core.utils.image_util import (
    get_pad_limit,
    pad_image,
    get_random_crop_x_and_y,
    crop_image,
)
from core.data import augmentator, normalizer
from abc import ABCMeta, abstractmethod

from core.logger import info


class BaseDataSetPt(Dataset, metaclass=ABCMeta):
    def __init__(self, config, mode):
        self.config = config
        root = self.config.root
        model_input_dim = self.config.model_input_dimension
        normalization = self.config.normalization
        transform = self.config.transformation

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

    @classmethod
    def get_train_data(cls, training_configuration):
        return DataLoader(
            dataset=cls(training_configuration, "train"),
            shuffle=True,
            num_workers=0,
            batch_size=training_configuration.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def get_val_data(cls, training_configuration):
        return DataLoader(
            dataset=cls(training_configuration, "val"),
            shuffle=True,
            num_workers=0,
            batch_size=training_configuration.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def get_test_data(cls, training_configuration):
        return DataLoader(
            dataset=cls(training_configuration, "test"),
            shuffle=True,
            num_workers=0,
            batch_size=training_configuration.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    def __len__(self):
        if len(self.images) != 0:
            return len(self.images)
        else:
            return len(self.labels)

    def __getitem__(self, idx):

        if self.mode in ["train", "val"]:
            img, _ = self.read_data(idx, self.images)
            mask, _ = self.read_data(idx, self.labels)

            input_dictionary = self.perform_image_operation_train_and_val(
                img=img, mask=mask
            )
            assert isinstance(input_dictionary, dict), "Return type should be dict"

            assert (
                "image" in input_dictionary and "label" in input_dictionary
            ), "while passing image use key-image and for label use key-label"

            return input_dictionary

        elif self.mode == "test":
            img, file_name = self.read_data(idx, self.images)
            input_dictionary = self.perform_image_operation_test(img=img)
            assert isinstance(input_dictionary, dict), "Return type should be dict"
            assert "image" in input_dictionary, "while passing image use key-image"

            return input_dictionary, str(file_name)
        else:
            raise NotImplementedError

    @info
    def load_transformation(self, transformation_param):
        transform_type = list(transformation_param.keys())[0]
        transformation_to_perform = list(transformation_param.values())[0]
        number_of_transformation = len(list(transformation_param.values())[0])

        transformation_to_applied = list()
        for _, transform_param in transformation_to_perform.items():
            transformation_to_applied.append(
                self._get_train_transformation(**transform_param)
            )
        if number_of_transformation == 1:
            transformation = transformation_to_applied[0]
        else:
            transformation = getattr(augmentator, transform_type)(
                transformation_to_applied, prob=0.5
            )
        return transformation

    @staticmethod
    def _get_train_transformation(to_perform, transform_type, augment_prob):
        transformation = []
        transforms_type = getattr(augmentator, transform_type)

        for trans in to_perform:
            transformation.append(getattr(augmentator, trans)(prob=augment_prob))

        train_transformation = transforms_type(transformation)
        return train_transformation

    @staticmethod
    @info
    def load_normalization(normalization_name):
        normalization = getattr(normalizer, normalization_name)()
        return normalization

    @staticmethod
    def handle_image_size(img, mask, model_input_dimension):
        if model_input_dimension < (img.shape[0], img.shape[1]):
            height, width = get_random_crop_x_and_y(model_input_dimension, img.shape)
            img = crop_image(img, model_input_dimension, (height, width))
            if mask is not None:
                mask = crop_image(mask, model_input_dimension, (height, width))
            return img, mask

        elif model_input_dimension > (img.shape[0], img.shape[1]):
            limit = get_pad_limit(model_input_dimension, img.shape)
            img = pad_image(img, limit)
            if mask is not None:
                mask = pad_image(mask, limit)
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

    def read_data(self, idx, data_list):
        if len(data_list) != 0:
            image_file_name = data_list[idx]
            image = self.load_image(str(image_file_name))
            return image, image_file_name
        else:
            return None, None

    @staticmethod
    def load_image(path: str):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @abstractmethod
    def perform_image_operation_train_and_val(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def perform_image_operation_test(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_label_normalization(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
