import os

from typing import Any

import numpy as np
from pathlib import Path

import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from core import augmentator
from abc import ABCMeta, abstractmethod

from core.logger import info
from utils.dict_ops import handle_dictionary
from utils.image_ops import handle_image_size, load_image
from utils.pt_tensor import to_input_image_tensor


@dataclass
class Data:
    train_data: Any
    val_data: Any
    test_data: Any


class BaseDataSetPt(Dataset, metaclass=ABCMeta):
    def __init__(self, config, mode):
        self.config = config
        root = self.config.root
        model_input_dim = self.config.model_input_dimension
        transform = self.config.transformation

        if mode == "train":
            self.transform = self.load_transformation(transform)
        else:
            self.transform = None

        self.mode = mode
        self.model_input_dimension = tuple(model_input_dim)

        self.root = Path(root)

        self.images = sorted(list((self.root / self.mode / "images").glob("*")))
        self.labels = sorted(list((self.root / self.mode / "labels").glob("*")))

    @classmethod
    def get_data_loader(cls, config):
        train_data = DataLoader(
            dataset=cls(config, "train"),
            shuffle=True,
            num_workers=0,
            batch_size=config.batch_size,
            pin_memory=torch.cuda.is_available(),
        )
        val_data = DataLoader(
            dataset=cls(config, "val"),
            shuffle=True,
            num_workers=0,
            batch_size=config.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        test_data = DataLoader(
            dataset=cls(config, "test"),
            shuffle=True,
            num_workers=0,
            batch_size=config.batch_size,
            pin_memory=torch.cuda.is_available(),
        )
        return Data(train_data, val_data, test_data)

    def __len__(self):
        if len(self.images) != 0:
            return len(self.images)
        else:
            return len(self.labels)

    def __getitem__(self, idx):

        if self.mode in ["train", "val"]:
            img, _ = self.read_data(idx, self.images)
            mask, _ = self.read_data(idx, self.labels)

            images, ground_truth = self.learner_data(img=img, mask=mask)
            assert isinstance(images, dict), "Return type should be dict"

            return images, ground_truth

        elif self.mode == "test":
            img, file_name = self.read_data(idx, self.images)
            images = self.evaluator_data(img=img)
            assert isinstance(images, dict), "Return type should be dict"
            return images, str(file_name)
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

    def transform_image(self, img, mask):
        if self.mode == "train":
            img, mask = self.transform(img, mask)
        return img, mask

    @staticmethod
    def read_data(idx, data_list):
        if len(data_list) != 0:
            image_file_name = data_list[idx]
            image = load_image(str(image_file_name))
            return image, image_file_name
        else:
            return None, None

    def learner_data(self, img, mask):
        ground_truth = dict()
        images = dict()
        data = self.adjust_learner_data(img, mask, self.model_input_dimension)
        for individual_data in data:
            keys = list(individual_data.keys())
            img = individual_data[keys[0]]
            mask = individual_data[keys[1]]
            assert len(mask.shape) == len(
                img.shape
            ), "Image and mask should have same Tensor dimension"

            img, mask = self.transform_image(img, mask)
            img = self.normalize_image(img)
            mask = self.normalize_label(mask=mask)

            assert len(mask.shape) == len(
                img.shape
            ), "Image and mask should have same Tensor dimension"

            images = handle_dictionary(images, keys[0], to_input_image_tensor(img))
            ground_truth = handle_dictionary(
                ground_truth, keys[1], to_input_image_tensor(mask)
            )
        return images, ground_truth

    def evaluator_data(self, img):
        images = dict()
        data = self.adjust_evaluator_data(img, self.model_input_dimension)
        for individual_data in data:
            keys = list(individual_data.keys())
            img = individual_data[keys[0]]
            img = self.normalize_image(img)
            images = handle_dictionary(images, keys[0], to_input_image_tensor(img))
        return images

    @staticmethod
    def adjust_learner_data(img, mask, dimension) -> [dict]:
        img, mask = handle_image_size(img, mask, dimension)
        return [{"image": img, "label": mask}]

    @staticmethod
    def adjust_evaluator_data(img, dimension) -> [dict]:
        img, _ = handle_image_size(img, None, dimension)
        return [{"image": img}]

    @abstractmethod
    def normalize_label(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def normalize_image(self, img) -> np.ndarray:
        raise NotImplementedError
