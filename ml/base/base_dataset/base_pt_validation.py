import os
import shutil

import numpy as np
from scipy import misc
import torch

from torch import nn

from utils.directory_handler import make_directory
from utils.print_format import (
    colored_single_string_print_with_brackets,
    tabular_print_handler,
)
from ml.commons.utils.torch_tensor_conversion import cuda_variable

from abc import ABCMeta, abstractmethod


class BaseValidationPt(metaclass=ABCMeta):
    @abstractmethod
    def compute_metric(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_computed_mean_metric(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_inference_output(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def perform_validation(self, model: nn.Module, loss_function, valid_loader):
        model.eval()
        losses = []
        for input_data in valid_loader:
            input_data = cuda_variable(input_data)

            targets = input_data["label"]
            loss, outputs = self.get_valid_loss(model, loss_function, **input_data)
            losses.append(loss.item())

            # TODO make get_prediction_as_per_instance abstract
            outputs = self.get_prediction_as_per_instance(outputs)
            self.compute_metric(targets=targets, outputs=outputs)

        valid_loss = np.mean(losses)
        valid_loss = {"valid_loss": valid_loss}

        validation_metric = self.get_computed_mean_metric()
        metrics = {**valid_loss, **validation_metric}

        colored_single_string_print_with_brackets(
            "  Validation Metrics  ", "green", attrs=["bold"]
        )
        tabular_print_handler(metrics, headers=["Metric", "Value"])
        return metrics

    @torch.no_grad()
    def inference(self, model, iteration, epoch, test_loader, save_path):
        save_path = make_directory(save_path, "test_prediction")
        model.eval()
        for i, (inputs, file_path) in enumerate(test_loader):

            image = cuda_variable(inputs)
            prediction = model(image)

            shutil.rmtree(save_path)
            os.makedirs(save_path)

            # TODO make get_prediction_as_per_instance abstract
            prediction = self.get_prediction_as_per_instance(prediction)

            for iterator, files in enumerate(file_path):
                inference_output = self.generate_inference_output(
                    img=prediction[iterator]
                )
                self.save_inference_output(
                    inference_output, save_path, iteration, epoch, files
                )

            stack_img = self.create_prediction_grid(inputs, prediction)
            return stack_img

    @staticmethod
    def create_prediction_grid(inputs, prediction):
        raise NotImplementedError

    @staticmethod
    def save_inference_output(img, save_path, iteration, epoch, files):
        save_image_path = os.path.join(
            save_path,
            "{}_{}_{}.png".format(
                str(files).split(os.sep)[-1].split(".")[0], epoch, iteration
            ),
        )
        misc.imsave(save_image_path, img)

    @staticmethod
    def get_valid_loss(model: nn.Module, loss_function, **input_data):
        input_data = cuda_variable(input_data)
        outputs = model(input_data)
        loss = loss_function(outputs, **input_data)
        return loss, outputs

    @staticmethod
    def get_prediction_as_per_instance(outputs):
        if isinstance(outputs, dict):
            assert "final_image" in outputs, "while passing image use key-final_image"
            return outputs["final_image"]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise NotImplementedError
