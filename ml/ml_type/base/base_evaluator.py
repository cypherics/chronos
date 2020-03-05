import os
import shutil

import numpy as np
from scipy import misc
import torch

from torch import nn
from torchvision.utils import make_grid

from utils.dictionary_set import set_key
from utils.directory_handler import make_directory
from ml.commons.utils.torch_tensor_conversion import cuda_variable

from abc import ABCMeta, abstractmethod


class BaseEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def compute_metric(self, **kwargs) -> dict:
        raise NotImplementedError

    @staticmethod
    def compute_mean_metric(metric: dict):
        mean_metric = dict()
        for key, value in metric.items():
            assert type(value) is list
            mean_value = np.mean(value)
            mean_metric = set_key(mean_metric, key, mean_value)
        return mean_metric

    @torch.no_grad()
    def perform_validation(self, model: nn.Module, loss_function, valid_loader):
        model.eval()
        losses = []
        metric = dict()
        for input_data in valid_loader:
            input_data = cuda_variable(input_data)

            targets = input_data["label"]
            loss, outputs = self.get_valid_loss(model, loss_function, **input_data)
            losses.append(loss.item())

            outputs = self.get_prediction_as_per_instance(outputs)
            met = self.compute_metric(targets=targets, outputs=outputs)
            if met is not None:
                for key, value in met.items():
                    metric = set_key(metric, key, value)

        valid_loss = np.mean(losses)
        valid_loss = {"valid_loss": valid_loss}

        validation_metric = self.compute_mean_metric(metric)
        metrics = {**valid_loss, **validation_metric}
        return metrics

    @torch.no_grad()
    def perform_test(self, model, test_loader):
        model.eval()
        for i, (inputs, file_path) in enumerate(test_loader):

            image = cuda_variable(inputs)
            prediction = model(image)

            prediction = self.handle_prediction(prediction)
            stack_img = self.create_prediction_grid(inputs, prediction)

            return stack_img

    def handle_prediction(self, prediction):
        prediction = self.get_prediction_as_per_instance(prediction)
        prediction = self.generate_image(prediction)
        return prediction

    @abstractmethod
    def generate_image(self, prediction):
        raise NotImplementedError

    @staticmethod
    def create_prediction_grid(inputs, prediction):
        display_image = cuda_variable(inputs)
        display_image = display_image["image"].cpu()
        grid = make_grid(prediction, nrow=2, normalize=True)
        nda = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        grid_sat = make_grid(display_image, nrow=2, normalize=True)
        grid_sat_nda = (
            grid_sat.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )

        return np.vstack((nda, grid_sat_nda))

    @staticmethod
    def save_inference_output(img, save_path, iteration, epoch):
        save_path = make_directory(save_path, "test_prediction")

        shutil.rmtree(save_path)
        os.makedirs(save_path)

        save_image_path = os.path.join(save_path, "{}_{}.png".format(epoch, iteration))
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
