import numpy as np
import torch

from torch import nn

from ml.commons.metrics import compute_metric
from ml.pt.logger import PtLogger
from utils.dictionary_set import handle_dictionary
from ml.commons.utils.tensor_util import cuda_variable

from abc import ABCMeta, abstractmethod

from utils.system_printer import SystemPrinter


class BaseEvaluator(metaclass=ABCMeta):
    @staticmethod
    @PtLogger(debug=True)
    def compute_mean_metric(metric: dict):
        mean_metric = dict()
        for key, value in metric.items():
            assert type(value) is list
            mean_value = np.mean(value)
            mean_metric = handle_dictionary(mean_metric, key, mean_value)
        return mean_metric

    @torch.no_grad()
    def perform_validation(self, model: nn.Module, loss_function, valid_loader):
        model.eval()
        losses = []
        metric = dict()
        ongoing_count = 1
        total_count = len(valid_loader)
        for input_data in valid_loader:
            SystemPrinter.dynamic_print(
                tag=str("Validation"), data="{}/{}".format(ongoing_count, total_count)
            )

            ongoing_count += 1
            input_data = cuda_variable(input_data)

            targets = input_data["label"]
            loss, outputs = self.get_valid_loss(model, loss_function, **input_data)
            losses.append(loss.item())

            outputs = self.get_prediction_as_per_instance(outputs)
            met = compute_metric(ground_truth=targets, prediction=outputs)
            if met is not None:
                for key, value in met.items():
                    metric = handle_dictionary(metric, key, value)

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
        prediction = self.classifier_activation(prediction)
        prediction = self.generate_image(prediction)
        return prediction

    @abstractmethod
    def generate_image(self, prediction):
        raise NotImplementedError

    @abstractmethod
    def classifier_activation(self, prediction):
        raise NotImplementedError

    @abstractmethod
    def create_prediction_grid(self, inputs, prediction):
        raise NotImplementedError

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
