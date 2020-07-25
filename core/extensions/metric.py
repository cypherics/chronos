import numpy as np
from torch import Tensor

from core.logger import ChronosLogger
from utils.dict_ops import handle_dictionary
from utils.pt_tensor import convert_tensor_to_numpy

logger = ChronosLogger.get_logger()


class MetricList:
    def __init__(self, metrics):
        metrics = metrics or []
        self.metrics = [c for c in metrics]
        if len(metrics) != 0:
            [
                logger.debug("Registered {}".format(c.__class__.__name__))
                for c in metrics
            ]
        self.metric_value = dict()

    def append(self, callback):
        logger.debug("Registered {}".format(callback.__class__.__name__))
        self.metrics.append(callback)

    def get_metrics(self, ground_truth: dict, prediction: dict):
        computed_metric = self.compute_metric(ground_truth, prediction)
        for key, value in computed_metric.items():
            self.metric_value = handle_dictionary(self.metric_value, key, value)

    def compute_metric(self, ground_truth: dict, prediction: dict):
        computed_metric = dict()
        for metric in self.metrics:
            value = metric.compute_metric(ground_truth, prediction)
            computed_metric[metric.__class__.__name__] = value
        return computed_metric

    def compute_mean(self):
        mean_metric = dict()
        for key, value in self.metric_value.items():
            assert type(value) is list
            mean_value = np.mean(value)
            mean_metric = handle_dictionary(mean_metric, key, mean_value)
        self.metric_value = dict()
        return mean_metric


class Metric:
    def compute_metric(self, ground_truth: dict, prediction: dict):
        raise NotImplementedError

    @staticmethod
    def get_numpy(ip):
        if type(ip) == Tensor:
            return convert_tensor_to_numpy(ip)
        elif type(ip) == np.ndarray:
            return ip
