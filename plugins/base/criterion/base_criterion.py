from abc import abstractmethod


class BaseCriterion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, ground_truth: dict, prediction: dict):
        return self.compute_criterion(ground_truth, prediction)

    @abstractmethod
    def compute_criterion(self, ground_truth: dict, prediction: dict):
        pass
