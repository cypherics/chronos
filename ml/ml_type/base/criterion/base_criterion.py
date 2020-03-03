from abc import abstractmethod


class BaseCriterion:
    def __init__(self, **kwargs):
        pass

    def __call__(self, outputs, **kwargs):
        return self.compute_criterion(outputs, **kwargs)

    @abstractmethod
    def compute_criterion(self, outputs, **kwargs):
        pass
