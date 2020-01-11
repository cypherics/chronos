from abc import abstractmethod


class BaseLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, outputs, **kwargs):
        return self.compute_loss(outputs, **kwargs)

    @abstractmethod
    def compute_loss(self, outputs, **kwargs):
        pass
