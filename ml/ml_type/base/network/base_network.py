from torch import nn
from abc import abstractmethod


class BaseNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return self.forward_propagate(x)

    @abstractmethod
    def forward_propagate(self, x):
        pass

