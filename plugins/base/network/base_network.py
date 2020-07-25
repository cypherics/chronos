import torch
from torch import nn
from abc import abstractmethod

from utils.network_util import adjust_model


class BaseNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return self.forward_propagate(x)

    @abstractmethod
    def forward_propagate(self, x) -> dict:
        pass

    def load_pre_trained(self, pre_trained):
        pre_trained_dict = torch.load(pre_trained)
        model_dict = self.state_dict()

        pre_trained_dict = adjust_model(pre_trained_dict)
        model_dict.update(pre_trained_dict)
        self.load_state_dict(model_dict)
