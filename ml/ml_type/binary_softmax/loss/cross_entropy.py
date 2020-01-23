import torch.nn as nn

from ml.base import BaseLoss


class CrossEntropyLoss2d(BaseLoss):
    def __init__(self):
        super().__init__()

        self.loss = nn.NLLLoss()

    def compute_loss(self, outputs, **kwargs):
        targets = kwargs["label"]
        return self.loss(outputs.log_softmax(dim=1), targets)
