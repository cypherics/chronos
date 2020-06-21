from plugins.base.criterion.base_criterion import BaseCriterion

from torch import nn


class BinaryCrossEntropy(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()

    def compute_criterion(self, outputs, **kwargs):
        targets = kwargs["label"]

        loss = self.nll_loss(outputs, targets)
        return loss
