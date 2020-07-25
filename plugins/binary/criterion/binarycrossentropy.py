from plugins.base.criterion.base_criterion import BaseCriterion

from torch import nn


class BinaryCrossEntropy(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()

    def compute_criterion(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]

        loss = self.nll_loss(prediction, ground_truth)
        return loss
