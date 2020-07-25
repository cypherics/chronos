from plugins.base.criterion.base_criterion import BaseCriterion

from torch import nn


def calculate_dice(outputs, targets):
    dice = (2.0 * (outputs * targets).sum() + 1) / (outputs.sum() + targets.sum() + 1)
    return dice


class Dice(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weights = kwargs["dice_weight"]

    def compute_criterion(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]

        bce_loss = self.nll_loss(prediction, ground_truth)

        outputs = prediction.sigmoid()
        dice_loss = calculate_dice(outputs, ground_truth)
        loss = bce_loss + self.dice_weights * (1 - dice_loss)
        return loss
