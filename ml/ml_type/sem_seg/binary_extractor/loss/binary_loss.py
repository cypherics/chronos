from torch import nn

from ml.ml_type.sem_seg.loss_utility import (
    calculate_iou,
    calculate_dice,
    calculate_lovasz_hinge,
)


class JaccardBinaryLoss:
    def __init__(self, jaccard_weight):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.binary_weight = 1 if self.jaccard_weight == 1 else 1 - self.jaccard_weight

    def __call__(self, outputs, **kwargs):
        targets = kwargs["label"]

        loss = self.binary_weight * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            targets = (targets == 1).float()
            outputs = outputs.sigmoid()
            loss -= self.jaccard_weight * calculate_iou(outputs, targets)
        return loss


class DiceBinaryLoss:
    """
    Loss defined as BCE + DICE
    """

    def __init__(self, dice_weight):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weights = dice_weight

    def __call__(self, outputs, **kwargs):
        targets = kwargs["label"]
        bce_loss = self.nll_loss(outputs, targets)

        outputs = outputs.sigmoid()
        dice_loss = calculate_dice(outputs, targets)
        loss = bce_loss + self.dice_weights * (1 - dice_loss)
        return loss


class WeightedBinaryLovaszLoss:
    def __init__(self, binary_weight=1, lovasz_weight=1):
        self.nll_loss = nn.BCEWithLogitsLoss()

        self.binary_weight = binary_weight
        self.lovasz_weight = lovasz_weight

    def __call__(self, outputs, **kwargs):
        targets = kwargs["label"]
        weight_map = kwargs["weight_map"] if "weight_map" in kwargs else None

        loss_binary = self.binary_weight * self.nll_loss(outputs, targets, weight_map)
        loss_lovasz = self.lovasz_weight * calculate_lovasz_hinge(outputs, targets)

        loss = loss_binary + loss_lovasz

        return loss
