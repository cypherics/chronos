import torch
from torch import nn

from ml.ml_type.base import BaseLoss


def calculate_iou(outputs, targets):
    eps = 1e-15
    iou_target = targets
    iou_output = outputs

    intersection = (iou_output * iou_target).sum()
    union = iou_output.sum() + iou_target.sum()

    iou = torch.log((intersection + eps) / (union - intersection + eps))
    return iou


class BuildingBoundarySigmoidLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.building_weight = kwargs["building_weight"]
        self.boundary_weight = kwargs["boundary_weight"]

    def compute_loss(self, outputs, **kwargs):
        targets = kwargs["label"]
        outputs = outputs.sigmoid()
        loss = (1 - self.building_weight) * self.nll_loss(outputs[:, 0:1, :, :], targets[:, 0:1, :, :])
        loss += ((1 - self.boundary_weight) * self.nll_loss(outputs[:, 1:2, :, :], targets[:, 1:2, :, :]))

        temp_targets = (targets[:, 0:1, :, :] == 1).float()
        loss -= self.building_weight * calculate_iou(
            outputs[:, 0:1, :, :], temp_targets
        )

        temp_targets = (targets[:, 1:2, :, :] == 1).float()
        loss -= self.boundary_weight * calculate_iou(
            outputs[:, 1:2, :, :], temp_targets
        )

        return loss
