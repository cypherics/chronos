import torch

from plugins.base.criterion.base_criterion import BaseCriterion

from torch import nn


def calculate_iou(outputs, targets):
    eps = 1e-15
    iou_target = targets
    iou_output = outputs

    intersection = (iou_output * iou_target).sum()
    union = iou_output.sum() + iou_target.sum()

    iou = torch.log((intersection + eps) / (union - intersection + eps))
    return iou


class Jaccard(BaseCriterion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = kwargs["jaccard_weight"]
        self.binary_weight = 1 if self.jaccard_weight == 1 else 1 - self.jaccard_weight

    def compute_criterion(self, ground_truth: dict, prediction: dict):
        prediction = prediction["output"]
        ground_truth = ground_truth["label"]

        loss = self.binary_weight * self.nll_loss(prediction, ground_truth)

        if self.jaccard_weight:
            targets = (ground_truth == 1).float()
            outputs = prediction.sigmoid()
            loss -= self.jaccard_weight * calculate_iou(outputs, targets)
        return loss
