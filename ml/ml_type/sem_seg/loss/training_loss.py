import torch
from torch import nn

from ml.ml_type.sem_seg.loss.loss_utility import (
    calculate_jaccard_binary,
    calculate_dice_binary,
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
            loss -= self.jaccard_weight * calculate_jaccard_binary(outputs, targets)
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
        dice_loss = calculate_dice_binary(outputs, targets)
        loss = bce_loss + self.dice_weights * (1 - dice_loss)
        return loss


class BuildingBoundarySigmoidLoss:
    def __init__(self, jaccard_weight_building, jaccard_weight_boundary):
        self.soft_jaccard_loss_building = JaccardBinaryLoss(jaccard_weight_building)

        self.soft_jaccard_loss_boundary = JaccardBinaryLoss(jaccard_weight_boundary)

    def __call__(self, outputs, **kwargs):
        final_image = outputs

        targets = kwargs["label"]

        building_image_output = final_image[:, 0:1, :, :]
        boundary_image_output = final_image[:, 1:2, :, :]

        building_image_targets = targets[:, 0:1, :, :]
        boundary_image_targets = targets[:, 1:2, :, :]

        loss_final_image_boundary = self.soft_jaccard_loss_boundary(
            boundary_image_output, **{"label": boundary_image_targets}
        )
        loss_final_image_building = self.soft_jaccard_loss_boundary(
            building_image_output, **{"label": building_image_targets}
        )

        loss_final_image = loss_final_image_building + loss_final_image_boundary

        return loss_final_image


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


class SingleClassSoftmaxLoss:
    def __init__(self, weight):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, **kwargs):
        targets = kwargs["label"]
        loss = self.loss(outputs, targets)
        return loss


class SingleClassSoftmaxJaccardLoss:
    def __init__(self, jaccard_weight=1):
        self.loss = nn.CrossEntropyLoss()
        self.jaccard_weight = jaccard_weight
        self.binary_weight = 1 if self.jaccard_weight == 1 else 1 - self.jaccard_weight

    def __call__(self, outputs, **kwargs):
        targets = kwargs["label"]
        loss = self.binary_weight * self.loss(outputs, targets)

        outputs = outputs.softmax(dim=1).float()
        # outputs = torch.argmax(outputs, dim=1).unsqueeze(dim=1).float()
        # targets = targets.unsqueeze(dim=1).float()

        if self.jaccard_weight:
            for cls in range(2):
                jaccard_targets = (targets == cls).float()
                jaccard_outputs = outputs[:, cls]
                loss -= self.jaccard_weight * calculate_jaccard_binary(
                    jaccard_outputs, jaccard_targets
                )

        return loss


class BinaryDistanceBoundaryLoss:
    def __init__(self, building_weight=1, boundary_weight=1, distance_weight=1):
        self.building_weight = building_weight
        self.boundary_weight = boundary_weight
        self.distance_weight = distance_weight

        self.nll_loss = nn.BCEWithLogitsLoss()
        self.l2 = nn.MSELoss(reduction="sum")

    def __call__(self, outputs, **kwargs):
        target = kwargs["label"]

        target_building = target[:, 0:1, :, :]
        target_boundary = target[:, 1:2, :, :]
        target_distance = target[:, 2:3, :, :]

        output_building = outputs[:, 0:1, :, :]
        output_boundary = outputs[:, 1:2, :, :]
        output_distance = outputs[:, 2:3, :, :]

        output_building = output_building.sigmoid()
        output_boundary = output_boundary.sigmoid()
        output_distance = output_distance.tanh()

        building_dice = calculate_dice_binary(output_building, target_building)
        loss_building = self.nll_loss(output_building, target_building)

        loss_building = loss_building + building_dice

        boundary_dice = calculate_dice_binary(output_boundary, target_boundary)
        loss_boundary = self.nll_loss(output_boundary, target_boundary)

        loss_boundary += boundary_dice

        loss_distance = self.l2(output_distance, target_distance)

        loss = (
            (self.building_weight * loss_building)
            + (self.boundary_weight * loss_boundary)
            + (self.distance_weight * loss_distance)
        )
        return loss
