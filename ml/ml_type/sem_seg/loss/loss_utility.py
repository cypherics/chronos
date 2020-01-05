import torch

from torch.nn import functional as F

from utils.torch_tensor_conversion import cuda_variable


def calculate_jaccard_binary(outputs, targets):
    eps = 1e-15
    jaccard_target = targets
    jaccard_output = outputs

    intersection = (jaccard_output * jaccard_target).sum()
    union = jaccard_output.sum() + jaccard_target.sum()

    jaccard = torch.log((intersection + eps) / (union - intersection + eps))
    return jaccard


def calculate_dice_binary(outputs, targets):
    dice = (2.0 * (outputs * targets).sum() + 1) / (outputs.sum() + targets.sum() + 1)
    return dice


def get_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    v_scores = scores[valid]
    v_labels = labels[valid]

    return v_scores, v_labels


def hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * cuda_variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = get_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), cuda_variable(grad))

    return loss


def calculate_lovasz_hinge(output, targets, ignore=255):
    loss = hinge_flat(*binary_scores(output, targets, ignore))
    return loss
