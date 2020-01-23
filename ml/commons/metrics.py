import numpy as np


def iou(true_values, predicted_values):
    epsilon = 1e-15
    intersection = (predicted_values * true_values).sum(dim=-2).sum(dim=-1)
    union = true_values.sum(dim=-2).sum(dim=-1) + predicted_values.sum(dim=-2).sum(
        dim=-1
    )

    return (intersection / (union - intersection + epsilon)).mean()


def precision(true_values, predicted_values):
    true_positives = (predicted_values * true_values).sum(dim=-2).sum(dim=-1)
    positives_predicted = predicted_values.sum(dim=-2).sum(dim=-1)
    return (true_positives / (positives_predicted + 1e-8)).mean()


def recall(true_values, predicted_values):
    true_positives = (predicted_values * true_values).sum(dim=-2).sum(dim=-1)
    positives_predicted = true_values.sum(dim=-2).sum(dim=-1)
    return (true_positives / (positives_predicted + 1e-8)).mean()


def f_score(true_values, predicted_values):
    precision_metric = precision(true_values, predicted_values)
    recall_metric = recall(true_values, predicted_values)
    f_score_metric = (2 * precision_metric * recall_metric) / (
        precision_metric + recall_metric + 1e-8
    )
    return f_score_metric


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten())).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)],
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou_on_confusion_matrix(confusion_matrix):
    inter_over_union_list = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denominator = true_positives + false_positives + false_negatives
        if denominator == 0:
            intersection_over_union = 0
        else:
            intersection_over_union = float(true_positives) / denominator
        inter_over_union_list.append(intersection_over_union)
    return inter_over_union_list
