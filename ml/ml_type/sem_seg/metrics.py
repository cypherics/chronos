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
