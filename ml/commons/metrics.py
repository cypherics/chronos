import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target
from torch import Tensor

from ml.commons.utils.tensor_util import convert_tensor_to_numpy
from ml.pt.logger import PtLogger


def get_numpy(ip):
    if type(ip) == Tensor:
        return convert_tensor_to_numpy(ip)
    elif type(ip) == np.ndarray:
        return ip


@PtLogger(debug=True)
def to_binary(prediction, cutoff=0.40):
    prediction[prediction >= cutoff] = 1
    prediction[prediction < cutoff] = 0
    return prediction


@PtLogger(debug=True)
def confusion_matrix_elements(mat):
    tp = mat[0][0]
    fp = mat[0][1]
    fn = mat[1][0]
    tn = mat[1][1]
    return tp, fp, fn, tn


@PtLogger(debug=True)
def compute_metric(prediction, ground_truth):
    prediction = get_numpy(prediction).flatten()
    ground_truth = get_numpy(ground_truth).flatten()
    if type_of_target(prediction) == "continuous":
        prediction = to_binary(prediction)
    tp, fp, fn, tn = confusion_matrix_elements(
        confusion_matrix(ground_truth, prediction)
    )
    metrics = {
        "Accuracy": accuracy(tp, fp, fn, tn),
        "F1": f1_score(tp, fp, fn, tn),
        "Precision": precision(tp, fp, fn, tn),
        "Recall": recall(tp, fp, fn, tn),
        "IOU": iou(tp, fp, fn, tn),
    }
    return metrics


@PtLogger(debug=True)
def accuracy(tp, fp, fn, tn):
    num = tp + tn
    den = tp + tn + fp + fn
    return num / den


@PtLogger(debug=True)
def f1_score(tp, fp, fn, tn):
    num = 2 * tp
    den = (2 * tp) + fp + fn
    return num / den


@PtLogger(debug=True)
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


@PtLogger(debug=True)
def recall(tp, fp, fn, tn):
    return tp / (tp + fn)


@PtLogger(debug=True)
def iou(tp, fp, fn, tn):
    denominator = tp + fp + fn
    if denominator == 0:
        value = 0
    else:
        value = float(tp) / denominator
    return value
