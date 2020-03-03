from ml.commons.metrics import iou, precision, recall, f_score
from ..base.base_evaluator import BaseEvaluator


class BinaryEvaluator(BaseEvaluator):
    def compute_metric(self, targets, outputs):
        jaccard = iou(targets, (outputs > 0).float()).item()
        precision_metric = precision(targets, (outputs > 0).float()).item()
        recall_metric = recall(targets, (outputs > 0).float()).item()
        f_score_metric = f_score(targets, (outputs > 0).float()).item()
        return {
            "Jaccard": jaccard,
            "Precision": precision_metric,
            "Recall": recall_metric,
            "F_Score": f_score_metric,
        }

    def get_accuracy(self, true_values, predicted_values):
        raise NotImplementedError

    def generate_image(self, prediction):
        prediction = prediction.sigmoid()
        return prediction
