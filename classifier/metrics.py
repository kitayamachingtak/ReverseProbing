#metrics: aucroc, auprc, F1
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def evaluate(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    both_classes = len(np.unique(y_true)) > 1
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auc": roc_auc_score(y_true, y_proba) if both_classes else 0.0,
        "auprc": average_precision_score(y_true, y_proba) if both_classes else 0.0,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }