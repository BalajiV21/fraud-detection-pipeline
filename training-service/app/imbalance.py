"""Class-imbalance utilities."""
import numpy as np
from sklearn.metrics import precision_recall_curve


def compute_scale_pos_weight(y) -> float:
    """For XGBoost/LightGBM. Ratio of negative to positive."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def find_threshold_for_recall(y_true, y_proba, target_recall: float = 0.80) -> dict:
    """Find the lowest probability threshold that still achieves target_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has len = len(precisions) - 1
    eligible = np.where(recalls[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        # fall back to threshold maximizing F1
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        idx = int(np.argmax(f1[:-1]))
    else:
        idx = int(eligible[-1])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precisions[idx]),
        "recall": float(recalls[idx]),
    }


def find_threshold_for_precision(y_true, y_proba, target_precision: float = 0.80) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    eligible = np.where(precisions[:-1] >= target_precision)[0]
    if len(eligible) == 0:
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        idx = int(np.argmax(f1[:-1]))
    else:
        idx = int(eligible[0])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precisions[idx]),
        "recall": float(recalls[idx]),
    }
