import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix
)

def find_best_threshold(errors, y_true):
    """Youden’s J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, errors)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    return thresholds[best_idx]


def evaluate_ae(errors, y_true, threshold):
    preds = (errors >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, output_dict=True)

    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, errors)
    ap = average_precision_score(y_true, errors)

    return preds, cm, report, roc_auc, ap
