"""
Evaluation metrics utility module.
Provides functions to compute classification metrics for imbalanced datasets.
"""
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification with imbalanced target.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities P(Bankrupt=1).
        threshold: Classification decision threshold.

    Returns:
        Dictionary containing ROC-AUC, PR-AUC, Precision, Recall, F1, Specificity, and Confusion Matrix values.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_thresholds(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, and F1 across a range of classification thresholds.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive probabilities.
        thresholds: List of float thresholds (defaults to 0.05 to 0.95 in 0.05 steps).

    Returns:
        DataFrame summarizing metrics for each threshold.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    records = []
    for t in thresholds:
        m = compute_classification_metrics(y_true, y_prob, threshold=t)
        records.append(m)

    return pd.DataFrame(records)
