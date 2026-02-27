"""Threshold optimization for shadow price classification pipeline."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 0.7,
    scaling_factor: float = 1.0,
) -> tuple[float, float]:
    """Find optimal classification threshold using F-beta score.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
    beta : float
        Beta parameter for F-beta score (0.7 = moderate recall/precision balance).
    scaling_factor : float
        Scale factor applied to the optimal threshold.

    Returns
    -------
    threshold : float
        Optimal threshold (scaled).
    max_fbeta : float
        Maximum F-beta score achieved.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns arrays where the last element of precision/recall
    # has no corresponding threshold, so we drop it
    precision = precision[:-1]
    recall = recall[:-1]

    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    fbeta = np.nan_to_num(fbeta, nan=0.0)

    if len(fbeta) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(fbeta)
    return float(thresholds[best_idx] * scaling_factor), float(fbeta[best_idx])


def apply_threshold(y_proba, threshold: float) -> np.ndarray:
    """Apply threshold to convert probabilities to binary predictions.

    Parameters
    ----------
    y_proba : array-like
        Predicted probabilities.
    threshold : float
        Classification threshold.

    Returns
    -------
    y_pred : np.ndarray
        Binary predictions (1 if proba > threshold, 0 otherwise).
    """
    return (np.asarray(y_proba) > threshold).astype(int)
