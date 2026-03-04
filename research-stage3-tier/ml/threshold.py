"""F-beta threshold optimization for binary classification.

Searches over a grid of thresholds to find the one maximizing
the F-beta score, allowing precision/recall trade-off via beta.
"""

import numpy as np
from sklearn.metrics import fbeta_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 0.7,
    n_thresholds: int = 200,
) -> float:
    """Find the threshold that maximizes the F-beta score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    beta : float, optional
        Beta parameter for F-beta score. beta < 1 weights precision
        more; beta > 1 weights recall more. Default 0.7.
    n_thresholds : int, optional
        Number of thresholds to evaluate between 0.01 and 0.99.
        Default 200.

    Returns
    -------
    float
        The threshold that maximizes F-beta. Returns 0.5 if no
        threshold produces any non-zero predictions with non-zero
        F-beta score.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        # Skip if no positive predictions (fbeta_score would be 0)
        if y_pred.sum() == 0:
            continue
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0)
        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold)


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a threshold.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities.
    threshold : float
        Decision threshold. Values >= threshold become 1, else 0.

    Returns
    -------
    np.ndarray
        Binary predictions as integer array.
    """
    return (y_proba >= threshold).astype(int)
