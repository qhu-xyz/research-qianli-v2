"""Evaluation harness for tier classification pipeline.

HUMAN-WRITE-ONLY — workers MUST NOT modify this file.

Group A (blocking): Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK.
Group B (monitor): Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1, Value-QWK.

All ranking metrics use tier_ev_score for ranking and actual_shadow_price
as relevance — preserving the downstream capital allocation objective.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

# No lower-is-better metrics in tier pipeline.
_LOWER_IS_BETTER: set[str] = set()


def _value_capture_at_k(
    actual: np.ndarray,
    ev_scores: np.ndarray,
    k: int,
) -> float:
    """Fraction of total actual value captured by top-k EV scores.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth shadow prices.
    ev_scores : np.ndarray
        Ranking scores (tier_ev_score).
    k : int
        Number of top positions to consider.

    Returns
    -------
    float
        Value capture ratio in [0, 1]. Returns 0 if total value <= 0.
    """
    total_value = actual.sum()
    if total_value <= 0:
        return 0.0
    k = min(k, len(ev_scores))
    top_k_idx = np.argsort(ev_scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total_value)


def _ndcg(actual: np.ndarray, ev_scores: np.ndarray) -> float:
    """Compute NDCG using actual_shadow_price as relevance, ranked by ev_scores.

    DCG  = sum(relevance_i / log2(i + 2))  for i = 0, 1, ...
    NDCG = DCG / ideal_DCG
    """
    n = len(actual)
    discounts = np.log2(np.arange(2, n + 2))

    ranked_idx = np.argsort(ev_scores)[::-1]
    dcg = float((actual[ranked_idx] / discounts).sum())

    ideal_idx = np.argsort(actual)[::-1]
    ideal_dcg = float((actual[ideal_idx] / discounts).sum())

    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def _quadratic_weighted_kappa(
    actual: np.ndarray,
    pred: np.ndarray,
    num_classes: int = 5,
) -> float:
    """Cohen's Quadratic Weighted Kappa for ordinal classification.

    Penalizes large tier mismatches quadratically. A perfect agreement
    returns 1.0; random agreement returns ~0; systematic disagreement < 0.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth tier labels in {0, 1, 2, 3, 4}.
    pred : np.ndarray
        Predicted tier labels in {0, 1, 2, 3, 4}.
    num_classes : int
        Number of ordinal categories.

    Returns
    -------
    float
        QWK score in (-inf, 1].
    """
    # Confusion matrix
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, p in zip(actual, pred):
        O[int(a), int(p)] += 1

    n = len(actual)
    if n == 0:
        return 0.0

    O = O / n

    # Expected (outer product of marginals)
    row_sums = O.sum(axis=1)
    col_sums = O.sum(axis=0)
    E = np.outer(row_sums, col_sums)

    # Quadratic weight matrix: w[i,j] = (i-j)^2 / (num_classes-1)^2
    indices = np.arange(num_classes)
    W = (indices[:, None] - indices[None, :]) ** 2
    W = W.astype(np.float64) / (num_classes - 1) ** 2

    numerator = (W * O).sum()
    denominator = (W * E).sum()

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(1.0 - numerator / denominator)


def _value_weighted_qwk(
    actual: np.ndarray,
    pred: np.ndarray,
    midpoints: list[float],
    num_classes: int = 5,
) -> float:
    """Value-Weighted Quadratic Weighted Kappa.

    Like standard QWK but each row (actual tier) of the confusion matrix is
    weighted by the tier's midpoint value.  Tier 0 ($4000) misclassifications
    are penalized ~80x more than tier 3 ($50) misclassifications, reflecting
    the capital allocation importance of high-value constraints.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth tier labels in {0, 1, 2, 3, 4}.
    pred : np.ndarray
        Predicted tier labels in {0, 1, 2, 3, 4}.
    midpoints : list[float]
        Tier midpoints, e.g. [4000, 2000, 550, 50, 0] for tiers 0-4.
    num_classes : int
        Number of ordinal categories.

    Returns
    -------
    float
        Value-weighted QWK score in (-inf, 1].
    """
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, p in zip(actual, pred):
        O[int(a), int(p)] += 1

    n = len(actual)
    if n == 0:
        return 0.0

    O = O / n

    row_sums = O.sum(axis=1)
    col_sums = O.sum(axis=0)
    E = np.outer(row_sums, col_sums)

    # Quadratic distance matrix
    indices = np.arange(num_classes)
    W = (indices[:, None] - indices[None, :]) ** 2
    W = W.astype(np.float64) / (num_classes - 1) ** 2

    # Value weighting: scale each row by tier midpoint
    mid = np.array(midpoints, dtype=np.float64)
    total_mid = mid.sum()
    if total_mid == 0:
        return 0.0
    row_weight = mid / total_mid  # normalize so weights sum to 1
    V = row_weight[:, None] * W  # broadcast: (num_classes, 1) * (num_classes, num_classes)

    numerator = (V * O).sum()
    denominator = (V * E).sum()

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(1.0 - numerator / denominator)


def _tier_recall(
    actual_tier: np.ndarray,
    pred_tier: np.ndarray,
    target_tier: int,
) -> float:
    """Recall for a specific tier.

    Returns
    -------
    float
        Recall in [0, 1]. Returns 0 if no samples of the target tier exist.
    """
    mask = actual_tier == target_tier
    n_actual = mask.sum()
    if n_actual == 0:
        return 0.0
    return float((pred_tier[mask] == target_tier).sum() / n_actual)


def evaluate_tier_pipeline(
    actual_shadow_price: np.ndarray,
    actual_tier: np.ndarray,
    pred_tier: np.ndarray,
    tier_proba: np.ndarray,
    tier_ev_score: np.ndarray,
    tier_midpoints: list[float] | None = None,
) -> dict:
    """Compute all tier classification metrics.

    Parameters
    ----------
    actual_shadow_price : np.ndarray
        Ground-truth shadow prices (continuous, for ranking evaluation).
    actual_tier : np.ndarray
        Ground-truth tier labels in {0, 1, 2, 3, 4}.
    pred_tier : np.ndarray
        Predicted tier labels in {0, 1, 2, 3, 4}.
    tier_proba : np.ndarray
        Predicted tier probabilities, shape ``(n_samples, 5)``.
    tier_ev_score : np.ndarray
        Probability-weighted expected shadow price for ranking.
    tier_midpoints : list[float], optional
        Tier midpoints for Value-QWK. Defaults to [4000, 2000, 550, 50, 0].

    Returns
    -------
    dict
        All tier metrics: Group A (blocking) + Group B (monitor) + monitoring.
    """
    n = len(actual_tier)

    # --- Group A: blocking gates ---
    tier_vc_100 = _value_capture_at_k(actual_shadow_price, tier_ev_score, 100)
    tier_vc_500 = _value_capture_at_k(actual_shadow_price, tier_ev_score, 500)
    tier_ndcg = _ndcg(actual_shadow_price, tier_ev_score)
    qwk = _quadratic_weighted_kappa(actual_tier, pred_tier)

    # --- Group B: monitor ---
    macro_f1 = float(f1_score(
        actual_tier, pred_tier, average="macro", zero_division=0,
    ))
    tier_accuracy = float(np.mean(actual_tier == pred_tier)) if n > 0 else 0.0
    adjacent_accuracy = (
        float(np.mean(np.abs(actual_tier.astype(int) - pred_tier.astype(int)) <= 1))
        if n > 0 else 0.0
    )
    tier_recall_0 = _tier_recall(actual_tier, pred_tier, 0)
    tier_recall_1 = _tier_recall(actual_tier, pred_tier, 1)

    if tier_midpoints is None:
        tier_midpoints = [4000, 2000, 550, 50, 0]
    value_qwk = _value_weighted_qwk(actual_tier, pred_tier, tier_midpoints)

    return {
        # Group A (blocking)
        "Tier-VC@100": tier_vc_100,
        "Tier-VC@500": tier_vc_500,
        "Tier-NDCG": tier_ndcg,
        "QWK": qwk,
        # Group B (monitor)
        "Macro-F1": macro_f1,
        "Tier-Accuracy": tier_accuracy,
        "Adjacent-Accuracy": adjacent_accuracy,
        "Tier-Recall@0": tier_recall_0,
        "Tier-Recall@1": tier_recall_1,
        "Value-QWK": value_qwk,
        # Monitoring
        "n_samples": n,
        "n_binding": int((actual_tier < 4).sum()),
        "binding_rate": float((actual_tier < 4).sum() / n) if n > 0 else 0.0,
    }


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into summary statistics.

    Parameters
    ----------
    per_month : dict[str, dict]
        Mapping of month strings to metric dicts.

    Returns
    -------
    dict
        Dictionary with keys "mean", "std", "min", "max", "bottom_2_mean".
        For all tier metrics, bottom_2_mean uses the 2 lowest values
        (worst months) since all are higher-is-better.
    """
    months = list(per_month.keys())
    if not months:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    metric_names = list(per_month[months[0]].keys())

    result: dict[str, dict] = {
        "mean": {},
        "std": {},
        "min": {},
        "max": {},
        "bottom_2_mean": {},
    }

    for metric in metric_names:
        values = [per_month[m][metric] for m in months]

        # Skip non-numeric metrics
        if not all(isinstance(v, (int, float)) for v in values):
            continue

        arr = np.array(values)
        result["mean"][metric] = float(np.mean(arr))
        result["std"][metric] = float(np.std(arr, ddof=0))
        result["min"][metric] = float(np.min(arr))
        result["max"][metric] = float(np.max(arr))

        sorted_values = np.sort(arr)
        if metric in _LOWER_IS_BETTER:
            worst_2 = sorted_values[-2:] if len(sorted_values) >= 2 else sorted_values
        else:
            worst_2 = sorted_values[:2] if len(sorted_values) >= 2 else sorted_values
        result["bottom_2_mean"][metric] = float(np.mean(worst_2))

    return result
