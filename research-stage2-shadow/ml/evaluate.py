"""EV-based evaluation harness (HUMAN-WRITE-ONLY).

Provides threshold-independent metrics for shadow-price prediction,
centred on Expected Value (EV) score ranking rather than binary
classification thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# Metrics where lower values are better (worst month = highest value).
_LOWER_IS_BETTER = {"C-RMSE", "C-MAE"}


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
        EV scores used for ranking.
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
    # If fewer than k samples, use all available
    k = min(k, len(ev_scores))
    top_k_idx = np.argsort(ev_scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total_value)


def _ndcg(actual: np.ndarray, ev_scores: np.ndarray) -> float:
    """Compute NDCG using actual_shadow_price as relevance, ranked by ev_scores.

    DCG  = sum(relevance_i / log2(i + 2))  for i = 0, 1, ...
    NDCG = DCG / ideal_DCG

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth shadow prices used as relevance scores.
    ev_scores : np.ndarray
        EV scores used for ranking.

    Returns
    -------
    float
        NDCG in [0, 1]. Returns 0 if ideal DCG is 0.
    """
    n = len(actual)
    discounts = np.log2(np.arange(2, n + 2))  # log2(i+2) for i = 0..n-1

    # DCG with ranking from ev_scores
    ranked_idx = np.argsort(ev_scores)[::-1]
    dcg = float((actual[ranked_idx] / discounts).sum())

    # Ideal DCG with perfect ranking by actual
    ideal_idx = np.argsort(actual)[::-1]
    ideal_dcg = float((actual[ideal_idx] / discounts).sum())

    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def _recall_at_k(
    actual: np.ndarray,
    ev_scores: np.ndarray,
    k: int,
) -> float:
    """Fraction of true binding constraints in top-k by EV score.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth shadow prices.
    ev_scores : np.ndarray
        EV scores used for ranking.
    k : int
        Number of top positions to consider.

    Returns
    -------
    float
        Recall in [0, 1]. Returns 0 if no binding samples.
    """
    binding_mask = actual > 0
    n_binding = binding_mask.sum()
    if n_binding == 0:
        return 0.0
    k = min(k, len(ev_scores))
    top_k_idx = np.argsort(ev_scores)[::-1][:k]
    top_k_set = set(top_k_idx)
    binding_in_top_k = sum(1 for i in np.where(binding_mask)[0] if i in top_k_set)
    return float(binding_in_top_k / n_binding)


def evaluate_pipeline(
    actual_shadow_price: np.ndarray,
    pred_proba: np.ndarray,
    pred_shadow_price: np.ndarray,
    ev_scores: np.ndarray,
) -> dict:
    """Compute EV-based, threshold-independent evaluation metrics.

    Parameters
    ----------
    actual_shadow_price : np.ndarray
        Ground-truth shadow prices.
    pred_proba : np.ndarray
        Predicted binding probabilities (unused for EV metrics but
        part of the pipeline interface).
    pred_shadow_price : np.ndarray
        Predicted shadow prices.
    ev_scores : np.ndarray
        Expected-value scores for ranking.

    Returns
    -------
    dict
        Dictionary with Group A (blocking), Group B (monitor), and
        monitoring metrics.
    """
    n = len(actual_shadow_price)
    binding_mask = actual_shadow_price > 0
    n_binding = int(binding_mask.sum())

    # --- Monitoring ---
    binding_rate = float(n_binding / n) if n > 0 else 0.0

    # --- Group A: EV-based, threshold-independent ---
    ev_vc_100 = _value_capture_at_k(actual_shadow_price, ev_scores, 100)
    ev_vc_500 = _value_capture_at_k(actual_shadow_price, ev_scores, 500)
    ev_ndcg = _ndcg(actual_shadow_price, ev_scores)

    if n_binding == 0:
        spearman_val = 0.0
    else:
        binding_actual = actual_shadow_price[binding_mask]
        binding_pred = pred_shadow_price[binding_mask]
        corr, _ = spearmanr(binding_actual, binding_pred)
        spearman_val = float(corr) if not np.isnan(corr) else 0.0

    # --- Group B: monitor ---
    if n_binding == 0:
        c_rmse = 0.0
        c_mae = 0.0
    else:
        binding_actual = actual_shadow_price[binding_mask]
        binding_pred = pred_shadow_price[binding_mask]
        residuals = binding_actual - binding_pred
        c_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        c_mae = float(np.mean(np.abs(residuals)))

    ev_vc_1000 = _value_capture_at_k(actual_shadow_price, ev_scores, 1000)
    r_rec_500 = _recall_at_k(actual_shadow_price, ev_scores, 500)

    return {
        # Group A (blocking)
        "EV-VC@100": ev_vc_100,
        "EV-VC@500": ev_vc_500,
        "EV-NDCG": ev_ndcg,
        "Spearman": spearman_val,
        # Group B (monitor)
        "C-RMSE": c_rmse,
        "C-MAE": c_mae,
        "EV-VC@1000": ev_vc_1000,
        "R-REC@500": r_rec_500,
        # Monitoring
        "binding_rate": binding_rate,
        "n_samples": n,
        "n_binding": n_binding,
    }


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into summary statistics.

    Parameters
    ----------
    per_month : dict[str, dict]
        Mapping of month strings (e.g. "2025-01") to metric dicts.

    Returns
    -------
    dict
        Dictionary with keys "mean", "std", "min", "max", "bottom_2_mean".

        For "lower is better" metrics (C-RMSE, C-MAE), bottom_2_mean
        uses the 2 *highest* values (worst months).
        For all other metrics, bottom_2_mean uses the 2 *lowest* values
        (worst months).
    """
    months = list(per_month.keys())
    if not months:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    # Collect all metric names from first month
    metric_names = list(per_month[months[0]].keys())

    result: dict[str, dict] = {
        "mean": {},
        "std": {},
        "min": {},
        "max": {},
        "bottom_2_mean": {},
    }

    for metric in metric_names:
        values = np.array([per_month[m][metric] for m in months])
        result["mean"][metric] = float(np.mean(values))
        result["std"][metric] = float(np.std(values, ddof=0))
        result["min"][metric] = float(np.min(values))
        result["max"][metric] = float(np.max(values))

        # bottom_2_mean: worst 2 months
        sorted_values = np.sort(values)
        if metric in _LOWER_IS_BETTER:
            # Lower is better -> worst = highest values
            worst_2 = sorted_values[-2:] if len(sorted_values) >= 2 else sorted_values
        else:
            # Higher is better -> worst = lowest values
            worst_2 = sorted_values[:2] if len(sorted_values) >= 2 else sorted_values
        result["bottom_2_mean"][metric] = float(np.mean(worst_2))

    return result
