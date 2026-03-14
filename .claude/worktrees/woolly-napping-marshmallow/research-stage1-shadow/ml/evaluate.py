# HUMAN-WRITE-ONLY — agents must NEVER modify this file
"""Standardized evaluation harness for shadow price classification pipeline.

Computes all 10 gate metrics required by registry/gates.json:
  S1-AUC, S1-AP, S1-VCAP@100, S1-VCAP@500, S1-VCAP@1000,
  S1-NDCG, S1-BRIER, S1-REC, S1-CAP@100, S1-CAP@500

Plus additional monitoring metrics (precision, f1, binding rates, etc.).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _value_capture_at_k(
    y_true_bin: np.ndarray,
    shadow_prices: np.ndarray,
    y_proba: np.ndarray,
    k: int,
) -> float:
    """Fraction of total shadow-price value captured in the top-K by predicted probability.

    VC@K = sum(actual_sp[top-K]) / sum(actual_sp[all])

    Threshold-free: ranks by probability, not binary prediction.
    """
    n = len(y_proba)
    if k > n:
        k = n
    total_value = float(shadow_prices.sum())
    if total_value <= 0:
        return 0.0
    order = np.argsort(-y_proba)
    captured = float(shadow_prices[order[:k]].sum())
    return captured / total_value


def _capture_at_k(
    shadow_prices: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> float:
    """Of the K constraints with highest actual shadow prices, how many did the model predict as binding?

    Capture@K = |{actual top-K by SP} intersect {predicted binding}| / K

    Threshold-dependent: uses binary predictions.
    """
    n = len(shadow_prices)
    if k > n:
        k = n
    if k == 0:
        return 0.0
    actual_order = np.argsort(-shadow_prices)
    top_k_indices = actual_order[:k]
    caught = int(y_pred[top_k_indices].sum())
    return caught / k


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    shadow_prices: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all gate metrics and monitoring metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (1 = binding).
    y_proba : np.ndarray
        Predicted probabilities of binding.
    y_pred : np.ndarray
        Binary predictions (after threshold).
    shadow_prices : np.ndarray
        Actual shadow prices (continuous).
    threshold : float
        Classification threshold used to produce y_pred.

    Returns
    -------
    metrics : dict
        Dictionary with all 10 gate metric keys (S1-AUC, S1-AP, etc.)
        plus additional monitoring metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)
    shadow_prices = np.asarray(shadow_prices, dtype=float)

    n = len(y_true)
    n_positive = int(y_true.sum())

    # --- Gate metrics ---

    # S1-AUC: ROC AUC (requires both classes present)
    if 0 < n_positive < n:
        auc_roc = float(roc_auc_score(y_true, y_proba))
    else:
        auc_roc = float("nan")

    # S1-AP: Average Precision (requires both classes present)
    if 0 < n_positive < n:
        avg_prec = float(average_precision_score(y_true, y_proba))
    else:
        avg_prec = float("nan")

    # S1-VCAP@K: Value Capture at K
    vcap_100 = _value_capture_at_k(y_true, shadow_prices, y_proba, k=100)
    vcap_500 = _value_capture_at_k(y_true, shadow_prices, y_proba, k=500)
    vcap_1000 = _value_capture_at_k(y_true, shadow_prices, y_proba, k=1000)

    # S1-NDCG: Normalized Discounted Cumulative Gain
    total_value = float(shadow_prices.sum())
    if total_value > 0 and n > 0:
        ndcg_val = float(
            ndcg_score(
                shadow_prices.reshape(1, -1),
                y_proba.reshape(1, -1),
            )
        )
    else:
        ndcg_val = float("nan")

    # S1-BRIER: Brier score loss (LOWER is better)
    if 0 < n_positive < n:
        brier = float(brier_score_loss(y_true, y_proba))
    else:
        brier = float("nan")

    # S1-REC: Recall
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # S1-CAP@K: Capture at K
    cap_100 = _capture_at_k(shadow_prices, y_pred, k=100)
    cap_500 = _capture_at_k(shadow_prices, y_pred, k=500)

    # --- Monitoring metrics (not gates, but useful for analysis) ---
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    binding_rate = n_positive / n if n > 0 else 0.0
    pred_binding_rate = float(y_pred.sum()) / n if n > 0 else 0.0

    return {
        # Gate metrics (10 total)
        "S1-AUC": round(auc_roc, 4) if not np.isnan(auc_roc) else auc_roc,
        "S1-AP": round(avg_prec, 4) if not np.isnan(avg_prec) else avg_prec,
        "S1-VCAP@100": round(vcap_100, 4),
        "S1-VCAP@500": round(vcap_500, 4),
        "S1-VCAP@1000": round(vcap_1000, 4),
        "S1-NDCG": round(ndcg_val, 4) if not np.isnan(ndcg_val) else ndcg_val,
        "S1-BRIER": round(brier, 4) if not np.isnan(brier) else brier,
        "S1-REC": round(rec, 4),
        "S1-CAP@100": round(cap_100, 4),
        "S1-CAP@500": round(cap_500, 4),
        # Monitoring metrics
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 6),
        "n_samples": n,
        "n_positive": n_positive,
        "binding_rate": round(binding_rate, 4),
        "pred_binding_rate": round(pred_binding_rate, 4),
    }


# Gate metrics that are "lower is better" (worst = highest values)
_LOWER_IS_BETTER = {"S1-BRIER"}


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into mean, std, min, max, bottom_2_mean.

    Parameters
    ----------
    per_month : dict
        {month_id: {metric_name: value, ...}, ...}

    Returns
    -------
    aggregate : dict
        {"mean": {...}, "std": {...}, "min": {...}, "max": {...}, "bottom_2_mean": {...}}
    """
    if not per_month:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    months = sorted(per_month.keys())
    all_keys = set()
    for m in months:
        all_keys.update(per_month[m].keys())

    mean_d, std_d, min_d, max_d, b2_d = {}, {}, {}, {}, {}

    for key in sorted(all_keys):
        vals = []
        for m in months:
            v = per_month[m].get(key)
            if v is not None and isinstance(v, (int, float)) and (not isinstance(v, float) or v == v):
                vals.append(v)
        if not vals:
            continue

        mean_d[key] = round(sum(vals) / len(vals), 4)
        min_d[key] = round(min(vals), 4)
        max_d[key] = round(max(vals), 4)

        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            std_d[key] = round((sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5, 4)
        else:
            std_d[key] = 0.0

        # bottom_2_mean: worst 2 values
        # For "lower is better" metrics, worst = highest values
        if key in _LOWER_IS_BETTER:
            sorted_vals = sorted(vals, reverse=True)  # worst first (highest)
        else:
            sorted_vals = sorted(vals)  # worst first (lowest)
        n_bottom = min(2, len(sorted_vals))
        b2_d[key] = round(sum(sorted_vals[:n_bottom]) / n_bottom, 4)

    return {"mean": mean_d, "std": std_d, "min": min_d, "max": max_d, "bottom_2_mean": b2_d}
