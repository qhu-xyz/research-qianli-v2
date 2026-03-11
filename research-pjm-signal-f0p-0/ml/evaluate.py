"""Evaluation harness for LTR ranking pipeline.

Group A (blocking, 12 gates):
  VC@20, VC@50, VC@100,
  Recall@20, Recall@50, Recall@100,
  NDCG, Spearman,
  NewBind_Recall@50, NewBind_Recall@100, NewBind_VC@50, NewBind_VC@100.

Group B (monitor): VC@10, VC@25, VC@200, Recall@10.

NewBind metrics require new_mask (BF-zero: not bound in prior 6 months).
Context metrics (n_new, new_value_share, new_row_share) reported but not gated.

All metrics are higher-is-better.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def value_capture_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of total actual value captured by top-k scored items."""
    total = actual.sum()
    if total <= 0:
        return 0.0
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total)


def recall_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the true top-k by actual value, how many are in model's top-k."""
    k = min(k, len(scores))
    true_top_k = set(np.argsort(actual)[::-1][:k].tolist())
    pred_top_k = set(np.argsort(scores)[::-1][:k].tolist())
    return len(true_top_k & pred_top_k) / k if k > 0 else 0.0


def ndcg(actual: np.ndarray, scores: np.ndarray) -> float:
    """NDCG using actual shadow price as relevance, ranked by scores."""
    n = len(actual)
    if n == 0:
        return 0.0
    discounts = np.log2(np.arange(2, n + 2))

    ranked_idx = np.argsort(scores)[::-1]
    dcg = float((actual[ranked_idx] / discounts).sum())

    ideal_idx = np.argsort(actual)[::-1]
    ideal_dcg = float((actual[ideal_idx] / discounts).sum())

    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def spearman_corr(actual: np.ndarray, scores: np.ndarray) -> float:
    """Spearman rank correlation between actual and predicted scores."""
    if len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(actual, scores)
    return float(corr) if not np.isnan(corr) else 0.0


def tier_ap(
    actual: np.ndarray,
    scores: np.ndarray,
    top_frac: float = 0.2,
) -> float:
    """Average Precision for top-frac% of constraints by actual value."""
    n = len(actual)
    k = max(1, int(n * top_frac))
    threshold = np.sort(actual)[::-1][min(k - 1, n - 1)]
    y_true = (actual >= threshold).astype(int)
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, scores))


def evaluate_ltr(
    actual_shadow_price: np.ndarray,
    scores: np.ndarray,
    new_mask: np.ndarray | None = None,
) -> dict:
    """Compute all LTR metrics.

    Parameters
    ----------
    actual_shadow_price : array of realized shadow prices (ground truth)
    scores : array of model scores (higher = more likely to bind)
    new_mask : boolean array, True for BF-zero constraints (not bound in
        prior 6 months). When provided, NewBind metrics are computed.
    """
    n = len(actual_shadow_price)

    result = {
        # Group A (blocking)
        "VC@20": value_capture_at_k(actual_shadow_price, scores, 20),
        "VC@50": value_capture_at_k(actual_shadow_price, scores, 50),
        "VC@100": value_capture_at_k(actual_shadow_price, scores, 100),
        "Recall@20": recall_at_k(actual_shadow_price, scores, 20),
        "Recall@50": recall_at_k(actual_shadow_price, scores, 50),
        "Recall@100": recall_at_k(actual_shadow_price, scores, 100),
        "NDCG": ndcg(actual_shadow_price, scores),
        "Spearman": spearman_corr(actual_shadow_price, scores),
        # Group B (monitor)
        "VC@10": value_capture_at_k(actual_shadow_price, scores, 10),
        "VC@25": value_capture_at_k(actual_shadow_price, scores, 25),
        "VC@200": value_capture_at_k(actual_shadow_price, scores, 200),
        "Recall@10": recall_at_k(actual_shadow_price, scores, 10),
        # Monitoring
        "n_samples": n,
    }

    # NewBind metrics for BF-zero cohort
    if new_mask is not None:
        new_actual = actual_shadow_price[new_mask]
        new_scores = scores[new_mask]
        new_binding = new_actual > 0

        # Context metrics (reported, not gated)
        n_new_binders = int(new_binding.sum())
        result["n_new"] = n_new_binders
        result["new_value_share"] = (
            float(new_actual.sum() / actual_shadow_price.sum())
            if actual_shadow_price.sum() > 0 else 0.0
        )
        result["new_row_share"] = float(new_mask.sum() / n) if n > 0 else 0.0

        # Performance metrics (gated) — only if there are new binders
        if n_new_binders > 0 and len(new_actual) >= 2:
            result["NewBind_Recall@50"] = recall_at_k(new_actual, new_scores, 50)
            result["NewBind_Recall@100"] = recall_at_k(new_actual, new_scores, 100)
            result["NewBind_VC@50"] = value_capture_at_k(new_actual, new_scores, 50)
            result["NewBind_VC@100"] = value_capture_at_k(new_actual, new_scores, 100)
        else:
            result["NewBind_Recall@50"] = 0.0
            result["NewBind_Recall@100"] = 0.0
            result["NewBind_VC@50"] = 0.0
            result["NewBind_VC@100"] = 0.0

    return result


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into summary statistics."""
    months = list(per_month.keys())
    if not months:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    metric_names = list(per_month[months[0]].keys())
    result: dict[str, dict] = {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    for metric in metric_names:
        values = [per_month[m][metric] for m in months]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        arr = np.array(values)
        result["mean"][metric] = float(np.mean(arr))
        result["std"][metric] = float(np.std(arr, ddof=0))
        result["min"][metric] = float(np.min(arr))
        result["max"][metric] = float(np.max(arr))
        # All metrics higher-is-better: worst = lowest
        sorted_vals = np.sort(arr)
        worst_2 = sorted_vals[:2] if len(sorted_vals) >= 2 else sorted_vals
        result["bottom_2_mean"][metric] = float(np.mean(worst_2))

    return result
