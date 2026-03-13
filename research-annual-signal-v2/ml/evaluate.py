"""Evaluation metrics, per-group evaluation, and gate checking.

Tier 1 (blocking gates): VC@K, Recall@K, NDCG, Abs_SP@K
Tier 2 (monitoring): Spearman, cohort contribution
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from ml.config import TIER1_GATE_METRICS, GATE_MIN_WINS

logger = logging.getLogger(__name__)


# ─── Tier 1 metric functions ────────────────────────────────────────────


def value_capture_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """VC@K: fraction of total SP captured by top-K scored branches."""
    total = actual.sum()
    if total <= 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total)


def recall_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Recall@K: fraction of binding branches in top-K."""
    n_binding = (actual > 0).sum()
    if n_binding == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float((actual[top_k_idx] > 0).sum() / n_binding)


def abs_sp_at_k(
    actual: np.ndarray, scores: np.ndarray, k: int, total_da_sp: float,
) -> float:
    """Abs_SP@K: top-K captured SP / total DA SP (cross-universe denominator)."""
    if total_da_sp <= 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total_da_sp)


def nb_recall_at_k(
    actual: np.ndarray, scores: np.ndarray, is_nb: np.ndarray, k: int,
) -> float:
    """NB12_Recall@K: fraction of NB binders captured in top-K."""
    nb_binders = (actual > 0) & is_nb
    n_nb_binders = nb_binders.sum()
    if n_nb_binders == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_mask = np.zeros(len(actual), dtype=bool)
    top_k_mask[top_k_idx] = True
    return float((nb_binders & top_k_mask).sum() / n_nb_binders)


def ndcg(relevance: np.ndarray, scores: np.ndarray) -> float:
    """NDCG using tiered labels (0/1/2/3) as relevance, NOT continuous SP."""
    if relevance.sum() <= 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    sorted_rel = relevance[order]
    # DCG
    discounts = np.log2(np.arange(2, len(sorted_rel) + 2))
    dcg = (sorted_rel / discounts).sum()
    # Ideal DCG
    ideal_order = np.argsort(relevance)[::-1]
    ideal_sorted = relevance[ideal_order]
    idcg = (ideal_sorted / discounts).sum()
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def spearman_corr(actual: np.ndarray, scores: np.ndarray) -> float:
    """Spearman rank correlation between actual SP and scores."""
    if len(actual) < 3 or actual.std() == 0:
        return 0.0
    corr, _ = scipy_stats.spearmanr(actual, scores)
    return float(corr)


# ─── Per-group evaluation ───────────────────────────────────────────────


def evaluate_group(group_df: pl.DataFrame) -> dict:
    """Compute all metrics for one (PY, quarter) group.

    Expects group_df to have: score, realized_shadow_price, total_da_sp_quarter,
    is_nb_12, cohort, onpeak_sp, offpeak_sp.
    """
    actual = group_df["realized_shadow_price"].to_numpy().astype(np.float64)
    label_tier = group_df["label_tier"].to_numpy().astype(np.float64)
    scores = group_df["score"].to_numpy().astype(np.float64)
    total_da_sp = float(group_df["total_da_sp_quarter"][0])

    is_nb = group_df["is_nb_12"].to_numpy()

    n = len(actual)
    metrics: dict = {"n_branches": n, "n_binding": int((actual > 0).sum())}

    # VC@K and Recall@K at multiple K (use continuous SP for value-weighted metrics)
    for k in [20, 50, 100]:
        if k <= n:
            metrics[f"VC@{k}"] = value_capture_at_k(actual, scores, k)
            metrics[f"Recall@{k}"] = recall_at_k(actual, scores, k)
            metrics[f"Abs_SP@{k}"] = abs_sp_at_k(actual, scores, k, total_da_sp)
            metrics[f"NB12_Recall@{k}"] = nb_recall_at_k(actual, scores, is_nb, k)

    # NDCG uses tiered labels (0/1/2/3), NOT continuous SP
    metrics["NDCG"] = ndcg(label_tier, scores)
    metrics["Spearman"] = spearman_corr(actual, scores)

    # Cohort contribution at K=50
    if 50 <= n:
        metrics["cohort_contribution"] = cohort_contribution(group_df, k=50)

    return metrics


def evaluate_all(model_table: pl.DataFrame) -> dict:
    """Evaluate all (PY, quarter) groups in the model table.

    Returns dict with:
      - per_group: {group_key: metrics_dict}
      - dev_mean: mean of dev group metrics
      - holdout_mean: mean of holdout group metrics
    """
    assert "score" in model_table.columns, "model_table must have 'score' column"

    per_group: dict = {}
    for (py, aq), group_df in model_table.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        key = f"{py}/{aq}"
        per_group[key] = evaluate_group(group_df)

    # Aggregate by split
    from ml.config import DEV_GROUPS, HOLDOUT_GROUPS

    result = {"per_group": per_group}
    for split_name, split_groups in [("dev_mean", DEV_GROUPS), ("holdout_mean", HOLDOUT_GROUPS)]:
        present = [g for g in split_groups if g in per_group]
        if present:
            agg: dict = {}
            # Average each metric across groups
            all_keys = set()
            for g in present:
                all_keys.update(per_group[g].keys())
            for mk in all_keys:
                vals = [per_group[g][mk] for g in present if mk in per_group[g]]
                if vals and isinstance(vals[0], (int, float)):
                    agg[mk] = sum(vals) / len(vals)
            result[split_name] = agg

    return result


# ─── Cohort contribution ────────────────────────────────────────────────


def cohort_contribution(group_df: pl.DataFrame, k: int) -> dict:
    """Tier 3: cohort breakdown of top-K branches."""
    scores = group_df["score"].to_numpy()
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_mask = np.zeros(len(group_df), dtype=bool)
    top_k_mask[top_k_idx] = True

    cohorts = group_df["cohort"].to_list()
    actual = group_df["realized_shadow_price"].to_numpy()

    result: dict = {}
    for cohort_name in ["established", "history_dormant", "history_zero"]:
        cohort_mask = np.array([c == cohort_name for c in cohorts])
        in_top_k = (cohort_mask & top_k_mask).sum()
        sp_captured = actual[cohort_mask & top_k_mask].sum()
        result[cohort_name] = {
            "count_in_top_k": int(in_top_k),
            "sp_captured": float(sp_captured),
        }

    return result


# ─── Gate checking ──────────────────────────────────────────────────────


def check_gates(
    candidate: dict,
    baseline: dict,
    baseline_name: str,
    holdout_groups: list[str],
) -> dict:
    """Check Tier 1 gate metrics: candidate vs baseline.

    Gate rule per metric:
      - Candidate must STRICTLY beat baseline on >= GATE_MIN_WINS of holdout groups
      - AND candidate mean > baseline mean (strict inequality)
      - Ties do NOT count as wins — prevents inert gates when baseline is 0.0

    Args:
        candidate: {group_key: {metric: value}}
        baseline: {group_key: {metric: value}}
        baseline_name: name for logging
        holdout_groups: list of group keys to gate on

    Returns:
        {metric_name: {passed, wins, candidate_mean, baseline_mean}}
    """
    gates: dict = {}

    # Only check Tier 1 gate metrics — not every metric in the dict
    for metric in TIER1_GATE_METRICS:
        cand_vals = []
        base_vals = []
        wins = 0
        for g in holdout_groups:
            c_val = candidate.get(g, {}).get(metric)
            b_val = baseline.get(g, {}).get(metric)
            if c_val is not None and b_val is not None:
                cand_vals.append(c_val)
                base_vals.append(b_val)
                if c_val > b_val:
                    wins += 1

        if not cand_vals:
            continue

        cand_mean = sum(cand_vals) / len(cand_vals)
        base_mean = sum(base_vals) / len(base_vals)
        passed = (wins >= GATE_MIN_WINS) and (cand_mean > base_mean)

        gates[metric] = {
            "passed": passed,
            "wins": wins,
            "n_groups": len(cand_vals),
            "candidate_mean": cand_mean,
            "baseline_mean": base_mean,
        }

        status = "PASS" if passed else "FAIL"
        logger.info(
            "Gate %s vs %s: %s %s (wins=%d/%d, mean=%.4f vs %.4f)",
            metric, baseline_name, status, metric,
            wins, len(cand_vals), cand_mean, base_mean,
        )

    return gates
