# ml/data_loader.py
"""Data loading for PJM LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 ml_pred features and
constraint_limits, and joins realized DA shadow prices as ground truth.

Data pipeline (2026-03-11 audit):
  V6.2B signal already contains: ori_mean (= density score), mix_mean,
  shadow_price_da, da_rank_value, density_ori_rank_value, density_mix_rank_value,
  rank_ori, branch_name, etc. — 21 columns total.

  We enrich with:
    - constraint_limit from spice6 limit.parquet (NOT in V6.2B)
    - ml_pred features: binding_probability, predicted_shadow_price,
      hist_da, prob_exceed_100 (NOT in V6.2B)
    - realized DA shadow prices (ground truth labels)
    - binding_freq (computed from DA history in script layer)

  We do NOT re-load density scores — ori_mean IS the density score and
  V6.2B already has it. The old density enrichment was 100% redundant.

KEY PJM DIFFERENCE: ground truth joins on branch_name (not constraint_id).
V6.2B parquet already has a branch_name column.

DEDUPLICATION: Multiple constraint_ids can share the same branch_name
(different contingencies, same physical line). After joining realized DA
on branch_name, all copies get identical realized_sp. We deduplicate to
one row per branch_name (keeping the row with lowest rank_ori) to avoid
inflated metrics and correlated training examples.
"""
from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.realized_da import load_realized_da
from ml.spice6_loader import load_constraint_limits, load_spice6_mlpred


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_MONTH_CACHE: dict[tuple[str, str, str, str | None], pl.DataFrame] = {}


def clear_month_cache() -> None:
    _MONTH_CACHE.clear()


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with constraint_limit + ml_pred + realized DA.

    V6.2B already contains density scores (ori_mean, mix_mean) and DA scores
    (shadow_price_da, da_rank_value). We only enrich with:
      - constraint_limit (from spice6 limit.parquet)
      - ml_pred features (binding_probability, predicted_shadow_price, hist_da, prob_exceed_100)
      - realized DA ground truth (joined on branch_name)
    """
    cache_key = (auction_month, period_type, class_type, cache_dir)
    if cache_key in _MONTH_CACHE:
        return _MONTH_CACHE[cache_key]

    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Ensure constraint_id and branch_name are strings
    df = df.with_columns(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
    )

    # Enrich with constraint_limit (NOT in V6.2B, comes from spice6 limit.parquet)
    limits = load_constraint_limits(auction_month, period_type)
    if len(limits) > 0:
        df = df.join(limits, on="constraint_id", how="left")
        df = df.with_columns(pl.col("constraint_limit").fill_null(0.0))
        n_matched = len(df.filter(pl.col("constraint_limit") > 0))
        print(f"[data_loader] constraint_limit: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no constraint_limit for {auction_month}")
        df = df.with_columns(pl.lit(0.0).alias("constraint_limit"))

    # Enrich with spice6 ml_pred features
    mlpred = load_spice6_mlpred(auction_month, period_type, class_type)
    if len(mlpred) > 0:
        df = df.join(mlpred, on=["constraint_id", "flow_direction"], how="left")
        mlpred_cols = [c for c in mlpred.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in mlpred_cols])
        n_matched = len(df.filter(pl.col("binding_probability") > 0))
        print(f"[data_loader] ml_pred: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no ml_pred for {auction_month}/{class_type}")
        for col in ["binding_probability", "predicted_shadow_price", "hist_da", "prob_exceed_100"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth on branch_name
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)

    # Map class_type to peak_type for DA fetch
    peak_type = class_type  # PJM uses same names: onpeak, dailyoffpeak, wkndonpeak
    da_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        realized = load_realized_da(gt_month, peak_type=peak_type, **da_kwargs)
        # Join on branch_name (PJM-specific)
        df = df.join(realized, on="branch_name", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
    except FileNotFoundError:
        print(f"[data_loader] WARNING: no realized DA for {gt_month}/{peak_type}")
        df = df.with_columns(pl.lit(0.0).alias("realized_sp"))

    # Deduplicate: keep one row per branch_name (lowest rank_ori = highest priority)
    n_pre = len(df)
    df = df.sort("rank_ori").unique(subset=["branch_name"], keep="first")
    n_post = len(df)
    n_binding = len(df.filter(pl.col("realized_sp") > 0))
    print(f"[data_loader] {auction_month}: {n_pre}→{n_post} rows (dedup by branch), "
          f"{n_binding} binding (gt={gt_month})")

    _MONTH_CACHE[cache_key] = df
    return df


def compute_new_mask(
    branch_names: list[str],
    eval_month: str,
    binding_sets: dict[str, set[str]],
    lookback: int = 6,
) -> "np.ndarray":
    """Return boolean mask: True if branch has NOT bound in prior `lookback` months (BF-zero).

    Uses same lag rules as BF: for eval_month M, checks binding history
    through prev_month(M) using months strictly before that cutoff.
    The caller is responsible for passing the correct eval_month
    (auction month, not delivery month).
    """
    cutoff = (pd.Timestamp(eval_month) - pd.DateOffset(months=1)).strftime("%Y-%m")
    prior = [m for m in sorted(binding_sets.keys()) if m < cutoff][-lookback:]
    bound_branches: set[str] = set()
    for m in prior:
        bound_branches |= binding_sets[m]
    return np.array([bn not in bound_branches for bn in branch_names], dtype=bool)


def compute_history_zero_mask(
    branch_names: list[str],
    eval_month: str,
    binding_sets: dict[str, set[str]],
) -> "np.ndarray":
    """Return boolean mask: True if branch has NEVER appeared in any binding set before eval_month.

    Diagnostic subset of BF-zero — too small/noisy for gating.
    """
    cutoff = (pd.Timestamp(eval_month) - pd.DateOffset(months=1)).strftime("%Y-%m")
    all_prior = [m for m in sorted(binding_sets.keys()) if m < cutoff]
    ever_bound: set[str] = set()
    for m in all_prior:
        ever_bound |= binding_sets[m]
    return np.array([bn not in ever_bound for bn in branch_names], dtype=bool)
