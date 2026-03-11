"""ML inference for PJM V7.0 signal generation.

Trains LightGBM LambdaRank on historical data and scores target month.
Returns (constraint_ids, scores).

PJM-specific: binding sets keyed on branch_name, 3 class types.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v70.cache import REALIZED_DA_CACHE
from ml.config import (
    LTRConfig, PipelineConfig, V62B_SIGNAL_BASE, V10E_FEATURES, V10E_MONOTONE,
    collect_usable_months,
)
from ml.data_loader import load_v62b_month
from ml.features import compute_query_groups, prepare_features
from ml.spice6_loader import load_constraint_limits, load_spice6_mlpred
from ml.train import predict_scores, train_ltr_model

# Per-slice blend weights: (w_da, w_dmix, w_dori) for v7_formula_score
# v0b optimized: 0.80/0.15/0.05 beats V6.2B default 0.60/0.30/0.10
BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.80, 0.15, 0.05),
    ("f0", "dailyoffpeak"): (0.80, 0.15, 0.05),
    ("f0", "wkndonpeak"): (0.80, 0.15, 0.05),
    ("f1", "onpeak"): (0.80, 0.15, 0.05),
    ("f1", "dailyoffpeak"): (0.80, 0.15, 0.05),
    ("f1", "wkndonpeak"): (0.80, 0.15, 0.05),
}

_DEFAULT_BLEND = (0.80, 0.15, 0.05)


def _prev_month(m: str) -> str:
    import pandas as pd
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load all cached DA into {month: set(branch_names)}.

    PJM-specific: keys are branch_name, not constraint_id.
    """
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["branch_name"].to_list()
        )
    return binding_sets


def _compute_bf(
    branch_names: list[str], month: str, bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    """PJM-specific: BF on branch_name, not constraint_id."""
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)
    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def _enrich_df(
    df: pl.DataFrame,
    month: str,
    bs: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    """Add binding_freq and formula score features. BF_LAG=1 always."""
    cutoff = _prev_month(month)
    w_da, w_dmix, w_dori = blend_weights
    branch_names = df["branch_name"].to_list()
    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = _compute_bf(branch_names, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def load_v62b_features_only(
    auction_month: str,
    period_type: str,
    class_type: str,
) -> pl.DataFrame:
    """Load V6.2B + constraint_limit + ml_pred WITHOUT realized DA (inference-only).

    V6.2B already contains density scores (ori_mean, mix_mean) and DA scores.
    We only enrich with constraint_limit and ml_pred features.
    """
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    df = df.with_columns(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
    )

    # Enrich with constraint_limit (NOT in V6.2B)
    limits = load_constraint_limits(auction_month, period_type)
    if len(limits) > 0:
        df = df.join(limits, on="constraint_id", how="left")
        df = df.with_columns(pl.col("constraint_limit").fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("constraint_limit"))

    # Enrich with ml_pred features
    mlpred = load_spice6_mlpred(auction_month, period_type, class_type)
    if len(mlpred) > 0:
        df = df.join(mlpred, on=["constraint_id", "flow_direction"], how="left")
        mlpred_cols = [c for c in mlpred.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in mlpred_cols])
    else:
        for col in ["binding_probability", "predicted_shadow_price", "hist_da", "prob_exceed_100"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

    return df


def score_ml_inference(
    auction_month: str,
    period_type: str,
    class_type: str,
    bs: dict[str, set[str]],
) -> tuple[list[str], np.ndarray]:
    """Train on history, score target month. Returns (constraint_ids, scores)."""
    blend = BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND)

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=list(V10E_FEATURES),
            monotone_constraints=list(V10E_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    train_month_strs = collect_usable_months(auction_month, period_type, n_months=8)
    if not train_month_strs:
        raise RuntimeError(
            f"Insufficient training history for {auction_month}/{period_type}"
        )
    train_month_strs = list(reversed(train_month_strs))

    parts = []
    for tm in train_month_strs:
        part = load_v62b_month(tm, period_type, class_type, cache_dir=REALIZED_DA_CACHE)
        part = part.with_columns(pl.lit(tm).alias("query_month"))
        part = _enrich_df(part, tm, bs, blend)
        parts.append(part)
    train_df = pl.concat(parts)
    train_df = train_df.sort("query_month")

    X_train, _ = prepare_features(train_df, cfg.ltr)
    y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
    groups = compute_query_groups(train_df)
    model = train_ltr_model(X_train, y_train, groups, cfg.ltr)

    target_df = load_v62b_features_only(auction_month, period_type, class_type)
    target_df = target_df.with_columns(pl.lit(auction_month).alias("query_month"))
    target_df = _enrich_df(target_df, auction_month, bs, blend)

    X_target, _ = prepare_features(target_df, cfg.ltr)
    scores = predict_scores(model, X_target)
    cids = target_df["constraint_id"].to_list()

    del train_df, parts, X_train, y_train, groups, model, X_target
    gc.collect()

    print(f"[inference] {period_type}/{class_type}: scored {len(cids)} constraints, "
          f"train={len(train_month_strs)} months")

    return cids, scores
