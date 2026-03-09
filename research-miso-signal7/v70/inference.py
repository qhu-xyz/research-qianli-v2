"""ML inference for V7.0 signal generation.

Trains LightGBM LambdaRank on historical data and scores target month.
Returns (constraint_ids, scores).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Ensure stage5 ml package is importable
_STAGE5 = Path(__file__).resolve().parent.parent.parent / "research-stage5-tier"
if str(_STAGE5) not in sys.path:
    sys.path.insert(0, str(_STAGE5))

from v70.cache import REALIZED_DA_CACHE
from ml.config import LTRConfig, PipelineConfig, V62B_SIGNAL_BASE, collect_usable_months
from ml.data_loader import load_v62b_month
from ml.features import compute_query_groups, prepare_features
from ml.spice6_loader import load_spice6_density
from ml.train import predict_scores, train_ltr_model

# ── Constants (from run_v10e_lagged.py) ──
V10E_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.85, 0.00, 0.15),
    ("f0", "offpeak"): (0.85, 0.00, 0.15),
    ("f1", "onpeak"): (0.70, 0.00, 0.30),
    ("f1", "offpeak"): (0.80, 0.00, 0.20),
}

_DEFAULT_BLEND = (0.85, 0.00, 0.15)


def _prev_month(m: str) -> str:
    import pandas as pd
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load all cached DA into {month: set(constraint_ids)}."""
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list()
        )
    return binding_sets


def _compute_bf(
    cids: list[str], month: str, bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(cids), dtype=np.float64)
    freq = np.zeros(len(cids), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, cid in enumerate(cids):
            if cid in s:
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
    cids = df["constraint_id"].to_list()
    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = _compute_bf(cids, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def load_v62b_features_only(
    auction_month: str,
    period_type: str,
    class_type: str,
) -> pl.DataFrame:
    """Load V6.2B + spice6 WITHOUT realized DA ground truth (inference-only)."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(spice6, on=["constraint_id", "flow_direction"], how="left")
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
    else:
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
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
            features=V10E_FEATURES,
            monotone_constraints=V10E_MONOTONE,
            backend="lightgbm",
            label_mode="tiered",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
        ),
        train_months=8,
        val_months=0,
    )

    # Collect training months
    train_month_strs = collect_usable_months(auction_month, period_type, n_months=8)
    if not train_month_strs:
        raise RuntimeError(
            f"Insufficient training history for {auction_month}/{period_type}"
        )
    train_month_strs = list(reversed(train_month_strs))  # chronological

    # Load and enrich training data
    parts = []
    for tm in train_month_strs:
        part = load_v62b_month(tm, period_type, class_type)
        part = part.with_columns(pl.lit(tm).alias("query_month"))
        part = _enrich_df(part, tm, bs, blend)
        parts.append(part)
    train_df = pl.concat(parts)
    train_df = train_df.sort("query_month")

    # Train model
    X_train, _ = prepare_features(train_df, cfg.ltr)
    y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
    groups = compute_query_groups(train_df)
    model = train_ltr_model(X_train, y_train, groups, cfg.ltr)

    # Score target month (no GT needed)
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
