"""Phase 6: Class-specific scoring — v0c formula, NB model, blend.

All scoring functions take a class-specific model table from
ml/phase6/features.py and produce per-branch scores.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ── v0c formula variants ──────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def score_v0a(group_df: pl.DataFrame) -> np.ndarray:
    """v0a: pure da_rank_value (class-specific)."""
    da_rank = group_df["da_rank_value"].to_numpy().astype(np.float64)
    return 1.0 - _minmax(da_rank)


def score_v0c(group_df: pl.DataFrame, bf_col: str) -> np.ndarray:
    """v0c: 0.40*da_rank + 0.30*rt_max + 0.30*class_bf.

    Args:
        bf_col: "bf_12" for onpeak, "bfo_12" for offpeak
    """
    da_rank = group_df["da_rank_value"].to_numpy().astype(np.float64)
    da_norm = 1.0 - _minmax(da_rank)

    rt_max = group_df.select(
        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                          "bin_100_cid_max", "bin_110_cid_max")
    ).to_series().to_numpy().astype(np.float64)
    rt_norm = _minmax(rt_max)

    bf = group_df[bf_col].to_numpy().astype(np.float64)
    bf_norm = _minmax(bf)

    return 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf_norm


def score_v0c_cross(group_df: pl.DataFrame, bf_col: str) -> np.ndarray:
    """v0c + cross-class BF: 0.35*da_rank + 0.25*rt_max + 0.25*class_bf + 0.15*cross_bf."""
    da_rank = group_df["da_rank_value"].to_numpy().astype(np.float64)
    da_norm = 1.0 - _minmax(da_rank)

    rt_max = group_df.select(
        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                          "bin_100_cid_max", "bin_110_cid_max")
    ).to_series().to_numpy().astype(np.float64)
    rt_norm = _minmax(rt_max)

    bf = group_df[bf_col].to_numpy().astype(np.float64)
    bf_norm = _minmax(bf)

    cross_bf = group_df["cross_class_bf"].to_numpy().astype(np.float64)
    cross_norm = _minmax(cross_bf)

    return 0.35 * da_norm + 0.25 * rt_norm + 0.25 * bf_norm + 0.15 * cross_norm


# ── NB model ──────────────────────────────────────────────────────────

NB_FEATURES = [
    "bin_80_cid_max", "bin_70_cid_max", "bin_90_cid_max",
    "count_active_cids", "bin_60_cid_max", "bin_100_cid_max", "bin_110_cid_max",
    "bin_-50_cid_max", "bin_120_cid_max", "bin_-100_cid_max", "bin_150_cid_max",
    "shadow_price_da", "da_rank_value",
    "cross_class_bf",
]


def train_nb_model(
    train_df: pl.DataFrame,
    weight_scheme: str = "sqrt",
) -> lgb.Booster:
    """Train class-specific NB model on dormant population.

    Uses class-specific target (already set as realized_shadow_price)
    and class-specific shadow_price_da/da_rank_value.
    Includes cross_class_bf as feature.
    """
    avail = [f for f in NB_FEATURES if f in train_df.columns]
    X = train_df.select(avail).to_numpy().astype(np.float64)
    sp = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    y = (sp > 0).astype(int)

    # Sample weights
    w = np.ones(len(sp), dtype=np.float64)
    pos = sp > 0
    if weight_scheme == "sqrt" and pos.sum() > 0:
        w[pos] = np.sqrt(sp[pos])
    elif weight_scheme == "tiered" and pos.sum() > 0:
        ranks = sp[pos].argsort().argsort()
        n = len(ranks)
        w[pos] = np.where(ranks < n // 3, 1.0, np.where(ranks < 2 * n // 3, 3.0, 10.0))

    # Class imbalance correction
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    cw = np.where(y == 1, n0 / max(n1, 1), 1.0)

    ds = lgb.Dataset(X, label=y, weight=cw * w, feature_name=avail, free_raw_data=False)
    model = lgb.train(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.03,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5,
         "num_threads": 4, "verbose": -1},
        ds, num_boost_round=200,
    )
    return model


def predict_nb(model: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    """Score dormant branches with NB model."""
    avail = [f for f in NB_FEATURES if f in df.columns]
    X = df.select(avail).to_numpy().astype(np.float64)
    return model.predict(X)


# ── Blend ──────────────────────────────────────────────────────────────

def blend_scores(
    group_df: pl.DataFrame,
    base_scores: np.ndarray,
    nb_model: lgb.Booster,
    alpha: float,
) -> np.ndarray:
    """Blend v0c + α×NB for dormant branches.

    Established and zero-history branches keep their base score.
    Dormant branches get base + α × normalized NB prediction.
    """
    cohorts = group_df["cohort"].to_list()
    is_dorm = np.array([c == "history_dormant" for c in cohorts])

    final = base_scores.copy()
    if alpha > 0 and is_dorm.sum() > 0:
        dorm_idx = np.where(is_dorm)[0]
        dorm_df = group_df[dorm_idx.tolist()]
        nb_raw = predict_nb(nb_model, dorm_df)

        base_range = base_scores.max() - base_scores.min()
        nb_range = nb_raw.max() - nb_raw.min()
        if nb_range > 0 and base_range > 0:
            nb_norm = (nb_raw - nb_raw.min()) / nb_range * base_range
            final[dorm_idx] += alpha * nb_norm

    return final
