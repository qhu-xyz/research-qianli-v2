"""7.2b release candidate: base (Bucket_6_20) + specialist (per-ctype dormant).

Copied exactly from research source of truth:
  /home/xyz/workspace/research-qianli-v2/.claude/worktrees/annual-worktree/
  research-annual-signal-v2/scripts/release_candidate_eval.py

Constants, params, feature lists, and helpers are byte-exact copies.
Research-only printing/report logic is excluded.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.markets.miso.config import CLASS_BF_COL, CROSS_CLASS_BF_COL, CLASS_NB_FLAG_COL

# ── Stage 1: Bucket_6_20 ─────────────────────────────────────────────

BASE_FEATURES = [
    "da_rank_value", "shadow_price_da", "bf", "count_active_cids",
    "bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max",
    "rt_max",
]

BASE_LGB = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 15, "learning_rate": 0.05,
    "min_child_samples": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
    "seed": 42, "deterministic": True,
}

BASE_BOOST_ROUNDS = 150

BUCKET_BOUNDS = [0, 200, 5000, 20000]
BUCKET_WEIGHTS = {0: 1, 1: 1, 2: 2, 3: 6, 4: 20}

# ── Stage 2: Dormant specialist ──────────────────────────────────────

STAGE2_FEATS = [
    "p_exc_40", "p_exc_60", "p_exc_80", "p_exc_100",
    "p_exc_max_80", "p_exc_max_100", "p_frac_80", "p_frac_100",
    "n_exc_n40", "n_exc_n60", "n_exc_n80", "n_frac_n40", "n_frac_n80",
    "p_dev", "n_dev", "dev_best",
    "top2_p_dev", "dev_gap_p", "dev_gap_n",
    "n_cids_hot_p", "n_cids_hot_n", "n_cids_total",
    "count_active_cids", "rt_max",
]

STAGE2_LGB = {
    "objective": "binary", "metric": "binary_logloss",
    "num_leaves": 11, "learning_rate": 0.05,
    "min_child_samples": 3, "subsample": 0.8,
    "colsample_bytree": 0.8, "num_threads": 4, "verbose": -1,
    "seed": 42, "deterministic": True,
}

STAGE2_BOOST_ROUNDS = 150

# ── Eval grid ────────────────────────────────────────────────────────

ALL_PYS = [
    "2017-06", "2018-06", "2019-06", "2020-06", "2021-06",
    "2022-06", "2023-06", "2024-06", "2025-06",
]
AQS_FULL = ["aq1", "aq2", "aq3", "aq4"]
AQS_NO_AQ4 = ["aq1", "aq2", "aq3"]

EVAL_CONFIGS = [
    ("2020-06", [py for py in ALL_PYS if py < "2020-06"], AQS_FULL),
    ("2021-06", [py for py in ALL_PYS if py < "2021-06"], AQS_FULL),
    ("2022-06", [py for py in ALL_PYS if py < "2022-06"], AQS_FULL),
    ("2023-06", [py for py in ALL_PYS if py < "2023-06"], AQS_FULL),
    ("2024-06", [py for py in ALL_PYS if py < "2024-06"], AQS_FULL),
    ("2025-06", [py for py in ALL_PYS if py < "2025-06"], AQS_NO_AQ4),
]

ROUNDS = [1, 2, 3]
K_LEVELS = [200, 400]
R_AT_K = {200: 50, 400: 100}

# Publication policy (production-only, not in research)
TIER_SIZE = 200
N_TIERS = 5
SPECIALIST_PER_TIER = 50
BASE_PER_TIER = 150


# ── Helpers (byte-exact copies from research) ────────────────────────

def assign_bucket_labels(sp: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(sp), dtype=np.int32)
    labels[sp > 0] = 1
    labels[sp > BUCKET_BOUNDS[1]] = 2
    labels[sp > BUCKET_BOUNDS[2]] = 3
    labels[sp > BUCKET_BOUNDS[3]] = 4
    return labels


def assign_bucket_weights(labels: np.ndarray) -> np.ndarray:
    return np.array([BUCKET_WEIGHTS[int(l)] for l in labels])


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def score_v0c(data: pl.DataFrame) -> np.ndarray:
    """v0c formula — for comparison baseline only."""
    da_rank = data["da_rank_value"].to_numpy().astype(np.float64)
    da_norm = 1.0 - _minmax(da_rank)
    rt_max = data["rt_max"].to_numpy().astype(np.float64)
    rt_norm = _minmax(rt_max)
    bf = data["bf"].to_numpy().astype(np.float64)
    bf_norm = _minmax(bf)
    return 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf_norm


def make_reserved_selection(
    primary_order: np.ndarray,
    specialist_order: np.ndarray,
    K: int,
    R: int,
) -> list[int]:
    """Research-parity reserved-slot selection. Returns list of selected indices."""
    spec_picks = set(specialist_order[:R].tolist())
    base_rem = [j for j in primary_order if j not in spec_picks][:K - R]
    return list(set(base_rem) | spec_picks)


def compute_metrics(
    sp: np.ndarray,
    is_nb12: np.ndarray,
    topk_indices: list[int],
    total_sp: float,
) -> dict:
    """Compute aggregate metrics for a selection. Uses total_da_sp_quarter as denominator."""
    mask = np.zeros(len(sp), dtype=bool)
    for j in topk_indices:
        mask[j] = True
    sel_sp = sp[mask]
    captured = float(sel_sp.sum())
    return {
        "sp": captured,
        "vc": captured / total_sp if total_sp > 0 else 0.0,
        "binders": int((sel_sp > 0).sum()),
        "precision": float((sel_sp > 0).sum() / max(len(sel_sp), 1)),
        "d20_count": int((sel_sp > 20000).sum()),
        "d20_sp": float(sel_sp[sel_sp > 20000].sum()),
        "d40_count": int((sel_sp > 40000).sum()),
        "d40_sp": float(sel_sp[sel_sp > 40000].sum()),
        "nb12_count": int((mask & is_nb12 & (sp > 0)).sum()),
        "nb12_sp": float(sp[mask & is_nb12 & (sp > 0)].sum()),
        "nb12_d20_count": int(((sp > 20000) & mask & is_nb12).sum()),
        "nb12_d20_sp": float(sp[(sp > 20000) & mask & is_nb12].sum()),
        "nb12_d40_count": int(((sp > 40000) & mask & is_nb12).sum()),
        "nb12_d40_sp": float(sp[(sp > 40000) & mask & is_nb12].sum()),
    }


def build_eval_table(
    eval_py: str,
    aq: str,
    ct: str,
    market_round: int,
) -> pl.DataFrame:
    """Build one eval table with deviation features. Matches research build_round_data()."""
    from ml.markets.miso.features import build_class_model_table
    from ml.markets.miso.deviation_profile import build_deviation_features

    table = build_class_model_table(eval_py, aq, ct, market_round=market_round)

    bf_col = CLASS_BF_COL[ct]
    bf_cross_col = CROSS_CLASS_BF_COL[ct]
    nb12_col = CLASS_NB_FLAG_COL[ct]

    table = table.with_columns(
        pl.col(bf_col).alias("bf"),
        pl.col(bf_cross_col).alias("bf_cross"),
        pl.col(nb12_col).cast(pl.Boolean).alias("is_nb12"),
        pl.max_horizontal(
            "bin_80_cid_max", "bin_90_cid_max",
            "bin_100_cid_max", "bin_110_cid_max",
        ).alias("rt_max"),
        pl.col("realized_shadow_price").alias("sp"),
    ).rename({"branch_name": "branch"})

    try:
        dev = build_deviation_features(eval_py, aq, market_round=market_round)
        dev = dev.rename({"branch_name": "branch"})
        table = table.join(dev, on="branch", how="left")
        dev_cols = [c for c in dev.columns if c != "branch"]
        for c in dev_cols:
            if c in table.columns:
                table = table.with_columns(pl.col(c).fill_null(0.0))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Deviation features failed for %s/%s/R%d: %s", eval_py, aq, market_round, e
        )
        for c in STAGE2_FEATS:
            if c not in table.columns:
                table = table.with_columns(pl.lit(0.0).alias(c))

    return table
