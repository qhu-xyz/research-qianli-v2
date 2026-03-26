"""7.2b release candidate: base (Bucket_6_20) + specialist (per-ctype dormant).

Copied exactly from research source of truth:
  /home/xyz/workspace/research-qianli-v2/.claude/worktrees/annual-worktree/
  research-annual-signal-v2/scripts/release_candidate_eval.py

Constants, params, feature lists, and helpers are byte-exact copies.
Research-only printing/report logic is excluded.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.markets.miso.config import CLASS_BF_COL, CROSS_CLASS_BF_COL, CLASS_NB_FLAG_COL

logger = logging.getLogger(__name__)

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

SPECIALIST_BRANCH_POOL_SIZE = SPECIALIST_PER_TIER * N_TIERS


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
    table = _prepare_release_candidate_table(table, ct)

    try:
        table = _join_deviation_features(
            table=table,
            planning_year=eval_py,
            aq=aq,
            market_round=market_round,
            strict=False,
        )
    except Exception as e:
        logger.warning(
            "Deviation features failed for %s/%s/R%d: %s", eval_py, aq, market_round, e
        )
        for c in STAGE2_FEATS:
            if c not in table.columns:
                table = table.with_columns(pl.lit(0.0).alias(c))

    return table


def _prepare_release_candidate_table(table: pl.DataFrame, ct: str) -> pl.DataFrame:
    """Alias model-table columns into the 7.2b scorer contract."""

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

    return table


def _join_deviation_features(
    table: pl.DataFrame,
    planning_year: str,
    aq: str,
    market_round: int,
    strict: bool,
) -> pl.DataFrame:
    """Join deviation features onto a prepared 7.2b branch table."""
    from ml.markets.miso.deviation_profile import build_deviation_features

    dev = build_deviation_features(planning_year, aq, market_round=market_round)
    dev = dev.rename({"branch_name": "branch"})
    table = table.join(dev, on="branch", how="left")
    dev_cols = [c for c in dev.columns if c != "branch"]

    for c in dev_cols:
        if c in table.columns:
            if strict and table.filter(pl.col(c).is_null()).height > 0:
                raise ValueError(
                    f"Deviation feature {c} has nulls for {planning_year}/{aq}/R{market_round}"
                )
            table = table.with_columns(pl.col(c).fill_null(0.0))

    missing = [c for c in STAGE2_FEATS if c not in table.columns]
    if missing:
        if strict:
            raise ValueError(
                f"Missing deviation features for {planning_year}/{aq}/R{market_round}: {missing}"
            )
        for c in missing:
            table = table.with_columns(pl.lit(0.0).alias(c))

    return table


def build_eval_table_strict(
    eval_py: str,
    aq: str,
    ct: str,
    market_round: int,
) -> pl.DataFrame:
    """Build one strict 7.2b training/eval table with no deviation fallback."""
    from ml.markets.miso.features import build_class_model_table

    table = build_class_model_table(eval_py, aq, ct, market_round=market_round)
    table = _prepare_release_candidate_table(table, ct)
    return _join_deviation_features(
        table=table,
        planning_year=eval_py,
        aq=aq,
        market_round=market_round,
        strict=True,
    )


def build_publish_table_72b(
    planning_year: str,
    aq: str,
    ct: str,
    market_round: int,
) -> pl.DataFrame:
    """Build one strict ex-ante 7.2b publish table for the target cell.

    Uses build_class_publish_table (no GT/NB) and only aliases the columns
    needed for scoring. Does NOT alias is_nb12 or sp (not available ex-ante).
    """
    from ml.markets.miso.features import build_class_publish_table

    table = build_class_publish_table(planning_year, aq, ct, market_round=market_round)

    bf_col = CLASS_BF_COL[ct]
    bf_cross_col = CROSS_CLASS_BF_COL[ct]

    table = table.with_columns(
        pl.col(bf_col).alias("bf"),
        pl.col(bf_cross_col).alias("bf_cross"),
        pl.max_horizontal(
            "bin_80_cid_max", "bin_90_cid_max",
            "bin_100_cid_max", "bin_110_cid_max",
        ).alias("rt_max"),
    )

    # Deviation features expect "branch" column name (research convention)
    table = table.rename({"branch_name": "branch"})
    table = _join_deviation_features(
        table=table,
        planning_year=planning_year,
        aq=aq,
        market_round=market_round,
        strict=True,
    )
    # Rename back to branch_name for publisher join compatibility
    return table.rename({"branch": "branch_name"})


_TRAIN_TABLE_CACHE: dict[tuple[str, str, str, int], pl.DataFrame] = {}
_PUBLISH_MODEL_CACHE: dict[tuple[str, str, int], dict] = {}
_PUBLISH_SCORE_CACHE: dict[tuple[str, str, str, int], pl.DataFrame] = {}


def _get_train_table(py: str, aq: str, ct: str, market_round: int) -> pl.DataFrame:
    key = (py, aq, ct, market_round)
    if key not in _TRAIN_TABLE_CACHE:
        _TRAIN_TABLE_CACHE[key] = build_eval_table_strict(py, aq, ct, market_round)
    return _TRAIN_TABLE_CACHE[key]


def get_publish_train_pys(planning_year: str) -> list[str]:
    """Historical planning years used to train the publish-time models."""
    train_pys = [py for py in ALL_PYS if py < planning_year]
    if not train_pys:
        raise ValueError(f"No historical training years available for {planning_year}")
    return train_pys


def train_publish_models_72b(
    planning_year: str,
    ct: str,
    market_round: int,
) -> dict:
    """Train and cache the base + specialist models for one publish year/ctype/round."""
    key = (planning_year, ct, market_round)
    if key in _PUBLISH_MODEL_CACHE:
        return _PUBLISH_MODEL_CACHE[key]

    train_pys = get_publish_train_pys(planning_year)
    train_frames: list[pl.DataFrame] = []
    for tpy in train_pys:
        for taq in AQS_FULL:
            t = _get_train_table(tpy, taq, ct, market_round)
            train_frames.append(
                t.with_columns(
                    pl.lit(tpy).alias("py"),
                    pl.lit(taq).alias("aq_label"),
                )
            )
    train_all = pl.concat(train_frames, how="diagonal")

    base_feats = [f for f in BASE_FEATURES if f in train_all.columns]
    if len(base_feats) != len(BASE_FEATURES):
        missing = [f for f in BASE_FEATURES if f not in base_feats]
        raise ValueError(f"Missing base features for publish training: {missing}")

    sp_t = train_all["sp"].to_numpy().astype(np.float64)
    labels_t = assign_bucket_labels(sp_t)
    weights_t = assign_bucket_weights(labels_t)
    groups_t = train_all.group_by(
        ["py", "aq_label"], maintain_order=True
    ).len()["len"].to_numpy()

    ds_base = lgb.Dataset(
        train_all.select(base_feats).to_numpy().astype(np.float64),
        label=labels_t,
        group=groups_t,
        weight=weights_t,
        feature_name=base_feats,
        free_raw_data=False,
    )
    model_base = lgb.train(BASE_LGB, ds_base, num_boost_round=BASE_BOOST_ROUNDS)

    s2_train_frames: list[pl.DataFrame] = []
    for tpy in train_pys:
        for taq in AQS_FULL:
            t = _get_train_table(tpy, taq, ct, market_round)
            bf_s = t["bf"].to_numpy()
            bf_c = t["bf_cross"].to_numpy()
            t_td = t.filter(pl.Series((bf_s == 0) & (bf_c == 0)))
            if len(t_td) > 0:
                s2_train_frames.append(t_td)

    if not s2_train_frames:
        raise ValueError(
            f"No specialist training rows for {planning_year}/{ct}/R{market_round}"
        )

    s2_train = pl.concat(s2_train_frames, how="diagonal")
    s2_feats = [f for f in STAGE2_FEATS if f in s2_train.columns]
    if len(s2_feats) != len(STAGE2_FEATS):
        missing = [f for f in STAGE2_FEATS if f not in s2_feats]
        raise ValueError(f"Missing specialist features for publish training: {missing}")

    s2_sp = s2_train["sp"].to_numpy().astype(np.float64)
    s2_lab = (s2_sp > 0).astype(np.int32)
    s2_wt = np.ones(len(s2_lab))
    s2_wt[s2_sp > 200] = 3.0
    s2_wt[s2_sp > 5000] = 10.0
    ds_s2 = lgb.Dataset(
        s2_train.select(s2_feats).to_numpy().astype(np.float64),
        label=s2_lab,
        weight=s2_wt,
        feature_name=s2_feats,
        free_raw_data=False,
    )
    model_s2 = lgb.train(STAGE2_LGB, ds_s2, num_boost_round=STAGE2_BOOST_ROUNDS)

    payload = {
        "train_pys": train_pys,
        "base_features": base_feats,
        "specialist_features": s2_feats,
        "base_model": model_base,
        "specialist_model": model_s2,
    }
    _PUBLISH_MODEL_CACHE[key] = payload
    return payload


def score_publish_branches_72b(
    planning_year: str,
    aq: str,
    ct: str,
    market_round: int,
) -> pl.DataFrame:
    """Score one publish cell and assign branch-level origin for 7.2b."""
    key = (planning_year, aq, ct, market_round)
    if key in _PUBLISH_SCORE_CACHE:
        return _PUBLISH_SCORE_CACHE[key]

    models = train_publish_models_72b(planning_year, ct, market_round)
    table = build_publish_table_72b(planning_year, aq, ct, market_round)

    base_scores = models["base_model"].predict(
        table.select(models["base_features"]).to_numpy().astype(np.float64)
    )
    specialist_scores = models["specialist_model"].predict(
        table.select(models["specialist_features"]).to_numpy().astype(np.float64)
    )

    bf = table["bf"].to_numpy().astype(np.float64)
    bf_cross = table["bf_cross"].to_numpy().astype(np.float64)
    is_true_dormant = (bf == 0) & (bf_cross == 0)
    td_idx = np.where(is_true_dormant)[0]
    td_order = td_idx[np.argsort(specialist_scores[td_idx])[::-1]]
    specialist_branch_idx = set(
        td_order[:SPECIALIST_BRANCH_POOL_SIZE].tolist()
    )

    out = table.with_columns(
        pl.Series("base_score", base_scores),
        pl.Series("specialist_score", specialist_scores),
        pl.Series("is_true_dormant", is_true_dormant),
        pl.Series(
            "origin",
            [
                "specialist" if i in specialist_branch_idx else "base"
                for i in range(len(table))
            ],
        ),
    )
    _PUBLISH_SCORE_CACHE[key] = out
    return out
