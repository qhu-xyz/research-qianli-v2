"""v13: Use backfill data (2017-04+) with strategies to maintain shadow_price_da importance.

Problem: naive backfill drops VC@20 from 0.3389 to 0.3013 because the model over-indexes
on stale BF patterns, pushing shadow_price_da from 55% to ~1% feature importance.

Strategies tested:
1. Exponentially-decayed BF (bfd_N): recent months weighted more, old months less
2. Two-model ensemble: Model A (SP-only) + Model B (BF+SP), blend at score level
3. Sample weighting: downweight old training years via LightGBM sample weights
4. Interaction constraints: force BF to only split alongside SP features
5. Combinations of the above

All use backfill data (floor_month="2017-04") unless noted.
"""
import json
import gc
import time
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    PipelineConfig, LTRConfig, EVAL_SPLITS, AQ_ROUNDS,
    DEFAULT_EVAL_GROUPS,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.features import prepare_features, compute_query_groups
from ml.train import train_ltr_model, predict_scores
from ml.binding_freq import (
    enrich_with_binding_freq,
    enrich_with_binding_freq_decayed,
    BF_WINDOWS,
)

REGISTRY_DIR = _PROJECT_ROOT / "registry"
DISPLAY_METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

# ── Year-level training/assessment ──
_year_groups: dict[str, list[str]] = OrderedDict()
for _g in DEFAULT_EVAL_GROUPS:
    _y = _g.split("/")[0]
    _year_groups.setdefault(_y, []).append(_g)


def _get_train_groups(eval_year: str) -> list[str]:
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == eval_year:
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError(f"No split for year: {eval_year}")


# ── Data loading ──

# Cache: (planning_year, aq_round, mode) -> DataFrame
_DATA_CACHE: dict[tuple[str, str, str], pl.DataFrame] = {}


def load_group(planning_year: str, aq_round: str, mode: str = "no_backfill",
               half_life: float = 12.0) -> pl.DataFrame:
    """Load enriched data with BF features.

    Modes:
    - "no_backfill": floor_month=None (uses DA_FLOOR_MONTH="2019-06")
    - "backfill": floor_month="2017-04", raw BF
    - "decayed_HL": floor_month="2017-04", decayed BF with half_life=HL
    """
    cache_key = (planning_year, aq_round, mode)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    # Start from base enriched data (spice6 features, no BF)
    df = load_v61_enriched(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)

    if mode == "no_backfill":
        df = enrich_with_binding_freq(df, planning_year, aq_round, floor_month=None)
    elif mode == "backfill":
        df = enrich_with_binding_freq(df, planning_year, aq_round, floor_month="2017-04")
    elif mode.startswith("decayed_"):
        df = enrich_with_binding_freq_decayed(
            df, planning_year, aq_round,
            half_life=half_life,
            floor_month="2017-04",
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Engineered features
    if "bf_3" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("bf_3") - pl.col("bf_12")).alias("bf_trend_3_12"))
    if "shadow_price_da" in df.columns and "bf_12" in df.columns:
        df = df.with_columns((pl.col("shadow_price_da") * pl.col("bf_12")).alias("sp_x_bf12"))
    if "da_rank_value" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(((1.0 - pl.col("da_rank_value")) * pl.col("bf_12")).alias("rank_x_bf12"))

    # Decayed-specific engineered features
    if "bfd_3" in df.columns and "bfd_12" in df.columns:
        df = df.with_columns((pl.col("bfd_3") - pl.col("bfd_12")).alias("bfd_trend_3_12"))
    if "shadow_price_da" in df.columns and "bfd_12" in df.columns:
        df = df.with_columns((pl.col("shadow_price_da") * pl.col("bfd_12")).alias("sp_x_bfd12"))
    if "da_rank_value" in df.columns and "bfd_12" in df.columns:
        df = df.with_columns(((1.0 - pl.col("da_rank_value")) * pl.col("bfd_12")).alias("rank_x_bfd12"))

    _DATA_CACHE[cache_key] = df
    return df


# ── Training helpers ──

def _compute_sample_weights(train_df: pl.DataFrame, eval_year: str, scheme: str) -> np.ndarray:
    """Compute per-sample weights for LightGBM.

    Schemes:
    - "uniform": all weight 1.0
    - "recency_linear": weight = 1 / (eval_year - train_year)
    - "recency_exp": weight = 2^(-(eval_year - train_year - 1))
    """
    if scheme == "uniform":
        return np.ones(len(train_df), dtype=np.float64)

    eval_yr = int(eval_year.split("-")[0])
    groups = train_df["query_group"].to_list()
    weights = np.ones(len(train_df), dtype=np.float64)

    for i, g in enumerate(groups):
        train_yr = int(g.split("-")[0])
        age = eval_yr - train_yr  # e.g., 2022 - 2019 = 3
        if scheme == "recency_linear":
            weights[i] = 1.0 / max(age, 1)
        elif scheme == "recency_exp":
            weights[i] = 2.0 ** (-(age - 1))  # most recent train year = 1.0
        else:
            raise ValueError(f"Unknown weight scheme: {scheme}")

    return weights


def train_weighted(X_train, y_train, groups_train, cfg, weights=None,
                   interaction_constraints=None):
    """Train LightGBM with optional sample weights and interaction constraints."""
    import lightgbm as lgb
    from ml.train import _tiered_labels

    y_rank = _tiered_labels(y_train, groups_train)
    max_label = int(y_rank.max())
    label_gain = list(range(max_label + 1))

    train_data = lgb.Dataset(
        X_train,
        label=y_rank,
        group=groups_train.tolist(),
        weight=weights,
        feature_name=cfg.features,
    )

    mono_str = ",".join(str(m) for m in cfg.monotone_constraints)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20, 100],
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_child_weight,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "label_gain": label_gain,
        "monotone_constraints": mono_str,
        "num_threads": 4,
        "verbose": -1,
        "seed": 42,
    }

    if interaction_constraints is not None:
        params["interaction_constraints"] = interaction_constraints

    model = lgb.train(params, train_data, num_boost_round=cfg.n_estimators)
    return model


# ── Experiment runners ──

def run_single_model(variant_name: str, features: list[str], monotone: list[int],
                     data_mode: str = "no_backfill", half_life: float = 12.0,
                     weight_scheme: str = "uniform",
                     interaction_constraints=None,
                     n_estimators: int = 200, learning_rate: float = 0.03,
                     num_leaves: int = 31) -> dict:
    """Train and assess a single-model variant."""
    config = PipelineConfig(
        ltr=LTRConfig(
            features=features,
            monotone_constraints=monotone,
            backend="lightgbm",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            label_mode="tiered",
        ),
    )

    per_group = {}
    model_cache = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_group_ids = _get_train_groups(year)
            dfs = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    dfs.append(load_group(py, aq, data_mode, half_life))
                except FileNotFoundError:
                    pass
            train_df = pl.concat(dfs, how="diagonal").sort("query_group")
            X_train, _ = prepare_features(train_df, config.ltr)
            y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
            groups_train = compute_query_groups(train_df)

            weights = _compute_sample_weights(train_df, year, weight_scheme)

            model = train_weighted(
                X_train, y_train, groups_train, config.ltr,
                weights=weights if weight_scheme != "uniform" else None,
                interaction_constraints=interaction_constraints,
            )
            model_cache[year] = model
            del train_df, X_train, y_train, groups_train

        model = model_cache[year]
        for gid in groups_in_year:
            py, aq = gid.split("/")
            df = load_group(py, aq, data_mode, half_life)
            X, _ = prepare_features(df, config.ltr)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = predict_scores(model, X)
            per_group[gid] = evaluate_ltr(actual, scores)

    # Feature importance from last model
    last_model = list(model_cache.values())[-1]
    feat_imp = {}
    if hasattr(last_model, "feature_importance"):
        importance = last_model.feature_importance(importance_type="gain")
        feat_imp = dict(zip(config.ltr.features, [float(x) for x in importance]))

    agg = aggregate_months(per_group)
    del model_cache
    gc.collect()

    return {
        "per_month": per_group,
        "aggregate": agg,
        "feature_importance": feat_imp,
        "n_months": len(per_group),
    }


def run_two_model_ensemble(name: str,
                           sp_features: list[str], sp_monotone: list[int],
                           bf_features: list[str], bf_monotone: list[int],
                           data_mode: str = "backfill", half_life: float = 12.0,
                           sp_weight: float = 0.5,
                           bf_weight_scheme: str = "uniform",
                           n_estimators: int = 200, learning_rate: float = 0.03,
                           ) -> dict:
    """Two-model ensemble: Model A (SP features) + Model B (BF features).

    Final score = sp_weight * minmax(A) + (1 - sp_weight) * minmax(B).
    Architecturally guarantees shadow_price_da contributes at least sp_weight.
    """
    cfg_sp = LTRConfig(
        features=sp_features, monotone_constraints=sp_monotone,
        backend="lightgbm", n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, label_mode="tiered",
    )
    cfg_bf = LTRConfig(
        features=bf_features, monotone_constraints=bf_monotone,
        backend="lightgbm", n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, label_mode="tiered",
    )

    per_group = {}
    sp_models = {}
    bf_models = {}

    for year, groups_in_year in _year_groups.items():
        if year not in sp_models:
            train_group_ids = _get_train_groups(year)
            # SP model uses no_backfill (doesn't need BF)
            dfs_sp = []
            dfs_bf = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    dfs_sp.append(load_group(py, aq, "no_backfill"))
                    dfs_bf.append(load_group(py, aq, data_mode, half_life))
                except FileNotFoundError:
                    pass

            # Train SP model
            train_sp = pl.concat(dfs_sp, how="diagonal").sort("query_group")
            X_sp, _ = prepare_features(train_sp, cfg_sp)
            y_sp = train_sp["realized_shadow_price"].to_numpy().astype(np.float64)
            g_sp = compute_query_groups(train_sp)
            sp_models[year] = train_weighted(X_sp, y_sp, g_sp, cfg_sp)
            del train_sp, X_sp, y_sp, g_sp

            # Train BF model (with optional sample weighting)
            train_bf = pl.concat(dfs_bf, how="diagonal").sort("query_group")
            X_bf, _ = prepare_features(train_bf, cfg_bf)
            y_bf = train_bf["realized_shadow_price"].to_numpy().astype(np.float64)
            g_bf = compute_query_groups(train_bf)
            weights = _compute_sample_weights(train_bf, year, bf_weight_scheme)
            bf_models[year] = train_weighted(
                X_bf, y_bf, g_bf, cfg_bf,
                weights=weights if bf_weight_scheme != "uniform" else None,
            )
            del train_bf, X_bf, y_bf, g_bf

        for gid in groups_in_year:
            py, aq = gid.split("/")
            df_sp = load_group(py, aq, "no_backfill")
            df_bf = load_group(py, aq, data_mode, half_life)
            actual = df_sp["realized_shadow_price"].to_numpy().astype(np.float64)

            X_sp, _ = prepare_features(df_sp, cfg_sp)
            X_bf, _ = prepare_features(df_bf, cfg_bf)

            scores_sp = predict_scores(sp_models[year], X_sp)
            scores_bf = predict_scores(bf_models[year], X_bf)

            # Minmax normalize and blend
            scores_sp_n = _minmax(scores_sp)
            scores_bf_n = _minmax(scores_bf)
            blended = sp_weight * scores_sp_n + (1 - sp_weight) * scores_bf_n

            per_group[gid] = evaluate_ltr(actual, blended)

    agg = aggregate_months(per_group)

    # Feature importance from BF model (SP model is trivial)
    last_bf = list(bf_models.values())[-1]
    feat_imp = {}
    if hasattr(last_bf, "feature_importance"):
        imp = last_bf.feature_importance(importance_type="gain")
        feat_imp = dict(zip(cfg_bf.features, [float(x) for x in imp]))
        feat_imp["__sp_weight"] = sp_weight

    del sp_models, bf_models
    gc.collect()

    return {
        "per_month": per_group,
        "aggregate": agg,
        "feature_importance": feat_imp,
        "n_months": len(per_group),
    }


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - mn) / (mx - mn)


def score_blend(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * _minmax(ml_scores) + (1.0 - alpha) * _minmax(formula_scores)


def run_formula_blend(base_results_fn, alpha_values, base_name: str) -> dict:
    """Run formula blending for an already-trained model.

    base_results_fn: callable returning (predictions_dict, test_data_dict)
    """
    predictions, test_data = base_results_fn()

    blend_results = {}
    for alpha_pct in alpha_values:
        alpha = alpha_pct / 100.0
        name = f"blend_{base_name}_a{alpha_pct}"
        per_month = {}
        for gid, df in test_data.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            formula = 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)
            blended = score_blend(predictions[gid], formula, alpha)
            per_month[gid] = evaluate_ltr(actual, blended)
        agg = aggregate_months(per_month)
        blend_results[name] = {"per_month": per_month, "aggregate": agg, "n_months": len(per_month)}

    return blend_results


# ── Main ──

def print_results(results: dict[str, dict]):
    """Print all variants sorted by VC@20."""
    print(f"\n{'='*160}")
    print("  V13 BACKFILL STRATEGIES RESULTS (mean over 12 dev groups)")
    print(f"{'='*160}")

    header = f"{'Variant':<55} {'#f':>3}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 160)

    entries = sorted(results.items(), key=lambda x: x[1]["aggregate"]["mean"].get("VC@20", 0), reverse=True)

    for name, res in entries:
        means = res["aggregate"]["mean"]
        fi = res.get("feature_importance", {})
        nf = len([k for k in fi if not k.startswith("__")]) if fi else "?"
        row = f"{name:<55} {nf:>3}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"

        if fi:
            total = sum(v for k, v in fi.items() if not k.startswith("__"))
            if total > 0:
                top3 = sorted([(k, v) for k, v in fi.items() if not k.startswith("__")],
                              key=lambda x: x[1], reverse=True)[:3]
                fi_str = ", ".join(f"{k}={v/total*100:.0f}%" for k, v in top3)
                row += f"  [{fi_str}]"

        print(row)

    # Reference baselines
    print(f"\n--- Reference baselines ---")
    for vid in ["v0b", "v8b"]:
        p = REGISTRY_DIR / vid / "metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            means = d["aggregate"]["mean"]
            row = f"{vid + ' (registry)':<55} {'':>3}"
            for m in DISPLAY_METRICS:
                row += f"  {means.get(m, 0):>10.4f}"
            print(row)


def main():
    t_total = time.time()

    # v10e features (pre-backfill best)
    v10e_features = ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15",
                     "rank_x_bf12", "bf_trend_3_12", "sp_x_bf12"]
    v10e_monotone = [1, -1, 1, 1, 1, 1, 0, 1]

    all_results = {}

    # ── 0. Reference: v10e pre-backfill (reproduce best) ──
    print(f"\n{'='*80}\n  v13_ref_no_backfill: reproduce v10e (no backfill)\n{'='*80}")
    t0 = time.time()
    r = run_single_model("v13_ref_no_backfill", v10e_features, v10e_monotone,
                         data_mode="no_backfill")
    all_results["v13_ref_no_backfill"] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 1. Decayed BF with different half-lives ──
    decayed_features_bf = ["shadow_price_da", "da_rank_value",
                           "bfd_6", "bfd_12", "bfd_15",
                           "rank_x_bfd12", "bfd_trend_3_12", "sp_x_bfd12"]
    decayed_monotone = [1, -1, 1, 1, 1, 1, 0, 1]

    for hl in [6, 9, 12, 18, 24]:
        name = f"v13a_decayed_hl{hl}"
        print(f"\n{'='*80}\n  {name}: decayed BF, half_life={hl}\n{'='*80}")
        t0 = time.time()
        r = run_single_model(name, decayed_features_bf, decayed_monotone,
                             data_mode=f"decayed_{hl}", half_life=float(hl))
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 2. Raw backfill + sample weighting ──
    for scheme in ["recency_linear", "recency_exp"]:
        name = f"v13b_{scheme}"
        print(f"\n{'='*80}\n  {name}: backfill + {scheme} weights\n{'='*80}")
        t0 = time.time()
        r = run_single_model(name, v10e_features, v10e_monotone,
                             data_mode="backfill", weight_scheme=scheme)
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 3. Interaction constraints: BF only splits after SP ──
    # Features: [0]=shadow_price_da, [1]=da_rank_value, [2]=bf_6, [3]=bf_12,
    #           [4]=bf_15, [5]=rank_x_bf12, [6]=bf_trend_3_12, [7]=sp_x_bf12
    # Constraint: SP features (0,1) can interact freely with everything.
    # BF features (2,3,4,5,6,7) can only interact with SP features (0,1), not each other.
    ic_sp_bf = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7]]
    name = "v13c_interact_constrained"
    print(f"\n{'='*80}\n  {name}: backfill + interaction constraints\n{'='*80}")
    t0 = time.time()
    r = run_single_model(name, v10e_features, v10e_monotone,
                         data_mode="backfill",
                         interaction_constraints=ic_sp_bf)
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 4. Two-model ensemble ──
    sp_feats = ["shadow_price_da", "da_rank_value"]
    sp_mono = [1, -1]
    bf_feats = ["bf_6", "bf_12", "bf_15", "bf_24", "bf_trend_3_12",
                "rank_x_bf12", "sp_x_bf12"]
    bf_mono = [1, 1, 1, 1, 0, 1, 1]

    for sp_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        name = f"v13d_ensemble_sp{int(sp_w*100)}"
        print(f"\n{'='*80}\n  {name}: 2-model ensemble, SP weight={sp_w}\n{'='*80}")
        t0 = time.time()
        r = run_two_model_ensemble(name, sp_feats, sp_mono, bf_feats, bf_mono,
                                  data_mode="backfill", sp_weight=sp_w)
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 5. Decayed BF + sample weighting (combo) ──
    best_hl = max(
        [k for k in all_results if k.startswith("v13a_")],
        key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"],
    )
    best_hl_val = int(best_hl.split("hl")[1])
    print(f"\n  Best decay half-life: {best_hl_val} ({best_hl})")

    for scheme in ["recency_linear", "recency_exp"]:
        name = f"v13e_decayed_hl{best_hl_val}_{scheme}"
        print(f"\n{'='*80}\n  {name}: decayed BF + {scheme}\n{'='*80}")
        t0 = time.time()
        r = run_single_model(name, decayed_features_bf, decayed_monotone,
                             data_mode=f"decayed_{best_hl_val}", half_life=float(best_hl_val),
                             weight_scheme=scheme)
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 6. Two-model ensemble with decayed BF ──
    bf_feats_d = ["bfd_6", "bfd_12", "bfd_15", "bfd_trend_3_12",
                  "rank_x_bfd12", "sp_x_bfd12"]
    bf_mono_d = [1, 1, 1, 0, 1, 1]

    for sp_w in [0.4, 0.5, 0.6]:
        name = f"v13f_ens_decayed_sp{int(sp_w*100)}"
        print(f"\n{'='*80}\n  {name}: 2-model ensemble + decayed BF, SP={sp_w}\n{'='*80}")
        t0 = time.time()
        r = run_two_model_ensemble(name, sp_feats, sp_mono, bf_feats_d, bf_mono_d,
                                  data_mode=f"decayed_{best_hl_val}", half_life=float(best_hl_val),
                                  sp_weight=sp_w)
        all_results[name] = r
        print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 7. Backfill + NO interaction features (test if interactions are the problem) ──
    lean_bf_features = ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_15", "bf_24"]
    lean_bf_monotone = [1, -1, 1, 1, 1, 1]
    name = "v13g_backfill_no_interactions"
    print(f"\n{'='*80}\n  {name}: backfill + lean BF (no interactions)\n{'='*80}")
    t0 = time.time()
    r = run_single_model(name, lean_bf_features, lean_bf_monotone, data_mode="backfill")
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 8. Backfill + lean + sample weighting (best simple combo?) ──
    name = "v13h_backfill_lean_recency_exp"
    print(f"\n{'='*80}\n  {name}: backfill + lean BF + exp weights\n{'='*80}")
    t0 = time.time()
    r = run_single_model(name, lean_bf_features, lean_bf_monotone,
                         data_mode="backfill", weight_scheme="recency_exp")
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── 9. Decayed lean (no interactions) ──
    decayed_lean = ["shadow_price_da", "da_rank_value", "bfd_6", "bfd_12", "bfd_15"]
    decayed_lean_mono = [1, -1, 1, 1, 1]
    name = f"v13i_decayed_lean_hl{best_hl_val}"
    print(f"\n{'='*80}\n  {name}: decayed lean BF\n{'='*80}")
    t0 = time.time()
    r = run_single_model(name, decayed_lean, decayed_lean_mono,
                         data_mode=f"decayed_{best_hl_val}", half_life=float(best_hl_val))
    all_results[name] = r
    print(f"  VC@20={r['aggregate']['mean']['VC@20']:.4f} ({time.time()-t0:.1f}s)")

    # ── Print final results ──
    print_results(all_results)

    # ── Find overall best and run formula blending ──
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"])
    best_vc20 = all_results[best_name]["aggregate"]["mean"]["VC@20"]
    ref_vc20 = all_results["v13_ref_no_backfill"]["aggregate"]["mean"]["VC@20"]
    print(f"\n  BEST: {best_name} (VC@20={best_vc20:.4f}, vs no_backfill={ref_vc20:.4f}, delta={best_vc20-ref_vc20:+.4f})")

    # Per-group comparison for top 3
    top3 = sorted(all_results.keys(), key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"], reverse=True)[:3]
    print(f"\n{'='*120}")
    print(f"  PER-GROUP VC@20 COMPARISON (top 3 vs reference)")
    print(f"{'='*120}")
    header = f"{'Group':<15}"
    header += f"{'no_backfill':>15}"
    for n in top3:
        short = n.replace("v13", "").replace("_", " ").strip()[:15]
        header += f"{short:>15}"
    print(header)
    print("-" * 120)

    for gid in sorted(DEFAULT_EVAL_GROUPS):
        row = f"{gid:<15}"
        ref_val = all_results["v13_ref_no_backfill"]["per_month"][gid]["VC@20"]
        row += f"{ref_val:>15.4f}"
        for n in top3:
            val = all_results[n]["per_month"][gid]["VC@20"]
            delta = val - ref_val
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
            row += f"{val:>12.4f}{marker:>3}"
        print(row)

    total = time.time() - t_total
    print(f"\n[main] Total walltime: {total:.1f}s")

    # Save results
    save_dir = REGISTRY_DIR / "v13_backfill_strategies"
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "mean": res["aggregate"]["mean"],
            "bottom_2_mean": res["aggregate"]["bottom_2_mean"],
            "feature_importance": res.get("feature_importance", {}),
        }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[main] Results saved to {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
