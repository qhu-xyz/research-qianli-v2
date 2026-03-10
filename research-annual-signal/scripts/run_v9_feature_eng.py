"""v9 feature engineering: push annual signal performance beyond v8b.

Strategy:
1. Expanded BF windows (1/3/6/12/15/24/36/48) with proper NaN for insufficient data
2. Engineered features:
   - bf_trend: bf_3 - bf_12 (recent trend -- accelerating or decaying?)
   - bf_max: max(bf_1..bf_48) (peak binding across any window)
   - sp_x_bf12: shadow_price_da * bf_12 (interaction: high hist SP + recent binding)
   - bf_months_avail: how many months of realized DA data exist (data quality signal)
3. Feature pruning: remove low-importance features, test minimal sets
4. Hyperparameter: try more/fewer trees, different learning rates

All use LightGBM LambdaRank with tiered labels.
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
from ml.data_loader import load_v61_enriched_bf
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.features import prepare_features, compute_query_groups
from ml.train import train_ltr_model, predict_scores

REGISTRY_DIR = _PROJECT_ROOT / "registry"
DISPLAY_METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]


def _get_train_groups(eval_year: str) -> list[str]:
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == eval_year:
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError(f"No split for year: {eval_year}")


def load_bf_group_engineered(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load BF-enriched data and add engineered features."""
    df = load_v61_enriched_bf(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)

    # -- Engineered features --

    # bf_trend: recent (bf_3) minus long-term (bf_12) -- positive = accelerating
    if "bf_3" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(
            (pl.col("bf_3") - pl.col("bf_12")).alias("bf_trend_3_12")
        )

    # bf_trend short: bf_1 - bf_6
    if "bf_1" in df.columns and "bf_6" in df.columns:
        df = df.with_columns(
            (pl.col("bf_1") - pl.col("bf_6")).alias("bf_trend_1_6")
        )

    # bf_max: max binding frequency across all windows (peak signal)
    bf_cols = [c for c in df.columns if c.startswith("bf_") and c != "bf_months_avail" and not c.startswith("bf_trend")]
    if bf_cols:
        df = df.with_columns(
            pl.max_horizontal(*[pl.col(c) for c in bf_cols]).alias("bf_max")
        )

    # sp_x_bf12: interaction of historical shadow price and recent binding
    if "shadow_price_da" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(
            (pl.col("shadow_price_da") * pl.col("bf_12")).alias("sp_x_bf12")
        )

    # sp_x_bf6: shorter window interaction
    if "shadow_price_da" in df.columns and "bf_6" in df.columns:
        df = df.with_columns(
            (pl.col("shadow_price_da") * pl.col("bf_6")).alias("sp_x_bf6")
        )

    # da_rank_x_bf12: interaction of DA rank with binding
    if "da_rank_value" in df.columns and "bf_12" in df.columns:
        df = df.with_columns(
            ((1.0 - pl.col("da_rank_value")) * pl.col("bf_12")).alias("rank_x_bf12")
        )

    return df


# -- Year-level training/assessment --

_year_groups: dict[str, list[str]] = OrderedDict()
for _g in DEFAULT_EVAL_GROUPS:
    _y = _g.split("/")[0]
    _year_groups.setdefault(_y, []).append(_g)


def train_and_assess(variant_name: str, features: list[str], monotone: list[int],
                     n_estimators: int = 100, learning_rate: float = 0.05,
                     num_leaves: int = 31) -> dict:
    """Train and assess a variant across all 12 dev groups."""
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
    all_data: dict[str, pl.DataFrame] = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_group_ids = _get_train_groups(year)
            dfs = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    if gid not in all_data:
                        all_data[gid] = load_bf_group_engineered(py, aq)
                    dfs.append(all_data[gid])
                except FileNotFoundError:
                    pass
            train_df = pl.concat(dfs, how="diagonal").sort("query_group")
            X_train, _ = prepare_features(train_df, config.ltr)
            y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
            groups_train = compute_query_groups(train_df)
            model_cache[year] = train_ltr_model(X_train, y_train, groups_train, config.ltr)
            del train_df, X_train, y_train, groups_train

        model = model_cache[year]
        for gid in groups_in_year:
            py, aq = gid.split("/")
            if gid not in all_data:
                all_data[gid] = load_bf_group_engineered(py, aq)
            df = all_data[gid]
            X, _ = prepare_features(df, config.ltr)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = predict_scores(model, X)
            metrics = evaluate_ltr(actual, scores)

            if hasattr(model, "feature_importance"):
                importance = model.feature_importance(importance_type="gain")
            else:
                importance = model.feature_importances_
            metrics["_feature_importance"] = dict(zip(config.ltr.features, [float(x) for x in importance]))
            per_group[gid] = metrics

    last_gid = list(per_group.keys())[-1]
    feat_imp = per_group[last_gid].get("_feature_importance", {})
    for gid in per_group:
        per_group[gid].pop("_feature_importance", None)

    agg = aggregate_months(per_group)

    del model_cache, all_data
    gc.collect()

    return {
        "per_month": per_group,
        "aggregate": agg,
        "feature_importance": feat_imp,
        "n_months": len(per_group),
    }


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def score_blend(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * _minmax_normalize(ml_scores) + (1.0 - alpha) * _minmax_normalize(formula_scores)


def run_blending_for_best(best_vid: str, best_features: list[str], best_monotone: list[int],
                          best_params: dict) -> dict:
    """Run blending between best variant and v0b formula."""
    config = PipelineConfig(
        ltr=LTRConfig(
            features=best_features,
            monotone_constraints=best_monotone,
            backend="lightgbm",
            label_mode="tiered",
            **best_params,
        ),
    )

    predictions: dict[str, np.ndarray] = {}
    test_data: dict[str, pl.DataFrame] = {}
    model_cache = {}

    for year, groups_in_year in _year_groups.items():
        if year not in model_cache:
            train_group_ids = _get_train_groups(year)
            dfs = []
            for gid in train_group_ids:
                py, aq = gid.split("/")
                try:
                    dfs.append(load_bf_group_engineered(py, aq))
                except FileNotFoundError:
                    pass
            train_df = pl.concat(dfs, how="diagonal").sort("query_group")
            X_train, _ = prepare_features(train_df, config.ltr)
            y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
            groups_train = compute_query_groups(train_df)
            model_cache[year] = train_ltr_model(X_train, y_train, groups_train, config.ltr)
            del train_df, X_train, y_train, groups_train

        model = model_cache[year]
        for gid in groups_in_year:
            py, aq = gid.split("/")
            df = load_bf_group_engineered(py, aq)
            test_data[gid] = df
            X, _ = prepare_features(df, config.ltr)
            predictions[gid] = predict_scores(model, X)

    del model_cache
    gc.collect()

    blend_results = {}
    for alpha_pct in [50, 60, 70, 75, 80, 85, 90, 95, 100]:
        alpha = alpha_pct / 100.0
        name = f"blend_{best_vid}_a{alpha_pct}"
        per_month = {}
        for gid, df in test_data.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            da_rank = df["da_rank_value"].to_numpy().astype(np.float64)
            formula = 1.0 - da_rank
            blended = score_blend(predictions[gid], formula, alpha)
            per_month[gid] = evaluate_ltr(actual, blended)
        agg = aggregate_months(per_month)
        blend_results[name] = {"per_month": per_month, "aggregate": agg, "n_months": len(per_month)}

    # v0b reference
    per_month = {}
    for gid, df in test_data.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        formula = 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)
        per_month[gid] = evaluate_ltr(actual, formula)
    blend_results["v0b_ref"] = {"per_month": per_month, "aggregate": aggregate_months(per_month), "n_months": len(per_month)}

    del test_data
    gc.collect()
    return blend_results


def print_results(results: dict[str, dict]):
    """Print all variants sorted by VC@20."""
    print(f"\n{'='*150}")
    print("  V9 FEATURE ENGINEERING RESULTS (mean over 12 dev groups)")
    print(f"{'='*150}")

    header = f"{'Variant':<40} {'#f':>3}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 150)

    entries = sorted(results.items(), key=lambda x: x[1]["aggregate"]["mean"].get("VC@20", 0), reverse=True)

    for name, res in entries:
        means = res["aggregate"]["mean"]
        nf = len(res.get("feature_importance", {})) or "?"
        row = f"{name:<40} {nf:>3}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"

        fi = res.get("feature_importance", {})
        if fi:
            total = sum(fi.values())
            top3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
            fi_str = ", ".join(f"{k}={v/total*100:.0f}%" for k, v in top3)
            row += f"  [{fi_str}]"

        print(row)

    print(f"\n--- Previous versions ---")
    for vid in ["v0b", "v8b"]:
        p = REGISTRY_DIR / vid / "metrics.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            means = d["aggregate"]["mean"]
            row = f"{vid + ' (prev)':<40} {'':>3}"
            for m in DISPLAY_METRICS:
                row += f"  {means.get(m, 0):>10.4f}"
            print(row)
    blend_p = REGISTRY_DIR / "v8_blending" / "best_blend_metrics.json"
    if blend_p.exists():
        with open(blend_p) as f:
            d = json.load(f)
        means = d["aggregate"]["mean"]
        row = f"{'blend_v8b_a80 (prev best)':<40} {'':>3}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"
        print(row)


def main():
    t_total = time.time()

    # -- Define all variants --
    bf_all = ["bf_1", "bf_3", "bf_6", "bf_12", "bf_15", "bf_24", "bf_36", "bf_48"]
    bf_all_mono = [1] * len(bf_all)

    bf_core = ["bf_3", "bf_6", "bf_12", "bf_24"]
    bf_core_mono = [1] * len(bf_core)

    eng_features = ["bf_trend_3_12", "bf_trend_1_6", "bf_max", "sp_x_bf12", "sp_x_bf6", "rank_x_bf12"]
    eng_mono = [0, 0, 1, 1, 1, 1]

    variants = OrderedDict()

    # v9a: lean + all 8 BF windows + bf_months_avail (NaN-aware)
    variants["v9a_lean_allbf"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_all + ["bf_months_avail"],
        "monotone": [1, -1] + bf_all_mono + [0],
    }

    # v9b: lean + core BF (4 windows) -- pruned
    variants["v9b_lean_corebf"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_core,
        "monotone": [1, -1] + bf_core_mono,
    }

    # v9c: lean + BF + engineered interactions
    variants["v9c_lean_bf_eng"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_core + eng_features,
        "monotone": [1, -1] + bf_core_mono + eng_mono,
    }

    # v9d: just top features -- shadow_price_da + bf_6 + bf_12 + sp_x_bf12
    variants["v9d_ultra_lean"] = {
        "features": ["shadow_price_da", "bf_6", "bf_12", "sp_x_bf12"],
        "monotone": [1, 1, 1, 1],
    }

    # v9e: shadow_price_da + da_rank_value + bf_6 + bf_12 + bf_trend
    variants["v9e_lean_trend"] = {
        "features": ["shadow_price_da", "da_rank_value", "bf_6", "bf_12", "bf_trend_3_12"],
        "monotone": [1, -1, 1, 1, 0],
    }

    # v9f: kitchen sink -- all BF + all engineered + V6.1 lean
    variants["v9f_kitchen_sink"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_all + ["bf_months_avail"] + eng_features,
        "monotone": [1, -1] + bf_all_mono + [0] + eng_mono,
    }

    # v9g: rank_x_bf12 as single interaction + sp_da + da_rank
    variants["v9g_rank_interact"] = {
        "features": ["shadow_price_da", "da_rank_value", "rank_x_bf12", "bf_6", "bf_12"],
        "monotone": [1, -1, 1, 1, 1],
    }

    # v9h: more trees (200) on lean + core BF
    variants["v9h_more_trees"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_core,
        "monotone": [1, -1] + bf_core_mono,
        "n_estimators": 200,
    }

    # v9i: deeper trees (63 leaves) on lean + core BF
    variants["v9i_deeper"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_core,
        "monotone": [1, -1] + bf_core_mono,
        "num_leaves": 63,
    }

    # v9j: slower learning rate (0.01) + 300 trees on lean + core BF
    variants["v9j_slow_lr"] = {
        "features": ["shadow_price_da", "da_rank_value"] + bf_core,
        "monotone": [1, -1] + bf_core_mono,
        "n_estimators": 300,
        "learning_rate": 0.01,
    }

    # -- Run all variants --
    all_results = {}
    for name, spec in variants.items():
        t0 = time.time()
        print(f"\n{'='*80}")
        print(f"  {name}: {len(spec['features'])}f")
        print(f"  Features: {spec['features']}")
        print(f"{'='*80}")

        result = train_and_assess(
            name,
            spec["features"],
            spec["monotone"],
            n_estimators=spec.get("n_estimators", 100),
            learning_rate=spec.get("learning_rate", 0.05),
            num_leaves=spec.get("num_leaves", 31),
        )
        all_results[name] = result

        vc20 = result["aggregate"]["mean"]["VC@20"]
        print(f"  [{name}] VC@20={vc20:.4f} ({time.time()-t0:.1f}s)")

    print_results(all_results)

    # -- Blending with best variant --
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["aggregate"]["mean"]["VC@20"])
    best_spec = variants[best_name]
    print(f"\n{'='*80}")
    print(f"  BLENDING: {best_name} + v0b formula")
    print(f"{'='*80}")

    blend_results = run_blending_for_best(
        best_name,
        best_spec["features"],
        best_spec["monotone"],
        {
            "n_estimators": best_spec.get("n_estimators", 100),
            "learning_rate": best_spec.get("learning_rate", 0.05),
            "num_leaves": best_spec.get("num_leaves", 31),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    )

    print(f"\n{'='*140}")
    print(f"  BLENDING RESULTS")
    print(f"{'='*140}")
    header = f"{'Blend':<40}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 140)

    blend_sorted = sorted(blend_results.items(), key=lambda x: x[1]["aggregate"]["mean"]["VC@20"], reverse=True)
    for name, res in blend_sorted:
        means = res["aggregate"]["mean"]
        row = f"{name:<40}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0):>10.4f}"
        print(row)

    # Save best blend
    blend_dir = REGISTRY_DIR / "v9_blending"
    blend_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: {"mean": v["aggregate"]["mean"], "bottom_2_mean": v["aggregate"]["bottom_2_mean"]} for k, v in blend_results.items()}
    with open(blend_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save all variant results
    for name, res in all_results.items():
        vdir = REGISTRY_DIR / name
        vdir.mkdir(parents=True, exist_ok=True)
        save_data = {
            "per_month": res["per_month"],
            "aggregate": res["aggregate"],
            "n_months": res["n_months"],
            "feature_importance": res.get("feature_importance", {}),
        }
        with open(vdir / "metrics.json", "w") as f:
            json.dump(save_data, f, indent=2)

    total = time.time() - t_total
    print(f"\n[main] Best ML: {best_name} (VC@20={all_results[best_name]['aggregate']['mean']['VC@20']:.4f})")
    best_blend_name = max(blend_results.keys(), key=lambda k: blend_results[k]["aggregate"]["mean"]["VC@20"])
    print(f"[main] Best blend: {best_blend_name} (VC@20={blend_results[best_blend_name]['aggregate']['mean']['VC@20']:.4f})")
    print(f"[main] Total walltime: {total:.1f}s")


if __name__ == "__main__":
    main()
