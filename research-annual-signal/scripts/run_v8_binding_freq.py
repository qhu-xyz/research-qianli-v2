"""v8 experiments: binding frequency features for annual constraint ranking.

Augments V6.1 annual signal with monthly binding frequency (bf_1/3/6/12/24)
derived from realized DA shadow prices. Annual auction submitted ~April,
so BF cutoff = months < planning_year-04 (March data fully available).

Variants:
  v8a: V6.1 + da_rank_value + BF (12 features) — full feature set
  v8b: shadow_price_da + da_rank_value + BF (7 features) — lean
  v8c: BF only (5 features) — pure binding frequency
  v8d: shadow_price_da + bf_12 (2 features) — minimal

All use LightGBM LambdaRank with tiered labels (0/1-4).
Blending: best v8 variant + v0b formula at various alpha.

Data flow:
  load_v61_enriched_bf() loads V6.1 + spice6 + BF features (cached).
  Training uses expanding window (all prior years).
  Eval on 12 dev groups (2022-06 through 2024-06, 4 quarters each).
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
    SET_V8_FEATURES, SET_V8_MONOTONE,
    SET_V8_LEAN_FEATURES, SET_V8_LEAN_MONOTONE,
    SET_V8_BF_ONLY_FEATURES, SET_V8_BF_ONLY_MONOTONE,
)
from ml.data_loader import load_v61_enriched_bf
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.features import prepare_features, compute_query_groups
from ml.train import train_ltr_model, predict_scores

REGISTRY_DIR = _PROJECT_ROOT / "registry"
REPORTS_DIR = _PROJECT_ROOT / "reports"
DISPLAY_METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

# Minimal 2-feature set
V8D_FEATURES = ["shadow_price_da", "bf_12"]
V8D_MONOTONE = [1, 1]


def _get_train_groups(eval_year: str) -> list[str]:
    """Get training groups for an eval year using expanding window."""
    for split_def in EVAL_SPLITS.values():
        if split_def["eval_year"] == eval_year:
            return [f"{y}/{aq}" for y in split_def["train_years"] for aq in AQ_ROUNDS]
    raise ValueError(f"No split for eval year: {eval_year}")


def load_bf_group(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load a single group with BF features + ground truth."""
    df = load_v61_enriched_bf(planning_year, aq_round)
    group_id = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(group_id).alias("query_group"))
    df = get_ground_truth(planning_year, aq_round, df, cache=True)
    return df


def train_for_year_bf(config: PipelineConfig, eval_year: str):
    """Train model for eval year using BF-enriched data."""
    train_group_ids = _get_train_groups(eval_year)
    print(f"\n[train_bf] Training for eval_year={eval_year}, {len(train_group_ids)} groups")

    dfs = []
    for gid in train_group_ids:
        py, aq = gid.split("/")
        try:
            df = load_bf_group(py, aq)
            dfs.append(df)
        except FileNotFoundError:
            print(f"[train_bf] WARNING: skipping {gid} (not found)")

    if not dfs:
        raise ValueError(f"No training data for {eval_year}")

    train_df = pl.concat(dfs, how="diagonal").sort("query_group")
    del dfs
    gc.collect()

    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    print(f"[train_bf] X={X_train.shape}, groups={groups_train}")

    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)

    del train_df, X_train, y_train, groups_train
    gc.collect()
    return model


def evaluate_group_bf(config: PipelineConfig, model, eval_group: str) -> dict:
    """Evaluate model on a single BF-enriched eval group."""
    py, aq = eval_group.split("/")
    df = load_bf_group(py, aq)

    X, _ = prepare_features(df, config.ltr)
    actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
    scores = predict_scores(model, X)
    metrics = evaluate_ltr(actual, scores)

    # Feature importance
    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    }

    del X, scores, actual, df
    gc.collect()
    return metrics


def run_v8_variants():
    """Run all v8 ML variants."""
    t0 = time.time()

    variants = {
        "v8a": {
            "desc": "V6.1 + da_rank + BF (12f), tiered",
            "features": list(SET_V8_FEATURES),
            "monotone": list(SET_V8_MONOTONE),
        },
        "v8b": {
            "desc": "Lean: shadow_price_da + da_rank + BF (7f), tiered",
            "features": list(SET_V8_LEAN_FEATURES),
            "monotone": list(SET_V8_LEAN_MONOTONE),
        },
        "v8c": {
            "desc": "BF only (5f), tiered",
            "features": list(SET_V8_BF_ONLY_FEATURES),
            "monotone": list(SET_V8_BF_ONLY_MONOTONE),
        },
        "v8d": {
            "desc": "shadow_price_da + bf_12 (2f), tiered",
            "features": list(V8D_FEATURES),
            "monotone": list(V8D_MONOTONE),
        },
    }

    # Group eval groups by year
    year_groups: dict[str, list[str]] = OrderedDict()
    for g in DEFAULT_EVAL_GROUPS:
        year = g.split("/")[0]
        year_groups.setdefault(year, []).append(g)

    all_results = {}

    for vid, spec in variants.items():
        print(f"\n{'='*80}")
        print(f"  {vid}: {spec['desc']}")
        print(f"{'='*80}")

        config = PipelineConfig(
            ltr=LTRConfig(
                features=spec["features"],
                monotone_constraints=spec["monotone"],
                backend="lightgbm",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                label_mode="tiered",
            ),
        )

        per_group = {}
        model_cache = {}

        for year, groups_in_year in year_groups.items():
            if year not in model_cache:
                model_cache[year] = train_for_year_bf(config, year)

            model = model_cache[year]
            for gid in groups_in_year:
                print(f"\n  [{vid}] Evaluating {gid}")
                metrics = evaluate_group_bf(config, model, gid)
                per_group[gid] = metrics
                # Print key metrics
                for m in ["VC@20", "VC@100", "Recall@20", "Spearman"]:
                    print(f"    {m}: {metrics[m]:.4f}")

        del model_cache
        gc.collect()

        # Extract and print feature importance from last group
        last_group = list(per_group.keys())[-1]
        feat_imp = per_group[last_group].pop("_feature_importance", {})
        for gid in per_group:
            per_group[gid].pop("_feature_importance", None)

        agg = aggregate_months(per_group)
        result = {
            "eval_config": {"eval_groups": DEFAULT_EVAL_GROUPS, "mode": "eval"},
            "per_month": per_group,
            "aggregate": agg,
            "n_months": len(per_group),
            "n_months_requested": len(DEFAULT_EVAL_GROUPS),
            "skipped_months": [],
        }

        # Save to registry
        version_dir = REGISTRY_DIR / vid
        version_dir.mkdir(parents=True, exist_ok=True)
        with open(version_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(version_dir / "config.json", "w") as f:
            json.dump({"ltr": config.ltr.to_dict(), "variant": spec["desc"]}, f, indent=2)

        all_results[vid] = result

        # Print feature importance
        if feat_imp:
            total_imp = sum(feat_imp.values())
            print(f"\n  [{vid}] Feature importance (gain):")
            for fname, imp in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
                pct = imp / total_imp * 100 if total_imp > 0 else 0
                print(f"    {fname:<30} {pct:6.1f}%")

    print(f"\n[v8_variants] Total walltime: {time.time() - t0:.1f}s")
    return all_results


# ── Blending ──

def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def score_blend(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * _minmax_normalize(ml_scores) + (1.0 - alpha) * _minmax_normalize(formula_scores)


def get_formula_v0b_scores(df: pl.DataFrame) -> np.ndarray:
    da_rank = df["da_rank_value"].to_numpy().astype(np.float64)
    return 1.0 - da_rank


def run_blending(all_results: dict):
    """Blend best v8 variant with v0b formula."""
    t0 = time.time()

    # Find best v8 variant by mean VC@20
    best_vid = max(
        all_results.keys(),
        key=lambda v: all_results[v]["aggregate"]["mean"]["VC@20"],
    )
    print(f"\n[blend] Best v8 variant: {best_vid} (VC@20={all_results[best_vid]['aggregate']['mean']['VC@20']:.4f})")

    # Also blend v8b (lean) — likely most practical
    blend_candidates = [best_vid]
    if "v8b" not in blend_candidates:
        blend_candidates.append("v8b")

    # Group eval groups by year
    year_groups: dict[str, list[str]] = OrderedDict()
    for g in DEFAULT_EVAL_GROUPS:
        year = g.split("/")[0]
        year_groups.setdefault(year, []).append(g)

    # Train models and get predictions
    ml_predictions: dict[str, dict[str, np.ndarray]] = {}

    for vid in blend_candidates:
        spec_map = {
            "v8a": (list(SET_V8_FEATURES), list(SET_V8_MONOTONE)),
            "v8b": (list(SET_V8_LEAN_FEATURES), list(SET_V8_LEAN_MONOTONE)),
            "v8c": (list(SET_V8_BF_ONLY_FEATURES), list(SET_V8_BF_ONLY_MONOTONE)),
            "v8d": (list(V8D_FEATURES), list(V8D_MONOTONE)),
        }
        features, monotone = spec_map[vid]
        config = PipelineConfig(
            ltr=LTRConfig(
                features=features,
                monotone_constraints=monotone,
                backend="lightgbm",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                label_mode="tiered",
            ),
        )

        print(f"\n[blend] Training {vid} for predictions...")
        ml_predictions[vid] = {}
        model_cache = {}

        for year, groups_in_year in year_groups.items():
            if year not in model_cache:
                model_cache[year] = train_for_year_bf(config, year)
            model = model_cache[year]
            for gid in groups_in_year:
                py, aq = gid.split("/")
                df = load_bf_group(py, aq)
                X, _ = prepare_features(df, config.ltr)
                ml_predictions[vid][gid] = predict_scores(model, X)

        del model_cache
        gc.collect()

    # Load test data for formula scores and ground truth
    test_groups: dict[str, pl.DataFrame] = {}
    for gid in DEFAULT_EVAL_GROUPS:
        py, aq = gid.split("/")
        test_groups[gid] = load_bf_group(py, aq)

    # Evaluate blends
    blend_results = {}

    for vid in blend_candidates:
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            name = f"blend_{vid}_a{int(alpha*100)}"
            per_month = {}
            for gid, df in test_groups.items():
                actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
                blended = score_blend(ml_predictions[vid][gid], get_formula_v0b_scores(df), alpha)
                per_month[gid] = evaluate_ltr(actual, blended)
            agg = aggregate_months(per_month)
            blend_results[name] = {"per_month": per_month, "aggregate": agg, "n_months": len(per_month)}

    # Also add standalone references
    for ref_name, score_fn in [("v0b_ref", lambda df: get_formula_v0b_scores(df))]:
        per_month = {}
        for gid, df in test_groups.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            per_month[gid] = evaluate_ltr(actual, score_fn(df))
        blend_results[ref_name] = {"per_month": per_month, "aggregate": aggregate_months(per_month), "n_months": len(per_month)}

    for vid in blend_candidates:
        per_month = {}
        for gid, df in test_groups.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            per_month[gid] = evaluate_ltr(actual, ml_predictions[vid][gid])
        blend_results[f"{vid}_ref"] = {"per_month": per_month, "aggregate": aggregate_months(per_month), "n_months": len(per_month)}

    del test_groups
    gc.collect()

    print(f"\n[blend] Blending done in {time.time() - t0:.1f}s")
    return blend_results


def print_comparison_table(all_results: dict, blend_results: dict):
    """Print comprehensive comparison table."""
    print(f"\n{'='*140}")
    print("  V8 RESULTS (mean over 12 dev groups)")
    print(f"{'='*140}")

    header = f"{'Name':<30}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 140)

    # Collect all entries
    entries = []

    # v0b reference
    if "v0b_ref" in blend_results:
        entries.append(("v0b (formula)", blend_results["v0b_ref"]["aggregate"]["mean"]))

    # v8 ML variants
    for vid in sorted(all_results.keys()):
        means = all_results[vid]["aggregate"]["mean"]
        entries.append((vid, means))

    # ML standalone references from blending
    for vid in sorted(all_results.keys()):
        ref_key = f"{vid}_ref"
        if ref_key in blend_results:
            means = blend_results[ref_key]["aggregate"]["mean"]
            # Skip — same as above (trained identically)

    # Blends
    for name in sorted(blend_results.keys()):
        if name.endswith("_ref"):
            continue
        if name == "v0b_ref":
            continue
        entries.append((name, blend_results[name]["aggregate"]["mean"]))

    # Sort by VC@20 descending
    entries.sort(key=lambda x: x[1].get("VC@20", 0), reverse=True)

    for name, means in entries:
        row = f"{name:<30}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)

    # Also print v7d and v0b from registry for context
    print(f"\n--- Previous versions (from registry) ---")
    for vid in ["v0b", "v7d"]:
        metrics_path = REGISTRY_DIR / vid / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            means = data.get("aggregate", {}).get("mean", {})
            row = f"{vid:<30}"
            for m in DISPLAY_METRICS:
                row += f"  {means.get(m, 0.0):>10.4f}"
            print(row)
    blend_path = REGISTRY_DIR / "v7_blending" / "best_blend_metrics.json"
    if blend_path.exists():
        with open(blend_path) as f:
            data = json.load(f)
        means = data.get("aggregate", {}).get("mean", {})
        row = f"{'blend_v7d_a70 (prev best)':<30}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)


def save_blend_results(blend_results: dict):
    """Save blend results to registry."""
    blend_dir = REGISTRY_DIR / "v8_blending"
    blend_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, result in blend_results.items():
        summary[name] = {
            "mean": result["aggregate"]["mean"],
            "bottom_2_mean": result["aggregate"]["bottom_2_mean"],
        }

    with open(blend_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_name = max(
        blend_results.keys(),
        key=lambda n: blend_results[n]["aggregate"]["mean"]["VC@20"],
    )
    best = blend_results[best_name]
    best_metrics = {
        "eval_config": {"eval_groups": DEFAULT_EVAL_GROUPS, "mode": "eval"},
        "per_month": best["per_month"],
        "aggregate": best["aggregate"],
        "n_months": best["n_months"],
        "n_months_requested": len(DEFAULT_EVAL_GROUPS),
        "skipped_months": [],
    }
    with open(blend_dir / "best_blend_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\n[save] Blend summary -> {blend_dir / 'summary.json'}")
    print(f"[save] Best blend ({best_name}) -> {blend_dir / 'best_blend_metrics.json'}")
    return best_name


def main():
    t_total = time.time()

    # Part 1: ML variants
    print("\n" + "="*80)
    print("  PART 1: V8 ML VARIANTS (binding frequency)")
    print("="*80)
    all_results = run_v8_variants()

    # Part 2: Blending
    print("\n" + "="*80)
    print("  PART 2: V8 BLENDING (best ML + v0b formula)")
    print("="*80)
    blend_results = run_blending(all_results)

    # Results
    print_comparison_table(all_results, blend_results)
    best_blend = save_blend_results(blend_results)
    print(f"\n[result] Best v8 blend: {best_blend}")

    total = time.time() - t_total
    print(f"\n[main] Total walltime: {total:.1f}s")


if __name__ == "__main__":
    main()
