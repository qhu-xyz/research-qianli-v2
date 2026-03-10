"""v6: Alternative baselines and ML+formula blending experiments.

Evaluates:
  Alternative baselines (no training):
    - da_raw:    shadow_price_da directly as score
    - da_only:   1 - da_rank_value (drop density components)
    - product:   shadow_price_da * mean_branch_max

  Blending (v1 ML + v0 formula):
    - score_blend_XX: alpha * norm(ML) + (1-alpha) * norm(formula), alpha in {0.3, 0.5, 0.7}
    - rank_blend:     average of rank positions
    - rrf:            Reciprocal Rank Fusion with k=60

Usage:
  PYTHONPATH=. python scripts/run_v6_baselines_and_blending.py --screen   # 4 groups
  PYTHONPATH=. python scripts/run_v6_baselines_and_blending.py            # 12 groups
"""
import argparse
import gc
import json
import resource
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

from ml.config import (
    PipelineConfig, LTRConfig,
    SET_A_FEATURES, SET_A_MONOTONE,
    SCREEN_EVAL_GROUPS, DEFAULT_EVAL_GROUPS,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.features import prepare_features
from ml.ground_truth import get_ground_truth
from ml.pipeline import train_for_year
from ml.train import predict_scores


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Scoring functions for alternative baselines
# ---------------------------------------------------------------------------

def score_da_raw(df: pl.DataFrame) -> np.ndarray:
    """Use shadow_price_da directly (higher = more binding)."""
    return df["shadow_price_da"].fill_null(0.0).to_numpy().astype(np.float64)


def score_da_only(df: pl.DataFrame) -> np.ndarray:
    """Use 1 - da_rank_value (dropping density components)."""
    return 1.0 - df["da_rank_value"].fill_null(1.0).to_numpy().astype(np.float64)


def score_product(df: pl.DataFrame) -> np.ndarray:
    """shadow_price_da * mean_branch_max (combine backward + forward signals)."""
    da = df["shadow_price_da"].fill_null(0.0).to_numpy().astype(np.float64)
    mbm = df["mean_branch_max"].fill_null(0.0).to_numpy().astype(np.float64)
    return da * mbm


def score_formula(df: pl.DataFrame) -> np.ndarray:
    """V6.1 formula: 1 - rank_ori (same as v0)."""
    return 1.0 - df["rank"].fill_null(1.0).to_numpy().astype(np.float64)


BASELINES = OrderedDict([
    ("v0_formula", score_formula),
    ("da_raw", score_da_raw),
    ("da_only", score_da_only),
    ("product", score_product),
])


# ---------------------------------------------------------------------------
# Blending helpers
# ---------------------------------------------------------------------------

def _normalize_01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Rank from 1 (lowest score) to N (highest score)."""
    order = np.argsort(np.argsort(x))  # 0-based rank of ascending
    return order.astype(np.float64) + 1.0


def blend_score(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * norm(ML) + (1-alpha) * norm(formula)."""
    return alpha * _normalize_01(ml_scores) + (1 - alpha) * _normalize_01(formula_scores)


def blend_rank(ml_scores: np.ndarray, formula_scores: np.ndarray) -> np.ndarray:
    """Average of rank positions (higher rank = better)."""
    return (_rankdata(ml_scores) + _rankdata(formula_scores)) / 2.0


def blend_rrf(ml_scores: np.ndarray, formula_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: 1/(k + rank_ML) + 1/(k + rank_formula)."""
    r_ml = _rankdata(ml_scores)
    r_form = _rankdata(formula_scores)
    # Higher rank number = better score, so RRF gives higher score to high-ranked items
    return 1.0 / (k + (len(r_ml) + 1 - r_ml)) + 1.0 / (k + (len(r_form) + 1 - r_form))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(eval_groups: list[str], mode: str, ml_version: str = "v1"):
    """Run all baseline and blending experiments."""
    registry_dir = Path(__file__).resolve().parent.parent / "registry"

    # ML config: use v1 (Set A, rank labels) as the ML component for blending
    ml_config = PipelineConfig(
        ltr=LTRConfig(
            features=SET_A_FEATURES,
            monotone_constraints=SET_A_MONOTONE,
            backend="lightgbm",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            label_mode="rank",
        ),
    )

    # Group by year for efficient model training
    year_groups: dict[str, list[str]] = OrderedDict()
    for g in eval_groups:
        year = g.split("/")[0]
        year_groups.setdefault(year, []).append(g)

    # All variants we evaluate
    variant_names = (
        list(BASELINES.keys())
        + ["ml_v1"]
        + [f"score_blend_{int(a*100):02d}" for a in [0.3, 0.5, 0.7]]
        + ["rank_blend", "rrf"]
    )
    # Per-group metrics for each variant
    all_results: dict[str, dict[str, dict]] = {v: {} for v in variant_names}

    model_cache = {}

    for year, groups_in_year in year_groups.items():
        # Train ML model once per year
        if year not in model_cache:
            print(f"\n[v6] Training ML model for eval_year={year} ... mem: {mem_mb():.0f} MB")
            model_cache[year] = train_for_year(ml_config, year)

        model = model_cache[year]

        for group_id in groups_in_year:
            planning_year, aq_round = group_id.split("/")
            print(f"\n[v6] === {group_id} === mem: {mem_mb():.0f} MB")

            # Load data + ground truth
            df = load_v61_enriched(planning_year, aq_round)
            df = df.with_columns(pl.lit(group_id).alias("query_group"))
            df = get_ground_truth(planning_year, aq_round, df, cache=True)
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)

            # 1) Alternative baselines
            for bname, bfunc in BASELINES.items():
                scores = bfunc(df)
                all_results[bname][group_id] = evaluate_ltr(actual, scores)

            # 2) ML scores
            X_test, _ = prepare_features(df, ml_config.ltr)
            ml_scores = predict_scores(model, X_test)
            all_results["ml_v1"][group_id] = evaluate_ltr(actual, ml_scores)

            # 3) Formula scores for blending
            formula_scores = score_formula(df)

            # 4) Score blending at different alphas
            for alpha in [0.3, 0.5, 0.7]:
                vname = f"score_blend_{int(alpha*100):02d}"
                blended = blend_score(ml_scores, formula_scores, alpha)
                all_results[vname][group_id] = evaluate_ltr(actual, blended)

            # 5) Rank blend
            blended = blend_rank(ml_scores, formula_scores)
            all_results["rank_blend"][group_id] = evaluate_ltr(actual, blended)

            # 6) RRF
            blended = blend_rrf(ml_scores, formula_scores, k=60)
            all_results["rrf"][group_id] = evaluate_ltr(actual, blended)

            del df, X_test, ml_scores, formula_scores, actual
            gc.collect()

    del model_cache
    gc.collect()

    # Aggregate and print comparison
    key_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG", "Spearman", "Tier0-AP"]

    print("\n" + "=" * 100)
    print("COMPARISON TABLE (mean across eval groups)")
    print("=" * 100)

    header = f"{'Variant':<22}" + "".join(f"{m:>12}" for m in key_metrics)
    print(header)
    print("-" * len(header))

    agg_all = {}
    for vname in variant_names:
        agg = aggregate_months(all_results[vname])
        agg_all[vname] = agg
        means = agg.get("mean", {})
        row = f"{vname:<22}"
        for m in key_metrics:
            val = means.get(m, 0.0)
            row += f"{val:>12.4f}"
        print(row)

    # Find best variant per metric
    print("\n" + "-" * len(header))
    best_row = f"{'BEST':<22}"
    for m in key_metrics:
        best_val = -1
        best_name = ""
        for vname in variant_names:
            val = agg_all[vname].get("mean", {}).get(m, 0.0)
            if val > best_val:
                best_val = val
                best_name = vname
        best_row += f"{best_name:>12}"
    print(best_row)

    # Delta vs v0_formula
    print(f"\n{'Deltas vs v0_formula':}")
    v0_means = agg_all["v0_formula"].get("mean", {})
    for vname in variant_names:
        if vname == "v0_formula":
            continue
        means = agg_all[vname].get("mean", {})
        row = f"  {vname:<20}"
        for m in key_metrics:
            v0_val = v0_means.get(m, 0.0)
            val = means.get(m, 0.0)
            delta_pct = 100 * (val - v0_val) / v0_val if v0_val != 0 else 0
            row += f"{delta_pct:>+11.1f}%"
        print(row)

    # Save all results to registry
    report_dir = Path(__file__).resolve().parent.parent / "registry" / "v6_exploration"
    report_dir.mkdir(parents=True, exist_ok=True)

    for vname in variant_names:
        result = {
            "eval_config": {"eval_groups": eval_groups, "mode": mode},
            "per_month": all_results[vname],
            "aggregate": agg_all[vname],
            "n_months": len(all_results[vname]),
            "n_months_requested": len(eval_groups),
            "skipped_months": [],
        }
        with open(report_dir / f"{vname}_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary JSON
    summary = {}
    for vname in variant_names:
        summary[vname] = {m: agg_all[vname].get("mean", {}).get(m, 0.0) for m in key_metrics}
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[v6] Results saved to {report_dir}/")
    print(f"[v6] Complete: {len(eval_groups)} groups, {len(variant_names)} variants")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()

    groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
    mode = "screen" if args.screen else "eval"

    run_experiment(groups, mode)
