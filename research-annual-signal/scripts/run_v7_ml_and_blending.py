"""v7 experiments: ML rebase on v0b + blending strategies.

Part 1 — ML variants against stronger v0b baseline:
  v7a: Set A (6 features) with tiered labels — same as v5 but without rank_ori
  v7b: Lean features (4f, drop density rank features) with tiered labels
  v7c: Lean + da_rank_value as 5th feature (replace rank_ori with pure DA rank)
  v7d: Set A + da_rank_value (7 features) — full features + clean formula feature

Part 2 — Blending ML + formula (v0b):
  Score blend: alpha * ML_score + (1-alpha) * formula_score (both min-max normalized)
  Rank blend:  average of ranks from ML and formula
  RRF:         1/(k+rank_ML) + 1/(k+rank_formula), k=60

Uses v1 (best gate-passing ML) and v5 (best mean ML) as blend candidates.
"""
import json
import gc
import time
import sys
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.benchmark import run_benchmark
from ml.config import (
    PipelineConfig, LTRConfig,
    SET_A_FEATURES, SET_A_MONOTONE,
    DEFAULT_EVAL_GROUPS,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.pipeline import train_for_year, evaluate_group
from ml.features import prepare_features, compute_query_groups
from ml.train import predict_scores
from ml.compare import run_comparison

REGISTRY_DIR = _PROJECT_ROOT / "registry"
REPORTS_DIR = _PROJECT_ROOT / "reports"
DISPLAY_METRICS = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]

# ── Feature sets ──

# Lean: drop density rank features (keep only signal + forward features)
LEAN_FEATURES = ["shadow_price_da", "mean_branch_max", "ori_mean", "mix_mean"]
LEAN_MONOTONE = [1, 1, 1, 1]

# Lean + da_rank_value as clean formula feature
LEAN_DA_FEATURES = LEAN_FEATURES + ["da_rank_value"]
LEAN_DA_MONOTONE = LEAN_MONOTONE + [-1]  # lower da_rank = more binding

# Set A + da_rank_value (full features + clean formula signal)
FULL_DA_FEATURES = list(SET_A_FEATURES) + ["da_rank_value"]
FULL_DA_MONOTONE = list(SET_A_MONOTONE) + [-1]


def load_all_test_groups() -> dict[str, pl.DataFrame]:
    """Load all 12 dev eval groups with ground truth."""
    groups = {}
    for group_id in DEFAULT_EVAL_GROUPS:
        planning_year, aq_round = group_id.split("/")
        df = load_v61_enriched(planning_year, aq_round)
        df = get_ground_truth(planning_year, aq_round, df, cache=True)
        groups[group_id] = df
    return groups


# ── Part 1: ML variants ──

def run_ml_variants():
    """Run ML variants v7a-v7d."""
    t0 = time.time()
    variants = {
        "v7a": {
            "desc": "Set A (6f), tiered labels — v5 without rank_ori",
            "features": list(SET_A_FEATURES),
            "monotone": list(SET_A_MONOTONE),
            "label_mode": "tiered",
        },
        "v7b": {
            "desc": "Lean (4f, no density rank), tiered labels",
            "features": list(LEAN_FEATURES),
            "monotone": list(LEAN_MONOTONE),
            "label_mode": "tiered",
        },
        "v7c": {
            "desc": "Lean + da_rank_value (5f), tiered labels",
            "features": list(LEAN_DA_FEATURES),
            "monotone": list(LEAN_DA_MONOTONE),
            "label_mode": "tiered",
        },
        "v7d": {
            "desc": "Set A + da_rank_value (7f), tiered labels",
            "features": list(FULL_DA_FEATURES),
            "monotone": list(FULL_DA_MONOTONE),
            "label_mode": "tiered",
        },
    }

    results = {}
    for vid, spec in variants.items():
        print(f"\n{'='*80}")
        print(f"  Running {vid}: {spec['desc']}")
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
                label_mode=spec["label_mode"],
            ),
        )

        result = run_benchmark(
            version_id=vid,
            eval_groups=DEFAULT_EVAL_GROUPS,
            config=config,
            mode="eval",
        )
        results[vid] = result
        gc.collect()

    print(f"\n[ml_variants] Total walltime: {time.time() - t0:.1f}s")
    return results


# ── Part 2: Blending ──

def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks (1 = best). Higher score = rank 1."""
    return len(scores) - np.argsort(np.argsort(scores)).astype(np.float64)


def score_blend(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * ML + (1-alpha) * formula, both min-max normalized."""
    ml_norm = _minmax_normalize(ml_scores)
    f_norm = _minmax_normalize(formula_scores)
    return alpha * ml_norm + (1.0 - alpha) * f_norm


def rank_blend(ml_scores: np.ndarray, formula_scores: np.ndarray) -> np.ndarray:
    """Average ranks from ML and formula. Return inverted (higher = better)."""
    ml_rank = _rank_scores(ml_scores)
    f_rank = _rank_scores(formula_scores)
    avg_rank = (ml_rank + f_rank) / 2.0
    return -avg_rank  # negate so higher = better


def rrf(ml_scores: np.ndarray, formula_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion. Higher = better."""
    ml_rank = _rank_scores(ml_scores)
    f_rank = _rank_scores(formula_scores)
    return 1.0 / (k + ml_rank) + 1.0 / (k + f_rank)


def get_formula_v0b_scores(df: pl.DataFrame) -> np.ndarray:
    """v0b formula: pure da_rank_value (inverted so higher = more binding)."""
    da_rank = df["da_rank_value"].to_numpy().astype(np.float64)
    return 1.0 - da_rank


def run_blending_experiments(test_groups: dict[str, pl.DataFrame]):
    """Run blending experiments between ML models and v0b formula."""
    t0 = time.time()

    # We need ML predictions. Train v1 and v5 models, get per-group scores
    ml_configs = {
        "v1": PipelineConfig(
            ltr=LTRConfig(
                features=list(SET_A_FEATURES),
                monotone_constraints=list(SET_A_MONOTONE),
                backend="lightgbm",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                label_mode="rank",
            ),
        ),
        "v7d": PipelineConfig(
            ltr=LTRConfig(
                features=list(FULL_DA_FEATURES),
                monotone_constraints=list(FULL_DA_MONOTONE),
                backend="lightgbm",
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                label_mode="tiered",
            ),
        ),
    }

    # Train models and get predictions per group
    ml_predictions: dict[str, dict[str, np.ndarray]] = {}  # ml_id -> {group_id -> scores}

    from collections import OrderedDict
    year_groups: dict[str, list[str]] = OrderedDict()
    for g in DEFAULT_EVAL_GROUPS:
        year = g.split("/")[0]
        year_groups.setdefault(year, []).append(g)

    for ml_id, config in ml_configs.items():
        print(f"\n[blend] Training {ml_id} for blending predictions...")
        ml_predictions[ml_id] = {}
        model_cache = {}

        for year, groups_in_year in year_groups.items():
            if year not in model_cache:
                model_cache[year] = train_for_year(config, year)

            model = model_cache[year]
            for group_id in groups_in_year:
                df = test_groups[group_id]
                X, _ = prepare_features(df, config.ltr)
                scores = predict_scores(model, X)
                ml_predictions[ml_id][group_id] = scores

        del model_cache
        gc.collect()

    # Define blending strategies
    blend_strategies = []

    for ml_id in ml_configs:
        # Score blends at various alpha
        for alpha in [0.3, 0.5, 0.7]:
            blend_strategies.append({
                "name": f"score_blend_{ml_id}_a{int(alpha*100)}",
                "ml_id": ml_id,
                "fn": lambda ml_s, f_s, a=alpha: score_blend(ml_s, f_s, a),
            })

        # Rank blend
        blend_strategies.append({
            "name": f"rank_blend_{ml_id}",
            "ml_id": ml_id,
            "fn": lambda ml_s, f_s: rank_blend(ml_s, f_s),
        })

        # RRF
        blend_strategies.append({
            "name": f"rrf_{ml_id}",
            "ml_id": ml_id,
            "fn": lambda ml_s, f_s: rrf(ml_s, f_s),
        })

    # Evaluate all blending strategies
    blend_results = {}

    for strat in blend_strategies:
        name = strat["name"]
        ml_id = strat["ml_id"]
        blend_fn = strat["fn"]

        per_month = {}
        for group_id, df in test_groups.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            ml_scores = ml_predictions[ml_id][group_id]
            formula_scores = get_formula_v0b_scores(df)
            blended = blend_fn(ml_scores, formula_scores)
            metrics = evaluate_ltr(actual, blended)
            per_month[group_id] = metrics

        agg = aggregate_months(per_month)
        blend_results[name] = {
            "per_month": per_month,
            "aggregate": agg,
            "n_months": len(per_month),
        }

    # Also evaluate v0b and ML standalone for comparison
    for ref_name, score_fn in [("v0b_ref", get_formula_v0b_scores)]:
        per_month = {}
        for group_id, df in test_groups.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = score_fn(df)
            metrics = evaluate_ltr(actual, scores)
            per_month[group_id] = metrics
        agg = aggregate_months(per_month)
        blend_results[ref_name] = {
            "per_month": per_month,
            "aggregate": agg,
            "n_months": len(per_month),
        }

    for ml_id in ml_configs:
        per_month = {}
        for group_id, df in test_groups.items():
            actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
            scores = ml_predictions[ml_id][group_id]
            metrics = evaluate_ltr(actual, scores)
            per_month[group_id] = metrics
        agg = aggregate_months(per_month)
        blend_results[f"{ml_id}_ref"] = {
            "per_month": per_month,
            "aggregate": agg,
            "n_months": len(per_month),
        }

    print(f"\n[blend] Blending experiments done in {time.time() - t0:.1f}s")
    return blend_results


def print_blend_table(blend_results: dict):
    """Print blending results comparison table."""
    print(f"\n{'='*120}")
    print("  BLENDING RESULTS (mean over 12 groups)")
    print(f"{'='*120}")

    header = f"{'Name':<35}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 120)

    # Sort by VC@20 descending
    sorted_names = sorted(
        blend_results.keys(),
        key=lambda n: blend_results[n]["aggregate"]["mean"]["VC@20"],
        reverse=True,
    )

    for name in sorted_names:
        means = blend_results[name]["aggregate"]["mean"]
        row = f"{name:<35}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)


def save_blend_results(blend_results: dict):
    """Save blending results to registry."""
    blend_dir = REGISTRY_DIR / "v7_blending"
    blend_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, result in blend_results.items():
        summary[name] = {
            "mean": result["aggregate"]["mean"],
            "bottom_2_mean": result["aggregate"]["bottom_2_mean"],
        }

    with open(blend_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full per-month data for the best blend
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

    print(f"[save] Blend summary -> {blend_dir / 'summary.json'}")
    print(f"[save] Best blend ({best_name}) -> {blend_dir / 'best_blend_metrics.json'}")
    return best_name


def print_ml_comparison():
    """Print comparison table for all ML variants."""
    print(f"\n{'='*120}")
    print("  ML VARIANT COMPARISON (mean over 12 groups)")
    print(f"{'='*120}")

    header = f"{'Version':<12}  {'Description':<45}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 120)

    versions_to_show = [
        ("v0", "V6.1 formula (0.60/0.30/0.10)"),
        ("v0b", "Pure da_rank_value (alpha=1.0)"),
        ("v1", "ML Set A (6f), rank labels"),
        ("v5", "ML Set AF (7f), tiered labels"),
        ("v7a", "ML Set A (6f), tiered labels"),
        ("v7b", "ML Lean (4f), tiered labels"),
        ("v7c", "ML Lean+da_rank (5f), tiered"),
        ("v7d", "ML Set A+da_rank (7f), tiered"),
    ]

    for vid, desc in versions_to_show:
        metrics_path = REGISTRY_DIR / vid / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        means = data.get("aggregate", {}).get("mean", data)
        row = f"{vid:<12}  {desc:<45}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)


def main():
    t_total = time.time()

    # Part 1: ML variants
    print("\n" + "="*80)
    print("  PART 1: ML VARIANTS (v7a-v7d)")
    print("="*80)
    run_ml_variants()

    # Print comparison
    print_ml_comparison()

    # Part 2: Blending
    print("\n" + "="*80)
    print("  PART 2: BLENDING EXPERIMENTS")
    print("="*80)

    print("[main] Loading test groups for blending...")
    test_groups = load_all_test_groups()

    blend_results = run_blending_experiments(test_groups)
    print_blend_table(blend_results)

    best_blend = save_blend_results(blend_results)
    print(f"\n[result] Best blend: {best_blend}")

    del test_groups
    gc.collect()

    # Run comparison with v0b gates
    print("\n[main] Running gate comparison...")
    run_comparison(
        batch_id="v7_rebase",
        iteration=1,
        gates_path=str(REGISTRY_DIR / "gates_v0b.json"),
        output_path=str(REPORTS_DIR / "v7_rebase_comparison.md"),
    )

    total = time.time() - t_total
    print(f"\n[main] Total walltime: {total:.1f}s")


if __name__ == "__main__":
    main()
