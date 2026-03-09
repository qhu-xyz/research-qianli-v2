"""Holdout evaluation (2025) for v0b, v7d, and score_blend_v7d_a70.

Trains v7d on 2019-2024 (all dev data), evaluates on 2025/aq1-aq4.
Computes v0b formula scores and blended scores on holdout.
Saves results to registry for comparison.
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

from ml.config import (
    PipelineConfig, LTRConfig, HOLDOUT_EVAL_GROUPS,
    SET_A_FEATURES, SET_A_MONOTONE,
)
from ml.data_loader import load_v61_enriched
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth
from ml.pipeline import train_for_year
from ml.features import prepare_features
from ml.train import predict_scores

REGISTRY_DIR = _PROJECT_ROOT / "registry"

# v7d feature set: Set A + da_rank_value
FULL_DA_FEATURES = list(SET_A_FEATURES) + ["da_rank_value"]
FULL_DA_MONOTONE = list(SET_A_MONOTONE) + [-1]

DISPLAY_METRICS = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG", "Spearman"]


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)


def get_v0b_scores(df: pl.DataFrame) -> np.ndarray:
    """v0b: pure da_rank_value, inverted (higher = more binding)."""
    return 1.0 - df["da_rank_value"].to_numpy().astype(np.float64)


def score_blend(ml_scores: np.ndarray, formula_scores: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * ML + (1-alpha) * formula, both min-max normalized."""
    return alpha * _minmax_normalize(ml_scores) + (1.0 - alpha) * _minmax_normalize(formula_scores)


def save_holdout_result(name: str, result: dict):
    """Save holdout metrics to registry."""
    version_dir = REGISTRY_DIR / name
    version_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "eval_config": {"eval_groups": HOLDOUT_EVAL_GROUPS, "mode": "holdout"},
        "per_month": result["per_month"],
        "aggregate": result["aggregate"],
        "n_months": result["n_months"],
        "n_months_requested": len(HOLDOUT_EVAL_GROUPS),
        "skipped_months": [],
    }
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] {name} -> {version_dir / 'metrics.json'}")


def main():
    t0 = time.time()

    # Load holdout groups
    print("[main] Loading holdout groups (2025/aq1-aq4)...")
    holdout_dfs = {}
    for group_id in HOLDOUT_EVAL_GROUPS:
        planning_year, aq_round = group_id.split("/")
        df = load_v61_enriched(planning_year, aq_round)
        df = get_ground_truth(planning_year, aq_round, df, cache=True)
        holdout_dfs[group_id] = df
        n_binding = int((df["realized_shadow_price"].to_numpy() > 0).sum())
        print(f"  {group_id}: {len(df)} constraints, {n_binding} binding ({100*n_binding/len(df):.1f}%)")

    # ── 1. v0b formula on holdout ──
    print("\n[main] Evaluating v0b (pure da_rank_value) on holdout...")
    v0b_per_month = {}
    for group_id, df in holdout_dfs.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        scores = get_v0b_scores(df)
        v0b_per_month[group_id] = evaluate_ltr(actual, scores)
    v0b_result = {
        "per_month": v0b_per_month,
        "aggregate": aggregate_months(v0b_per_month),
        "n_months": len(v0b_per_month),
    }
    save_holdout_result("v0b_holdout", v0b_result)

    # ── 2. Train v7d model on all dev data, predict holdout ──
    print("\n[main] Training v7d model for holdout (train: 2019-2024)...")
    v7d_config = PipelineConfig(
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
    )

    model = train_for_year(v7d_config, "2025-06")

    v7d_per_month = {}
    ml_predictions = {}  # save for blending
    for group_id, df in holdout_dfs.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        X, _ = prepare_features(df, v7d_config.ltr)
        ml_scores = predict_scores(model, X)
        ml_predictions[group_id] = ml_scores
        v7d_per_month[group_id] = evaluate_ltr(actual, ml_scores)
    v7d_result = {
        "per_month": v7d_per_month,
        "aggregate": aggregate_months(v7d_per_month),
        "n_months": len(v7d_per_month),
    }
    save_holdout_result("v7d_holdout", v7d_result)

    del model
    gc.collect()

    # ── 3. score_blend_v7d_a70 on holdout ──
    print("\n[main] Evaluating score_blend_v7d_a70 on holdout...")
    blend_per_month = {}
    for group_id, df in holdout_dfs.items():
        actual = df["realized_shadow_price"].to_numpy().astype(np.float64)
        ml_scores = ml_predictions[group_id]
        formula_scores = get_v0b_scores(df)
        blended = score_blend(ml_scores, formula_scores, alpha=0.70)
        blend_per_month[group_id] = evaluate_ltr(actual, blended)
    blend_result = {
        "per_month": blend_per_month,
        "aggregate": aggregate_months(blend_per_month),
        "n_months": len(blend_per_month),
    }
    save_holdout_result("blend_v7d_a70_holdout", blend_result)

    # ── 4. Print comparison table ──
    # Load existing holdout results for v0 and v1
    existing_holdouts = {}
    for name in ["v0", "v1_holdout"]:
        path = REGISTRY_DIR / name / "metrics.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            existing_holdouts[name] = data.get("aggregate", {}).get("mean", data)

    all_results = {
        "v0": existing_holdouts.get("v0"),
        "v0b": v0b_result["aggregate"]["mean"],
        "v1": existing_holdouts.get("v1_holdout"),
        "v7d": v7d_result["aggregate"]["mean"],
        "blend_v7d_a70": blend_result["aggregate"]["mean"],
    }

    print(f"\n{'='*120}")
    print("  HOLDOUT RESULTS (2025, 4 quarters)")
    print(f"{'='*120}")
    header = f"{'Version':<20}"
    for m in DISPLAY_METRICS:
        header += f"  {m:>10}"
    print(header)
    print("-" * 120)

    for name, means in all_results.items():
        if means is None:
            continue
        row = f"{name:<20}"
        for m in DISPLAY_METRICS:
            row += f"  {means.get(m, 0.0):>10.4f}"
        print(row)

    # Delta table vs v0b
    v0b_means = v0b_result["aggregate"]["mean"]
    print(f"\n  Delta vs v0b (holdout):")
    for name, means in all_results.items():
        if means is None or name == "v0b":
            continue
        vc20_delta = (means["VC@20"] - v0b_means["VC@20"]) / v0b_means["VC@20"] * 100
        r20_delta = (means["Recall@20"] - v0b_means["Recall@20"]) / v0b_means["Recall@20"] * 100 if v0b_means["Recall@20"] > 0 else 0
        print(f"    {name:<20} VC@20: {vc20_delta:+.1f}%   Recall@20: {r20_delta:+.1f}%")

    # Per-quarter breakdown
    print(f"\n{'='*120}")
    print("  PER-QUARTER BREAKDOWN (VC@20)")
    print(f"{'='*120}")
    header = f"{'Quarter':<16}  {'v0b':>10}  {'v7d':>10}  {'blend_a70':>10}"
    print(header)
    print("-" * 60)
    for group_id in HOLDOUT_EVAL_GROUPS:
        v0b_val = v0b_per_month[group_id]["VC@20"]
        v7d_val = v7d_per_month[group_id]["VC@20"]
        blend_val = blend_per_month[group_id]["VC@20"]
        print(f"{group_id:<16}  {v0b_val:>10.4f}  {v7d_val:>10.4f}  {blend_val:>10.4f}")

    total = time.time() - t0
    print(f"\n[main] Total walltime: {total:.1f}s")


if __name__ == "__main__":
    main()
