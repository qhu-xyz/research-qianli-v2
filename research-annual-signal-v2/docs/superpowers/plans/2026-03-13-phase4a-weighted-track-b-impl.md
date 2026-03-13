# Phase 4a: Value-Weighted Track B Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 3's unweighted binary Track B classifier with a value-weighted variant trained on history_dormant only, to improve NB12_SP capture quality.

**Architecture:** Fork Phase 3's `run_two_track_experiment.py` into a new `run_phase4a_experiment.py` that adds sample weighting (tiered or continuous) to `train_track_b_model()`, restricts Track B population to history_dormant, and sweeps both schemes on dev before validating the winner on holdout. No library (`ml/`) changes needed.

**Tech Stack:** LightGBM binary, sklearn LogisticRegression, polars, numpy

**Spec:** `docs/superpowers/specs/2026-03-13-phase4a-weighted-track-b-design.md`

---

## Chunk 1: Weight computation + experiment script

### Task 1: Unit test for `compute_sample_weights`

**Files:**
- Create: `tests/test_phase4a_weights.py`

- [ ] **Step 1: Write failing tests for both weighting schemes**

```python
"""Tests for Phase 4a sample weight computation."""
import numpy as np
import pytest


def test_tiered_weights_basic():
    """Tiered scheme: bottom 1/3 = 1.0, middle 1/3 = 3.0, top 1/3 = 10.0."""
    # Import deferred to step 3
    from scripts.run_phase4a_experiment import compute_sample_weights

    # 3 negatives + 6 positives with known SP ordering
    sp = np.array([0.0, 0.0, 0.0, 10.0, 20.0, 100.0, 200.0, 500.0, 1000.0])
    w = compute_sample_weights(sp, "tiered")

    # Negatives: weight 1.0
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 1.0

    # 6 positives: bottom 2 = 1.0, middle 2 = 3.0, top 2 = 10.0
    pos_weights = w[3:]
    assert pos_weights[0] == 1.0   # SP=10, rank 0
    assert pos_weights[1] == 1.0   # SP=20, rank 1
    assert pos_weights[2] == 3.0   # SP=100, rank 2
    assert pos_weights[3] == 3.0   # SP=200, rank 3
    assert pos_weights[4] == 10.0  # SP=500, rank 4
    assert pos_weights[5] == 10.0  # SP=1000, rank 5


def test_tiered_weights_all_negative():
    """No positives -> all weights = 1.0."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 0.0, 0.0])
    w = compute_sample_weights(sp, "tiered")
    np.testing.assert_array_equal(w, [1.0, 1.0, 1.0])


def test_continuous_weights_basic():
    """Continuous scheme: negatives=1.0, positives=1.0+min(log1p(SP), 12.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 100.0, 1e6])
    w = compute_sample_weights(sp, "continuous")

    assert w[0] == 1.0  # negative
    assert abs(w[1] - (1.0 + np.log1p(100.0))) < 1e-10  # log1p(100) ≈ 4.62
    assert abs(w[2] - 13.0) < 1e-10  # capped: 1.0 + 12.0 = 13.0


def test_continuous_weights_cap():
    """Cap kicks in at SP ≈ $162k (log1p(162754.79) ≈ 12.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([200000.0, 500000.0])
    w = compute_sample_weights(sp, "continuous")
    # Both above cap -> weight = 13.0
    assert w[0] == 13.0
    assert w[1] == 13.0


def test_tiered_weights_single_positive():
    """Single positive -> always top tier (weight=10.0)."""
    from scripts.run_phase4a_experiment import compute_sample_weights

    sp = np.array([0.0, 0.0, 42.0])
    w = compute_sample_weights(sp, "tiered")
    assert w[0] == 1.0
    assert w[1] == 1.0
    assert w[2] == 10.0  # single positive: ranks < 0 always False -> top tier
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_phase4a_weights.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` since `run_phase4a_experiment` doesn't exist yet.

---

### Task 2: Create `scripts/run_phase4a_experiment.py`

**Files:**
- Create: `scripts/run_phase4a_experiment.py`

This is a fork of `scripts/run_two_track_experiment.py` with these changes:
1. `compute_sample_weights()` function added
2. `train_track_b_model()` gains `scheme` param and applies sample weights
3. Training population: `cohort == "history_dormant"` (not `is_in(["history_dormant", "history_zero"])`)
4. Inference population in `run_two_track_group()`: `cohort == "history_dormant"` only
5. Dev sweep iterates over `schemes = ["tiered", "continuous"]`
6. CLI: `--scheme` for holdout, `--r50`/`--r100` for fixed R

- [ ] **Step 1: Create the full script**

```python
"""Phase 4a: Value-weighted Track B on dormant subset.

Replaces Phase 3's unweighted binary classifier with value-weighted variants.
Trains on history_dormant only (history_zero excluded).
Sweeps tiered vs continuous weighting on dev, validates winner on holdout.

Usage:
    # Dev sweep (all schemes x models x R combos)
    PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c

    # Holdout validation with specific scheme and R values
    PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c \
        --holdout --scheme tiered --r50 5 --r100 15 --version p4a_tiered_r5_r15
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS, AQ_QUARTERS,
    REGISTRY_DIR, TWO_TRACK_GATE_METRICS,
)
from ml.features import build_model_table_all
from ml.evaluate import evaluate_group, check_gates, check_nb_threshold
from ml.registry import save_experiment, load_metrics
from ml.merge import merge_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


R_VALUES_50 = [0, 5, 10]
R_VALUES_100 = [0, 10, 15, 20]
SCHEMES = ["tiered", "continuous"]


# ── Sample weight computation ──────────────────────────────────────────


def compute_sample_weights(
    sp: np.ndarray,
    scheme: str,
) -> np.ndarray:
    """Compute value-aware sample weights for Track B training.

    Args:
        sp: realized_shadow_price array (full training set, not per-group).
        scheme: "tiered" (1/3/10 by tertile) or "continuous" (1 + min(log1p(SP), 12)).

    Returns:
        Per-sample weight array. Negatives (SP=0) always get weight 1.0.
    """
    weights = np.ones(len(sp), dtype=np.float64)
    pos_mask = sp > 0

    if scheme == "tiered":
        if pos_mask.sum() > 0:
            pos_sp = sp[pos_mask]
            # ranks: 0 = lowest SP, n-1 = highest SP
            ranks = pos_sp.argsort().argsort()
            n = len(ranks)
            t1, t2 = n // 3, 2 * n // 3
            tier_w = np.where(ranks < t1, 1.0, np.where(ranks < t2, 3.0, 10.0))
            weights[pos_mask] = tier_w
    elif scheme == "continuous":
        weights[pos_mask] = 1.0 + np.minimum(np.log1p(sp[pos_mask]), 12.0)

    return weights


# ── Reused from Phase 3 (unchanged) ────────────────────────────────────


def load_track_b_features() -> list[str]:
    """Load selected Track B features from Phase 3.1."""
    path = REGISTRY_DIR / "nb_analysis" / "selected_features.json"
    with open(path) as f:
        return json.load(f)["track_b_features"]


def compute_v0c_scores(group_df: pl.DataFrame) -> np.ndarray:
    """Compute v0c formula scores for a single group."""
    def _minmax(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    da_rank = group_df["da_rank_value"].to_numpy().astype(np.float64)
    da_norm = 1.0 - _minmax(da_rank)

    rt_max = group_df.select(
        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                          "bin_100_cid_max", "bin_110_cid_max")
    ).to_series().to_numpy().astype(np.float64)
    rt_norm = _minmax(rt_max)

    bf = group_df["bf_combined_12"].to_numpy().astype(np.float64)
    bf_norm = _minmax(bf)

    return 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf_norm


def predict_track_b(model, df: pl.DataFrame, features: list[str], model_type: str) -> np.ndarray:
    """Score Track B candidates."""
    X = df.select(features).to_numpy().astype(np.float64)
    if model_type == "lgbm":
        return model.predict(X)
    else:
        return model.predict_proba(X)[:, 1]


# ── Changed from Phase 3 ───────────────────────────────────────────────


def train_track_b_model(
    train_df: pl.DataFrame,
    features: list[str],
    model_type: str,
    scheme: str,
) -> object:
    """Train Track B model with value-aware sample weights.

    Changes from Phase 3:
      - train_df is history_dormant only (caller filters)
      - sample weights computed via compute_sample_weights(scheme)
      - LightGBM: class correction folded into weight array (no scale_pos_weight)
      - Logistic: value weights passed as sample_weight; class_weight="balanced" handles ratio
    """
    X = train_df.select(features).to_numpy().astype(np.float64)
    sp = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    y = (sp > 0).astype(int)

    value_weights = compute_sample_weights(sp, scheme)

    if model_type == "lgbm":
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        # Fold class-imbalance correction into per-sample weights
        class_ratio = n_neg / n_pos if n_pos > 0 else 1.0
        per_sample_class_w = np.where(y == 1, class_ratio, 1.0)
        combined_weight = per_sample_class_w * value_weights

        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 5,
            # NO scale_pos_weight — already in combined_weight
            "num_threads": 4, "verbose": -1,
        }
        ds = lgb.Dataset(X, label=y, weight=combined_weight,
                         feature_name=features, free_raw_data=False)
        model = lgb.train(params, ds, num_boost_round=200)
    else:
        # Logistic: class_weight="balanced" handles class ratio; value_weights via sample_weight
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X, y, sample_weight=value_weights)

    return model


def run_two_track_group(
    group_df: pl.DataFrame,
    track_a_model: str,
    track_b_model,
    track_b_features: list[str],
    track_b_model_type: str,
    r50: int,
    r100: int,
) -> dict:
    """Run two-track merge + evaluation at both K=50 and K=100.

    Change from Phase 3: Track B population is history_dormant only.
    history_zero is excluded from reserved slots.
    """
    # Split into Track A and Track B (dormant only)
    track_a_df = group_df.filter(pl.col("cohort") == "established")
    track_b_df = group_df.filter(pl.col("cohort") == "history_dormant")

    # Score Track A
    if track_a_model == "v0c":
        a_scores = compute_v0c_scores(track_a_df)
    elif track_a_model == "v3a":
        raise NotImplementedError("v3a not implemented for Phase 4a")

    track_a_scored = track_a_df.with_columns(pl.Series("score", a_scores))

    # Score Track B (dormant only)
    b_scores = predict_track_b(track_b_model, track_b_df, track_b_features, track_b_model_type)
    track_b_scored = track_b_df.with_columns(pl.Series("score", b_scores))

    # Merge and evaluate at K=50
    merged_50, idx_50 = merge_tracks(track_a_scored, track_b_scored, k=50, r=r50)
    m50 = evaluate_group(merged_50, k=50, top_k_override=idx_50)

    # Merge and evaluate at K=100
    merged_100, idx_100 = merge_tracks(track_a_scored, track_b_scored, k=100, r=r100)
    m100 = evaluate_group(merged_100, k=100, top_k_override=idx_100)

    # Combine: take @50 from m50, @100 from m100, plus global metrics from m50
    metrics: dict = {}
    for key, val in m50.items():
        if "@50" in str(key) or key in ("n_branches", "n_binding", "NDCG", "Spearman"):
            metrics[key] = val
    if "cohort_contribution" in m50:
        metrics["cohort_contribution_50"] = m50["cohort_contribution"]

    for key, val in m100.items():
        if "@100" in str(key):
            metrics[key] = val
    if "cohort_contribution" in m100:
        metrics["cohort_contribution_100"] = m100["cohort_contribution"]

    return metrics


def _print_table(
    r_values: list[int],
    k: int,
    all_results: dict[int, dict],
    eval_groups: list[str],
    scheme_label: str = "",
):
    """Print the results table for one K level."""
    for r in r_values:
        label = f" [{scheme_label}]" if scheme_label else ""
        print(f"\n--- K={k}, R={r}{label} ---")
        header = (
            f"{'Group':<16} {'VC':>8} {'Recall':>8} {'Abs_SP':>8} "
            f"{'NB12_C':>7} {'NB12_SP':>8} {'NB12_R':>7}  "
            f"{'Dorm_C':>7} {'Dorm_SP':>10} {'Zero_C':>7} {'Zero_SP':>10}"
        )
        print(header)
        print("-" * len(header))

        per_group = all_results[r]
        tot_vc, tot_nb_sp, tot_nb_cnt, cnt = 0.0, 0.0, 0, 0
        for g in eval_groups:
            if g not in per_group:
                continue
            m = per_group[g]
            vc = m.get(f"VC@{k}", 0)
            recall = m.get(f"Recall@{k}", 0)
            abs_sp = m.get(f"Abs_SP@{k}", 0)
            nb_cnt = m.get(f"NB12_Count@{k}", 0)
            nb_sp = m.get(f"NB12_SP@{k}", 0)
            nb_r = m.get(f"NB12_Recall@{k}", 0)

            cc = m.get(f"cohort_contribution_{k}", {})
            dorm = cc.get("history_dormant", {})
            zero = cc.get("history_zero", {})
            dorm_cnt = dorm.get("count_in_top_k", 0)
            dorm_sp = dorm.get("sp_captured", 0)
            zero_cnt = zero.get("count_in_top_k", 0)
            zero_sp = zero.get("sp_captured", 0)

            print(
                f"{g:<16} {vc:>8.4f} {recall:>8.4f} {abs_sp:>8.4f} "
                f"{nb_cnt:>7d} {nb_sp:>8.4f} {nb_r:>7.4f}  "
                f"{dorm_cnt:>7d} {dorm_sp:>10.1f} {zero_cnt:>7d} {zero_sp:>10.1f}"
            )
            tot_vc += vc
            tot_nb_sp += nb_sp
            tot_nb_cnt += nb_cnt
            cnt += 1

        if cnt:
            print(f"\n  Mean VC@{k}={tot_vc/cnt:.4f}, Mean NB12_SP@{k}={tot_nb_sp/cnt:.4f}, "
                  f"Mean NB12_Count@{k}={tot_nb_cnt/cnt:.1f}")


def _run_sweep(
    model_table: pl.DataFrame,
    eval_groups: list[str],
    is_holdout: bool,
    track_a_model: str,
    track_b_features: list[str],
    model_type: str,
    scheme: str,
    r50_values: list[int],
    r100_values: list[int],
) -> dict[tuple[int, int], dict]:
    """Run sweep for one (scheme, model_type) combo. Returns {(r50, r100): {group: metrics}}."""
    target_split = "holdout" if is_holdout else "dev"

    # Train one model per split (model doesn't depend on R)
    split_models: dict[str, object] = {}
    for eval_key, split_info in EVAL_SPLITS.items():
        if split_info["split"] != target_split:
            continue
        train_df = model_table.filter(
            pl.col("planning_year").is_in(split_info["train_pys"])
            & (pl.col("cohort") == "history_dormant")
        )
        split_models[eval_key] = train_track_b_model(
            train_df, track_b_features, model_type, scheme,
        )

    # Sweep R combos reusing trained models
    combos = [(r50, r100) for r50 in r50_values for r100 in r100_values]
    all_results: dict[tuple[int, int], dict] = {}

    for r50, r100 in combos:
        combo_results: dict[str, dict] = {}

        for eval_key, split_info in EVAL_SPLITS.items():
            if split_info["split"] != target_split:
                continue
            tb_model = split_models[eval_key]

            for py in split_info["eval_pys"]:
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    if key not in eval_groups:
                        continue

                    gdf = model_table.filter(
                        (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                    )
                    metrics = run_two_track_group(
                        gdf, track_a_model, tb_model,
                        track_b_features, model_type,
                        r50=r50, r100=r100,
                    )
                    combo_results[key] = metrics

        all_results[(r50, r100)] = combo_results

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a", default="v0c", choices=["v0c", "v3a"])
    parser.add_argument("--holdout", action="store_true", help="Run on holdout groups")
    parser.add_argument("--scheme", default=None, choices=["tiered", "continuous"],
                        help="Fixed scheme (skip scheme sweep)")
    parser.add_argument("--model-type", default=None, choices=["logistic", "lgbm"],
                        help="Fixed model type (skip model sweep)")
    parser.add_argument("--r50", type=int, default=None, help="Fixed R for K=50")
    parser.add_argument("--r100", type=int, default=None, help="Fixed R for K=100")
    parser.add_argument("--version", default=None, help="Version ID for registry save")
    args = parser.parse_args()

    t0 = time.time()

    track_b_features = load_track_b_features()
    logger.info("Track B features (%d): %s", len(track_b_features), track_b_features)

    # Build model tables
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")
    model_table = build_model_table_all(sorted(all_needed))

    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS
    r50_values = [args.r50] if args.r50 is not None else R_VALUES_50
    r100_values = [args.r100] if args.r100 is not None else R_VALUES_100
    schemes = [args.scheme] if args.scheme else SCHEMES
    model_types = [args.model_type] if args.model_type else ["logistic", "lgbm"]

    split_label = "HOLDOUT" if args.holdout else "DEV"

    # Run sweep for each (scheme, model_type)
    for scheme in schemes:
        for model_type in model_types:
            label = f"{scheme}/{model_type}"
            logger.info("Running sweep: %s", label)

            results = _run_sweep(
                model_table, eval_groups, args.holdout,
                args.track_a, track_b_features, model_type, scheme,
                r50_values, r100_values,
            )

            # Print results
            print(f"\n{'='*120}")
            print(f"  Phase 4a: Track A={args.track_a}, Track B={model_type}, Scheme={scheme}")
            print(f"  Split: {split_label}, Population: dormant-only")
            print(f"{'='*120}")

            # K=50 table
            print(f"\n  K=50 RESULTS")
            print(f"  {'─'*60}")
            r50_results: dict[int, dict] = {}
            for r50 in r50_values:
                r50_results[r50] = results[(r50, r100_values[0])]
            _print_table(r50_values, 50, r50_results, eval_groups, label)

            # K=100 table
            print(f"\n  K=100 RESULTS")
            print(f"  {'─'*60}")
            r100_results: dict[int, dict] = {}
            for r100 in r100_values:
                r100_results[r100] = results[(r50_values[0], r100)]
            _print_table(r100_values, 100, r100_results, eval_groups, label)

    # Holdout mode: gate checks and registry save
    if args.holdout and args.r50 is not None and args.r100 is not None and args.version:
        assert args.scheme is not None, "--scheme required for holdout save"
        assert args.model_type is not None or len(model_types) == 1, \
            "--model-type required when sweeping multiple model types"
        final_model_type = args.model_type or model_types[0]

        # Re-run with final config (single combo)
        final_results = _run_sweep(
            model_table, eval_groups, True,
            args.track_a, track_b_features, final_model_type, args.scheme,
            [args.r50], [args.r100],
        )
        per_group = final_results[(args.r50, args.r100)]

        # Gate check vs v0c at both K levels
        baseline_metrics = load_metrics("v0c")
        gate_metrics_50 = TWO_TRACK_GATE_METRICS
        gate_metrics_100 = ["VC@100", "Recall@100", "Abs_SP@100"]

        gate_results_50 = check_gates(
            candidate=per_group,
            baseline=baseline_metrics["per_group"],
            baseline_name="v0c",
            holdout_groups=HOLDOUT_GROUPS,
            gate_metrics=gate_metrics_50,
        )
        gate_results_100 = check_gates(
            candidate=per_group,
            baseline=baseline_metrics["per_group"],
            baseline_name="v0c",
            holdout_groups=HOLDOUT_GROUPS,
            gate_metrics=gate_metrics_100,
        )
        gate_results = {**gate_results_50, **gate_results_100}

        nb_results_50 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=50)
        nb_results_100 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=100)

        # Compare vs Phase 3 baseline
        phase3_version = "tt_v0c_r5_r15"
        try:
            phase3_metrics = load_metrics(phase3_version)
            phase3_pg = phase3_metrics["per_group"]
            p3_nb_sp_50 = [phase3_pg[g].get("NB12_SP@50", 0) for g in HOLDOUT_GROUPS if g in phase3_pg]
            p4_nb_sp_50 = [per_group[g].get("NB12_SP@50", 0) for g in HOLDOUT_GROUPS if g in per_group]
            p3_nb_sp_100 = [phase3_pg[g].get("NB12_SP@100", 0) for g in HOLDOUT_GROUPS if g in phase3_pg]
            p4_nb_sp_100 = [per_group[g].get("NB12_SP@100", 0) for g in HOLDOUT_GROUPS if g in per_group]
            p3_vc_50 = [phase3_pg[g].get("VC@50", 0) for g in HOLDOUT_GROUPS if g in phase3_pg]
            p4_vc_50 = [per_group[g].get("VC@50", 0) for g in HOLDOUT_GROUPS if g in per_group]

            print(f"\n{'='*80}")
            print(f"  Phase 4a vs Phase 3 ({phase3_version})")
            print(f"{'='*80}")
            if p3_nb_sp_50:
                print(f"  NB12_SP@50:  Phase3={sum(p3_nb_sp_50)/len(p3_nb_sp_50):.4f}  "
                      f"Phase4a={sum(p4_nb_sp_50)/len(p4_nb_sp_50):.4f}")
            if p3_nb_sp_100:
                print(f"  NB12_SP@100: Phase3={sum(p3_nb_sp_100)/len(p3_nb_sp_100):.4f}  "
                      f"Phase4a={sum(p4_nb_sp_100)/len(p4_nb_sp_100):.4f}")
            if p3_vc_50:
                p3_mean = sum(p3_vc_50) / len(p3_vc_50)
                p4_mean = sum(p4_vc_50) / len(p4_vc_50)
                delta = p4_mean - p3_mean
                print(f"  VC@50:       Phase3={p3_mean:.4f}  Phase4a={p4_mean:.4f}  "
                      f"delta={delta:+.4f} {'OK' if delta > -0.01 else 'REGRESSED'}")
        except (AssertionError, FileNotFoundError):
            logger.warning("Phase 3 baseline %s not found, skipping comparison", phase3_version)

        # Print gate results
        print(f"\n{'='*80}")
        print(f"  Gate Check vs v0c")
        print(f"{'='*80}")
        for metric, gate in gate_results.items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {metric:<20} {status:>4}  wins={gate['wins']}/{gate['n_groups']}")

        print(f"\n  NB Threshold @50:  {'PASS' if nb_results_50['passed'] else 'FAIL'} "
              f"(total={nb_results_50['total_count']}, min={nb_results_50['min_total_count']})")
        print(f"  NB Threshold @100: {'PASS' if nb_results_100['passed'] else 'FAIL'} "
              f"(total={nb_results_100['total_count']}, min={nb_results_100['min_total_count']})")

        # Save to registry
        config = {
            "version": args.version,
            "phase": "4a",
            "track_a_model": args.track_a,
            "track_b_model": final_model_type,
            "track_b_features": track_b_features,
            "scheme": args.scheme,
            "population": "history_dormant",
            "r50": args.r50, "r100": args.r100,
            "gate_metrics_50": gate_metrics_50,
            "gate_metrics_100": gate_metrics_100,
        }
        metrics_out = {"per_group": per_group}
        nb_results_combined = {"k50": nb_results_50, "k100": nb_results_100}
        save_experiment(
            args.version, config, metrics_out,
            gate_results=gate_results,
            baseline_version="v0c",
            nb_gate_results=nb_results_combined,
        )
        logger.info("Saved to registry/%s/", args.version)

    logger.info("Done (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run weight unit tests to verify they pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_phase4a_weights.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/ -v`
Expected: All tests PASS (98 existing + 5 new = 103).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_phase4a_experiment.py tests/test_phase4a_weights.py
git commit -m "phase4a: add weighted Track B experiment script and weight tests

Value-weighted binary classification on history_dormant only.
Two schemes: tiered (1/3/10) and continuous (log1p, capped at 12).
LightGBM folds class correction into per-sample weights.
Evaluates at both K=50 and K=100 with independent R sweeps."
```

---

## Chunk 2: Dev sweep + holdout validation

### Task 3: Run dev sweep

- [ ] **Step 1: Run the full dev sweep (all schemes × models)**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c 2>&1 | tee /tmp/phase4a_dev_sweep.txt`
Expected: 4 blocks of output (tiered/logistic, tiered/lgbm, continuous/logistic, continuous/lgbm). Each block has K=50 and K=100 tables. Runtime ~4-8 minutes (4 scheme×model combos × 3 splits, model training hoisted outside R loop).

- [ ] **Step 2: Analyze Pareto frontier**

From the dev sweep output, for each scheme:
- Compare Mean NB12_SP@50 and Mean NB12_SP@100 against Phase 3 baselines (NB12_SP@50 ≈ 0.011, NB12_SP@100 ≈ 0.016)
- Check if any scheme improves NB12_SP without extra VC@50 degradation
- Select best (scheme, model_type) combo by NB12_SP improvement
- Note: if tiered and continuous are close, prefer tiered (reference/simpler)

Key questions to answer:
1. Does weighting shift top-R Track B picks toward higher-SP binders?
2. Does dormant-only training improve or degrade AUC vs Phase 3 (dormant+zero)?
3. Which R values are on the Pareto frontier for each K level?

- [ ] **Step 3: Select winner and R values**

Pick (scheme, model_type, R50, R100) based on:
- Primary: highest NB12_SP@50 and NB12_SP@100
- Secondary: VC@50 and VC@100 within 0.01 of Phase 3
- If no scheme improves NB12_SP, declare Phase 4a negative and stop

---

### Task 4: Run holdout validation

- [ ] **Step 1: Run holdout with winning config**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c --holdout --scheme <WINNER_SCHEME> --model-type <WINNER_MODEL> --r50 <R50> --r100 <R100> --version p4a_<scheme>_r<R50>_r<R100>`

Replace `<WINNER_SCHEME>`, `<WINNER_MODEL>`, `<R50>`, `<R100>` with values from Task 3 analysis.

Expected output includes:
- K=50 and K=100 tables with dormant/zero breakdown
- Phase 4a vs Phase 3 comparison (NB12_SP and VC deltas)
- Gate check vs v0c
- NB threshold at both K levels
- Registry save

- [ ] **Step 2: Evaluate success criteria**

Check against spec §6:
1. NB12_SP@50 > Phase 3 NB12_SP@50? (at matched R50)
2. NB12_SP@100 > Phase 3 NB12_SP@100? (at matched R100)
3. VC@50 delta > -0.01 absolute?
4. NB threshold passes at both K=50 and K=100?

If all pass → Phase 4a is positive.
If NB12_SP did not improve → Phase 4a is negative (approach doesn't earn complexity).

- [ ] **Step 3: Commit results**

```bash
git add scripts/run_phase4a_experiment.py registry/p4a_*/
git commit -m "phase4a: <WINNER_SCHEME> weighted Track B — <RESULT>

<WINNER_SCHEME>/<WINNER_MODEL> on dormant-only population.
R50=<R50>, R100=<R100>.
NB12_SP@50: Phase3=X.XXXX → Phase4a=Y.YYYY (<delta>)
NB12_SP@100: Phase3=X.XXXX → Phase4a=Y.YYYY (<delta>)
VC@50 delta: <delta> (<OK or REGRESSED>)"
```
