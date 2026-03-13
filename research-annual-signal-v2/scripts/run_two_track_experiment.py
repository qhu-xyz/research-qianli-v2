"""Phase 3.3-3.4: Two-track merge sweep + holdout validation.

Sweeps R in {0, 5, 10, 15} for K=50 on dev, validates best R on holdout.
Tests both v0c and v3a as Track A models.

Usage:
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v3a
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c --holdout --r 10
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


R_VALUES = [0, 5, 10, 15]
K = 50


def load_track_b_features() -> list[str]:
    """Load selected Track B features from Phase 3.1."""
    path = REGISTRY_DIR / "nb_analysis" / "selected_features.json"
    with open(path) as f:
        return json.load(f)["track_b_features"]


def load_track_b_model_choice() -> str:
    """Load Track B model choice from Phase 3.2."""
    path = REGISTRY_DIR / "track_b_experiment" / "results.json"
    with open(path) as f:
        results = json.load(f)
    # Pick model with higher mean AUC
    lgbm_aucs = [v["AUC"] for v in results["lgbm"].values()]
    lr_aucs = [v["AUC"] for v in results["logistic"].values()]
    lgbm_mean = sum(lgbm_aucs) / len(lgbm_aucs) if lgbm_aucs else 0
    lr_mean = sum(lr_aucs) / len(lr_aucs) if lr_aucs else 0
    # Use logistic unless LightGBM beats it by > 3%
    if lgbm_mean > lr_mean + 0.03:
        return "lgbm"
    return "logistic"


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


def train_track_b_model(
    train_df: pl.DataFrame,
    features: list[str],
    model_type: str,
) -> object:
    """Train Track B model on training data."""
    X = train_df.select(features).to_numpy().astype(np.float64)
    y = (train_df["realized_shadow_price"].to_numpy() > 0).astype(int)

    if model_type == "lgbm":
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 5,
            "scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0,
            "num_threads": 4, "verbose": -1,
        }
        ds = lgb.Dataset(X, label=y, feature_name=features, free_raw_data=False)
        model = lgb.train(params, ds, num_boost_round=200)
    else:
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X, y)

    return model


def predict_track_b(model, df: pl.DataFrame, features: list[str], model_type: str) -> np.ndarray:
    """Score Track B candidates."""
    X = df.select(features).to_numpy().astype(np.float64)
    if model_type == "lgbm":
        return model.predict(X)
    else:
        return model.predict_proba(X)[:, 1]


def run_two_track_group(
    group_df: pl.DataFrame,
    track_a_model: str,
    track_b_model,
    track_b_features: list[str],
    track_b_model_type: str,
    r: int,
) -> dict:
    """Run two-track merge + evaluation for one (PY, quarter) group."""
    # Split into Track A and Track B
    track_a_df = group_df.filter(pl.col("cohort") == "established")
    track_b_df = group_df.filter(pl.col("cohort").is_in(["history_dormant", "history_zero"]))

    # Score Track A
    if track_a_model == "v0c":
        a_scores = compute_v0c_scores(track_a_df)
    elif track_a_model == "v3a":
        raise NotImplementedError(
            "v3a Track A scoring requires per-split LightGBM training. "
            "Implement after v0c baseline confirms approach viability (spec Phase 3.3.3)."
        )

    track_a_scored = track_a_df.with_columns(pl.Series("score", a_scores))

    # Score Track B
    b_scores = predict_track_b(track_b_model, track_b_df, track_b_features, track_b_model_type)
    track_b_scored = track_b_df.with_columns(pl.Series("score", b_scores))

    # Merge
    merged, top_k_idx = merge_tracks(track_a_scored, track_b_scored, k=K, r=r)

    # Evaluate
    metrics = evaluate_group(merged, k=K, top_k_override=top_k_idx)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a", default="v0c", choices=["v0c", "v3a"])
    parser.add_argument("--holdout", action="store_true", help="Run on holdout groups")
    parser.add_argument("--r", type=int, default=None, help="Fixed R value (skip sweep)")
    parser.add_argument("--version", default=None, help="Version ID for registry save")
    args = parser.parse_args()

    t0 = time.time()

    track_b_features = load_track_b_features()
    track_b_model_type = load_track_b_model_choice()
    logger.info("Track B features: %s", track_b_features)
    logger.info("Track B model: %s", track_b_model_type)

    # Build model tables
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")
    model_table = build_model_table_all(sorted(all_needed))

    # Determine eval groups
    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS
    r_values = [args.r] if args.r is not None else R_VALUES

    # Results: {R: {group: metrics}}
    all_results: dict[int, dict] = {}

    for r in r_values:
        all_results[r] = {}

        for eval_key, split_info in EVAL_SPLITS.items():
            target_split = "holdout" if args.holdout else "dev"
            if split_info["split"] != target_split:
                continue

            # Train Track B on training PYs (Track B population only)
            train_df = model_table.filter(
                pl.col("planning_year").is_in(split_info["train_pys"])
                & pl.col("cohort").is_in(["history_dormant", "history_zero"])
            )
            tb_model = train_track_b_model(train_df, track_b_features, track_b_model_type)

            # Eval on each quarter
            for py in split_info["eval_pys"]:
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    if key not in eval_groups:
                        continue

                    gdf = model_table.filter(
                        (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                    )
                    metrics = run_two_track_group(
                        gdf, args.track_a, tb_model,
                        track_b_features, track_b_model_type, r,
                    )
                    all_results[r][key] = metrics

    # Print results
    print(f"\n{'='*100}")
    print(f"  Two-Track Sweep: Track A={args.track_a}, Track B={track_b_model_type}")
    split_label = "HOLDOUT" if args.holdout else "DEV"
    print(f"  Split: {split_label}")
    print(f"{'='*100}\n")

    for r in r_values:
        print(f"\n--- R={r} ---")
        header = f"{'Group':<16} {'VC@50':>8} {'Recall@50':>10} {'Abs_SP@50':>10} {'NB12_Cnt':>9} {'NB12_SP':>8} {'NB12_R':>8}"
        print(header)
        print("-" * len(header))

        per_group = all_results[r]
        for g in eval_groups:
            if g not in per_group:
                continue
            m = per_group[g]
            print(
                f"{g:<16} {m.get(f'VC@{K}', 0):>8.4f} {m.get(f'Recall@{K}', 0):>10.4f} "
                f"{m.get(f'Abs_SP@{K}', 0):>10.4f} {m.get(f'NB12_Count@{K}', 0):>9d} "
                f"{m.get(f'NB12_SP@{K}', 0):>8.4f} {m.get(f'NB12_Recall@{K}', 0):>8.4f}"
            )

        # Mean
        vals = list(per_group.values())
        if vals:
            mean_vc = sum(m.get(f"VC@{K}", 0) for m in vals) / len(vals)
            mean_nb_cnt = sum(m.get(f"NB12_Count@{K}", 0) for m in vals) / len(vals)
            print(f"\n  Mean VC@50={mean_vc:.4f}, Mean NB12_Count@50={mean_nb_cnt:.1f}")

    # If holdout mode with specific R, do gate checks and save
    if args.holdout and args.r is not None and args.version:
        r = args.r
        per_group = all_results[r]

        # Gate check vs v0c
        baseline_metrics = load_metrics("v0c")
        gate_results = check_gates(
            candidate=per_group,
            baseline=baseline_metrics["per_group"],
            baseline_name="v0c",
            holdout_groups=HOLDOUT_GROUPS,
            gate_metrics=TWO_TRACK_GATE_METRICS,
        )

        # NB threshold check
        nb_results = check_nb_threshold(per_group, HOLDOUT_GROUPS)

        # Print gate results
        print(f"\n{'='*60}")
        print(f"  Gate Check vs v0c (TWO_TRACK_GATE_METRICS)")
        print(f"{'='*60}")
        for metric, gate in gate_results.items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {metric:<20} {status:>4}  wins={gate['wins']}/{gate['n_groups']}")

        print(f"\n  NB Threshold: {'PASS' if nb_results['passed'] else 'FAIL'} "
              f"(total={nb_results['total_count']}, min={nb_results['min_total_count']})")

        # Save to registry
        config = {
            "version": args.version,
            "track_a_model": args.track_a,
            "track_b_model": track_b_model_type,
            "track_b_features": track_b_features,
            "r": r, "k": K,
            "gate_metrics": TWO_TRACK_GATE_METRICS,
        }
        metrics_out = {"per_group": per_group}
        save_experiment(
            args.version, config, metrics_out,
            gate_results=gate_results,
            baseline_version="v0c",
            nb_gate_results=nb_results,
        )
        logger.info("Saved to registry/%s/", args.version)

    logger.info("Done (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
