"""Phase 3.3-3.4: Two-track merge sweep + holdout validation.

Sweeps R values at K=50 and K=100 independently on dev, validates on holdout.
Each K level gets its own R sweep since different K budgets warrant different
NB slot allocations.

Usage:
    # Dev sweep at both K levels
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c

    # Holdout validation with fixed R per K level
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c \
        --holdout --r50 5 --r100 15 --version tt_v0c_r5_r15
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


R_VALUES_50 = [0, 5, 10, 15]
R_VALUES_100 = [0, 10, 15, 20]


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
    lgbm_aucs = [v["AUC"] for v in results["lgbm"].values()]
    lr_aucs = [v["AUC"] for v in results["logistic"].values()]
    lgbm_mean = sum(lgbm_aucs) / len(lgbm_aucs) if lgbm_aucs else 0
    lr_mean = sum(lr_aucs) / len(lr_aucs) if lr_aucs else 0
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
    r50: int,
    r100: int,
) -> dict:
    """Run two-track merge + evaluation at both K=50 and K=100.

    Returns combined metrics dict with @50 from K=50 merge and @100 from K=100 merge,
    plus cohort breakdowns at both levels.
    """
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
    # Cohort contribution at K=50
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
):
    """Print the results table for one K level."""
    for r in r_values:
        print(f"\n--- K={k}, R={r} ---")
        header = (
            f"{'Group':<16} {'VC':>8} {'Recall':>8} {'Abs_SP':>8} "
            f"{'NB12_C':>7} {'NB12_SP':>8} {'NB12_R':>7}  "
            f"{'Dorm_C':>7} {'Dorm_SP':>10} {'Zero_C':>7} {'Zero_SP':>10}"
        )
        print(header)
        print("-" * len(header))

        per_group = all_results[r]
        tot_vc, tot_nb, cnt = 0.0, 0, 0
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

            # Cohort breakdown
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
            tot_nb += nb_cnt
            cnt += 1

        if cnt:
            print(f"\n  Mean VC@{k}={tot_vc/cnt:.4f}, Mean NB12_Count@{k}={tot_nb/cnt:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a", default="v0c", choices=["v0c", "v3a"])
    parser.add_argument("--holdout", action="store_true", help="Run on holdout groups")
    parser.add_argument("--r50", type=int, default=None, help="Fixed R for K=50 (skip sweep)")
    parser.add_argument("--r100", type=int, default=None, help="Fixed R for K=100 (skip sweep)")
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
    model_table = build_model_table_all(sorted(all_needed), market_round=1)

    # Determine eval groups
    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS
    r50_values = [args.r50] if args.r50 is not None else R_VALUES_50
    r100_values = [args.r100] if args.r100 is not None else R_VALUES_100

    # Results: {(r50, r100): {group: metrics}}
    all_results: dict[tuple[int, int], dict] = {}

    # Sweep all (r50, r100) combos
    combos = [(r50, r100) for r50 in r50_values for r100 in r100_values]

    for r50, r100 in combos:
        combo_results: dict[str, dict] = {}

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
                        track_b_features, track_b_model_type,
                        r50=r50, r100=r100,
                    )
                    combo_results[key] = metrics

        all_results[(r50, r100)] = combo_results

    # Print results
    split_label = "HOLDOUT" if args.holdout else "DEV"
    print(f"\n{'='*120}")
    print(f"  Two-Track Sweep: Track A={args.track_a}, Track B={track_b_model_type}")
    print(f"  Split: {split_label}")
    print(f"{'='*120}")

    # Print K=50 table (grouping by r50, averaging over r100 values — but since
    # @50 metrics only depend on r50, just pick any r100)
    print(f"\n{'='*120}")
    print("  K=50 RESULTS")
    print(f"{'='*120}")
    r50_results: dict[int, dict] = {}
    for r50 in r50_values:
        # Pick first r100 combo for @50 metrics (they're the same for all r100)
        r100_first = r100_values[0]
        r50_results[r50] = all_results[(r50, r100_first)]
    _print_table(r50_values, 50, r50_results, eval_groups)

    # Print K=100 table (grouping by r100)
    print(f"\n{'='*120}")
    print("  K=100 RESULTS")
    print(f"{'='*120}")
    r100_results: dict[int, dict] = {}
    for r100 in r100_values:
        # Pick first r50 combo for @100 metrics (they're the same for all r50)
        r50_first = r50_values[0]
        r100_results[r100] = all_results[(r50_first, r100)]
    _print_table(r100_values, 100, r100_results, eval_groups)

    # If holdout mode with fixed R values, do gate checks and save
    if args.holdout and args.r50 is not None and args.r100 is not None and args.version:
        per_group = all_results[(args.r50, args.r100)]

        # Gate check vs v0c at both K levels
        baseline_metrics = load_metrics("v0c")

        gate_metrics_50 = TWO_TRACK_GATE_METRICS  # ["VC@50", "Recall@50", "Abs_SP@50"]
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

        # NB threshold check at K=50 and K=100
        nb_results_50 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=50)
        nb_results_100 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=100)

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
            "track_a_model": args.track_a,
            "track_b_model": track_b_model_type,
            "track_b_features": track_b_features,
            "r50": args.r50, "r100": args.r100,
            "gate_metrics_50": gate_metrics_50,
            "gate_metrics_100": gate_metrics_100,
        }
        metrics_out = {"per_group": per_group}
        nb_results_combined = {
            "k50": nb_results_50,
            "k100": nb_results_100,
        }
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
