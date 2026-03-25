"""ML experiment runner — trains expanding-window LambdaRank and evaluates.

Usage:
    python scripts/run_ml_experiment.py --version v2a --features history
    python scripts/run_ml_experiment.py --version v2b --features history,density_core
    python scripts/run_ml_experiment.py --version v2b_mono --features history,density_core --monotone
    python scripts/run_ml_experiment.py --version v2f --custom-features da_rank_value,bf_6,bf_12
"""
from __future__ import annotations

import argparse
import logging
import time

import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS,
    HISTORY_FEATURES, AQ_QUARTERS,
)
from ml.features import build_model_table_all
from ml.train import train_and_predict
from ml.evaluate import evaluate_all, check_gates
from ml.registry import save_experiment, load_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Feature group definitions ───────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "history": HISTORY_FEATURES,
    "density_core": [
        "bin_80_cid_max", "bin_100_cid_max", "bin_110_cid_max",
        "bin_120_cid_max", "bin_150_cid_max",
    ],
    "density_counter": [
        "bin_-100_cid_max", "bin_-50_cid_max", "bin_60_cid_max",
    ],
    "limits": ["limit_min", "limit_mean", "limit_max", "limit_std"],
    "metadata": ["count_cids", "count_active_cids"],
}


def resolve_features(feature_spec: str, custom_features: str | None) -> list[str]:
    """Resolve feature specification into a flat list of column names."""
    if custom_features:
        cols = [f.strip() for f in custom_features.split(",")]
        assert len(cols) > 0, "No custom features specified"
        return cols

    group_names = [g.strip() for g in feature_spec.split(",")]
    cols: list[str] = []
    for name in group_names:
        assert name in FEATURE_GROUPS, (
            f"Unknown feature group '{name}'. Available: {list(FEATURE_GROUPS.keys())}"
        )
        for col in FEATURE_GROUPS[name]:
            if col not in cols:
                cols.append(col)
    return cols


def collect_all_needed_groups() -> list[str]:
    """Collect all PY/aq groups needed for training + eval, excluding 2025-06/aq4."""
    all_pys: set[str] = set()
    for split_info in EVAL_SPLITS.values():
        all_pys.update(split_info["train_pys"])
        all_pys.update(split_info["eval_pys"])

    groups = []
    for py in sorted(all_pys):
        for aq in AQ_QUARTERS:
            key = f"{py}/{aq}"
            # Exclude incomplete 2025-06/aq4
            if key == "2025-06/aq4":
                continue
            groups.append(key)
    return groups


def aggregate_importance(
    per_split_info: dict[str, dict],
    splits: list[str],
) -> dict[str, float]:
    """Train-row-weighted average of feature importance across splits."""
    total_rows = 0
    weighted: dict[str, float] = {}
    for split_key in splits:
        info = per_split_info[split_key]
        n = info["n_train_rows"]
        total_rows += n
        for feat, imp in info["feature_importance"].items():
            weighted[feat] = weighted.get(feat, 0.0) + imp * n

    if total_rows == 0:
        return weighted
    return {f: v / total_rows for f, v in weighted.items()}


def print_report(
    metrics: dict,
    feature_importance_agg: dict[str, float],
    feature_importance_dev: dict[str, float],
    gate_results: dict,
    version: str,
    feature_cols: list[str],
) -> None:
    """Print comprehensive experiment report."""
    print(f"\n{'='*80}")
    print(f"  Experiment: {version}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"{'='*80}\n")

    # Per-group metrics table
    eval_groups = DEV_GROUPS + HOLDOUT_GROUPS
    header = f"{'Group':<16} {'VC@50':>8} {'VC@100':>8} {'Recall@50':>10} {'NDCG':>8} {'NB12_R@50':>10} {'Abs_SP@50':>10} {'n_bind':>7}"
    print(header)
    print("-" * len(header))

    per_group = metrics["per_group"]
    for g in eval_groups:
        if g not in per_group:
            continue
        m = per_group[g]
        split = "HO" if g in HOLDOUT_GROUPS else "dev"
        print(
            f"{g:<14}{split:>2} "
            f"{m.get('VC@50', 0):>8.4f} {m.get('VC@100', 0):>8.4f} "
            f"{m.get('Recall@50', 0):>10.4f} {m.get('NDCG', 0):>8.4f} "
            f"{m.get('NB12_Recall@50', 0):>10.4f} {m.get('Abs_SP@50', 0):>10.4f} "
            f"{m.get('n_binding', 0):>7d}"
        )

    # Dev and holdout means
    print()
    for split_name in ["dev_mean", "holdout_mean"]:
        if split_name in metrics:
            m = metrics[split_name]
            label = "DEV MEAN" if split_name == "dev_mean" else "HOLDOUT MEAN"
            print(
                f"{label:<16} "
                f"{m.get('VC@50', 0):>8.4f} {m.get('VC@100', 0):>8.4f} "
                f"{m.get('Recall@50', 0):>10.4f} {m.get('NDCG', 0):>8.4f} "
                f"{m.get('NB12_Recall@50', 0):>10.4f} {m.get('Abs_SP@50', 0):>10.4f}"
            )

    # Feature importance (dev-averaged for decisions)
    print(f"\n--- Feature Importance (dev-averaged, for decisions) ---")
    sorted_imp = sorted(feature_importance_dev.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:>6.1%}  {bar}")

    print(f"\n--- Feature Importance (all-split aggregated, for monitoring) ---")
    sorted_imp_all = sorted(feature_importance_agg.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp_all:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:>6.1%}  {bar}")

    # Gate results
    print(f"\n--- Gate Check vs v0c ---")
    for metric, gate in gate_results.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(
            f"  {metric:<20} {status:>4}  "
            f"wins={gate['wins']}/{gate['n_groups']}  "
            f"mean={gate['candidate_mean']:.4f} vs {gate['baseline_mean']:.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ML experiment runner")
    parser.add_argument("--version", required=True, help="Version ID (e.g. v2a)")
    parser.add_argument("--features", default="history",
                        help="Comma-separated feature groups (e.g. history,density_core)")
    parser.add_argument("--custom-features", default=None,
                        help="Comma-separated individual feature names (overrides --features)")
    parser.add_argument("--monotone", action="store_true",
                        help="Enable monotone constraints")
    args = parser.parse_args()

    feature_cols = resolve_features(args.features, args.custom_features)
    logger.info("Version: %s, Features (%d): %s", args.version, len(feature_cols), feature_cols)

    # Step 1: Build model tables for all needed groups
    t0 = time.time()
    all_groups = collect_all_needed_groups()
    logger.info("Building model tables for %d groups...", len(all_groups))
    model_table = build_model_table_all(all_groups, market_round=1)
    logger.info("Model table built: %d rows, %.1fs", len(model_table), time.time() - t0)

    # Verify requested features exist
    missing = [f for f in feature_cols if f not in model_table.columns]
    assert not missing, f"Features not in model table: {missing}. Available: {sorted(model_table.columns)}"

    # Step 2: Train expanding-window models
    scored_frames: list[pl.DataFrame] = []
    per_split_info: dict[str, dict] = {}

    for eval_key, split_info in EVAL_SPLITS.items():
        logger.info("Training split %s (train=%s)", eval_key, split_info["train_pys"])
        scored, train_info = train_and_predict(
            model_table=model_table,
            train_pys=split_info["train_pys"],
            eval_pys=split_info["eval_pys"],
            feature_cols=feature_cols,
            use_monotone=args.monotone,
        )
        per_split_info[eval_key] = train_info

        # Filter to DEV_GROUPS + HOLDOUT_GROUPS only (drops 2025-06/aq4)
        valid_groups = set(DEV_GROUPS + HOLDOUT_GROUPS)
        scored = scored.filter(
            (pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(valid_groups)
        )
        scored_frames.append(scored)
        logger.info("  -> %d eval rows (after filter), walltime=%.1fs",
                     len(scored), train_info["walltime"])

    scored_all = pl.concat(scored_frames, how="diagonal")
    logger.info("Total scored rows: %d", len(scored_all))

    # Step 3: Evaluate
    metrics = evaluate_all(scored_all)

    # Step 4: Feature importance aggregation
    dev_splits = [k for k, v in EVAL_SPLITS.items() if v["split"] == "dev"]
    all_splits = list(EVAL_SPLITS.keys())
    fi_dev = aggregate_importance(per_split_info, dev_splits)
    fi_all = aggregate_importance(per_split_info, all_splits)

    # Step 5: Gate check vs v0c
    baseline_metrics = load_metrics("v0c")
    gate_results = check_gates(
        candidate=metrics["per_group"],
        baseline=baseline_metrics["per_group"],
        baseline_name="v0c",
        holdout_groups=HOLDOUT_GROUPS,
    )

    # Step 6: Print report
    print_report(metrics, fi_all, fi_dev, gate_results, args.version, feature_cols)

    # Step 7: Save to registry
    config = {
        "version": args.version,
        "features": feature_cols,
        "feature_groups": args.features,
        "monotone": args.monotone,
        "feature_importance_dev_averaged": fi_dev,
        "feature_importance_all_averaged": fi_all,
        "feature_importance_per_split": {
            k: v["feature_importance"] for k, v in per_split_info.items()
        },
        "walltimes": {k: v["walltime"] for k, v in per_split_info.items()},
        "n_train_rows": {k: v["n_train_rows"] for k, v in per_split_info.items()},
    }
    save_experiment(args.version, config, metrics)
    logger.info("Done. Results saved to registry/%s/", args.version)


if __name__ == "__main__":
    main()
