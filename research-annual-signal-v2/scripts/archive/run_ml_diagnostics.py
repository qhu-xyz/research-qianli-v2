"""ML objective diagnostics — test whether the LambdaRank setup is the problem.

Experiments:
  d1_reg_sp:    LightGBM regressor on log1p(realized_shadow_price)
  d2_reg_tier:  LightGBM regressor on label_tier (0/1/2/3)
  d3_rank_cont: LambdaRank with continuous SP as relevance (no tiering)
  d4_blend_v0c: LightGBM regressor on log1p(SP) with v0c_score as a feature

Usage:
    python scripts/run_ml_diagnostics.py --diag d1_reg_sp
    python scripts/run_ml_diagnostics.py --diag d2_reg_tier
    python scripts/run_ml_diagnostics.py --diag d3_rank_cont
    python scripts/run_ml_diagnostics.py --diag d4_blend_v0c
"""
from __future__ import annotations

import argparse
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS,
    HISTORY_FEATURES, AQ_QUARTERS,
)
from ml.features import build_model_table_all
from ml.train import build_query_groups, tiered_labels
from ml.evaluate import evaluate_all, check_gates
from ml.registry import save_experiment, load_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DIAG_CONFIGS = {
    "d1_reg_sp": {
        "description": "Regressor on log1p(realized_shadow_price)",
        "objective": "regression",
        "metric": "rmse",
        "target": "log1p_sp",
    },
    "d2_reg_tier": {
        "description": "Regressor on label_tier (0/1/2/3)",
        "objective": "regression",
        "metric": "rmse",
        "target": "tier",
    },
    "d3_rank_cont": {
        "description": "LambdaRank with continuous SP as relevance (capped at 255)",
        "objective": "lambdarank",
        "metric": "ndcg",
        "target": "continuous_rank",
    },
    "d4_blend_v0c": {
        "description": "Regressor on log1p(SP) with v0c_score as extra feature",
        "objective": "regression",
        "metric": "rmse",
        "target": "log1p_sp",
        "extra_feature": "v0c_score",
    },
}


def collect_all_needed_groups() -> list[str]:
    """Collect all PY/aq groups needed, excluding 2025-06/aq4."""
    all_pys: set[str] = set()
    for split_info in EVAL_SPLITS.values():
        all_pys.update(split_info["train_pys"])
        all_pys.update(split_info["eval_pys"])

    groups = []
    for py in sorted(all_pys):
        for aq in AQ_QUARTERS:
            key = f"{py}/{aq}"
            if key == "2025-06/aq4":
                continue
            groups.append(key)
    return groups


def compute_v0c_scores(df: pl.DataFrame) -> pl.Series:
    """Reproduce exact v0c formula: 0.40*da_rank_norm + 0.30*right_tail_norm + 0.30*bf_combined_12_norm.

    Matches scripts/run_v0c_full_blend.py exactly:
    - da_rank_norm: inverted min-max of da_rank_value (lower da_rank = more binding = 1.0)
    - right_tail_norm: min-max of max(bin_80_cid_max, bin_90_cid_max, bin_100_cid_max, bin_110_cid_max)
    - bf_combined_12_norm: min-max of bf_combined_12
    - All normalization is per-group (planning_year, aq_quarter)
    - Edge case: if range=0, norm=0.5
    """
    result_scores = np.zeros(len(df))

    groups = df.group_by(["planning_year", "aq_quarter"], maintain_order=True).agg(
        pl.len().alias("count")
    )
    offset = 0
    for row in groups.iter_rows(named=True):
        n = row["count"]
        chunk = df.slice(offset, n)

        # da_rank_value: lower = more binding -> invert for normalization
        da_vals = chunk["da_rank_value"].to_numpy().astype(np.float64)
        da_inv = da_vals.max() - da_vals
        da_range = da_inv.max() - da_inv.min()
        da_norm = (da_inv - da_inv.min()) / da_range if da_range > 0 else np.full_like(da_inv, 0.5)

        # right_tail: max(bin_80, bin_90, bin_100, bin_110) cid_max
        rt_cols = ["bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max"]
        rt_vals = np.column_stack([chunk[c].to_numpy().astype(np.float64) for c in rt_cols]).max(axis=1)
        rt_range = rt_vals.max() - rt_vals.min()
        rt_norm = (rt_vals - rt_vals.min()) / rt_range if rt_range > 0 else np.full_like(rt_vals, 0.5)

        # bf_combined_12: higher = more binding
        bf12_vals = chunk["bf_combined_12"].to_numpy().astype(np.float64)
        bf12_range = bf12_vals.max() - bf12_vals.min()
        bf12_norm = (bf12_vals - bf12_vals.min()) / bf12_range if bf12_range > 0 else np.full_like(bf12_vals, 0.5)

        score = 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf12_norm
        result_scores[offset:offset + n] = score
        offset += n

    return pl.Series("v0c_score", result_scores)


def make_target(
    y_raw: np.ndarray,
    groups: np.ndarray,
    target_type: str,
) -> np.ndarray:
    """Create target array based on diagnostic config."""
    if target_type == "log1p_sp":
        return np.log1p(y_raw)
    elif target_type == "tier":
        return tiered_labels(y_raw, groups).astype(np.float64)
    elif target_type == "continuous_rank":
        # Cap at 255 (LightGBM max label for ranking)
        # Use log1p to compress range, then scale to 0-255
        log_vals = np.log1p(y_raw)
        max_log = log_vals.max()
        if max_log > 0:
            scaled = (log_vals / max_log) * 255.0
        else:
            scaled = np.zeros_like(log_vals)
        return scaled.astype(np.float64)
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def train_diagnostic(
    model_table: pl.DataFrame,
    train_pys: list[str],
    eval_pys: list[str],
    feature_cols: list[str],
    diag_config: dict,
) -> tuple[pl.DataFrame, dict]:
    """Train a diagnostic model and return scored eval df + train info."""
    t0 = time.time()

    model_table = model_table.sort(["planning_year", "aq_quarter", "branch_name"])

    train_df = model_table.filter(pl.col("planning_year").is_in(train_pys))
    eval_df = model_table.filter(pl.col("planning_year").is_in(eval_pys))

    assert len(train_df) > 0, f"No training data for {train_pys}"
    assert len(eval_df) > 0, f"No eval data for {eval_pys}"

    X_train = train_df.select(feature_cols).to_numpy()
    X_eval = eval_df.select(feature_cols).to_numpy()

    y_train_raw = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    y_eval_raw = eval_df["realized_shadow_price"].to_numpy().astype(np.float64)

    train_groups = build_query_groups(train_df)
    eval_groups = build_query_groups(eval_df)

    y_train = make_target(y_train_raw, train_groups, diag_config["target"])
    y_eval = make_target(y_eval_raw, eval_groups, diag_config["target"])

    objective = diag_config["objective"]
    metric = diag_config["metric"]

    params = {
        "objective": objective,
        "metric": metric,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "num_threads": 4,
        "verbose": -1,
    }

    n_estimators = params.pop("n_estimators", 200)

    if objective == "lambdarank":
        train_dataset = lgb.Dataset(
            X_train, label=y_train, group=train_groups,
            feature_name=feature_cols, free_raw_data=False,
        )
        eval_dataset = lgb.Dataset(
            X_eval, label=y_eval, group=eval_groups,
            reference=train_dataset, free_raw_data=False,
        )
    else:
        train_dataset = lgb.Dataset(
            X_train, label=y_train,
            feature_name=feature_cols, free_raw_data=False,
        )
        eval_dataset = lgb.Dataset(
            X_eval, label=y_eval,
            reference=train_dataset, free_raw_data=False,
        )

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=n_estimators,
        valid_sets=[eval_dataset],
        valid_names=["eval"],
    )

    scores = model.predict(X_eval)

    # Feature importance
    raw_imp = model.feature_importance(importance_type="gain")
    total_imp = raw_imp.sum()
    if total_imp > 0:
        norm_imp = raw_imp / total_imp
    else:
        norm_imp = raw_imp
    feature_importance = dict(zip(feature_cols, norm_imp.tolist()))

    walltime = time.time() - t0
    logger.info(
        "Train %s -> eval %s: %d train rows, %d eval rows, %.1fs",
        train_pys, eval_pys, len(train_df), len(eval_df), walltime,
    )

    result = eval_df.with_columns(pl.Series("score", scores))
    train_info = {
        "feature_importance": feature_importance,
        "walltime": walltime,
        "n_train_rows": len(train_df),
    }
    return result, train_info


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
    fi_dev: dict[str, float],
    gate_results: dict,
    version: str,
    feature_cols: list[str],
    diag_config: dict,
) -> None:
    """Print comprehensive diagnostic report."""
    print(f"\n{'='*80}")
    print(f"  Diagnostic: {version}")
    print(f"  Description: {diag_config['description']}")
    print(f"  Objective: {diag_config['objective']}, Target: {diag_config['target']}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"{'='*80}\n")

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

    print(f"\n--- Feature Importance (dev-averaged) ---")
    sorted_imp = sorted(fi_dev.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:>6.1%}  {bar}")

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
    parser = argparse.ArgumentParser(description="ML objective diagnostics")
    parser.add_argument("--diag", required=True, choices=list(DIAG_CONFIGS.keys()),
                        help="Diagnostic experiment to run")
    args = parser.parse_args()

    diag_config = DIAG_CONFIGS[args.diag]
    version = args.diag
    logger.info("Running diagnostic: %s — %s", version, diag_config["description"])

    # Feature columns
    feature_cols = list(HISTORY_FEATURES)
    if "extra_feature" in diag_config:
        feature_cols.append(diag_config["extra_feature"])

    # Build model tables
    t0 = time.time()
    all_groups = collect_all_needed_groups()
    logger.info("Building model tables for %d groups...", len(all_groups))
    model_table = build_model_table_all(all_groups, market_round=1)
    logger.info("Model table built: %d rows, %.1fs", len(model_table), time.time() - t0)

    # Add v0c_score if needed
    if "extra_feature" in diag_config and diag_config["extra_feature"] == "v0c_score":
        logger.info("Computing v0c_score feature...")
        # Need bf_combined_6 and bf_combined_12 for v0c formula
        assert "bf_combined_6" in model_table.columns, "Need bf_combined_6 for v0c_score"
        assert "bf_combined_12" in model_table.columns, "Need bf_combined_12 for v0c_score"
        assert "da_rank_value" in model_table.columns, "Need da_rank_value for v0c_score"
        v0c_scores = compute_v0c_scores(model_table)
        model_table = model_table.with_columns(v0c_scores)

    # Verify features exist
    missing = [f for f in feature_cols if f not in model_table.columns]
    assert not missing, f"Features not in model table: {missing}. Available: {sorted(model_table.columns)}"

    # Train expanding-window models
    scored_frames: list[pl.DataFrame] = []
    per_split_info: dict[str, dict] = {}

    for eval_key, split_info in EVAL_SPLITS.items():
        logger.info("Training split %s (train=%s)", eval_key, split_info["train_pys"])
        scored, train_info = train_diagnostic(
            model_table=model_table,
            train_pys=split_info["train_pys"],
            eval_pys=split_info["eval_pys"],
            feature_cols=feature_cols,
            diag_config=diag_config,
        )
        per_split_info[eval_key] = train_info

        valid_groups = set(DEV_GROUPS + HOLDOUT_GROUPS)
        scored = scored.filter(
            (pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(valid_groups)
        )
        scored_frames.append(scored)
        logger.info("  -> %d eval rows, walltime=%.1fs", len(scored), train_info["walltime"])

    scored_all = pl.concat(scored_frames, how="diagonal")
    logger.info("Total scored rows: %d", len(scored_all))

    # Evaluate
    metrics = evaluate_all(scored_all)

    # Feature importance (dev-only for decisions)
    dev_splits = [k for k, v in EVAL_SPLITS.items() if v["split"] == "dev"]
    fi_dev = aggregate_importance(per_split_info, dev_splits)

    # Gate check vs v0c
    baseline_metrics = load_metrics("v0c")
    gate_results = check_gates(
        candidate=metrics["per_group"],
        baseline=baseline_metrics["per_group"],
        baseline_name="v0c",
        holdout_groups=HOLDOUT_GROUPS,
    )

    # Print report
    print_report(metrics, fi_dev, gate_results, version, feature_cols, diag_config)

    # Save
    config = {
        "version": version,
        "diagnostic": args.diag,
        "description": diag_config["description"],
        "objective": diag_config["objective"],
        "target": diag_config["target"],
        "features": feature_cols,
        "feature_importance_dev_averaged": fi_dev,
    }
    save_experiment(version, config, metrics)
    logger.info("Done. Results saved to registry/%s/", version)


if __name__ == "__main__":
    main()
