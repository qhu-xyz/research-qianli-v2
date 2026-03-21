"""Phase 4b Experiment 1: log1p(SP) regression on dormant subset with dynamic R.

Value-aware Track B: LightGBM regression predicting log1p(realized_shadow_price)
on history_dormant branches. Dynamic R via score threshold tau.

Usage:
    # Dev sweep (all tau values at K=50 and K=100)
    PYTHONPATH=. uv run python scripts/run_phase4b_regression.py

    # Holdout validation
    PYTHONPATH=. uv run python scripts/run_phase4b_regression.py \
        --holdout --tau50 0.5 --tau100 0.3 --r50 5 --r100 15 \
        --version p4b_reg_t50_t30
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS, AQ_QUARTERS,
    REGISTRY_DIR, TWO_TRACK_GATE_METRICS,
    get_bf_cutoff_month,
)
from ml.features import build_model_table_all
from ml.features_trackb import (
    compute_constraint_propagation, compute_recency_features,
    compute_density_shape, PHASE4B_TRACK_B_FEATURES,
)
from ml.history_features import compute_history_features
from ml.evaluate import evaluate_group, check_gates, check_nb_threshold
from ml.registry import save_experiment, load_metrics
from ml.merge import merge_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

R_MAX_50 = 5
R_MAX_100 = 15
TAU_CANDIDATES = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]


# ── Reused from Phase 4a ───────────────────────────────────────────────

def compute_v0c_scores(group_df: pl.DataFrame) -> np.ndarray:
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


def _append_zero_history(merged: pl.DataFrame, group_df: pl.DataFrame) -> pl.DataFrame:
    zero_df = group_df.filter(pl.col("cohort") == "history_zero")
    if len(zero_df) == 0:
        return merged
    zero_scored = zero_df.with_columns([
        pl.lit(0.0).alias("score"),
        pl.lit("Z").alias("track"),
    ])
    return pl.concat([merged, zero_scored], how="diagonal")


# ── Phase 4b specific ──────────────────────────────────────────────────

def enrich_with_trackb_features(
    model_table: pl.DataFrame,
    planning_year: str,
    aq_quarter: str,
) -> pl.DataFrame:
    """Add Phase 4b features (recency + shape) to dormant branches in model_table."""
    dormant = model_table.filter(
        pl.col("cohort") == "history_dormant"
    )["branch_name"].to_list()

    if not dormant:
        return model_table

    cutoff = get_bf_cutoff_month(planning_year)
    _, monthly_binding = compute_history_features(
        planning_year, aq_quarter, model_table["branch_name"].to_list(),
    )

    # Recency features
    rec = compute_recency_features(monthly_binding, dormant, cutoff)

    # Shape features
    shp = compute_density_shape(model_table, dormant)

    # Join to model table (left join — only dormant branches get these features)
    result = model_table.join(rec, on="branch_name", how="left")
    result = result.join(shp, on="branch_name", how="left")

    # Fill nulls for non-dormant branches
    for col in rec.columns:
        if col != "branch_name" and col in result.columns:
            if col == "months_since_last_bind":
                result = result.with_columns(pl.col(col).fill_null(0))
            elif col == "n_historical_binding_months":
                result = result.with_columns(pl.col(col).fill_null(0))
            else:
                result = result.with_columns(pl.col(col).fill_null(0.0))
    for col in shp.columns:
        if col != "branch_name" and col in result.columns:
            result = result.with_columns(pl.col(col).fill_null(0.0))

    return result


def train_regression_model(
    train_df: pl.DataFrame,
    features: list[str],
) -> lgb.Booster:
    """Train LightGBM regression on log1p(realized_shadow_price) for dormant branches."""
    X = train_df.select(features).to_numpy().astype(np.float64)
    sp = train_df["realized_shadow_price"].to_numpy().astype(np.float64)
    y = np.log1p(sp)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "num_threads": 4,
        "verbose": -1,
    }
    ds = lgb.Dataset(X, label=y, feature_name=features, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=200)
    return model


def run_two_track_group(
    group_df: pl.DataFrame,
    track_b_model: lgb.Booster,
    track_b_features: list[str],
    r50: int,
    r100: int,
    tau50: float | None,
    tau100: float | None,
) -> dict:
    """Run two-track merge + evaluation at both K=50 and K=100."""
    track_a_df = group_df.filter(pl.col("cohort") == "established")
    track_b_df = group_df.filter(pl.col("cohort") == "history_dormant")

    # Score Track A
    a_scores = compute_v0c_scores(track_a_df)
    track_a_scored = track_a_df.with_columns(pl.Series("score", a_scores))

    # Score Track B (predicted log1p(SP))
    X_b = track_b_df.select(track_b_features).to_numpy().astype(np.float64)
    b_scores = track_b_model.predict(X_b)
    track_b_scored = track_b_df.with_columns(pl.Series("score", b_scores))

    # Merge + evaluate at K=50 with dynamic R
    merged_50, idx_50 = merge_tracks(track_a_scored, track_b_scored, k=50, r=r50, tau=tau50)
    full_50 = _append_zero_history(merged_50, group_df)
    m50 = evaluate_group(full_50, k=50, top_k_override=idx_50)

    # Merge + evaluate at K=100 with dynamic R
    merged_100, idx_100 = merge_tracks(track_a_scored, track_b_scored, k=100, r=r100, tau=tau100)
    full_100 = _append_zero_history(merged_100, group_df)
    m100 = evaluate_group(full_100, k=100, top_k_override=idx_100)

    # Combine metrics
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
    tau_values: list[float],
    k: int,
    r_max: int,
    all_results: dict[float, dict],
    eval_groups: list[str],
):
    for tau in tau_values:
        print(f"\n--- K={k}, R_max={r_max}, tau={tau:.1f} ---")
        header = (
            f"{'Group':<16} {'VC':>8} {'Recall':>8} {'Abs_SP':>8} "
            f"{'NB12_C':>7} {'NB12_SP':>8} {'NB12_R':>7}  "
            f"{'Dorm_C':>7} {'Dorm_SP':>10} {'Zero_C':>7} {'Zero_SP':>10}"
        )
        print(header)
        print("-" * len(header))

        per_group = all_results[tau]
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

            print(
                f"{g:<16} {vc:>8.4f} {recall:>8.4f} {abs_sp:>8.4f} "
                f"{nb_cnt:>7d} {nb_sp:>8.4f} {nb_r:>7.4f}  "
                f"{dorm.get('count_in_top_k', 0):>7d} {dorm.get('sp_captured', 0):>10.1f} "
                f"{zero.get('count_in_top_k', 0):>7d} {zero.get('sp_captured', 0):>10.1f}"
            )
            tot_vc += vc
            tot_nb_sp += nb_sp
            tot_nb_cnt += nb_cnt
            cnt += 1

        if cnt:
            print(f"\n  Mean VC@{k}={tot_vc/cnt:.4f}, Mean NB12_SP@{k}={tot_nb_sp/cnt:.4f}, "
                  f"Mean NB12_Count@{k}={tot_nb_cnt/cnt:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--tau50", type=float, default=None, help="Fixed tau for K=50")
    parser.add_argument("--tau100", type=float, default=None, help="Fixed tau for K=100")
    parser.add_argument("--r50", type=int, default=R_MAX_50)
    parser.add_argument("--r100", type=int, default=R_MAX_100)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    t0 = time.time()

    features = [f for f in PHASE4B_TRACK_B_FEATURES
                if not f.startswith("max_cid_") and not f.startswith("mean_cid_")
                and not f.startswith("sum_cid_") and not f.startswith("n_active_cids")
                and not f.startswith("active_cid_ratio")]
    logger.info("Features (%d): %s", len(features), features)

    # Build model tables
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")
    model_table = build_model_table_all(sorted(all_needed))

    # Enrich with Phase 4b features
    logger.info("Enriching with Phase 4b features...")
    enriched_parts = []
    for (py, aq), part in model_table.group_by(["planning_year", "aq_quarter"], maintain_order=True):
        enriched = enrich_with_trackb_features(part, str(py), str(aq))
        enriched_parts.append(enriched)
    model_table = pl.concat(enriched_parts, how="diagonal")
    logger.info("Enrichment done. Shape: %s", model_table.shape)

    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS
    target_split = "holdout" if args.holdout else "dev"
    tau50_values = [args.tau50] if args.tau50 is not None else TAU_CANDIDATES
    tau100_values = [args.tau100] if args.tau100 is not None else TAU_CANDIDATES

    # Train one model per split
    split_models: dict[str, lgb.Booster] = {}
    for eval_key, split_info in EVAL_SPLITS.items():
        if split_info["split"] != target_split:
            continue
        train_df = model_table.filter(
            pl.col("planning_year").is_in(split_info["train_pys"])
            & (pl.col("cohort") == "history_dormant")
        )
        logger.info("Training %s: %d dormant rows, %d positive",
                     eval_key, len(train_df),
                     int((train_df["realized_shadow_price"] > 0).sum()))
        split_models[eval_key] = train_regression_model(train_df, features)

    # Sweep tau for K=50
    results_50: dict[float, dict] = {}
    for tau50 in tau50_values:
        per_group: dict[str, dict] = {}
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
                        gdf, tb_model, features,
                        r50=args.r50, r100=args.r100,
                        tau50=tau50, tau100=0.0,
                    )
                    per_group[key] = metrics
        results_50[tau50] = per_group

    # Sweep tau for K=100
    results_100: dict[float, dict] = {}
    for tau100 in tau100_values:
        per_group = {}
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
                        gdf, tb_model, features,
                        r50=args.r50, r100=args.r100,
                        tau50=0.0, tau100=tau100,
                    )
                    per_group[key] = metrics
        results_100[tau100] = per_group

    # Print results
    split_label = "HOLDOUT" if args.holdout else "DEV"
    print(f"\n{'='*120}")
    print(f"  Phase 4b Exp1: log1p(SP) regression, dormant-only, dynamic R")
    print(f"  Split: {split_label}, Features: {len(features)}")
    print(f"{'='*120}")

    print(f"\n  K=50 RESULTS (R_max={args.r50})")
    print(f"  {'─'*60}")
    _print_table(tau50_values, 50, args.r50, results_50, eval_groups)

    print(f"\n  K=100 RESULTS (R_max={args.r100})")
    print(f"  {'─'*60}")
    _print_table(tau100_values, 100, args.r100, results_100, eval_groups)

    # Feature importance from first split
    first_model = list(split_models.values())[0]
    raw_imp = first_model.feature_importance(importance_type="gain")
    total_imp = raw_imp.sum()
    if total_imp > 0:
        fi = sorted(zip(features, raw_imp / total_imp), key=lambda x: -x[1])
        print(f"\n  Feature Importance (top 10):")
        for name, imp in fi[:10]:
            print(f"    {name:<30} {imp:.4f}")

    # Holdout save
    if args.holdout and args.tau50 is not None and args.tau100 is not None and args.version:
        # Re-run with final tau combo
        per_group = {}
        for eval_key, split_info in EVAL_SPLITS.items():
            if split_info["split"] != "holdout":
                continue
            tb_model = split_models[eval_key]
            for py in split_info["eval_pys"]:
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    if key not in HOLDOUT_GROUPS:
                        continue
                    gdf = model_table.filter(
                        (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                    )
                    metrics = run_two_track_group(
                        gdf, tb_model, features,
                        r50=args.r50, r100=args.r100,
                        tau50=args.tau50, tau100=args.tau100,
                    )
                    per_group[key] = metrics

        # Gates
        baseline_metrics = load_metrics("v0c")
        gate_metrics_50 = TWO_TRACK_GATE_METRICS
        gate_metrics_100 = ["VC@100", "Recall@100", "Abs_SP@100"]
        gate_results = {
            **check_gates(per_group, baseline_metrics["per_group"], "v0c", HOLDOUT_GROUPS, gate_metrics_50),
            **check_gates(per_group, baseline_metrics["per_group"], "v0c", HOLDOUT_GROUPS, gate_metrics_100),
        }
        nb_50 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=50)
        nb_100 = check_nb_threshold(per_group, HOLDOUT_GROUPS, k=100)

        # Phase 3 comparison
        try:
            p3 = load_metrics("tt_v0c_r5_r15")["per_group"]
            p3_nb50 = np.mean([p3[g].get("NB12_SP@50", 0) for g in HOLDOUT_GROUPS if g in p3])
            p4_nb50 = np.mean([per_group[g].get("NB12_SP@50", 0) for g in HOLDOUT_GROUPS if g in per_group])
            p3_nb100 = np.mean([p3[g].get("NB12_SP@100", 0) for g in HOLDOUT_GROUPS if g in p3])
            p4_nb100 = np.mean([per_group[g].get("NB12_SP@100", 0) for g in HOLDOUT_GROUPS if g in per_group])
            p3_vc50 = np.mean([p3[g].get("VC@50", 0) for g in HOLDOUT_GROUPS if g in p3])
            p4_vc50 = np.mean([per_group[g].get("VC@50", 0) for g in HOLDOUT_GROUPS if g in per_group])
            print(f"\n{'='*80}")
            print(f"  Phase 4b vs Phase 3 (tt_v0c_r5_r15)")
            print(f"{'='*80}")
            print(f"  NB12_SP@50:  Phase3={p3_nb50:.4f}  Phase4b={p4_nb50:.4f}")
            print(f"  NB12_SP@100: Phase3={p3_nb100:.4f}  Phase4b={p4_nb100:.4f}")
            print(f"  VC@50:       Phase3={p3_vc50:.4f}  Phase4b={p4_vc50:.4f}  delta={p4_vc50-p3_vc50:+.4f}")
        except Exception:
            pass

        # Print gates
        print(f"\n{'='*80}")
        print(f"  Gate Check vs v0c")
        print(f"{'='*80}")
        for metric, gate in gate_results.items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {metric:<20} {status:>4}  wins={gate['wins']}/{gate['n_groups']}")
        print(f"\n  NB @50:  {'PASS' if nb_50['passed'] else 'FAIL'} (total={nb_50['total_count']})")
        print(f"  NB @100: {'PASS' if nb_100['passed'] else 'FAIL'} (total={nb_100['total_count']})")

        # Save
        config = {
            "version": args.version, "phase": "4b", "experiment": "regression",
            "features": features, "tau50": args.tau50, "tau100": args.tau100,
            "r50": args.r50, "r100": args.r100,
        }
        save_experiment(args.version, config, {"per_group": per_group},
                        gate_results=gate_results, baseline_version="v0c",
                        nb_gate_results={"k50": nb_50, "k100": nb_100})
        logger.info("Saved to registry/%s/", args.version)

    logger.info("Done (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
