"""Phase 3.1: NB population analysis and Track B feature profiling.

3.1.1 - NB populations at multiple windows (6, 12, 24)
3.1.2 - Track B feature profiling: base rates, feature distributions, per-feature AUC
3.1.3 - Feature correlation within Track B, prune |r| > 0.85

Usage:
    PYTHONPATH=. uv run python scripts/run_nb_analysis.py
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, AQ_QUARTERS, REGISTRY_DIR,
    DENSITY_MAX_FEATURES, DENSITY_MIN_FEATURES,
    LIMIT_FEATURES, METADATA_FEATURES,
)
from ml.features import build_model_table_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Track B candidate features (from spec — density + limits + metadata)
TRACK_B_FEATURES: list[str] = (
    DENSITY_MAX_FEATURES + DENSITY_MIN_FEATURES
    + LIMIT_FEATURES + METADATA_FEATURES
)


def analyze_nb_populations(model_table: pl.DataFrame, groups: list[str]) -> dict:
    """3.1.1: NB populations at multiple windows."""
    results: dict = {}
    for g in groups:
        py, aq = g.split("/")
        gdf = model_table.filter(
            (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
        )
        n_total = len(gdf)
        n_binding = gdf.filter(pl.col("realized_shadow_price") > 0).height
        total_sp = gdf["realized_shadow_price"].sum()

        group_result: dict = {"n_total": n_total, "n_binding": n_binding, "total_sp": float(total_sp)}
        for window in [6, 12, 24]:
            col = f"is_nb_{window}"
            nb_mask = gdf[col].to_numpy()
            sp = gdf["realized_shadow_price"].to_numpy()
            n_nb = int(nb_mask.sum())
            nb_sp = float(sp[nb_mask].sum())
            group_result[f"nb{window}_count"] = n_nb
            group_result[f"nb{window}_sp"] = nb_sp
            group_result[f"nb{window}_sp_share"] = nb_sp / total_sp if total_sp > 0 else 0.0

        # Cohort breakdown
        for cohort in ["established", "history_dormant", "history_zero"]:
            c_mask = gdf["cohort"] == cohort
            group_result[f"cohort_{cohort}_count"] = int(c_mask.sum())

        results[g] = group_result

    return results


def profile_track_b(model_table: pl.DataFrame, groups: list[str]) -> dict:
    """3.1.2: Track B feature profiling."""
    track_b = model_table.filter(
        (pl.col("cohort").is_in(["history_dormant", "history_zero"]))
        & ((pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(groups))
    )

    n_total = len(track_b)
    target = (track_b["realized_shadow_price"].to_numpy() > 0).astype(int)
    n_binders = int(target.sum())
    base_rate = n_binders / n_total if n_total > 0 else 0.0

    # Per-feature AUC
    feature_auc: dict[str, float] = {}
    for feat in TRACK_B_FEATURES:
        if feat not in track_b.columns:
            continue
        vals = track_b[feat].to_numpy().astype(np.float64)
        if np.std(vals) == 0 or n_binders == 0 or n_binders == n_total:
            feature_auc[feat] = 0.5
            continue
        try:
            auc = roc_auc_score(target, vals)
            feature_auc[feat] = float(auc)
        except ValueError:
            feature_auc[feat] = 0.5

    return {
        "n_track_b": n_total,
        "n_binders": n_binders,
        "base_rate": base_rate,
        "feature_auc": feature_auc,
    }


def compute_correlation_matrix(
    model_table: pl.DataFrame,
    groups: list[str],
    features: list[str],
) -> tuple[dict, list[str]]:
    """3.1.3: Feature correlation within Track B, prune |r| > 0.85."""
    track_b = model_table.filter(
        (pl.col("cohort").is_in(["history_dormant", "history_zero"]))
        & ((pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(groups))
    )

    available = [f for f in features if f in track_b.columns]
    X = track_b.select(available).to_numpy().astype(np.float64)
    corr = np.corrcoef(X, rowvar=False)

    high_corr_pairs: list[dict] = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if abs(corr[i, j]) > 0.85:
                high_corr_pairs.append({
                    "feat_a": available[i],
                    "feat_b": available[j],
                    "correlation": float(corr[i, j]),
                })

    return {"high_corr_pairs": high_corr_pairs, "n_features": len(available)}, available


def main():
    t0 = time.time()

    # Build model tables — need all PYs for build_model_table_all
    all_needed: set[str] = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")

    logger.info("Building model tables for %d groups...", len(all_needed))
    model_table = build_model_table_all(sorted(all_needed))

    # 3.1.1: NB populations
    logger.info("=== Phase 3.1.1: NB Population Analysis ===")
    pop_results = analyze_nb_populations(model_table, DEV_GROUPS)

    print("\n" + "=" * 90)
    print("  Phase 3.1.1: NB Populations at Multiple Windows (Dev Only)")
    print("=" * 90)
    header = f"{'Group':<16} {'N':>6} {'Bind':>6} {'NB6':>5} {'NB12':>5} {'NB24':>5} {'NB12_SP%':>8} {'Dormant':>8} {'Zero':>8}"
    print(header)
    print("-" * len(header))
    for g in DEV_GROUPS:
        r = pop_results[g]
        print(
            f"{g:<16} {r['n_total']:>6} {r['n_binding']:>6} "
            f"{r['nb6_count']:>5} {r['nb12_count']:>5} {r['nb24_count']:>5} "
            f"{r['nb12_sp_share']:>7.1%} "
            f"{r['cohort_history_dormant_count']:>8} {r['cohort_history_zero_count']:>8}"
        )

    # 3.1.2: Track B profiling
    logger.info("=== Phase 3.1.2: Track B Feature Profiling ===")
    profile = profile_track_b(model_table, DEV_GROUPS)

    print(f"\n{'=' * 70}")
    print("  Phase 3.1.2: Track B Profiling (Dev)")
    print(f"{'=' * 70}")
    print(f"  Track B total: {profile['n_track_b']}")
    print(f"  Track B binders (NB12): {profile['n_binders']}")
    print(f"  Base rate: {profile['base_rate']:.2%}")
    print("\n  Per-Feature AUC (descending):")
    sorted_auc = sorted(profile["feature_auc"].items(), key=lambda x: x[1], reverse=True)
    for feat, auc in sorted_auc:
        direction = "^" if auc > 0.5 else "v" if auc < 0.5 else "="
        print(f"    {feat:<30} AUC={auc:.4f} {direction}")

    # 3.1.3: Correlation
    logger.info("=== Phase 3.1.3: Feature Correlation ===")
    corr_results, available_feats = compute_correlation_matrix(
        model_table, DEV_GROUPS, TRACK_B_FEATURES,
    )

    print(f"\n{'=' * 70}")
    print("  Phase 3.1.3: High Correlation Pairs (|r| > 0.85)")
    print(f"{'=' * 70}")
    for pair in corr_results["high_corr_pairs"]:
        print(f"  {pair['feat_a']:<30} x {pair['feat_b']:<30} r={pair['correlation']:.3f}")
    if not corr_results["high_corr_pairs"]:
        print("  (none)")

    # Save all results
    out_dir = REGISTRY_DIR / "nb_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "population.json", "w") as f:
        json.dump(pop_results, f, indent=2, default=str)
    with open(out_dir / "track_b_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    with open(out_dir / "correlation.json", "w") as f:
        json.dump(corr_results, f, indent=2, default=str)

    logger.info("Results saved to %s (%.1fs)", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
