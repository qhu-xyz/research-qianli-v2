"""Phase 7: Learned weights — logistic/ridge with incremental features.

Runs 6 feature steps × 2 models per class_type. Compares against v0c baseline.

Usage:
    PYTHONPATH=. uv run python scripts/phase6/run_learned_weights.py
    PYTHONPATH=. uv run python scripts/phase6/run_learned_weights.py --class-type onpeak
    PYTHONPATH=. uv run python scripts/phase6/run_learned_weights.py --holdout
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge

from ml.config import (
    EVAL_SPLITS, AQ_QUARTERS, DEV_GROUPS, HOLDOUT_GROUPS,
    CLASS_TYPES, CLASS_BF_COL, DANGEROUS_THRESHOLD_CLASS,
)
from ml.phase6.features import build_class_model_table_all
from ml.phase6.scoring import score_v0c

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

K_LEVELS = [150, 200, 300, 400]


# ── Feature steps ──────────────────────────────────────────────────────

def get_feature_steps(bf_col: str) -> list[tuple[str, list[str]]]:
    """Return (step_name, feature_list) for incremental feature ladder."""
    steps = [
        ("S1_v0c_3f", ["da_rank_value", "rt_max", bf_col]),
        ("S2_+spda_4f", ["da_rank_value", "rt_max", bf_col, "shadow_price_da"]),
        ("S3_+bins_7f", ["da_rank_value", "rt_max", bf_col, "shadow_price_da",
                         "bin_60_cid_max", "bin_70_cid_max", "bin_120_cid_max"]),
        ("S4_+struct_9f", ["da_rank_value", "rt_max", bf_col, "shadow_price_da",
                           "bin_60_cid_max", "bin_70_cid_max", "bin_120_cid_max",
                           "count_active_cids", "limit_mean"]),
        ("S5_+cross_11f", ["da_rank_value", "rt_max", bf_col, "shadow_price_da",
                            "bin_60_cid_max", "bin_70_cid_max", "bin_120_cid_max",
                            "count_active_cids", "limit_mean",
                            "cross_class_bf", "cross_shadow_price_da"]),
        ("S6_+counter_13f", ["da_rank_value", "rt_max", bf_col, "shadow_price_da",
                              "bin_60_cid_max", "bin_70_cid_max", "bin_120_cid_max",
                              "count_active_cids", "limit_mean",
                              "cross_class_bf", "cross_shadow_price_da",
                              "bin_-50_cid_max", "bin_-100_cid_max"]),
    ]
    return steps


# ── Normalization ──────────────────────────────────────────────────────

def normalize_features(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Min-max normalize features within the group (matching v0c contract)."""
    X = df.select(feature_cols).to_numpy().astype(np.float64)
    for j in range(X.shape[1]):
        mn, mx = X[:, j].min(), X[:, j].max()
        if mx > mn:
            X[:, j] = (X[:, j] - mn) / (mx - mn)
        else:
            X[:, j] = 0.5
    return X


# ── Evaluation ─────────────────────────────────────────────────────────

def eval_at_k(sp, scores, is_nb12, is_dorm, total_da, K):
    n = len(sp)
    topk = np.argsort(scores)[::-1][:min(K, n)]
    mask = np.zeros(n, dtype=bool)
    mask[topk] = True
    total_sp = sp.sum()
    n_bind = (sp > 0).sum()
    nb12_sp_tot = sp[is_nb12].sum()

    m = {
        f"VC@{K}": sp[mask].sum() / total_sp if total_sp > 0 else 0,
        f"Recall@{K}": (sp[mask] > 0).sum() / n_bind if n_bind > 0 else 0,
        f"NB12_Count@{K}": int(is_nb12[mask].sum()),
        f"NB12_SP@{K}": sp[mask & is_nb12].sum() / nb12_sp_tot if nb12_sp_tot > 0 else 0,
        f"Dorm_inK@{K}": int((mask & is_dorm).sum()),
    }
    for label, thresh in DANGEROUS_THRESHOLD_CLASS.items():
        dang = sp > thresh
        m[f"Dang{int(thresh/1000)}k_Recall@{K}"] = (mask & dang).sum() / dang.sum() if dang.sum() > 0 else 0
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-type", default=None, choices=CLASS_TYPES)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    ctypes = [args.class_type] if args.class_type else CLASS_TYPES
    target_split = "holdout" if args.holdout else "dev"
    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS

    for class_type in ctypes:
        t0 = time.time()
        bf_col = CLASS_BF_COL[class_type]
        steps = get_feature_steps(bf_col)

        # Build model tables
        all_needed = set()
        for si in EVAL_SPLITS.values():
            for py in si["train_pys"] + si["eval_pys"]:
                for aq in AQ_QUARTERS:
                    all_needed.add(f"{py}/{aq}")
        all_needed.discard("2025-06/aq4")

        mt = build_class_model_table_all(sorted(all_needed), class_type)

        # Derive rt_max
        mt = mt.with_columns(
            pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                              "bin_100_cid_max", "bin_110_cid_max").alias("rt_max")
        )

        # Add cross-class shadow_price_da (build other class table for Step 5)
        other_class = "offpeak" if class_type == "onpeak" else "onpeak"
        mt_other = build_class_model_table_all(sorted(all_needed), other_class)
        cross_spda = mt_other.select([
            "branch_name", "planning_year", "aq_quarter",
            pl.col("shadow_price_da").alias("cross_shadow_price_da"),
        ])
        mt = mt.join(cross_spda, on=["branch_name", "planning_year", "aq_quarter"], how="left")
        mt = mt.with_columns(pl.col("cross_shadow_price_da").fill_null(0.0))

        logger.info("Data built for %s (%.1fs)", class_type, time.time() - t0)

        # Results
        all_results = {}

        # v0c baseline
        all_results["v0c"] = {K: {} for K in K_LEVELS}
        for (py, aq), gdf in mt.group_by(["planning_year", "aq_quarter"], maintain_order=True):
            key = f"{py}/{aq}"
            if key not in eval_groups:
                continue
            sp = gdf["realized_shadow_price"].to_numpy().astype(np.float64)
            is_nb12 = gdf["is_nb_12"].to_numpy()
            is_dorm = np.array([c == "history_dormant" for c in gdf["cohort"].to_list()])
            total_da = float(gdf["total_da_sp_quarter"][0])
            scores = score_v0c(gdf, bf_col)
            for K in K_LEVELS:
                all_results["v0c"][K][key] = eval_at_k(sp, scores, is_nb12, is_dorm, total_da, K)

        # Per step × model
        for step_name, feature_cols in steps:
            avail_cols = [f for f in feature_cols if f in mt.columns]
            if len(avail_cols) < len(feature_cols):
                missing = set(feature_cols) - set(avail_cols)
                logger.warning("Step %s: missing features %s, skipping", step_name, missing)
                continue

            for model_name, model_cls, target_fn in [
                ("logistic", LogisticRegression, lambda sp: (sp > 0).astype(int)),
                ("ridge", Ridge, lambda sp: np.log1p(sp)),
            ]:
                full_name = f"{step_name}_{model_name}"
                all_results[full_name] = {K: {} for K in K_LEVELS}

                for ek, si in EVAL_SPLITS.items():
                    if si["split"] != target_split:
                        continue

                    # Train: pool normalized rows from training PYs
                    train_groups = mt.filter(pl.col("planning_year").is_in(si["train_pys"]))
                    X_train_parts, y_train_parts = [], []
                    for (py, aq), gdf in train_groups.group_by(
                        ["planning_year", "aq_quarter"], maintain_order=True
                    ):
                        X = normalize_features(gdf, avail_cols)
                        sp = gdf["realized_shadow_price"].to_numpy().astype(np.float64)
                        X_train_parts.append(X)
                        y_train_parts.append(target_fn(sp))

                    if not X_train_parts:
                        continue
                    X_train = np.vstack(X_train_parts)
                    y_train = np.concatenate(y_train_parts)

                    # Fit
                    if model_name == "logistic":
                        mdl = LogisticRegression(C=1.0, class_weight="balanced",
                                                  max_iter=1000, solver="lbfgs")
                        mdl.fit(X_train, y_train)
                    else:
                        mdl = Ridge(alpha=1.0)
                        mdl.fit(X_train, y_train)

                    # Eval per group
                    for py in si["eval_pys"]:
                        for aq in AQ_QUARTERS:
                            key = f"{py}/{aq}"
                            if key not in eval_groups:
                                continue
                            gdf = mt.filter(
                                (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                            )
                            X_eval = normalize_features(gdf, avail_cols)
                            sp = gdf["realized_shadow_price"].to_numpy().astype(np.float64)
                            is_nb12 = gdf["is_nb_12"].to_numpy()
                            is_dorm = np.array([c == "history_dormant" for c in gdf["cohort"].to_list()])
                            total_da = float(gdf["total_da_sp_quarter"][0])

                            if model_name == "logistic":
                                scores = mdl.predict_proba(X_eval)[:, 1]
                            else:
                                scores = mdl.predict(X_eval)

                            for K in K_LEVELS:
                                all_results[full_name][K][key] = eval_at_k(
                                    sp, scores, is_nb12, is_dorm, total_da, K)

        # Print
        split_label = target_split.upper()
        print(f"\n{'='*130}")
        print(f"  Phase 7: {class_type} ({split_label})")
        print(f"{'='*130}")

        for K in K_LEVELS:
            print(f"\n  K={K}")
            print(f"  {'Model':<30} {'VC':>8} {'Recall':>8} {'NB12_C':>7} {'NB12_SP':>8} {'Dg20k':>7} {'Dg40k':>7} {'Dorm':>6}")
            print(f"  {'-'*85}")
            for name in all_results:
                pg = all_results[name][K]
                if not pg:
                    continue
                vals = list(pg.values())
                ng = len(vals)
                print(f"  {name:<30} "
                      f"{sum(v.get(f'VC@{K}',0) for v in vals)/ng:>8.4f} "
                      f"{sum(v.get(f'Recall@{K}',0) for v in vals)/ng:>8.4f} "
                      f"{sum(v.get(f'NB12_Count@{K}',0) for v in vals)/ng:>7.1f} "
                      f"{sum(v.get(f'NB12_SP@{K}',0) for v in vals)/ng:>8.4f} "
                      f"{sum(v.get(f'Dang20k_Recall@{K}',0) for v in vals)/ng:>7.4f} "
                      f"{sum(v.get(f'Dang40k_Recall@{K}',0) for v in vals)/ng:>7.4f} "
                      f"{sum(v.get(f'Dorm_inK@{K}',0) for v in vals)/ng:>6.1f}")

        # Save
        out_dir = f"registry/{class_type}/phase7_{target_split}"
        os.makedirs(out_dir, exist_ok=True)
        summary = {}
        for name in all_results:
            summary[name] = {}
            for K in K_LEVELS:
                pg = all_results[name][K]
                if not pg:
                    continue
                vals = list(pg.values())
                ng = len(vals)
                summary[name][f"K={K}"] = {
                    mk: sum(v.get(mk, 0) for v in vals) / ng for mk in vals[0]
                }
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved to %s (%.1fs)", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
