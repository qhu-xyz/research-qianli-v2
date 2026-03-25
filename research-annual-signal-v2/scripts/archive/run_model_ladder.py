"""Phase 6: Class-specific model ladder — v0a → v0c → v0c+cross → NB → blend.

Runs the full model ladder for both onpeak and offpeak, evaluates at
K=150/200/300/400, and saves evidence per milestone.

Usage:
    PYTHONPATH=. uv run python scripts/phase6/run_model_ladder.py
    PYTHONPATH=. uv run python scripts/phase6/run_model_ladder.py --class-type onpeak
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import polars as pl

from ml.config import (
    EVAL_SPLITS, AQ_QUARTERS, DEV_GROUPS, HOLDOUT_GROUPS,
    CLASS_TYPES, CLASS_BF_COL, DANGEROUS_THRESHOLD_CLASS,
)
from ml.phase6.features import build_class_model_table_all
from ml.phase6.scoring import (
    score_v0a, score_v0c, score_v0c_cross,
    train_nb_model, blend_scores,
)
from ml.evaluate import evaluate_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

K_LEVELS = [150, 200, 300, 400]
BLEND_ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20]


def eval_at_k(group_df, scores, K, dang_thresh):
    """Evaluate scoring at a single K with class-specific dangerous threshold."""
    sp = group_df["realized_shadow_price"].to_numpy().astype(np.float64)
    total_sp = sp.sum()
    n_bind = (sp > 0).sum()
    total_da = float(group_df["total_da_sp_quarter"][0])
    is_nb12 = group_df["is_nb_12"].to_numpy() if "is_nb_12" in group_df.columns else np.zeros(len(sp), dtype=bool)
    cohorts = group_df["cohort"].to_list()
    is_dorm = np.array([c == "history_dormant" for c in cohorts])

    n = len(sp)
    topk = np.argsort(scores)[::-1][:min(K, n)]
    mask = np.zeros(n, dtype=bool)
    mask[topk] = True

    nb12_sp_tot = sp[is_nb12].sum()
    dang = sp > dang_thresh

    return {
        f"VC@{K}": sp[mask].sum() / total_sp if total_sp > 0 else 0,
        f"Recall@{K}": (sp[mask] > 0).sum() / n_bind if n_bind > 0 else 0,
        f"Abs_SP@{K}": sp[mask].sum() / total_da if total_da > 0 else 0,
        f"NB12_Count@{K}": int(is_nb12[mask].sum()),
        f"NB12_SP@{K}": sp[mask & is_nb12].sum() / nb12_sp_tot if nb12_sp_tot > 0 else 0,
        f"Dang_Recall@{K}": (mask & dang).sum() / dang.sum() if dang.sum() > 0 else 0,
        f"Dorm_inK@{K}": int((mask & is_dorm).sum()),
    }


def run_model_ladder(class_type: str, target_split: str = "dev"):
    t0 = time.time()
    eval_groups = DEV_GROUPS if target_split == "dev" else HOLDOUT_GROUPS
    bf_col = CLASS_BF_COL[class_type]

    # Build all model tables
    all_needed = set()
    for si in EVAL_SPLITS.values():
        for py in si["train_pys"] + si["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")

    logger.info("Building class-specific model tables for %s...", class_type)
    mt = build_class_model_table_all(sorted(all_needed), class_type)
    logger.info("Built %d rows in %.1fs", len(mt), time.time() - t0)

    # Models to evaluate
    models = {
        "v0a": lambda gdf: score_v0a(gdf),
        "v0c": lambda gdf: score_v0c(gdf, bf_col),
        "v0c_cross": lambda gdf: score_v0c_cross(gdf, bf_col),
    }

    # Results: {model_name: {K: {group: metrics}}}
    all_results = {}

    for model_name, scorer in models.items():
        all_results[model_name] = {K: {} for K in K_LEVELS}

        for (py, aq), gdf in mt.group_by(["planning_year", "aq_quarter"], maintain_order=True):
            key = f"{py}/{aq}"
            if key not in eval_groups:
                continue

            scores = scorer(gdf)

            for K in K_LEVELS:
                for dang_label, dang_thresh in DANGEROUS_THRESHOLD_CLASS.items():
                    m = eval_at_k(gdf, scores, K, dang_thresh)
                    # Rename dang metrics with threshold label
                    renamed = {}
                    for mk, mv in m.items():
                        if "Dang" in mk:
                            renamed[mk.replace("Dang", f"Dang{int(dang_thresh/1000)}k")] = mv
                        else:
                            renamed[mk] = mv
                    if key not in all_results[model_name][K]:
                        all_results[model_name][K][key] = {}
                    all_results[model_name][K][key].update(renamed)

    # NB model + blend (train per split)
    for alpha in BLEND_ALPHAS:
        model_name = f"blend_a{alpha:.2f}" if alpha > 0 else "v0c_solo"
        if model_name == "v0c_solo" and "v0c" in all_results:
            continue  # Already computed as v0c
        all_results[model_name] = {K: {} for K in K_LEVELS}

        for ek, si in EVAL_SPLITS.items():
            if si["split"] != target_split:
                continue

            # Train NB on dormant
            train_df = mt.filter(
                pl.col("planning_year").is_in(si["train_pys"])
                & (pl.col("cohort") == "history_dormant")
            )
            if len(train_df) < 10:
                continue
            nb_model = train_nb_model(train_df)

            for py in si["eval_pys"]:
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    if key not in eval_groups:
                        continue

                    gdf = mt.filter(
                        (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                    )
                    base = score_v0c(gdf, bf_col)
                    scores = blend_scores(gdf, base, nb_model, alpha)

                    for K in K_LEVELS:
                        for dang_label, dang_thresh in DANGEROUS_THRESHOLD_CLASS.items():
                            m = eval_at_k(gdf, scores, K, dang_thresh)
                            renamed = {}
                            for mk, mv in m.items():
                                if "Dang" in mk:
                                    renamed[mk.replace("Dang", f"Dang{int(dang_thresh/1000)}k")] = mv
                                else:
                                    renamed[mk] = mv
                            if key not in all_results[model_name][K]:
                                all_results[model_name][K][key] = {}
                            all_results[model_name][K][key].update(renamed)

    # Print results
    split_label = target_split.upper()
    print(f"\n{'='*140}")
    print(f"  Phase 6 Model Ladder: {class_type} ({split_label})")
    print(f"{'='*140}")

    for K in K_LEVELS:
        print(f"\n  K={K}")
        header = f"  {'Model':<18} {'VC':>8} {'Recall':>8} {'NB12_C':>7} {'NB12_SP':>8} {'Dg20k_R':>8} {'Dg40k_R':>8} {'Dorm_K':>7}"
        print(header)
        print(f"  {'-'*75}")

        for model_name in all_results:
            per_group = all_results[model_name][K]
            if not per_group:
                continue
            vals = list(per_group.values())
            ng = len(vals)
            avg_vc = sum(v.get(f"VC@{K}", 0) for v in vals) / ng
            avg_rec = sum(v.get(f"Recall@{K}", 0) for v in vals) / ng
            avg_nb_c = sum(v.get(f"NB12_Count@{K}", 0) for v in vals) / ng
            avg_nb_sp = sum(v.get(f"NB12_SP@{K}", 0) for v in vals) / ng
            avg_dg20 = sum(v.get(f"Dang20k_Recall@{K}", 0) for v in vals) / ng
            avg_dg40 = sum(v.get(f"Dang40k_Recall@{K}", 0) for v in vals) / ng
            avg_dorm = sum(v.get(f"Dorm_inK@{K}", 0) for v in vals) / ng

            print(f"  {model_name:<18} {avg_vc:>8.4f} {avg_rec:>8.4f} {avg_nb_c:>7.1f} {avg_nb_sp:>8.4f} "
                  f"{avg_dg20:>8.4f} {avg_dg40:>8.4f} {avg_dorm:>7.1f}")

    # Save evidence
    out_dir = f"registry/{class_type}/m2_{target_split}"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for model_name in all_results:
        summary[model_name] = {}
        for K in K_LEVELS:
            per_group = all_results[model_name][K]
            if not per_group:
                continue
            vals = list(per_group.values())
            ng = len(vals)
            summary[model_name][f"K={K}"] = {
                mk: sum(v.get(mk, 0) for v in vals) / ng
                for mk in vals[0]
            }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Evidence saved to %s (%.1fs total)", out_dir, time.time() - t0)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-type", default=None, choices=CLASS_TYPES)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    ctypes = [args.class_type] if args.class_type else CLASS_TYPES
    split = "holdout" if args.holdout else "dev"

    for ct in ctypes:
        run_model_ladder(ct, split)


if __name__ == "__main__":
    main()
