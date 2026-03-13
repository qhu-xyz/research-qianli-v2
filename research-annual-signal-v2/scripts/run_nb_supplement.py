"""Compute NB supplement metrics for existing registry entries.

Phase 3.0.5: adds NB12_Count@50, NB12_SP@50, NB6_Recall@50, NB24_Recall@50
to existing versions WITHOUT overwriting frozen metrics.json.

Usage:
    PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v0c
    PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v3a
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS,
    REGISTRY_DIR,
)
from ml.features import build_model_table_all
from ml.evaluate import evaluate_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_col(df: pl.DataFrame, col: str, invert: bool = False) -> pl.Series:
    vals = df[col]
    v_min, v_max = vals.min(), vals.max()
    v_range = v_max - v_min
    if v_range > 0:
        if invert:
            return (1.0 - (vals - v_min) / v_range).alias(f"_{col}_norm")
        else:
            return ((vals - v_min) / v_range).alias(f"_{col}_norm")
    else:
        return pl.Series(f"_{col}_norm", [0.5] * len(df))


def compute_v0c_scores(model_table: pl.DataFrame) -> pl.DataFrame:
    """Replicate v0c formula scoring. See baseline_contract.json."""
    frames = []
    for (py, aq), gdf in model_table.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        gdf = gdf.with_columns(
            pl.max_horizontal(
                "bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max",
            ).alias("_right_tail_max")
        )
        da_norm = _normalize_col(gdf, "da_rank_value", invert=True)
        rt_norm = _normalize_col(gdf, "_right_tail_max", invert=False)
        bf_norm = _normalize_col(gdf, "bf_combined_12", invert=False)
        gdf = gdf.with_columns([da_norm, rt_norm, bf_norm])
        gdf = gdf.with_columns(
            (0.40 * pl.col("_da_rank_value_norm")
             + 0.30 * pl.col("__right_tail_max_norm")
             + 0.30 * pl.col("_bf_combined_12_norm")).alias("score")
        ).drop([
            "_right_tail_max",
            "_da_rank_value_norm", "__right_tail_max_norm", "_bf_combined_12_norm",
        ])
        frames.append(gdf)
    return pl.concat(frames, how="diagonal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=["v0c", "v3a"])
    args = parser.parse_args()

    t0 = time.time()

    eval_groups = DEV_GROUPS + HOLDOUT_GROUPS
    logger.info("Building model tables for %d eval groups...", len(eval_groups))
    model_table = build_model_table_all(eval_groups)

    if args.version == "v0c":
        scored = compute_v0c_scores(model_table)
    elif args.version == "v3a":
        from ml.registry import load_config
        from ml.train import train_and_predict

        config = load_config("v3a")
        feature_cols = config["features"]

        # Need full model_table with train PYs too
        all_groups = []
        all_pys: set[str] = set()
        for split_info in EVAL_SPLITS.values():
            all_pys.update(split_info["train_pys"])
            all_pys.update(split_info["eval_pys"])
        for py in sorted(all_pys):
            for aq in ["aq1", "aq2", "aq3", "aq4"]:
                key = f"{py}/{aq}"
                if key == "2025-06/aq4":
                    continue
                all_groups.append(key)

        full_table = build_model_table_all(all_groups)

        scored_frames = []
        for eval_key, split_info in EVAL_SPLITS.items():
            scored_split, _ = train_and_predict(
                model_table=full_table,
                train_pys=split_info["train_pys"],
                eval_pys=split_info["eval_pys"],
                feature_cols=feature_cols,
            )
            valid_groups = set(eval_groups)
            scored_split = scored_split.filter(
                (pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(valid_groups)
            )
            scored_frames.append(scored_split)
        scored = pl.concat(scored_frames, how="diagonal")

    # Evaluate each group with new NB metrics
    nb_supplement: dict = {"per_group": {}}
    for (py, aq), gdf in scored.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        key = f"{py}/{aq}"
        m = evaluate_group(gdf)
        nb_metrics = {
            k: v for k, v in m.items()
            if k.startswith("NB") or k == "cohort_contribution"
        }
        nb_supplement["per_group"][key] = nb_metrics

    # Aggregate
    for split_name, split_groups in [("dev_mean", DEV_GROUPS), ("holdout_mean", HOLDOUT_GROUPS)]:
        present = [g for g in split_groups if g in nb_supplement["per_group"]]
        if present:
            agg: dict = {}
            all_keys: set = set()
            for g in present:
                all_keys.update(nb_supplement["per_group"][g].keys())
            for mk in all_keys:
                vals = [nb_supplement["per_group"][g][mk] for g in present
                        if mk in nb_supplement["per_group"][g]
                        and isinstance(nb_supplement["per_group"][g][mk], (int, float))]
                if vals:
                    agg[mk] = sum(vals) / len(vals)
            nb_supplement[split_name] = agg

    # Save
    out_path = REGISTRY_DIR / args.version / "nb_metrics_supplement.json"
    with open(out_path, "w") as f:
        json.dump(nb_supplement, f, indent=2, default=str)

    logger.info("Saved NB supplement to %s (%.1fs)", out_path, time.time() - t0)

    # Print summary
    for split_name in ["dev_mean", "holdout_mean"]:
        if split_name in nb_supplement:
            print(f"\n{split_name}:")
            for k, v in sorted(nb_supplement[split_name].items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
