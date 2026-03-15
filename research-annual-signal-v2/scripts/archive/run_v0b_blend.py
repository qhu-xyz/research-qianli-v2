"""v0b: da_rank + right_tail blend formula baseline.

Score = 0.60 * da_rank_norm + 0.40 * right_tail_norm
Both normalized to [0, 1] within each (PY, quarter) group.
  - da_rank_norm: 1.0 = most binding (inverted: lower rank -> higher norm)
  - right_tail_norm: 1.0 = highest density signal (higher -> higher norm)
"""
from __future__ import annotations

import time

import polars as pl

from ml.features import build_model_table_all
from ml.evaluate import evaluate_all
from ml.registry import save_experiment
from ml.config import DEV_GROUPS, HOLDOUT_GROUPS


def _score_group(group_df: pl.DataFrame) -> pl.DataFrame:
    """Compute v0b score for one (PY, quarter) group."""
    # right_tail_max = max(bin_80..bin_110 cid_max)
    group_df = group_df.with_columns(
        pl.max_horizontal(
            "bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max",
        ).alias("_right_tail_max")
    )

    # Normalize da_rank_value: lower rank = more binding = 1.0
    da_max = group_df["da_rank_value"].max()
    da_min = group_df["da_rank_value"].min()
    da_range = da_max - da_min
    if da_range > 0:
        group_df = group_df.with_columns(
            (1.0 - (pl.col("da_rank_value") - da_min) / da_range).alias("_da_norm")
        )
    else:
        group_df = group_df.with_columns(pl.lit(0.5).alias("_da_norm"))

    # Normalize right_tail_max: higher = more binding = 1.0
    rt_max = group_df["_right_tail_max"].max()
    rt_min = group_df["_right_tail_max"].min()
    rt_range = rt_max - rt_min
    if rt_range > 0:
        group_df = group_df.with_columns(
            ((pl.col("_right_tail_max") - rt_min) / rt_range).alias("_rt_norm")
        )
    else:
        group_df = group_df.with_columns(pl.lit(0.5).alias("_rt_norm"))

    # Blend
    group_df = group_df.with_columns(
        (0.60 * pl.col("_da_norm") + 0.40 * pl.col("_rt_norm")).alias("score")
    ).drop(["_right_tail_max", "_da_norm", "_rt_norm"])

    return group_df


def main():
    start = time.time()

    print("Building model tables...")
    all_groups = DEV_GROUPS + HOLDOUT_GROUPS
    model_table = build_model_table_all(all_groups)

    # Score per group (normalization is group-local)
    scored_groups = []
    for (py, aq), group_df in model_table.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        scored_groups.append(_score_group(group_df))
    model_table = pl.concat(scored_groups, how="diagonal")

    print("Evaluating...")
    metrics = evaluate_all(model_table)

    walltime = time.time() - start
    print(f"\nWalltime: {walltime:.1f}s")

    config = {
        "version": "v0b",
        "formula": "0.60*da_rank_norm + 0.40*right_tail_norm",
        "features": ["da_rank_value", "bin_80_cid_max", "bin_90_cid_max",
                      "bin_100_cid_max", "bin_110_cid_max"],
    }
    save_experiment("v0b", config, metrics)
    print("Saved to registry/v0b/")

    _print_summary(metrics)


def _print_summary(metrics):
    print("\n" + "=" * 60)
    print("v0b: 0.60*da_rank + 0.40*right_tail")
    print("=" * 60)
    for group, m in sorted(metrics["per_group"].items()):
        print(f"\n{group}:")
        for k in ["VC@50", "VC@100", "Recall@50", "NDCG", "Abs_SP@50", "NB12_Recall@50"]:
            if k in m:
                print(f"  {k}: {m[k]:.4f}")

    for split in ["dev_mean", "holdout_mean"]:
        if split in metrics:
            print(f"\n{split}:")
            for k in ["VC@50", "VC@100", "Recall@50", "NDCG"]:
                if k in metrics[split]:
                    print(f"  {k}: {metrics[split][k]:.4f}")


if __name__ == "__main__":
    main()
