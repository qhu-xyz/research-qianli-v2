"""v0c: da_rank + right_tail + bf_combined_12 blend formula baseline.

Score = 0.40 * da_rank_norm + 0.30 * right_tail_norm + 0.30 * bf_combined_12_norm
All normalized to [0, 1] within each (PY, quarter) group.
  - da_rank_norm: 1.0 = most binding (inverted)
  - right_tail_norm: 1.0 = highest density signal
  - bf_combined_12_norm: 1.0 = highest binding frequency
"""
from __future__ import annotations

import time

import polars as pl

from ml.features import build_model_table_all
from ml.evaluate import evaluate_all
from ml.registry import save_experiment
from ml.config import DEV_GROUPS, HOLDOUT_GROUPS


def _normalize_col(df: pl.DataFrame, col: str, invert: bool = False) -> pl.Series:
    """Normalize a column to [0, 1]. If invert=True, lower values get 1.0."""
    vals = df[col]
    v_min = vals.min()
    v_max = vals.max()
    v_range = v_max - v_min
    if v_range > 0:
        if invert:
            return (1.0 - (vals - v_min) / v_range).alias(f"_{col}_norm")
        else:
            return ((vals - v_min) / v_range).alias(f"_{col}_norm")
    else:
        return pl.Series(f"_{col}_norm", [0.5] * len(df))


def _score_group(group_df: pl.DataFrame) -> pl.DataFrame:
    """Compute v0c score for one (PY, quarter) group."""
    # right_tail_max
    group_df = group_df.with_columns(
        pl.max_horizontal(
            "bin_80_cid_max", "bin_90_cid_max", "bin_100_cid_max", "bin_110_cid_max",
        ).alias("_right_tail_max")
    )

    # Normalize
    da_norm = _normalize_col(group_df, "da_rank_value", invert=True)
    rt_norm = _normalize_col(group_df, "_right_tail_max", invert=False)
    bf_norm = _normalize_col(group_df, "bf_combined_12", invert=False)

    group_df = group_df.with_columns([da_norm, rt_norm, bf_norm])

    # Blend
    group_df = group_df.with_columns(
        (0.40 * pl.col("_da_rank_value_norm")
         + 0.30 * pl.col("__right_tail_max_norm")
         + 0.30 * pl.col("_bf_combined_12_norm")).alias("score")
    ).drop([
        "_right_tail_max",
        "_da_rank_value_norm", "__right_tail_max_norm", "_bf_combined_12_norm",
    ])

    return group_df


def main():
    start = time.time()

    print("Building model tables...")
    all_groups = DEV_GROUPS + HOLDOUT_GROUPS
    model_table = build_model_table_all(all_groups)

    # Score per group
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
        "version": "v0c",
        "formula": "0.40*da_rank_norm + 0.30*right_tail_norm + 0.30*bf_combined_12_norm",
        "features": ["da_rank_value", "bin_80_cid_max", "bin_90_cid_max",
                      "bin_100_cid_max", "bin_110_cid_max", "bf_combined_12"],
    }
    save_experiment("v0c", config, metrics)
    print("Saved to registry/v0c/")

    _print_summary(metrics)


def _print_summary(metrics):
    print("\n" + "=" * 60)
    print("v0c: 0.40*da_rank + 0.30*right_tail + 0.30*bf_combined_12")
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
