"""v0a: pure da_rank_value formula baseline.

Score = -da_rank_value (lower rank = more binding = higher score).
"""
from __future__ import annotations

import time

import polars as pl

from ml.features import build_model_table_all
from ml.evaluate import evaluate_all
from ml.registry import save_experiment
from ml.config import DEV_GROUPS, HOLDOUT_GROUPS


def main():
    start = time.time()

    # Build model tables for all groups
    print("Building model tables...")
    all_groups = DEV_GROUPS + HOLDOUT_GROUPS
    model_table = build_model_table_all(all_groups)

    # Score: -da_rank_value
    model_table = model_table.with_columns(
        (-pl.col("da_rank_value")).alias("score")
    )

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_all(model_table)

    walltime = time.time() - start
    print(f"\nWalltime: {walltime:.1f}s")

    # Save
    config = {"version": "v0a", "formula": "-da_rank_value", "features": ["da_rank_value"]}
    save_experiment("v0a", config, metrics)
    print("Saved to registry/v0a/")

    # Print summary
    _print_summary(metrics)


def _print_summary(metrics):
    print("\n" + "=" * 60)
    print("v0a: pure da_rank_value")
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
