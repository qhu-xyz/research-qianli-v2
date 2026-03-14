"""Evaluate V6.2B's own ranking as the v0 baseline.

Uses V6.2B's rank column (inverted: 1-rank = score) to compute
all LTR metrics. Saves results to registry/v0/.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.evaluate import aggregate_months, evaluate_ltr


def evaluate_v62b_month(auction_month: str) -> dict:
    """Evaluate V6.2B ranking for one month."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))
    actual = df["shadow_price_da"].to_numpy().astype(np.float64)
    scores = 1.0 - df["rank"].to_numpy().astype(np.float64)  # invert: higher = better
    return evaluate_ltr(actual, scores)


def main():
    eval_months = _DEFAULT_EVAL_MONTHS
    per_month = {}
    for m in eval_months:
        print(f"Evaluating V6.2B on {m}...")
        per_month[m] = evaluate_v62b_month(m)

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "model": "v62b_baseline",
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": [],
    }

    # Save to registry
    v0_dir = Path("registry/v0")
    v0_dir.mkdir(parents=True, exist_ok=True)
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(v0_dir / "config.json", "w") as f:
        json.dump({"model": "v62b_baseline", "note": "V6.2B formula ranking, no ML"}, f, indent=2)
    with open(v0_dir / "meta.json", "w") as f:
        json.dump({"version_id": "v0", "model": "v62b_baseline"}, f, indent=2)

    # Calibrate gates
    gates = {}
    group_a_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG"]
    group_b_metrics = ["VC@10", "VC@25", "VC@50", "VC@200",
                       "Recall@10", "Recall@50", "Spearman", "Tier0-AP", "Tier01-AP"]

    for metric in group_a_metrics + group_b_metrics:
        group = "A" if metric in group_a_metrics else "B"
        mean_val = agg["mean"].get(metric, 0)
        min_val = agg["min"].get(metric, 0)
        gates[metric] = {
            "floor": round(0.9 * mean_val, 6),
            "tail_floor": round(min_val, 6),
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "version": 1,
        "note": "Calibrated from V6.2B baseline",
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": eval_months},
        "gates": gates,
    }

    registry_dir = Path("registry")
    registry_dir.mkdir(exist_ok=True)
    with open(registry_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)
    with open(registry_dir / "champion.json", "w") as f:
        json.dump({"version": "v0"}, f, indent=2)
    with open(registry_dir / "version_counter.json", "w") as f:
        json.dump({"next_id": 1}, f, indent=2)

    # Print summary
    print("\n=== V6.2B Baseline (v0) ===")
    for metric in group_a_metrics:
        mean = agg["mean"].get(metric, 0)
        mn = agg["min"].get(metric, 0)
        mx = agg["max"].get(metric, 0)
        print(f"  {metric}: mean={mean:.4f} min={mn:.4f} max={mx:.4f}")
    print("\nGates calibrated. Champion set to v0.")


if __name__ == "__main__":
    main()
