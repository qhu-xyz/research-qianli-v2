"""Evaluate V6.2B forecast-only ranking as the v0 baseline.

V6.2B formula: rank_ori = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value
The da_rank_value component (60%) is realized DA shadow price rank = LEAKAGE.

Fair baseline uses only the forecast components:
  score = 0.30 * (1 - density_mix_rank_value) + 0.10 * (1 - density_ori_rank_value)
         (inverted so higher score = more binding)
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.evaluate import aggregate_months, evaluate_ltr


def evaluate_v62b_month(auction_month: str) -> dict:
    """Evaluate V6.2B fair (forecast-only) ranking for one month."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))
    actual = df["shadow_price_da"].to_numpy().astype(np.float64)

    # Fair score: only forecast components, no da_rank_value
    # density_*_rank_value are inverted (lower = more binding), so use (1 - x)
    dmix = df["density_mix_rank_value"].to_numpy().astype(np.float64)
    dori = df["density_ori_rank_value"].to_numpy().astype(np.float64)
    scores = 0.30 * (1.0 - dmix) + 0.10 * (1.0 - dori)

    return evaluate_ltr(actual, scores)


def main():
    eval_months = _DEFAULT_EVAL_MONTHS
    per_month = {}
    for m in eval_months:
        print(f"Evaluating V6.2B (fair) on {m}...")
        per_month[m] = evaluate_v62b_month(m)

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "model": "v62b_fair_baseline",
            "note": "Forecast-only: 0.30*(1-dmix) + 0.10*(1-dori). No da_rank_value.",
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
        json.dump({
            "model": "v62b_fair_baseline",
            "note": "V6.2B forecast-only ranking (no da_rank_value leakage)",
            "formula": "score = 0.30*(1-density_mix_rank_value) + 0.10*(1-density_ori_rank_value)",
        }, f, indent=2)
    with open(v0_dir / "meta.json", "w") as f:
        json.dump({"version_id": "v0", "model": "v62b_fair_baseline"}, f, indent=2)

    # Calibrate gates from fair baseline
    gates = {}
    group_a_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    group_b_metrics = ["VC@10", "VC@25", "VC@50", "VC@200",
                       "Recall@10", "Spearman", "Tier0-AP", "Tier01-AP"]

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
        "version": 2,
        "note": "Calibrated from V6.2B fair baseline (no leakage)",
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
    print("\n=== V6.2B Fair Baseline (v0) ===")
    for metric in group_a_metrics:
        mean = agg["mean"].get(metric, 0)
        mn = agg["min"].get(metric, 0)
        mx = agg["max"].get(metric, 0)
        print(f"  {metric}: mean={mean:.4f} min={mn:.4f} max={mx:.4f}")
    for metric in group_b_metrics:
        mean = agg["mean"].get(metric, 0)
        print(f"  {metric}: mean={mean:.4f}")
    print("\nGates calibrated from fair baseline. Champion set to v0.")


if __name__ == "__main__":
    main()
