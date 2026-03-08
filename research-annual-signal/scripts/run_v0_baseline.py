"""v0 Baseline: V6.1 formula rank evaluated against realized DA.

No training. Just loads V6.1 rank and evaluates against ground truth.
Produces registry/v0/metrics.json and calibrates gates.

Formula: rank_ori = 0.60*da_rank + 0.30*density_mix_rank + 0.10*density_ori_rank
Score for evaluation: 1 - rank  (higher = more binding)
"""
import json
import gc
from pathlib import Path

from ml.config import DEFAULT_EVAL_GROUPS, SCREEN_EVAL_GROUPS
from ml.data_loader import load_v61_group
from ml.evaluate import evaluate_ltr, aggregate_months
from ml.ground_truth import get_ground_truth

import numpy as np


def run_v0_baseline(eval_groups: list[str], registry_dir: str = "registry") -> dict:
    per_group = {}

    for group_id in eval_groups:
        planning_year, aq_round = group_id.split("/")
        print(f"\n[v0] === {group_id} ===")

        v61 = load_v61_group(planning_year, aq_round)
        v61 = get_ground_truth(planning_year, aq_round, v61, cache=True)

        # V6.1 formula score: rank column (lower = more binding)
        # Invert: 1 - rank so higher = more binding for evaluate_ltr
        scores = 1.0 - v61["rank"].to_numpy().astype(np.float64)
        actual = v61["realized_shadow_price"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_group[group_id] = metrics

        n_binding = (actual > 0).sum()
        print(f"  Binding: {n_binding}/{len(actual)} ({100*n_binding/len(actual):.1f}%)")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        gc.collect()

    agg = aggregate_months(per_group)

    result = {
        "eval_config": {"eval_groups": eval_groups, "mode": "eval"},
        "per_month": per_group,
        "aggregate": agg,
        "n_months": len(per_group),
        "n_months_requested": len(eval_groups),
        "skipped_months": [],
    }

    # Save to registry
    version_dir = Path(registry_dir) / "v0"
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "formula": "rank_ori = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value",
        "score": "1 - rank (inverted so higher = more binding)",
        "note": "No training. Evaluates stored V6.1 formula rank against realized DA ground truth.",
    }
    with open(version_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[v0] === AGGREGATE (mean across {len(per_group)} groups) ===")
    for k, v in agg.get("mean", {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Calibrate gates from v0 results
    _calibrate_gates(agg, registry_dir)

    return result


def _calibrate_gates(agg: dict, registry_dir: str) -> None:
    """Calibrate 3-layer gates from v0 aggregate metrics."""
    blocking_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    monitor_metrics = ["Spearman", "Tier0-AP", "Tier01-AP"]

    gates = {}
    for metric in blocking_metrics + monitor_metrics:
        mean_val = agg.get("mean", {}).get(metric)
        min_val = agg.get("min", {}).get(metric)
        if mean_val is None:
            continue
        group = "A" if metric in blocking_metrics else "B"
        gates[metric] = {
            "floor": round(0.9 * mean_val, 4),         # L1: 90% of v0 mean
            "tail_floor": round(min_val, 4) if min_val is not None else None,  # L2: v0 min
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
    }

    gates_path = Path(registry_dir) / "gates.json"
    with open(gates_path, "w") as f:
        json.dump(gates_data, f, indent=2)
    print(f"[v0] Calibrated gates from v0 -> {gates_path}")

    champion_path = Path(registry_dir) / "champion.json"
    with open(champion_path, "w") as f:
        json.dump({"version": "v0"}, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()

    groups = SCREEN_EVAL_GROUPS if args.screen else DEFAULT_EVAL_GROUPS
    run_v0_baseline(groups)
