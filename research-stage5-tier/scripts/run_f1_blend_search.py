#!/usr/bin/env python
"""F1 blend search: grid-search da/dmix/dori weights on f1 dev months.

Mirrors the f0 v7 experiment but on f1 data with correct GT (delivery_month).
v1 dev metrics are in-sample (weights selected on dev). Holdout is the fair comparison.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import has_period_type, delivery_month, _FULL_EVAL_MONTHS
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.registry_paths import registry_root, holdout_root

ROOT = Path(__file__).resolve().parent.parent
REALIZED_DA_DIR = ROOT / "data" / "realized_da"
HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def _has_gt(auction_month: str, period_type: str, class_type: str = "onpeak") -> bool:
    """Check if realized DA ground truth exists for this auction_month + period_type."""
    gt_month = delivery_month(auction_month, period_type)
    if class_type == "onpeak":
        return (REALIZED_DA_DIR / f"{gt_month}.parquet").exists()
    return (REALIZED_DA_DIR / f"{gt_month}_{class_type}.parquet").exists()


def search_blend(eval_months: list[str], period_type: str = "f1", class_type: str = "onpeak"):
    """Grid search over (da, dmix, dori) triplets summing to 1.0, step 0.05."""
    # Preload all months
    data = {}
    for m in eval_months:
        df = load_v62b_month(m, period_type, class_type)
        data[m] = {
            "da": df["da_rank_value"].to_numpy(),
            "dmix": df["density_mix_rank_value"].to_numpy(),
            "dori": df["density_ori_rank_value"].to_numpy(),
            "actual": df["realized_sp"].to_numpy().astype(np.float64),
        }

    # Grid: step 0.05, sum to 1.0
    steps = [round(x * 0.05, 2) for x in range(21)]  # 0.00 to 1.00
    triplets = [(a, b, c) for a in steps for b in steps for c in steps
                if abs(a + b + c - 1.0) < 1e-9]
    print(f"[blend] Searching {len(triplets)} triplets on {len(eval_months)} months")

    best_vc20, best_weights = -1.0, None
    for w_da, w_dmix, w_dori in triplets:
        per_month = {}
        for m, d in data.items():
            scores = -(w_da * d["da"] + w_dmix * d["dmix"] + w_dori * d["dori"])
            metrics = evaluate_ltr(d["actual"], scores)
            per_month[m] = metrics
        agg = aggregate_months(per_month)
        mean_vc20 = agg["mean"]["VC@20"]
        if mean_vc20 > best_vc20:
            best_vc20 = mean_vc20
            best_weights = (w_da, w_dmix, w_dori)

    print(f"[blend] Best: da={best_weights[0]}, dmix={best_weights[1]}, "
          f"dori={best_weights[2]}, VC@20={best_vc20:.4f}")
    return best_weights


def evaluate_blend(weights: tuple, eval_months: list[str], period_type: str, class_type: str) -> dict:
    """Evaluate a specific blend on given months."""
    w_da, w_dmix, w_dori = weights
    per_month = {}
    for m in eval_months:
        df = load_v62b_month(m, period_type, class_type)
        scores = -(w_da * df["da_rank_value"].to_numpy()
                   + w_dmix * df["density_mix_rank_value"].to_numpy()
                   + w_dori * df["density_ori_rank_value"].to_numpy())
        actual = df["realized_sp"].to_numpy().astype(np.float64)
        metrics = evaluate_ltr(actual, scores)
        n_binding = int((actual > 0).sum())
        print(f"  {m}: n={len(df)}, binding={n_binding}, "
              f"VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}")
        per_month[m] = metrics
    return per_month


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default="f1")
    parser.add_argument("--class-type", default="onpeak")
    args = parser.parse_args()

    period_type = args.ptype
    class_type = args.class_type
    t0 = time.time()

    # Filter eval months (must have period type AND realized DA ground truth)
    dev_months = [m for m in _FULL_EVAL_MONTHS
                  if has_period_type(m, period_type) and _has_gt(m, period_type, class_type)]
    holdout_months = [m for m in HOLDOUT_MONTHS
                      if has_period_type(m, period_type) and _has_gt(m, period_type, class_type)]
    print(f"[v1] Blend search for {period_type}/{class_type}: {len(dev_months)} dev, {len(holdout_months)} holdout")

    # Grid search on dev
    best_weights = search_blend(dev_months, period_type, class_type)
    w_da, w_dmix, w_dori = best_weights

    # Evaluate best blend on dev (for metrics recording)
    print(f"\n[v1] Evaluating best blend ({w_da}, {w_dmix}, {w_dori}) on dev...")
    dev_pm = evaluate_blend(best_weights, dev_months, period_type, class_type)
    dev_agg = aggregate_months(dev_pm)
    dev_means = dev_agg["mean"]
    print(f"  => VC@20={dev_means['VC@20']:.4f} VC@100={dev_means['VC@100']:.4f} "
          f"Spearman={dev_means['Spearman']:.4f}")

    # Evaluate on holdout
    print(f"\n[v1] Evaluating on holdout ({len(holdout_months)} months)...")
    ho_pm = evaluate_blend(best_weights, holdout_months, period_type, class_type)
    ho_agg = aggregate_months(ho_pm)
    ho_means = ho_agg["mean"]
    print(f"  => VC@20={ho_means['VC@20']:.4f} VC@100={ho_means['VC@100']:.4f} "
          f"Spearman={ho_means['Spearman']:.4f}")

    # Compare with v0
    reg_slice = registry_root(period_type, class_type, base_dir=ROOT / "registry")
    v0_path = reg_slice / "v0" / "metrics.json"
    if v0_path.exists():
        v0_dev = json.load(open(v0_path))["aggregate"]["mean"]
        print(f"\n[v1] vs v0 (dev): VC@20 {v0_dev['VC@20']:.4f} -> {dev_means['VC@20']:.4f} "
              f"({100*(dev_means['VC@20']/v0_dev['VC@20']-1):+.1f}%)")

    # Save dev results
    version_id = "v1"
    v1_dir = reg_slice / version_id
    v1_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "eval_config": {
            "eval_months": dev_months,
            "class_type": class_type,
            "period_type": period_type,
            "mode": "eval",
            "note": "weights selected on dev — holdout is the fair comparison",
        },
        "blend_weights": {"da": w_da, "dmix": w_dmix, "dori": w_dori},
        "per_month": dev_pm,
        "aggregate": dev_agg,
        "n_months": len(dev_pm),
    }
    with open(v1_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    config_out = {
        "method": "optimized_blend",
        "formula": f"-({w_da}*da_rank_value + {w_dmix}*density_mix_rank_value + {w_dori}*density_ori_rank_value)",
        "ground_truth": f"realized_da (abs sum of DA shadow prices, {class_type})",
        "features_used": ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"],
        "search": "grid_step_0.05",
        "optimized_on": "mean VC@20 over dev months",
    }
    with open(v1_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    meta_out = {
        "version_id": version_id,
        "description": f"Optimized blend weights for {period_type}/{class_type}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(v1_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\n[v1] Saved dev to {v1_dir}")

    # Save holdout results
    ho_slice = holdout_root(period_type, class_type, base_dir=ROOT / "holdout")
    ho_dir = ho_slice / version_id
    ho_dir.mkdir(parents=True, exist_ok=True)

    ho_out = {
        "eval_config": {
            "eval_months": holdout_months,
            "class_type": class_type,
            "period_type": period_type,
            "mode": "holdout",
        },
        "blend_weights": {"da": w_da, "dmix": w_dmix, "dori": w_dori},
        "per_month": ho_pm,
        "aggregate": ho_agg,
        "n_months": len(ho_pm),
    }
    with open(ho_dir / "metrics.json", "w") as f:
        json.dump(ho_out, f, indent=2)
    print(f"[v1] Saved holdout to {ho_dir}")

    print(f"\n[v1] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
