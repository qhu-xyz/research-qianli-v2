#!/usr/bin/env python
"""V0 formula baseline for PJM: evaluate V6.2B formula against realized DA.

Runs all 6 ML slices: f0×{onpeak,dailyoffpeak,wkndonpeak} + f1×{same}.
Saves to registry/{ptype}/{ctype}/v0/ and calibrates gates.json + champion.json.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v0_formula_baseline.py
    python scripts/run_v0_formula_baseline.py --ptype f0 --class-type onpeak  # single slice
    python scripts/run_v0_formula_baseline.py --holdout  # include holdout
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    V62B_SIGNAL_BASE, PJM_CLASS_TYPES, _FULL_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
    HOLDOUT_MONTHS, has_period_type,
)
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.registry_paths import registry_root, holdout_root
from ml.v62b_formula import v62b_score

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def evaluate_month(month: str, class_type: str, period_type: str) -> dict:
    """Evaluate V6.2B formula on one month against realized DA."""
    from ml.config import delivery_month as _delivery_month

    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
    df = pl.read_parquet(str(path))
    df = df.with_columns(pl.col("branch_name").cast(pl.String))

    gt_month = _delivery_month(month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
    df = df.join(realized, on="branch_name", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    actual = df["realized_sp"].to_numpy().astype(np.float64)

    # Negate because lower rank_value = more binding, but eval expects higher = better
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )

    metrics = evaluate_ltr(actual, scores)
    n_binding = int((actual > 0).sum())
    print(f"  {month}: n={len(df)}, binding={n_binding}, "
          f"VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}")

    del df, realized, actual, scores
    gc.collect()
    return metrics


def build_gates(agg: dict) -> dict:
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    gates = {}
    means = agg["mean"]
    mins = agg["min"]
    for metric in group_a:
        if metric in means:
            gates[metric] = {
                "floor": round(0.9 * means[metric], 4),
                "tail_floor": round(mins[metric], 4),
                "direction": "higher",
                "group": "A",
            }
    return {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_slice(
    period_type: str,
    class_type: str,
    eval_months: list[str],
    holdout: bool = False,
):
    """Run v0 for one (ptype, ctype) slice."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[v0] {period_type}/{class_type} ({len(eval_months)} eval months)")
    print(f"{'='*60}")

    per_month: dict[str, dict] = {}
    for month in eval_months:
        try:
            per_month[month] = evaluate_month(month, class_type, period_type)
        except FileNotFoundError as e:
            print(f"  {month}: SKIP ({e})")

    if not per_month:
        print(f"[v0] No valid months for {period_type}/{class_type}")
        return

    agg = aggregate_months(per_month)
    means = agg["mean"]
    print(f"\n[v0] Aggregate ({len(per_month)} months):")
    for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
        print(f"  {metric:<12} {means.get(metric, 0):.4f}")

    # Save to registry
    v0_dir = registry_root(period_type, class_type, base_dir=REGISTRY) / "v0"
    v0_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "eval_config": {"eval_months": sorted(per_month.keys()), "class_type": class_type,
                        "period_type": period_type, "mode": "eval"},
        "per_month": per_month, "aggregate": agg,
        "n_months": len(per_month), "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(per_month.keys())),
    }
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    config_out = {
        "method": "v62b_formula",
        "formula": "-(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)",
        "ground_truth": f"realized_da (branch-level sum, {class_type})",
    }
    with open(v0_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Gates + champion
    slice_dir = registry_root(period_type, class_type, base_dir=REGISTRY)
    gates_data = build_gates(agg)
    with open(slice_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)

    champion_data = {
        "version": "v0",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reason": f"initial baseline ({period_type}/{class_type})",
    }
    with open(slice_dir / "champion.json", "w") as f:
        json.dump(champion_data, f, indent=2)

    print(f"[v0] Saved to {v0_dir} ({time.time()-t0:.1f}s)")

    # Holdout
    if holdout:
        ho_months = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]
        print(f"\n[v0] Holdout ({len(ho_months)} months)...")
        ho_pm: dict[str, dict] = {}
        for month in ho_months:
            try:
                ho_pm[month] = evaluate_month(month, class_type, period_type)
            except FileNotFoundError:
                print(f"  {month}: SKIP")

        if ho_pm:
            ho_agg = aggregate_months(ho_pm)
            ho_dir = holdout_root(period_type, class_type, base_dir=HOLDOUT_DIR) / "v0"
            ho_dir.mkdir(parents=True, exist_ok=True)
            ho_out = {
                "eval_config": {"eval_months": sorted(ho_pm.keys()), "class_type": class_type,
                                "period_type": period_type, "mode": "holdout"},
                "per_month": ho_pm, "aggregate": ho_agg,
                "n_months": len(ho_pm), "n_months_requested": len(ho_months),
                "skipped_months": sorted(set(ho_months) - set(ho_pm.keys())),
            }
            with open(ho_dir / "metrics.json", "w") as f:
                json.dump(ho_out, f, indent=2)
            ho_means = ho_agg["mean"]
            print(f"[v0] Holdout aggregate ({len(ho_pm)} months):")
            for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                print(f"  {metric:<12} {ho_means.get(metric, 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None, help="Period type (f0 or f1). Default: both.")
    parser.add_argument("--class-type", default=None, help="Class type. Default: all 3.")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use 36 eval months")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES

    for ptype in ptypes:
        eval_months = _FULL_EVAL_MONTHS if args.full else _DEFAULT_EVAL_MONTHS
        eval_months = [m for m in eval_months if has_period_type(m, ptype)]
        for ctype in ctypes:
            run_slice(ptype, ctype, eval_months, holdout=args.holdout)

    print("\n[v0] All slices complete.")


if __name__ == "__main__":
    main()
