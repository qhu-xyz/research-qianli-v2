#!/usr/bin/env python
"""V0/V0b formula baseline for PJM: evaluate formula against realized DA.

v0:  V6.2B default weights (0.60/0.30/0.10)
v0b: Optimized weights (0.80/0.15/0.05) — stronger baseline

Runs all 6 ML slices: f0×{onpeak,dailyoffpeak,wkndonpeak} + f1×{same}.
Saves to registry/{ptype}/{ctype}/{version}/ with gates and NewBind metrics.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v0_formula_baseline.py                          # v0, all slices
    python scripts/run_v0_formula_baseline.py --version v0b --holdout  # v0b with holdout
    python scripts/run_v0_formula_baseline.py --version v0b --holdout --calibrate-gates
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
    REALIZED_DA_CACHE, V62B_SIGNAL_BASE, PJM_CLASS_TYPES,
    _FULL_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
    HOLDOUT_MONTHS, has_period_type,
)
from ml.data_loader import compute_new_mask
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.registry_paths import registry_root, holdout_root

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"

# Blend weights per version
BLEND_CONFIGS = {
    "v0":  (0.60, 0.30, 0.10),  # V6.2B default
    "v0b": (0.80, 0.15, 0.05),  # Optimized for PJM
}

# Group A metrics for gate calibration (12 gates)
GROUP_A_METRICS = [
    "VC@20", "VC@50", "VC@100",
    "Recall@20", "Recall@50", "Recall@100",
    "NDCG", "Spearman",
    "NewBind_Recall@50", "NewBind_Recall@100",
    "NewBind_VC@50", "NewBind_VC@100",
]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["branch_name"].to_list()
        )
    print(f"[bf] Loaded {len(binding_sets)} months of {peak_type} binding sets")
    return binding_sets


def evaluate_month(
    month: str, class_type: str, period_type: str,
    blend: tuple[float, float, float],
    bs: dict[str, set[str]],
) -> dict:
    """Evaluate formula on one month against realized DA with NewBind metrics."""
    from ml.config import delivery_month as _delivery_month

    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
    df = pl.read_parquet(str(path))
    df = df.with_columns(pl.col("branch_name").cast(pl.String))

    gt_month = _delivery_month(month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
    df = df.join(realized, on="branch_name", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    # Deduplicate: keep one row per branch_name (lowest rank_ori = highest priority)
    n_pre = len(df)
    df = df.sort("rank_ori").unique(subset=["branch_name"], keep="first")

    actual = df["realized_sp"].to_numpy().astype(np.float64)
    branch_names = df["branch_name"].to_list()

    # Compute new_mask (BF-zero: not bound in prior 6 months)
    new_mask = compute_new_mask(branch_names, month, bs, lookback=6)

    # Negate because lower rank_value = more binding, but eval expects higher = better
    w_da, w_dmix, w_dori = blend
    scores = -(
        w_da * df["da_rank_value"].to_numpy().astype(np.float64)
        + w_dmix * df["density_mix_rank_value"].to_numpy().astype(np.float64)
        + w_dori * df["density_ori_rank_value"].to_numpy().astype(np.float64)
    )

    metrics = evaluate_ltr(actual, scores, new_mask=new_mask)
    n_binding = int((actual > 0).sum())
    n_new = metrics.get("n_new", 0)
    print(f"  {month}: n={n_pre}→{len(df)}, binding={n_binding}, new={n_new}, "
          f"VC@20={metrics['VC@20']:.4f}, NB_R@50={metrics.get('NewBind_Recall@50', 0):.3f}")

    del df, realized, actual, scores
    gc.collect()
    return metrics


def build_gates(agg: dict, version: str) -> dict:
    gates = {}
    means = agg["mean"]
    stds = agg.get("std", {})
    b2m = agg.get("bottom_2_mean", {})
    for metric in GROUP_A_METRICS:
        if metric in means:
            mean_val = means[metric]
            std_val = stds.get(metric, 0)
            # Floor = mean - 1 std (with 2% noise tolerance applied at check time)
            floor = round(max(0, mean_val - std_val), 4)
            # Tail floor = bottom_2_mean (p5 proxy)
            tail_floor = round(b2m.get(metric, 0), 4)
            gates[metric] = {
                "floor": floor,
                "tail_floor": tail_floor,
                "direction": "higher",
                "group": "A",
            }
    return {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": version,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_slice(
    period_type: str,
    class_type: str,
    eval_months: list[str],
    version: str,
    blend: tuple[float, float, float],
    bs: dict[str, set[str]],
    holdout: bool = False,
    calibrate_gates: bool = False,
):
    """Run formula baseline for one (ptype, ctype) slice."""
    t0 = time.time()
    w_da, w_dmix, w_dori = blend
    print(f"\n{'='*60}")
    print(f"[{version}] {period_type}/{class_type} ({len(eval_months)} months) "
          f"blend=({w_da:.2f}/{w_dmix:.2f}/{w_dori:.2f})")
    print(f"{'='*60}")

    per_month: dict[str, dict] = {}
    for month in eval_months:
        try:
            per_month[month] = evaluate_month(month, class_type, period_type, blend, bs)
        except FileNotFoundError as e:
            print(f"  {month}: SKIP ({e})")

    if not per_month:
        print(f"[{version}] No valid months for {period_type}/{class_type}")
        return

    agg = aggregate_months(per_month)
    means = agg["mean"]
    print(f"\n[{version}] Aggregate ({len(per_month)} months):")
    for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50",
                    "NDCG", "Spearman", "NewBind_Recall@50", "NewBind_VC@50"]:
        print(f"  {metric:<20} {means.get(metric, 0):.4f}")

    # Save to registry
    ver_dir = registry_root(period_type, class_type, base_dir=REGISTRY) / version
    ver_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "eval_config": {"eval_months": sorted(per_month.keys()), "class_type": class_type,
                        "period_type": period_type, "mode": "eval", "version": version},
        "per_month": per_month, "aggregate": agg,
        "n_months": len(per_month), "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(per_month.keys())),
    }
    with open(ver_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    formula_str = f"-({w_da:.2f}*da + {w_dmix:.2f}*dmix + {w_dori:.2f}*dori)"
    config_out = {
        "method": "formula",
        "version": version,
        "formula": formula_str,
        "blend_weights": {"da": w_da, "dmix": w_dmix, "dori": w_dori},
        "ground_truth": f"realized_da (branch-level sum, {class_type})",
    }
    with open(ver_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Gates + champion (only if --calibrate-gates)
    if calibrate_gates:
        slice_dir = registry_root(period_type, class_type, base_dir=REGISTRY)
        gates_data = build_gates(agg, version)
        with open(slice_dir / "gates.json", "w") as f:
            json.dump(gates_data, f, indent=2)
        print(f"[{version}] Gates calibrated ({len(gates_data['gates'])} metrics)")

        champion_data = {
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": f"formula baseline ({period_type}/{class_type}, blend={w_da:.2f}/{w_dmix:.2f}/{w_dori:.2f})",
        }
        with open(slice_dir / "champion.json", "w") as f:
            json.dump(champion_data, f, indent=2)
        print(f"[{version}] Promoted as champion")

    print(f"[{version}] Saved to {ver_dir} ({time.time()-t0:.1f}s)")

    # Holdout
    if holdout:
        ho_months = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]
        print(f"\n[{version}] Holdout ({len(ho_months)} months)...")
        ho_pm: dict[str, dict] = {}
        for month in ho_months:
            try:
                ho_pm[month] = evaluate_month(month, class_type, period_type, blend, bs)
            except FileNotFoundError:
                print(f"  {month}: SKIP")

        if ho_pm:
            ho_agg = aggregate_months(ho_pm)
            ho_dir = holdout_root(period_type, class_type, base_dir=HOLDOUT_DIR) / version
            ho_dir.mkdir(parents=True, exist_ok=True)
            ho_out = {
                "eval_config": {"eval_months": sorted(ho_pm.keys()), "class_type": class_type,
                                "period_type": period_type, "mode": "holdout", "version": version},
                "per_month": ho_pm, "aggregate": ho_agg,
                "n_months": len(ho_pm), "n_months_requested": len(ho_months),
                "skipped_months": sorted(set(ho_months) - set(ho_pm.keys())),
            }
            with open(ho_dir / "metrics.json", "w") as f:
                json.dump(ho_out, f, indent=2)
            ho_means = ho_agg["mean"]
            print(f"[{version}] Holdout aggregate ({len(ho_pm)} months):")
            for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG",
                            "Spearman", "NewBind_Recall@50", "NewBind_VC@50"]:
                print(f"  {metric:<20} {ho_means.get(metric, 0):.4f}")

            # Temporal segmentation: 2024 vs 2025
            for year in ["2024", "2025"]:
                year_pm = {m: v for m, v in ho_pm.items() if m.startswith(year)}
                if year_pm:
                    year_agg = aggregate_months(year_pm)
                    year_means = year_agg["mean"]
                    print(f"[{version}] Holdout {year} ({len(year_pm)} months):")
                    for metric in ["VC@20", "VC@50", "Recall@20", "NewBind_Recall@50"]:
                        print(f"  {metric:<20} {year_means.get(metric, 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None, help="Period type (f0 or f1). Default: both.")
    parser.add_argument("--class-type", default=None, help="Class type. Default: all 3.")
    parser.add_argument("--version", default="v0", choices=list(BLEND_CONFIGS.keys()))
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--calibrate-gates", action="store_true",
                        help="Calibrate gates.json and promote as champion")
    parser.add_argument("--full", action="store_true", help="Use 36 eval months")
    args = parser.parse_args()

    version = args.version
    blend = BLEND_CONFIGS[version]
    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES

    for ptype in ptypes:
        eval_months = _FULL_EVAL_MONTHS if args.full else _DEFAULT_EVAL_MONTHS
        eval_months = [m for m in eval_months if has_period_type(m, ptype)]
        for ctype in ctypes:
            bs = load_all_binding_sets(peak_type=ctype)
            run_slice(ptype, ctype, eval_months, version=version, blend=blend,
                      bs=bs, holdout=args.holdout, calibrate_gates=args.calibrate_gates)

    print(f"\n[{version}] All slices complete.")


if __name__ == "__main__":
    main()
