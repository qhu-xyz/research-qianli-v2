#!/usr/bin/env python
"""V0 formula baseline: evaluate V6.2B formula against realized DA.

Evaluates over 12 default eval months, saves to registry/v0/,
and recalibrates gates.json + champion.json.

Expected results (from experiment-setup.md):
  VC@20  ~0.2817
  VC@100 ~0.6008
  Spearman ~0.2045
"""
from __future__ import annotations

import gc
import json
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.registry_paths import registry_root, holdout_root
from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS, _FULL_EVAL_MONTHS, REALIZED_DA_CACHE
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.v62b_formula import v62b_score


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def evaluate_month(month: str, class_type: str = "onpeak") -> dict:
    """Evaluate V6.2B formula on one month against realized DA."""
    path = Path(V62B_SIGNAL_BASE) / month / "f0" / class_type
    df = pl.read_parquet(str(path))

    realized = load_realized_da(month, peak_type=class_type)
    df = df.join(realized, on="constraint_id", how="left")
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
    print(
        f"  {month}: n={len(df)}, binding={n_binding}, "
        f"VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}, "
        f"Spearman={metrics['Spearman']:.4f}"
    )

    del df, realized, actual, scores
    gc.collect()
    return metrics


def build_gates(agg: dict) -> dict:
    """Build gates.json from v0 aggregate metrics.

    floor = 0.9 * mean, tail_floor = min, direction = "higher".
    """
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    group_b = [
        "VC@10", "VC@25", "VC@50", "VC@200",
        "Recall@10", "Spearman", "Tier0-AP", "Tier01-AP",
    ]

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

    for metric in group_b:
        if metric in means:
            gates[metric] = {
                "floor": round(0.9 * means[metric], 4),
                "tail_floor": round(mins[metric], 4),
                "direction": "higher",
                "group": "B",
            }

    return {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-type", default="onpeak", choices=["onpeak", "offpeak"])
    parser.add_argument("--full", action="store_true", help="Use all 36 months instead of 12")
    parser.add_argument("--holdout", action="store_true", help="Also run on 2024-2025 holdout")
    args = parser.parse_args()
    class_type = args.class_type

    HOLDOUT_MONTHS = [f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]

    print(f"[v0] Starting V6.2B formula baseline ({class_type}), mem: {mem_mb():.0f} MB")
    eval_months = _FULL_EVAL_MONTHS if args.full else _DEFAULT_EVAL_MONTHS
    print(f"[v0] Eval months ({len(eval_months)}): {eval_months}")

    per_month: dict[str, dict] = {}
    for month in eval_months:
        per_month[month] = evaluate_month(month, class_type=class_type)

    agg = aggregate_months(per_month)

    # ── Print summary ──
    print("\n[v0] Aggregate Results (12-month eval):")
    print(f"  {'Metric':<15} {'Mean':>8} {'Min':>8} {'Max':>8}")
    for metric in ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100",
                    "NDCG", "Spearman", "Tier0-AP", "Tier01-AP"]:
        m = agg["mean"].get(metric, float("nan"))
        mn = agg["min"].get(metric, float("nan"))
        mx = agg["max"].get(metric, float("nan"))
        print(f"  {metric:<15} {m:>8.4f} {mn:>8.4f} {mx:>8.4f}")

    # ── Check against expected numbers (onpeak only) ──
    if class_type == "onpeak":
        expected = {"VC@20": 0.2817, "VC@100": 0.6008, "Spearman": 0.2045}
        print("\n[v0] Validation against expected (tolerance=0.01):")
        all_ok = True
        for metric, exp_val in expected.items():
            got = agg["mean"].get(metric, float("nan"))
            diff = abs(got - exp_val)
            status = "OK" if diff <= 0.01 else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  {metric}: expected={exp_val:.4f}, got={got:.4f}, diff={diff:.4f} [{status}]")

        if not all_ok:
            print("\n[v0] ERROR: Numbers differ by more than 0.01. Investigate before proceeding.")
            sys.exit(1)

    # ── Save to registry/f0/{class_type}/v0/ ──
    base_registry = Path(__file__).resolve().parent.parent / "registry"
    version_id = "v0"
    v0_dir = registry_root("f0", class_type, base_dir=base_registry) / version_id
    v0_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json (v2 format: per_month + aggregate)
    metrics_out = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": class_type,
            "period_type": "f0",
            "mode": "eval",
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": [],
    }
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[v0] Wrote {v0_dir / 'metrics.json'}")

    # config.json
    config_out = {
        "method": "v62b_formula",
        "formula": "-(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)",
        "ground_truth": f"realized_da (abs sum of DA shadow prices, {class_type})",
        "features_used": ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"],
    }
    with open(v0_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # meta.json
    meta_out = {
        "version_id": version_id,
        "description": f"V6.2B formula baseline evaluated against realized DA ({class_type})",
        "n_months": len(per_month),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(v0_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # ── Gates (only recalibrate for onpeak) ──
    slice_dir = registry_root("f0", class_type, base_dir=base_registry)
    if class_type == "onpeak":
        gates_data = build_gates(agg)
        with open(slice_dir / "gates.json", "w") as f:
            json.dump(gates_data, f, indent=2)
        print(f"[v0] Wrote {slice_dir / 'gates.json'}")

        champion_data = {
            "version": "v0",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": "initial baseline (V6.2B formula vs realized DA)",
        }
        with open(slice_dir / "champion.json", "w") as f:
            json.dump(champion_data, f, indent=2)
        print(f"[v0] Wrote {slice_dir / 'champion.json'}")

    # ── Holdout ──
    if args.holdout:
        print(f"\n[v0] Running holdout ({len(HOLDOUT_MONTHS)} months)...")
        ho_per_month: dict[str, dict] = {}
        for month in HOLDOUT_MONTHS:
            try:
                ho_per_month[month] = evaluate_month(month, class_type=class_type)
            except FileNotFoundError:
                print(f"  {month}: SKIP (not found)")
        ho_agg = aggregate_months(ho_per_month)
        ho_means = ho_agg["mean"]
        print(f"\n[v0] Holdout aggregate ({len(ho_per_month)} months):")
        for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
            print(f"  {metric:<12} {ho_means.get(metric, 0):.4f}")

        base_holdout = Path(__file__).resolve().parent.parent / "holdout"
        holdout_dir = holdout_root("f0", class_type, base_dir=base_holdout) / version_id
        holdout_dir.mkdir(parents=True, exist_ok=True)
        ho_out = {
            "eval_config": {"eval_months": HOLDOUT_MONTHS, "class_type": class_type,
                            "period_type": "f0", "mode": "holdout"},
            "per_month": ho_per_month, "aggregate": ho_agg,
            "n_months": len(ho_per_month),
        }
        with open(holdout_dir / "metrics.json", "w") as f:
            json.dump(ho_out, f, indent=2)
        print(f"[v0] Wrote {holdout_dir / 'metrics.json'}")

    print(f"\n[v0] Done, mem: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
