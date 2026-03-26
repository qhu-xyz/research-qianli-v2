"""DEPRECATED — archived 2026-03-17. Uses old data source (pjm_annual_with_mcp.parquet).
New work should use canonical get_trades_of_given_duration().
See ../../../task_plan.md for the porting plan.

PJM Annual V1: Asymmetric Empirical Quantile Bands.

Same method as MISO V10: asymmetric signed quantile pairs, per-(bin, class) calibration.
5 quantile bins × 3 classes = 15 cells per round.
No sign stratification, no correction.
Temporal expanding CV, min_train_pys=2 for dev.
8 coverage levels: P10-P99.

PJM-specific:
- 4 rounds (R1-R4), single period_type "a"
- 3 production classes: onpeak, dailyoffpeak, wkndonpeak
- Annual scale: target = mcp, baseline = mtm_1st_mean * 12
- All rounds use mtm_1st_mean (no nodal stitch needed)
- PY 2017-2022 R1 is onpeak-only (dailyoff/wkndon fall back to pooled)
- hedge_type already filtered to obligation in the parquet

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/run_v1_bands.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Reuse core functions from MISO
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "miso" / "scripts"))

from run_v9_bands import (
    calibrate_asymmetric_per_class,
    apply_asymmetric_bands_per_class_fast,
    compute_quantile_boundaries,
    assign_bins,
    evaluate_coverage,
    evaluate_per_class_coverage,
    sanitize_for_json,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
)

# ─── PJM Constants ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path("/opt/temp/qianli/annual_research/pjm_annual_with_mcp.parquet")

PJM_CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]
MCP_COL = "mcp"  # annual total clearing price (native column)
BASELINE_COL = "baseline"  # mtm_1st_mean * 12 (created in loader)
PY_COL = "planning_year"
CLASS_COL = "class_type"

N_BINS = 5
MIN_CELL_ROWS = 200  # lowered from 500 — R1 PY2017-2022 has only onpeak
MIN_TRAIN_PYS = int(os.environ.get("BANDS_MIN_TRAIN_PYS", "2"))
VERSION_ID = os.environ.get("BANDS_VERSION_ID", "v1_dev")

_max_py = int(os.environ.get("BANDS_MAX_PY", "2024"))
DEV_PYS = list(range(2017, _max_py + 1))

# Override MISO CLASSES for PJM
import run_v9_bands
run_v9_bands.CLASSES = PJM_CLASSES
run_v9_bands.MIN_CELL_ROWS = MIN_CELL_ROWS


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ─── Data Loader ─────────────────────────────────────────────────────────────


def load_round(round_num: int) -> pl.DataFrame:
    """Load PJM annual data for one round, scale to annual."""
    df = (
        pl.scan_parquet(DATA_PATH)
        .filter(
            (pl.col("round") == round_num)
            & pl.col("class_type").is_in(PJM_CLASSES)
            & pl.col("mtm_1st_mean").is_not_null()
            & pl.col("mcp").is_not_null()
            & pl.col(PY_COL).is_in(DEV_PYS)
        )
        .select(["mtm_1st_mean", "mcp", PY_COL, CLASS_COL, "source_id", "sink_id"])
        .collect()
    )
    # Scale baseline to annual
    df = df.with_columns((pl.col("mtm_1st_mean") * 12).alias(BASELINE_COL))
    return df


# ─── Experiment Runner (adapted for PJM) ────────────────────────────────────


def run_experiment(
    df: pl.DataFrame,
    pys: list[int],
    n_bins: int = N_BINS,
    min_train_pys: int = MIN_TRAIN_PYS,
) -> dict:
    """Temporal expanding CV for one round."""
    # Validate class_type
    actual_classes = set(df[CLASS_COL].unique().to_list())
    if not actual_classes <= set(PJM_CLASSES):
        raise ValueError(f"Unexpected class_type values: {actual_classes - set(PJM_CLASSES)}")

    per_py = {}
    all_test_dfs = []
    filtered_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) < test_py)
        test = df.filter(pl.col(PY_COL) == test_py)
        if train.height == 0 or test.height == 0:
            continue

        train_pys = sorted(train[PY_COL].unique().to_list())
        meets_min = len(train_pys) >= min_train_pys

        boundaries, bin_labels = compute_quantile_boundaries(train[BASELINE_COL], n_bins)
        bin_pairs = calibrate_asymmetric_per_class(
            train, BASELINE_COL, MCP_COL, CLASS_COL,
            boundaries, bin_labels, COVERAGE_LEVELS,
        )
        test_banded = apply_asymmetric_bands_per_class_fast(
            test, bin_pairs, BASELINE_COL, CLASS_COL, boundaries, bin_labels,
        )

        all_test_dfs.append(test_banded)
        if meets_min:
            filtered_test_dfs.append(test_banded)

        # Overall coverage
        mcp = test_banded[MCP_COL].to_numpy()
        overall_cov = {}
        for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
            lo = test_banded[f"lower_{clabel}"].to_numpy()
            hi = test_banded[f"upper_{clabel}"].to_numpy()
            import numpy as np
            cov = float(np.mean((mcp >= lo) & (mcp <= hi)) * 100)
            overall_cov[clabel] = {"target": round(level * 100, 1), "actual": round(cov, 2),
                                   "error": round(cov - level * 100, 2)}

        # Per-class P95
        per_class_cov = {}
        for cls in PJM_CLASSES:
            mask = test_banded[CLASS_COL].to_numpy() == cls
            if mask.sum() > 0:
                m = mcp[mask]
                lo = test_banded["lower_p95"].to_numpy()[mask]
                hi = test_banded["upper_p95"].to_numpy()[mask]
                per_class_cov[cls] = round(float(np.mean((m >= lo) & (m <= hi)) * 100), 1)

        # Per-bin P95
        test_bins = assign_bins(test_banded[BASELINE_COL].abs(), boundaries, bin_labels).to_numpy()
        per_bin_cov = {}
        for bl in bin_labels:
            mask = test_bins == bl
            if mask.sum() > 0:
                m = mcp[mask]
                lo = test_banded["lower_p95"].to_numpy()[mask]
                hi = test_banded["upper_p95"].to_numpy()[mask]
                per_bin_cov[bl] = round(float(np.mean((m >= lo) & (m <= hi)) * 100), 1)

        # Width
        import numpy as np
        width_summary = {}
        for clabel in COVERAGE_LABELS:
            half_widths = []
            for label in bin_labels:
                for cls in PJM_CLASSES:
                    lo_hi = bin_pairs[label].get(cls, {}).get(clabel)
                    if isinstance(lo_hi, (list, tuple)):
                        half_widths.append((lo_hi[1] - lo_hi[0]) / 2)
            width_summary[clabel] = {
                "mean_width": round(sum(half_widths) / len(half_widths), 1) if half_widths else None,
            }

        p95_cov = overall_cov["p95"]["actual"]
        p95_w = width_summary["p95"]["mean_width"]
        flag = "" if meets_min else " [excluded]"
        print(f"    PY{test_py}: train={len(train_pys)}PY P95={p95_cov:.1f}% hw={p95_w:.0f}{flag}")

        per_py[str(test_py)] = {
            "train_pys": train_pys, "meets_min_train_pys": meets_min,
            "n_train": train.height, "n_test": test.height,
            "coverage": {"overall": overall_cov, "per_bin": per_bin_cov, "per_class": per_class_cov},
            "widths": width_summary,
            "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        }

    # Aggregate
    use_dfs = filtered_test_dfs if filtered_test_dfs else all_test_dfs
    agg_pys = [py for py in per_py if per_py[py].get("meets_min_train_pys", True)]

    if use_dfs:
        all_test = pl.concat(use_dfs)
        agg_cov = evaluate_coverage(
            all_test, MCP_COL, BASELINE_COL, COVERAGE_LEVELS,
            *compute_quantile_boundaries(df[BASELINE_COL], n_bins),
        )
        agg_per_class = evaluate_per_class_coverage(all_test, MCP_COL, CLASS_COL, COVERAGE_LEVELS)
        agg_cov["per_class"] = agg_per_class
    else:
        agg_cov = {"overall": {}, "per_bin": {}, "per_class": {}}

    # Aggregate widths
    agg_widths = {}
    for clabel in COVERAGE_LABELS:
        fold_means = [
            per_py[py]["widths"][clabel]["mean_width"]
            for py in agg_pys
            if per_py[py]["widths"][clabel]["mean_width"] is not None
        ]
        agg_widths[clabel] = {
            "mean_width": round(sum(fold_means) / len(fold_means), 1) if fold_means else None,
        }

    # Stability
    p95_coverages = [per_py[py]["coverage"]["overall"]["p95"]["actual"] for py in agg_pys]
    stability = {}
    if len(p95_coverages) >= 2:
        stability = {
            "p95_coverage_range": round(max(p95_coverages) - min(p95_coverages), 2),
            "p95_worst_py": agg_pys[p95_coverages.index(min(p95_coverages))],
            "p95_worst_py_coverage": round(min(p95_coverages), 2),
        }

    return {
        "per_py": per_py,
        "aggregate": {"coverage": agg_cov, "widths": agg_widths},
        "stability": stability,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    print(f"PJM Annual V1: Asymmetric Bands")
    print(f"Classes: {PJM_CLASSES}")
    print(f"min_train_pys={MIN_TRAIN_PYS}, n_bins={N_BINS}, MIN_CELL_ROWS={MIN_CELL_ROWS}")
    print(f"Dev PYs: {DEV_PYS}")
    print(f"mem={mem_mb():.0f}MB")

    for rnd in [1, 2, 3, 4]:
        print(f"\n{'#' * 80}")
        print(f"  ROUND {rnd}")
        print(f"{'#' * 80}")

        df = load_round(rnd)
        print(f"  Loaded: {df.height:,} rows, mem={mem_mb():.0f}MB")
        for cls in PJM_CLASSES:
            n = df.filter(pl.col(CLASS_COL) == cls).height
            print(f"    {cls}: {n:,}")

        result = run_experiment(df, DEV_PYS)

        # Print summary
        agg = result["aggregate"]
        print(f"\n  Coverage (all levels):")
        header = "  "
        for cl in COVERAGE_LABELS:
            header += f" {cl:>7}"
        print(header)
        line = "  "
        for cl in COVERAGE_LABELS:
            c = agg["coverage"]["overall"].get(cl, {}).get("actual", 0)
            line += f" {c:>6.1f}%"
        print(line)

        # Per-PY
        print(f"\n  Per-PY P95:")
        for py in sorted(result["per_py"].keys()):
            pd = result["per_py"][py]
            if pd["meets_min_train_pys"]:
                cls_str = " ".join(f"{pd['coverage']['per_class'].get(c, 0):.0f}" for c in PJM_CLASSES)
                bin_str = " ".join(f"{pd['coverage']['per_bin'].get(f'q{i+1}', 0):.0f}" for i in range(N_BINS))
                print(f"    PY{py}: P95={pd['coverage']['overall']['p95']['actual']:.1f}% "
                      f"hw={pd['widths']['p95']['mean_width']:.0f} "
                      f"bins=[{bin_str}] cls=[{cls_str}]")

        # Per-class
        print(f"\n  Per-class P95:")
        for cls in PJM_CLASSES:
            c = agg["coverage"].get("per_class", {}).get(cls, {}).get("p95", {}).get("actual", 0)
            print(f"    {cls}: {c:.1f}%")

        # Per-bin
        print(f"\n  Per-bin P95:")
        for bl in [f"q{i+1}" for i in range(N_BINS)]:
            c = agg["coverage"].get("per_bin", {}).get(bl, {}).get("p95", {}).get("actual", 0)
            print(f"    {bl}: {c:.1f}%")

        # Width
        print(f"\n  P95 half-width: {agg['widths'].get('p95', {}).get('mean_width', 0):.0f} annual")

        # Save
        v_dir = ROOT / "versions" / VERSION_ID / f"r{rnd}"
        v_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "coverage": {"a": agg["coverage"]},
            "widths": {"a": {"overall": agg["widths"]}},
            "stability": {"a": result["stability"]},
            "per_class_coverage": {"a": agg["coverage"].get("per_class", {})},
            "per_py": {"a": {py: {
                "train_pys": pdata["train_pys"],
                "meets_min_train_pys": pdata["meets_min_train_pys"],
                "n_train": pdata["n_train"], "n_test": pdata["n_test"],
                "p95_coverage": pdata["coverage"]["overall"]["p95"]["actual"],
                "p95_mean_width": pdata["widths"]["p95"]["mean_width"],
                "per_bin_p95": pdata["coverage"]["per_bin"],
                "per_class_p95": pdata["coverage"]["per_class"],
            } for py, pdata in result["per_py"].items()}},
            "temporal_validation": {"a": {
                "aggregate_coverage": agg["coverage"]["overall"],
                "aggregate_widths": agg["widths"],
                "stability": result["stability"],
            }},
        }

        with open(v_dir / "metrics.json", "w") as f:
            json.dump(sanitize_for_json(metrics), f, indent=2); f.write("\n")

        config = {
            "schema_version": 1, "version": VERSION_ID, "rto": "pjm",
            "description": f"PJM R{rnd} asymmetric bands ({N_BINS} bins, per-class, no correction)",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "part": "bands", "round": f"r{rnd}", "period_type": "a",
            "baseline": "mtm_1st_mean * 12",
            "method": {
                "band_type": "asymmetric",
                "calibration": "signed quantile pair of (mcp - baseline) per (bin, class)",
                "cv_method": "temporal_expanding",
                "n_bins": N_BINS, "min_cell_rows": MIN_CELL_ROWS,
                "min_train_pys": MIN_TRAIN_PYS,
                "classes": PJM_CLASSES,
            },
        }
        with open(v_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2); f.write("\n")

        print(f"  Saved: {v_dir}")

        del df; gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
