# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Segmented band coverage analysis.

Breaks down band coverage by:
  1. Direction: prevail (sign(baseline)==sign(mcp)) vs counter
  2. Class: onpeak vs offpeak
  3. Magnitude: |baseline| quartiles
  4. Cross-cuts: direction × class, direction × magnitude

Uses LOO CV to measure coverage per segment.
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_bands import (
    assign_bins,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
)
from run_phase3_v2_bands import compute_quantile_boundaries
from run_v3_bands import (
    calibrate_bin_widths_per_class,
    apply_bands_per_class_fast,
    MCP_COL, PY_COL, CLASS_COL, CLASSES,
    MIN_CLASS_BIN_ROWS,
    R1_PYS, R2R3_PYS,
)

import resource
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

ROOT = Path(__file__).resolve().parent.parent
R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]


def derive_direction(baseline: pl.Series, mcp: pl.Series) -> pl.Series:
    """Classify each path as prevail, counter, or zero."""
    bl_sign = baseline.sign()
    mcp_sign = mcp.sign()
    return (
        pl.when((bl_sign == mcp_sign) & (bl_sign != 0))
        .then(pl.lit("prevail"))
        .when((bl_sign != mcp_sign) & (bl_sign != 0) & (mcp_sign != 0))
        .then(pl.lit("counter"))
        .when(bl_sign == 0)
        .then(pl.lit("zero_baseline"))
        .otherwise(pl.lit("zero_mcp"))
    )


def derive_magnitude_bin(baseline: pl.Series, n_bins: int = 4) -> pl.Series:
    """Bin |baseline| into quantile-based magnitude groups."""
    abs_bl = baseline.abs()
    cuts = [0.0]
    for i in range(1, n_bins):
        q = i / n_bins
        cuts.append(round(float(abs_bl.quantile(q)), 1))
    cuts.append(float("inf"))
    labels = [f"mag_q{i+1}" for i in range(n_bins)]

    # Use assign_bins
    tmp = pl.DataFrame({"_abs_bl": abs_bl})
    exprs = []
    for i, label in enumerate(labels):
        lo, hi = cuts[i], cuts[i + 1]
        if math.isinf(hi):
            exprs.append(pl.when(pl.col("_abs_bl") >= lo).then(pl.lit(label)))
        else:
            exprs.append(pl.when((pl.col("_abs_bl") >= lo) & (pl.col("_abs_bl") < hi)).then(pl.lit(label)))
    result = tmp.with_columns(pl.coalesce(exprs).alias("mag_bin"))["mag_bin"]
    return result


def compute_segment_coverage(
    df: pl.DataFrame,
    segment_col: str,
    mcp_col: str = MCP_COL,
) -> dict:
    """Compute P95 coverage per segment value."""
    results = {}
    for seg_val in sorted(df[segment_col].unique().to_list()):
        if seg_val is None:
            continue
        subset = df.filter(pl.col(segment_col) == seg_val)
        n = subset.height
        if n == 0:
            continue

        mcp_s = subset[mcp_col]
        seg_result = {"n": n, "pct": round(n / df.height * 100, 1)}

        # Mean |residual|
        if "abs_residual" in subset.columns:
            seg_result["mean_abs_residual"] = round(float(subset["abs_residual"].mean()), 1)
            seg_result["median_abs_residual"] = round(float(subset["abs_residual"].median()), 1)
            seg_result["p95_abs_residual"] = round(float(subset["abs_residual"].quantile(0.95)), 1)

        for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
            lower = subset[f"lower_{clabel}"]
            upper = subset[f"upper_{clabel}"]
            covered = ((mcp_s >= lower) & (mcp_s <= upper)).mean()
            actual = round(float(covered) * 100, 2)
            target = round(level * 100, 1)
            seg_result[clabel] = {
                "target": target,
                "actual": actual,
                "error": round(actual - target, 2),
            }

            # Mean width for this level
            widths = (upper - lower)
            seg_result[f"{clabel}_mean_width"] = round(float(widths.mean()), 1)

        results[seg_val] = seg_result
    return results


def run_loo_segmented(
    df: pl.DataFrame,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
) -> dict:
    """LOO CV with segmented coverage analysis."""
    all_test_dfs = []

    for test_py in pys:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)
        if train.height == 0 or test.height == 0:
            continue

        boundaries, labels = compute_quantile_boundaries(train[baseline_col], n_bins)
        bin_widths = calibrate_bin_widths_per_class(
            train, baseline_col, MCP_COL, CLASS_COL,
            boundaries, labels, COVERAGE_LEVELS,
        )
        test_banded = apply_bands_per_class_fast(
            test, bin_widths, baseline_col, CLASS_COL, boundaries, labels,
        )
        all_test_dfs.append(test_banded)

    if not all_test_dfs:
        return {}

    all_test = pl.concat(all_test_dfs)

    # Add derived columns
    baseline_s = all_test[baseline_col]
    mcp_s = all_test[MCP_COL]
    abs_res = (mcp_s - baseline_s).abs()

    all_test = all_test.with_columns([
        derive_direction(baseline_s, mcp_s).alias("direction"),
        derive_magnitude_bin(baseline_s).alias("mag_bin"),
        pl.Series("abs_residual", abs_res),
    ])

    # Also create cross-cut columns
    all_test = all_test.with_columns([
        (pl.col("direction") + "_" + pl.col(CLASS_COL)).alias("dir_class"),
        (pl.col("direction") + "_" + pl.col("mag_bin")).alias("dir_mag"),
    ])

    results = {
        "n_total": all_test.height,
        "by_direction": compute_segment_coverage(all_test, "direction"),
        "by_class": compute_segment_coverage(all_test, CLASS_COL),
        "by_magnitude": compute_segment_coverage(all_test, "mag_bin"),
        "by_dir_class": compute_segment_coverage(all_test, "dir_class"),
        "by_dir_mag": compute_segment_coverage(all_test, "dir_mag"),
    }

    # Overall stats
    overall_cov = {}
    for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
        lower = all_test[f"lower_{clabel}"]
        upper = all_test[f"upper_{clabel}"]
        covered = ((mcp_s >= lower) & (mcp_s <= upper)).mean()
        actual = round(float(covered) * 100, 2)
        target = round(level * 100, 1)
        overall_cov[clabel] = {"target": target, "actual": actual, "error": round(actual - target, 2)}
    results["overall"] = overall_cov

    # Magnitude bin boundaries for reporting
    abs_bl = baseline_s.abs()
    results["mag_boundaries"] = {
        "p25": round(float(abs_bl.quantile(0.25)), 0),
        "p50": round(float(abs_bl.quantile(0.50)), 0),
        "p75": round(float(abs_bl.quantile(0.75)), 0),
        "p95": round(float(abs_bl.quantile(0.95)), 0),
    }

    # Direction distribution
    dir_counts = all_test.group_by("direction").len().sort("direction")
    results["direction_dist"] = {
        row["direction"]: row["len"]
        for row in dir_counts.iter_rows(named=True)
    }

    return results


def load_r1(quarter: str) -> pl.DataFrame:
    path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col(PY_COL) >= 2019)
            & pl.col("nodal_f0").is_not_null()
            & pl.col(MCP_COL).is_not_null()
            & pl.col(CLASS_COL).is_not_null()
        )
        .collect()
    )


def load_r2r3(round_num: int, quarter: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(R2R3_DATA_PATH)
        .filter(
            (pl.col("round") == round_num)
            & (pl.col("period_type") == quarter)
            & (pl.col(PY_COL) >= 2019)
            & pl.col("mtm_1st_mean").is_not_null()
            & pl.col(MCP_COL).is_not_null()
            & pl.col(CLASS_COL).is_not_null()
        )
        .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL])
        .collect()
    )


def main():
    print(f"Segmented band coverage analysis")
    print(f"Memory at start: {mem_mb():.0f} MB\n")

    all_results = {}

    # R1
    print("=" * 60)
    print("  R1 (baseline=nodal_f0, 4 bins)")
    print("=" * 60)
    for q in QUARTERS:
        df = load_r1(q)
        print(f"\n  {q}: {df.height:,} rows, mem={mem_mb():.0f} MB")
        results = run_loo_segmented(df, R1_PYS, n_bins=4, baseline_col="nodal_f0")
        all_results[f"r1_{q}"] = results

        # Print direction summary
        dd = results.get("direction_dist", {})
        total = sum(dd.values())
        for d in ["prevail", "counter", "zero_baseline", "zero_mcp"]:
            n = dd.get(d, 0)
            pct = n / total * 100 if total else 0
            p95 = results["by_direction"].get(d, {}).get("p95", {})
            cov = p95.get("actual", 0)
            err = p95.get("error", 0)
            print(f"    {d:15s}: {n:>8,} ({pct:5.1f}%)  P95 cov={cov:.2f}% err={err:+.2f}pp")

        del df
        gc.collect()

    # R2
    print(f"\n{'=' * 60}")
    print("  R2 (baseline=mtm_1st_mean, 6 bins)")
    print("=" * 60)
    for q in QUARTERS:
        df = load_r2r3(2, q)
        print(f"\n  {q}: {df.height:,} rows, mem={mem_mb():.0f} MB")
        results = run_loo_segmented(df, R2R3_PYS, n_bins=6, baseline_col="mtm_1st_mean")
        all_results[f"r2_{q}"] = results

        dd = results.get("direction_dist", {})
        total = sum(dd.values())
        for d in ["prevail", "counter", "zero_baseline", "zero_mcp"]:
            n = dd.get(d, 0)
            pct = n / total * 100 if total else 0
            p95 = results["by_direction"].get(d, {}).get("p95", {})
            cov = p95.get("actual", 0)
            err = p95.get("error", 0)
            print(f"    {d:15s}: {n:>8,} ({pct:5.1f}%)  P95 cov={cov:.2f}% err={err:+.2f}pp")

        del df
        gc.collect()

    # R3
    print(f"\n{'=' * 60}")
    print("  R3 (baseline=mtm_1st_mean, 6 bins)")
    print("=" * 60)
    for q in QUARTERS:
        df = load_r2r3(3, q)
        print(f"\n  {q}: {df.height:,} rows, mem={mem_mb():.0f} MB")
        results = run_loo_segmented(df, R2R3_PYS, n_bins=6, baseline_col="mtm_1st_mean")
        all_results[f"r3_{q}"] = results

        dd = results.get("direction_dist", {})
        total = sum(dd.values())
        for d in ["prevail", "counter", "zero_baseline", "zero_mcp"]:
            n = dd.get(d, 0)
            pct = n / total * 100 if total else 0
            p95 = results["by_direction"].get(d, {}).get("p95", {})
            cov = p95.get("actual", 0)
            err = p95.get("error", 0)
            print(f"    {d:15s}: {n:>8,} ({pct:5.1f}%)  P95 cov={cov:.2f}% err={err:+.2f}pp")

        del df
        gc.collect()

    # Save raw results
    out_path = ROOT / "segment_analysis.json"
    # Sanitize for JSON
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(sanitize(all_results), f, indent=2)
        f.write("\n")
    print(f"\nResults saved to {out_path}")
    print(f"Done. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
