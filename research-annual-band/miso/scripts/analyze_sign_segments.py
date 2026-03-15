"""Segmented band analysis by sign(baseline) — no data leakage.

Prevail = baseline > 0, Counter = baseline < 0.
This is observable at prediction time (no MCP needed).

For each segment: residual distribution + band coverage + band widths.
"""
from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_bands import COVERAGE_LEVELS, COVERAGE_LABELS
from run_phase3_v2_bands import compute_quantile_boundaries
from run_v3_bands import (
    calibrate_bin_widths_per_class,
    apply_bands_per_class_fast,
    MCP_COL, PY_COL, CLASS_COL,
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


def segment_by_sign(baseline: pl.Series) -> pl.Series:
    """prevail (>0), counter (<0), zero (==0)."""
    return (
        pl.when(baseline > 0).then(pl.lit("prevail"))
        .when(baseline < 0).then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
    )


def measure_segment(df: pl.DataFrame, seg_col: str, mcp_col: str = MCP_COL) -> dict:
    """Per-segment: coverage, widths, residual stats."""
    results = {}
    for seg_val in sorted(df[seg_col].unique().to_list()):
        if seg_val is None:
            continue
        s = df.filter(pl.col(seg_col) == seg_val)
        n = s.height
        if n == 0:
            continue

        mcp = s[mcp_col]
        res = mcp - s["_baseline"]
        abs_res = res.abs()

        entry = {
            "n": n,
            "pct": round(n / df.height * 100, 1),
            "bias": round(float(res.mean()), 1),
            "mean_ae": round(float(abs_res.mean()), 1),
            "median_ae": round(float(abs_res.median()), 1),
            "p95_ae": round(float(abs_res.quantile(0.95)), 1),
        }

        for level, clabel in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
            lower = s[f"lower_{clabel}"]
            upper = s[f"upper_{clabel}"]
            covered = float(((mcp >= lower) & (mcp <= upper)).mean()) * 100
            target = level * 100
            entry[clabel] = {
                "coverage": round(covered, 2),
                "error": round(covered - target, 2),
            }
            width = (upper - lower)
            entry[f"{clabel}_width"] = round(float(width.mean()), 1)

        results[seg_val] = entry
    return results


def run_loo(
    df: pl.DataFrame,
    pys: list[int],
    n_bins: int,
    baseline_col: str,
) -> dict:
    """LOO CV, then segment by sign(baseline)."""
    all_test = []

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
        all_test.append(test_banded)

    if not all_test:
        return {}

    combined = pl.concat(all_test)
    combined = combined.with_columns([
        pl.col(baseline_col).alias("_baseline"),
        segment_by_sign(pl.col(baseline_col)).alias("sign_seg"),
        (pl.col(baseline_col).abs()).alias("_abs_bl"),
    ])

    result = {
        "n_total": combined.height,
        "by_sign": measure_segment(combined, "sign_seg"),
    }

    # Also cross-cut: sign × class
    combined = combined.with_columns(
        (pl.col("sign_seg") + "_" + pl.col(CLASS_COL)).alias("sign_class"),
    )
    result["by_sign_class"] = measure_segment(combined, "sign_class")

    # Residual asymmetry: upper vs lower residuals within each sign segment
    for seg_val in ["prevail", "counter"]:
        s = combined.filter(pl.col("sign_seg") == seg_val)
        if s.height == 0:
            continue
        res = s[MCP_COL] - s["_baseline"]
        upper_res = res.filter(res > 0)
        lower_res = res.filter(res < 0).abs()
        result[f"{seg_val}_residual_shape"] = {
            "n": s.height,
            "upper_mean": round(float(upper_res.mean()), 1) if upper_res.len() > 0 else None,
            "upper_p95": round(float(upper_res.quantile(0.95)), 1) if upper_res.len() > 0 else None,
            "upper_count_pct": round(upper_res.len() / s.height * 100, 1),
            "lower_mean": round(float(lower_res.mean()), 1) if lower_res.len() > 0 else None,
            "lower_p95": round(float(lower_res.quantile(0.95)), 1) if lower_res.len() > 0 else None,
            "lower_count_pct": round(lower_res.len() / s.height * 100, 1),
        }

    return result


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


def fmt(seg: dict, label: str) -> str:
    """One-line summary."""
    p50 = seg.get("p50", {})
    p95 = seg.get("p95", {})
    return (
        f"  {label:22s}: n={seg['n']:>10,} ({seg['pct']:5.1f}%)  "
        f"bias={seg['bias']:+7.1f}  MAE={seg['mean_ae']:7.1f}  MedAE={seg['median_ae']:7.1f}  P95AE={seg['p95_ae']:7.1f}  "
        f"P50w={seg.get('p50_width', 0):7.1f}  P95w={seg.get('p95_width', 0):7.1f}  "
        f"P50err={p50.get('error', 0):+.2f}pp  P95err={p95.get('error', 0):+.2f}pp"
    )


def main():
    print("Sign-based segmented analysis (prevail=bl>0, counter=bl<0)")
    print(f"Memory at start: {mem_mb():.0f} MB\n")

    all_results = {}

    for rnd, rnd_label, baseline_col, n_bins, pys in [
        (1, "R1 (nodal_f0, 4 bins)", "nodal_f0", 4, R1_PYS),
        (2, "R2 (mtm_1st_mean, 6 bins)", "mtm_1st_mean", 6, R2R3_PYS),
        (3, "R3 (mtm_1st_mean, 6 bins)", "mtm_1st_mean", 6, R2R3_PYS),
    ]:
        print(f"{'=' * 70}")
        print(f"  {rnd_label}")
        print(f"{'=' * 70}")

        for q in QUARTERS:
            if rnd == 1:
                df = load_r1(q)
            else:
                df = load_r2r3(rnd, q)

            print(f"\n  {q}: {df.height:,} rows, mem={mem_mb():.0f} MB")
            results = run_loo(df, pys, n_bins, baseline_col)
            all_results[f"r{rnd}_{q}"] = results

            # Print by sign
            for seg in ["prevail", "counter", "zero"]:
                if seg in results["by_sign"]:
                    print(fmt(results["by_sign"][seg], seg))

            # Print residual shape
            for seg in ["prevail", "counter"]:
                shape = results.get(f"{seg}_residual_shape")
                if shape:
                    print(
                        f"    {seg} residual shape: "
                        f"upper {shape['upper_count_pct']:.0f}% (mean={shape['upper_mean']:.0f}, p95={shape['upper_p95']:.0f}) | "
                        f"lower {shape['lower_count_pct']:.0f}% (mean={shape['lower_mean']:.0f}, p95={shape['lower_p95']:.0f})"
                    )

            del df
            gc.collect()

        print()

    # Save
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

    out_path = ROOT / "sign_segment_analysis.json"
    with open(out_path, "w") as f:
        json.dump(sanitize(all_results), f, indent=2)
        f.write("\n")
    print(f"Results saved to {out_path}")
    print(f"Done. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
