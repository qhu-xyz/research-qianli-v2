# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Investigate sign(baseline) as band stratification axis.

Does splitting calibration by prevail (baseline > 0) vs counter (baseline < 0)
within each |baseline| bin produce meaningfully different residual distributions
and improve out-of-sample coverage?

Steps:
    1. Detailed per-cell residual profiles (R1 aq1 onpeak)
    2. LOO coverage comparison (R1 aq1 onpeak) — pooled vs split
    3. Extend to all quarters (R1 aq1-aq4, onpeak)
    4. Extend to R2/R3 (aq1 onpeak)
    5. Summary report with decision

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/investigate_sign_split.py
"""
from __future__ import annotations

import gc
import resource
import statistics
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_bands import (
    assign_bins,
    calibrate_bin_widths,
    apply_bands,
    mem_mb,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
)
from run_phase3_v2_bands import compute_quantile_boundaries

# ─── Constants ───────────────────────────────────────────────────────────────

R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")

QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

MCP_COL = "mcp_mean"
PY_COL = "planning_year"
CLASS_COL = "class_type"
BASELINE_COL_R1 = "nodal_f0"
BASELINE_COL_R2R3 = "mtm_1st_mean"

TARGET = 0.95  # P95
SEGS = ["prevail", "counter"]


# ─── Data loaders ────────────────────────────────────────────────────────────


def load_r1(quarter: str, class_type: str = "onpeak") -> pl.DataFrame:
    path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col(PY_COL) >= 2019)
            & pl.col(BASELINE_COL_R1).is_not_null()
            & pl.col(MCP_COL).is_not_null()
            & pl.col(CLASS_COL).is_not_null()
            & (pl.col(CLASS_COL) == class_type)
        )
        .collect()
    )


def load_r2r3(round_num: int, quarter: str, class_type: str = "onpeak") -> pl.DataFrame:
    return (
        pl.scan_parquet(R2R3_DATA_PATH)
        .filter(
            (pl.col("round") == round_num)
            & (pl.col("period_type") == quarter)
            & (pl.col(PY_COL) >= 2019)
            & pl.col(BASELINE_COL_R2R3).is_not_null()
            & pl.col(MCP_COL).is_not_null()
            & pl.col(CLASS_COL).is_not_null()
            & (pl.col(CLASS_COL) == class_type)
        )
        .select([BASELINE_COL_R2R3, MCP_COL, PY_COL, "period_type", CLASS_COL])
        .collect()
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def add_sign_seg(df: pl.DataFrame, baseline_col: str) -> pl.DataFrame:
    """Add sign_seg column: prevail/counter/zero."""
    return df.with_columns(
        pl.when(pl.col(baseline_col) > 0).then(pl.lit("prevail"))
        .when(pl.col(baseline_col) < 0).then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
        .alias("sign_seg")
    )


def calibrate_width(residuals: pl.Series, target: float) -> float:
    """Empirical quantile of |residual|."""
    if residuals.len() == 0:
        return 0.0
    return float(residuals.abs().quantile(target))


# ─── Step 1: Residual profiles ──────────────────────────────────────────────


def step1_residual_profiles(df: pl.DataFrame, baseline_col: str, n_bins: int, label: str):
    """Per-cell residual distribution for prevail vs counter."""
    print(f"\n{'#'*90}")
    print(f"  STEP 1: Residual Profiles — {label}")
    print(f"{'#'*90}")

    # Use full data for boundaries (diagnostic, not CV)
    boundaries, labels = compute_quantile_boundaries(df[baseline_col], n_bins)
    print(f"\n  Bin boundaries: {[round(b, 1) for b in boundaries]}")
    print(f"  Labels: {labels}")

    # Overall sign distribution
    n_prevail = df.filter(pl.col("sign_seg") == "prevail").height
    n_counter = df.filter(pl.col("sign_seg") == "counter").height
    n_zero = df.filter(pl.col("sign_seg") == "zero").height
    print(f"\n  Sign distribution: prevail={n_prevail:,} ({n_prevail/df.height*100:.1f}%)  "
          f"counter={n_counter:,} ({n_counter/df.height*100:.1f}%)  zero={n_zero:,}")

    residual = df[MCP_COL] - df[baseline_col]
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)

    work = df.with_columns([
        residual.alias("residual"),
        residual.abs().alias("abs_residual"),
        pl.Series("bin", bins),
    ])

    # Header
    print(f"\n  {'Bin':>4} {'Seg':>8} {'N':>8} {'Mean AE':>8} {'Med AE':>8} {'P95 AE':>8} "
          f"{'%pos_res':>8} {'P95 upper':>10} {'P95 lower':>10} "
          f"{'Pooled W':>9} {'Split W':>9} {'Ratio':>7}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*10} {'-'*10} "
          f"{'-'*9} {'-'*9} {'-'*7}")

    for b in labels:
        bin_data = work.filter(pl.col("bin") == b)
        if bin_data.height == 0:
            continue

        # Pooled P95 width for this bin
        pooled_w = calibrate_width(bin_data["residual"], TARGET)

        for seg in SEGS:
            seg_data = bin_data.filter(pl.col("sign_seg") == seg)
            if seg_data.height == 0:
                continue

            n = seg_data.height
            res = seg_data["residual"]
            abs_res = seg_data["abs_residual"]

            mean_ae = float(abs_res.mean())
            med_ae = float(abs_res.median())
            p95_ae = float(abs_res.quantile(0.95))

            # % of residuals that are positive (mcp > baseline)
            pct_pos = float((res > 0).mean()) * 100

            # Upper tail: P95 of max(0, mcp - baseline) = overshoot
            overshoot = res.clip(lower_bound=0)
            p95_upper = float(overshoot.quantile(0.95))

            # Lower tail: P95 of max(0, baseline - mcp) = undershoot
            undershoot = (-res).clip(lower_bound=0)
            p95_lower = float(undershoot.quantile(0.95))

            # Split P95 width
            split_w = calibrate_width(res, TARGET)

            ratio = split_w / pooled_w if pooled_w > 0 else 0

            print(
                f"  {b:>4} {seg:>8} {n:>8,} {mean_ae:>8.1f} {med_ae:>8.1f} {p95_ae:>8.1f} "
                f"{pct_pos:>7.1f}% {p95_upper:>10.1f} {p95_lower:>10.1f} "
                f"{pooled_w:>9.1f} {split_w:>9.1f} {ratio:>7.3f}"
            )

        # Blank line between bins
        print()


# ─── Step 2 & 3: LOO coverage comparison ────────────────────────────────────


def loo_sign_split_comparison(
    df: pl.DataFrame,
    baseline_col: str,
    n_bins: int,
    pys: list[int],
    label: str,
) -> dict:
    """LOO comparison: pooled vs sign-split calibration.

    Returns summary dict with per-cell and per-PY results.
    """
    available_pys = sorted(df[PY_COL].unique().to_list())
    pys_to_use = [py for py in pys if py in available_pys]

    # Accumulators: list of dicts per test row
    rows = []

    for test_py in pys_to_use:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)
        if train.height == 0 or test.height == 0:
            continue

        # Bin boundaries from training data
        boundaries, bin_labels = compute_quantile_boundaries(train[baseline_col], n_bins)

        abs_bl_train = train[baseline_col].abs()
        abs_bl_test = test[baseline_col].abs()
        train_bins = assign_bins(abs_bl_train, boundaries, bin_labels)
        test_bins = assign_bins(abs_bl_test, boundaries, bin_labels)

        train_w = train.with_columns(pl.Series("bin", train_bins))
        test_w = test.with_columns(pl.Series("bin", test_bins))

        for b in bin_labels:
            train_bin = train_w.filter(pl.col("bin") == b)
            test_bin = test_w.filter(pl.col("bin") == b)
            if train_bin.height == 0 or test_bin.height == 0:
                continue

            # Pooled P95 width
            train_res = train_bin[MCP_COL] - train_bin[baseline_col]
            pooled_w = calibrate_width(train_res, TARGET)

            # P50 pooled width
            pooled_w50 = calibrate_width(train_res, 0.50)

            # Split widths per sign segment
            split_widths = {}
            split_widths50 = {}
            for seg in SEGS:
                seg_train = train_bin.filter(pl.col("sign_seg") == seg)
                if seg_train.height > 0:
                    seg_res = seg_train[MCP_COL] - seg_train[baseline_col]
                    split_widths[seg] = calibrate_width(seg_res, TARGET)
                    split_widths50[seg] = calibrate_width(seg_res, 0.50)
                else:
                    split_widths[seg] = pooled_w
                    split_widths50[seg] = pooled_w50

            # Evaluate on test data per sign segment
            for seg in SEGS:
                seg_test = test_bin.filter(pl.col("sign_seg") == seg)
                if seg_test.height == 0:
                    continue

                baseline = seg_test[baseline_col]
                mcp = seg_test[MCP_COL]
                abs_res = (mcp - baseline).abs()

                # P95 coverage
                cov_pooled_95 = ((mcp >= baseline - pooled_w) & (mcp <= baseline + pooled_w))
                sw = split_widths[seg]
                cov_split_95 = ((mcp >= baseline - sw) & (mcp <= baseline + sw))

                # P50 coverage
                cov_pooled_50 = ((mcp >= baseline - pooled_w50) & (mcp <= baseline + pooled_w50))
                sw50 = split_widths50[seg]
                cov_split_50 = ((mcp >= baseline - sw50) & (mcp <= baseline + sw50))

                for i in range(seg_test.height):
                    rows.append({
                        "bin": b,
                        "sign_seg": seg,
                        "test_py": test_py,
                        "covered_pooled_95": bool(cov_pooled_95[i]),
                        "covered_split_95": bool(cov_split_95[i]),
                        "covered_pooled_50": bool(cov_pooled_50[i]),
                        "covered_split_50": bool(cov_split_50[i]),
                        "width_pooled_95": pooled_w * 2,
                        "width_split_95": sw * 2,
                        "width_pooled_50": pooled_w50 * 2,
                        "width_split_50": sw50 * 2,
                        "abs_res": float(abs_res[i]),
                    })

    rdf = pl.DataFrame(rows)
    return rdf


def print_loo_results(rdf: pl.DataFrame, label: str, pys: list[int]):
    """Print LOO coverage comparison tables."""
    print(f"\n{'='*110}")
    print(f"  LOO Coverage: Pooled vs Sign-Split — {label}")
    print(f"{'='*110}")
    print(f"  Total test rows: {rdf.height:,}\n")

    # Per (bin × sign_seg) summary
    print(f"  {'Bin':>4} {'Seg':>8} {'N':>8}  "
          f"{'P95 Pool':>9} {'P95 Split':>10} {'P95 Diff':>9}  "
          f"{'P50 Pool':>9} {'P50 Split':>10} {'P50 Diff':>9}  "
          f"{'P95 W Pool':>11} {'P95 W Split':>12} {'W chg':>7}")
    print(f"  {'-'*4} {'-'*8} {'-'*8}  "
          f"{'-'*9} {'-'*10} {'-'*9}  "
          f"{'-'*9} {'-'*10} {'-'*9}  "
          f"{'-'*11} {'-'*12} {'-'*7}")

    bins_order = sorted(rdf["bin"].unique().to_list())
    for b in bins_order:
        for seg in SEGS:
            sub = rdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
            if sub.height == 0:
                continue

            n = sub.height
            p95_pool = float(sub["covered_pooled_95"].mean()) * 100
            p95_split = float(sub["covered_split_95"].mean()) * 100
            p50_pool = float(sub["covered_pooled_50"].mean()) * 100
            p50_split = float(sub["covered_split_50"].mean()) * 100
            w_pool = float(sub["width_pooled_95"].mean())
            w_split = float(sub["width_split_95"].mean())
            w_chg = (w_split / w_pool - 1) * 100 if w_pool > 0 else 0

            print(
                f"  {b:>4} {seg:>8} {n:>8,}  "
                f"{p95_pool:>8.2f}% {p95_split:>9.2f}% {p95_split - p95_pool:>+8.2f}  "
                f"{p50_pool:>8.2f}% {p50_split:>9.2f}% {p50_split - p50_pool:>+8.2f}  "
                f"{w_pool:>11.1f} {w_split:>12.1f} {w_chg:>+6.1f}%"
            )
        print()

    # Per-PY stability for P95
    print(f"\n  Per-PY P95 coverage (sign-split method):")
    print(f"  {'PY':>6} {'N':>8} {'Prevail':>9} {'Counter':>9} {'All':>9}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")

    available_pys = sorted(rdf["test_py"].unique().to_list())
    for py in available_pys:
        py_sub = rdf.filter(pl.col("test_py") == py)
        n = py_sub.height
        all_cov = float(py_sub["covered_split_95"].mean()) * 100
        prev_sub = py_sub.filter(pl.col("sign_seg") == "prevail")
        ctr_sub = py_sub.filter(pl.col("sign_seg") == "counter")
        prev_cov = float(prev_sub["covered_split_95"].mean()) * 100 if prev_sub.height > 0 else 0
        ctr_cov = float(ctr_sub["covered_split_95"].mean()) * 100 if ctr_sub.height > 0 else 0
        print(f"  {py:>6} {n:>8,} {prev_cov:>8.2f}% {ctr_cov:>8.2f}% {all_cov:>8.2f}%")

    # Overall aggregates
    print(f"\n  Overall:")
    for seg in SEGS + ["all"]:
        if seg == "all":
            sub = rdf
        else:
            sub = rdf.filter(pl.col("sign_seg") == seg)
        if sub.height == 0:
            continue
        p95_pool = float(sub["covered_pooled_95"].mean()) * 100
        p95_split = float(sub["covered_split_95"].mean()) * 100
        p50_pool = float(sub["covered_pooled_50"].mean()) * 100
        p50_split = float(sub["covered_split_50"].mean()) * 100
        w_pool = float(sub["width_pooled_95"].mean())
        w_split = float(sub["width_split_95"].mean())
        w_chg = (w_split / w_pool - 1) * 100 if w_pool > 0 else 0
        print(
            f"    {seg:>8}: n={sub.height:>8,}  "
            f"P95: pool={p95_pool:.2f}% split={p95_split:.2f}% diff={p95_split-p95_pool:+.2f}pp  "
            f"P50: pool={p50_pool:.2f}% split={p50_split:.2f}%  "
            f"W95: pool={w_pool:.0f} split={w_split:.0f} ({w_chg:+.1f}%)"
        )


# ─── Step 5: Summary report ─────────────────────────────────────────────────


def print_summary_report(all_results: dict):
    """Print cross-round summary table."""
    print(f"\n{'#'*90}")
    print(f"  STEP 5: CROSS-ROUND SUMMARY")
    print(f"{'#'*90}")

    print(f"\n  {'Round':>6} {'Quarter':>8} {'Seg':>8} {'N':>8} "
          f"{'P95 Pool':>9} {'P95 Split':>10} {'Diff':>7} "
          f"{'W Pool':>8} {'W Split':>9} {'W chg':>7}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*9} {'-'*10} {'-'*7} "
          f"{'-'*8} {'-'*9} {'-'*7}")

    for key, rdf in sorted(all_results.items()):
        round_name, quarter = key
        for seg in SEGS:
            sub = rdf.filter(pl.col("sign_seg") == seg)
            if sub.height == 0:
                continue
            n = sub.height
            p95_pool = float(sub["covered_pooled_95"].mean()) * 100
            p95_split = float(sub["covered_split_95"].mean()) * 100
            w_pool = float(sub["width_pooled_95"].mean())
            w_split = float(sub["width_split_95"].mean())
            w_chg = (w_split / w_pool - 1) * 100 if w_pool > 0 else 0
            print(
                f"  {round_name:>6} {quarter:>8} {seg:>8} {n:>8,} "
                f"{p95_pool:>8.2f}% {p95_split:>9.2f}% {p95_split-p95_pool:>+6.2f} "
                f"{w_pool:>8.0f} {w_split:>9.0f} {w_chg:>+6.1f}%"
            )
        # Also print "all" row for this round/quarter
        n = rdf.height
        p95_pool = float(rdf["covered_pooled_95"].mean()) * 100
        p95_split = float(rdf["covered_split_95"].mean()) * 100
        w_pool = float(rdf["width_pooled_95"].mean())
        w_split = float(rdf["width_split_95"].mean())
        w_chg = (w_split / w_pool - 1) * 100 if w_pool > 0 else 0
        print(
            f"  {round_name:>6} {quarter:>8} {'all':>8} {n:>8,} "
            f"{p95_pool:>8.2f}% {p95_split:>9.2f}% {p95_split-p95_pool:>+6.2f} "
            f"{w_pool:>8.0f} {w_split:>9.0f} {w_chg:>+6.1f}%"
        )
        print()

    # Decision framework
    print(f"\n  {'='*70}")
    print(f"  DECISION CRITERIA")
    print(f"  {'='*70}")
    print(f"  1. Counter P95 coverage improves by >1pp in majority of cells")
    print(f"  2. Overall coverage does not degrade")
    print(f"  3. Effect is consistent across quarters and rounds")
    print(f"  4. Min cell size after split is >500 rows (LOO training fold)")
    print()

    # Compute decision metrics
    improvements = []
    degradations = []
    min_cell_sizes = []
    for key, rdf in sorted(all_results.items()):
        round_name, quarter = key
        for seg in SEGS:
            sub = rdf.filter(pl.col("sign_seg") == seg)
            if sub.height == 0:
                continue
            p95_pool = float(sub["covered_pooled_95"].mean()) * 100
            p95_split = float(sub["covered_split_95"].mean()) * 100
            diff = p95_split - p95_pool
            if seg == "counter":
                improvements.append(diff)
            elif seg == "prevail":
                # Check prevail doesn't degrade much
                degradations.append(diff)

        # Track min cell size per (bin × seg) in test data
        for b in rdf["bin"].unique().to_list():
            for seg in SEGS:
                cell = rdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
                if cell.height > 0:
                    min_cell_sizes.append(cell.height)

    if improvements:
        avg_counter_improvement = sum(improvements) / len(improvements)
        n_counter_improved = sum(1 for x in improvements if x > 0)
        print(f"  Counter P95 coverage improvement: avg={avg_counter_improvement:+.2f}pp, "
              f"{n_counter_improved}/{len(improvements)} cells improved")
    if degradations:
        avg_prevail_change = sum(degradations) / len(degradations)
        n_prevail_degraded = sum(1 for x in degradations if x < -1)
        print(f"  Prevail P95 coverage change: avg={avg_prevail_change:+.2f}pp, "
              f"{n_prevail_degraded}/{len(degradations)} cells degraded >1pp")
    if min_cell_sizes:
        print(f"  Min test cell size: {min(min_cell_sizes):,} rows")

    # Overall net
    all_dfs = list(all_results.values())
    if all_dfs:
        combined = pl.concat(all_dfs)
        overall_pool = float(combined["covered_pooled_95"].mean()) * 100
        overall_split = float(combined["covered_split_95"].mean()) * 100
        print(f"\n  Overall net P95 coverage change: {overall_split - overall_pool:+.2f}pp "
              f"(pooled={overall_pool:.2f}%, split={overall_split:.2f}%)")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    print(f"Investigate sign(baseline) as band stratification axis")
    print(f"Focus: onpeak only")
    print(f"Memory at start: {mem_mb():.0f} MB")

    all_results = {}

    # ─── Step 1: Residual profiles (R1 aq1 onpeak) ───────────────────

    print(f"\nLoading R1 aq1 onpeak...")
    df_r1_aq1 = load_r1("aq1", "onpeak")
    df_r1_aq1 = add_sign_seg(df_r1_aq1, BASELINE_COL_R1)
    print(f"  {df_r1_aq1.height:,} rows, Memory: {mem_mb():.0f} MB")

    step1_residual_profiles(df_r1_aq1, BASELINE_COL_R1, n_bins=4, label="R1 aq1 onpeak")

    # ─── Step 2: LOO coverage (R1 aq1 onpeak) ────────────────────────

    print(f"\n{'#'*90}")
    print(f"  STEP 2: LOO Coverage Comparison — R1 aq1 onpeak")
    print(f"{'#'*90}")

    rdf = loo_sign_split_comparison(df_r1_aq1, BASELINE_COL_R1, n_bins=4, pys=R1_PYS, label="R1 aq1 onpeak")
    print_loo_results(rdf, "R1 aq1 onpeak", R1_PYS)
    all_results[("R1", "aq1")] = rdf

    del df_r1_aq1
    gc.collect()
    print(f"\n  Memory: {mem_mb():.0f} MB")

    # ─── Step 3: Extend to all quarters (R1 aq2-aq4, onpeak) ────────

    print(f"\n{'#'*90}")
    print(f"  STEP 3: Extend to all R1 quarters (onpeak)")
    print(f"{'#'*90}")

    for quarter in ["aq2", "aq3", "aq4"]:
        print(f"\n  Loading R1 {quarter} onpeak...")
        df = load_r1(quarter, "onpeak")
        df = add_sign_seg(df, BASELINE_COL_R1)
        print(f"    {df.height:,} rows, Memory: {mem_mb():.0f} MB")

        rdf = loo_sign_split_comparison(df, BASELINE_COL_R1, n_bins=4, pys=R1_PYS, label=f"R1 {quarter} onpeak")
        print_loo_results(rdf, f"R1 {quarter} onpeak", R1_PYS)
        all_results[("R1", quarter)] = rdf

        del df
        gc.collect()
        print(f"  Memory: {mem_mb():.0f} MB")

    # ─── Step 4: Extend to R2/R3 (aq1 onpeak) ────────────────────────

    print(f"\n{'#'*90}")
    print(f"  STEP 4: R2 and R3 aq1 onpeak")
    print(f"{'#'*90}")

    for round_num in [2, 3]:
        print(f"\n  Loading R{round_num} aq1 onpeak...")
        df = load_r2r3(round_num, "aq1", "onpeak")
        df = add_sign_seg(df, BASELINE_COL_R2R3)
        print(f"    {df.height:,} rows, Memory: {mem_mb():.0f} MB")

        # R2/R3 use 6 bins
        rdf = loo_sign_split_comparison(
            df, BASELINE_COL_R2R3, n_bins=6, pys=R2R3_PYS,
            label=f"R{round_num} aq1 onpeak",
        )
        print_loo_results(rdf, f"R{round_num} aq1 onpeak", R2R3_PYS)
        all_results[(f"R{round_num}", "aq1")] = rdf

        del df
        gc.collect()
        print(f"  Memory: {mem_mb():.0f} MB")

    # ─── Step 5: Summary report ───────────────────────────────────────

    print_summary_report(all_results)

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
