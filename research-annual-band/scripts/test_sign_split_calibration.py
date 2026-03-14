"""Test: does splitting calibration by sign(baseline) within each bin help?

R1 aq1 only. LOO CV.
Compares:
  A) Pooled: calibrate P95 width from all paths in the bin (current method)
  B) Split: calibrate P95 width from prevail/counter paths separately

Reports per-bin: coverage on test data, band widths, for each method.
Zooms into a specific bin (50 < |baseline| < 100) as example.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_v2_bands import compute_quantile_boundaries
from run_v3_bands import (
    MCP_COL, PY_COL, CLASS_COL,
    R1_PYS,
)

import resource
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
BASELINE_COL = "nodal_f0"
N_BINS = 4
TARGET = 0.95  # P95


def load_r1_aq1() -> pl.DataFrame:
    path = R1_DATA_DIR / "aq1_all_baselines.parquet"
    return (
        pl.scan_parquet(path)
        .filter(
            (pl.col(PY_COL) >= 2019)
            & pl.col(BASELINE_COL).is_not_null()
            & pl.col(MCP_COL).is_not_null()
            & pl.col(CLASS_COL).is_not_null()
        )
        .collect()
    )


def assign_to_bins(df: pl.DataFrame, boundaries: list[float], labels: list[str], col: str) -> pl.DataFrame:
    """Assign rows to |baseline| bins."""
    abs_bl = df[col].abs()
    exprs = []
    for i, label in enumerate(labels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi == float("inf"):
            exprs.append(pl.when(abs_bl >= lo).then(pl.lit(label)))
        else:
            exprs.append(pl.when((abs_bl >= lo) & (abs_bl < hi)).then(pl.lit(label)))
    return df.with_columns(pl.coalesce(exprs).alias("bin"))


def calibrate_width(residuals: pl.Series, target: float) -> float:
    """Empirical quantile of |residual|."""
    if residuals.len() == 0:
        return 0.0
    abs_r = residuals.abs()
    return float(abs_r.quantile(target))


def main():
    print("Sign-split calibration test: R1 aq1")
    print(f"Memory: {mem_mb():.0f} MB\n")

    df = load_r1_aq1()
    df = df.with_columns(
        pl.when(pl.col(BASELINE_COL) > 0).then(pl.lit("prevail"))
        .when(pl.col(BASELINE_COL) < 0).then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
        .alias("sign_seg")
    )
    print(f"Loaded {df.height:,} rows\n")

    # Collect test results across all LOO folds
    pooled_results = []  # (bin, sign_seg, covered_pooled, width_pooled, abs_res)
    split_results = []   # (bin, sign_seg, covered_split, width_split, abs_res)

    for test_py in R1_PYS:
        train = df.filter(pl.col(PY_COL) != test_py)
        test = df.filter(pl.col(PY_COL) == test_py)
        if train.height == 0 or test.height == 0:
            continue

        # Compute bin boundaries from training data
        boundaries, labels = compute_quantile_boundaries(train[BASELINE_COL], N_BINS)

        # Assign bins
        train_b = assign_to_bins(train, boundaries, labels, BASELINE_COL)
        test_b = assign_to_bins(test, boundaries, labels, BASELINE_COL)

        for bin_label in labels:
            train_bin = train_b.filter(pl.col("bin") == bin_label)
            test_bin = test_b.filter(pl.col("bin") == bin_label)
            if train_bin.height == 0 or test_bin.height == 0:
                continue

            # Method A: Pooled calibration
            train_res = train_bin[MCP_COL] - train_bin[BASELINE_COL]
            pooled_width = calibrate_width(train_res, TARGET)

            # Method B: Split calibration
            split_widths = {}
            for seg in ["prevail", "counter"]:
                seg_train = train_bin.filter(pl.col("sign_seg") == seg)
                if seg_train.height > 0:
                    seg_res = seg_train[MCP_COL] - seg_train[BASELINE_COL]
                    split_widths[seg] = calibrate_width(seg_res, TARGET)
                else:
                    split_widths[seg] = pooled_width  # fallback

            # Evaluate on test data
            for seg in ["prevail", "counter"]:
                seg_test = test_bin.filter(pl.col("sign_seg") == seg)
                if seg_test.height == 0:
                    continue

                baseline = seg_test[BASELINE_COL]
                mcp = seg_test[MCP_COL]
                abs_res = (mcp - baseline).abs()

                # Pooled coverage
                lower_p = baseline - pooled_width
                upper_p = baseline + pooled_width
                covered_p = ((mcp >= lower_p) & (mcp <= upper_p))

                # Split coverage
                sw = split_widths[seg]
                lower_s = baseline - sw
                upper_s = baseline + sw
                covered_s = ((mcp >= lower_s) & (mcp <= upper_s))

                for i in range(seg_test.height):
                    pooled_results.append({
                        "bin": bin_label,
                        "sign_seg": seg,
                        "test_py": test_py,
                        "covered": bool(covered_p[i]),
                        "width": pooled_width * 2,
                        "abs_res": float(abs_res[i]),
                        "baseline": float(baseline[i]),
                    })
                    split_results.append({
                        "bin": bin_label,
                        "sign_seg": seg,
                        "test_py": test_py,
                        "covered": bool(covered_s[i]),
                        "width": sw * 2,
                        "abs_res": float(abs_res[i]),
                        "baseline": float(baseline[i]),
                    })

    # Convert to DataFrames
    pdf = pl.DataFrame(pooled_results)
    sdf = pl.DataFrame(split_results)

    print(f"Total test rows: {pdf.height:,}\n")

    # Summary by bin × sign_seg
    print(f"{'='*100}")
    print(f"{'Bin':>6s}  {'Seg':>8s}  {'N':>8s}  {'Pooled Cov':>10s}  {'Split Cov':>10s}  {'Diff':>7s}  {'Pooled W':>9s}  {'Split W':>9s}  {'W Diff':>8s}  {'Mean AE':>8s}")
    print(f"{'='*100}")

    bins_order = sorted(pdf["bin"].unique().to_list())
    for b in bins_order:
        for seg in ["prevail", "counter"]:
            p_sub = pdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
            s_sub = sdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
            if p_sub.height == 0:
                continue

            n = p_sub.height
            p_cov = p_sub["covered"].mean() * 100
            s_cov = s_sub["covered"].mean() * 100
            p_w = p_sub["width"].mean()
            s_w = s_sub["width"].mean()
            mae = p_sub["abs_res"].mean()

            print(
                f"{b:>6s}  {seg:>8s}  {n:>8,}  {p_cov:>9.2f}%  {s_cov:>9.2f}%  {s_cov - p_cov:>+6.2f}  "
                f"{p_w:>9.1f}  {s_w:>9.1f}  {(s_w/p_w - 1)*100:>+7.1f}%  {mae:>8.1f}"
            )
        print()

    # Per-PY detail for a specific bin
    print(f"\n{'='*100}")
    print(f"ZOOM: Per-PY detail for bin q2 (approx 50-166 range)")
    print(f"{'='*100}")

    # Find actual bin boundaries
    # Just use one fold to show the boundaries
    sample_train = df.filter(pl.col(PY_COL) != R1_PYS[0])
    boundaries, labels = compute_quantile_boundaries(sample_train[BASELINE_COL], N_BINS)
    print(f"Bin boundaries (sample fold): {list(zip(['lo'] + labels, boundaries))}")
    print()

    for b in ["q2"]:  # The bin that likely contains 50-166
        for seg in ["prevail", "counter"]:
            p_sub = pdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
            s_sub = sdf.filter((pl.col("bin") == b) & (pl.col("sign_seg") == seg))
            if p_sub.height == 0:
                continue

            print(f"  {b} / {seg}:")
            for py in sorted(R1_PYS):
                pp = p_sub.filter(pl.col("test_py") == py)
                ss = s_sub.filter(pl.col("test_py") == py)
                if pp.height == 0:
                    continue
                print(
                    f"    PY {py}: n={pp.height:>6,}  "
                    f"pooled_cov={pp['covered'].mean()*100:.1f}%  split_cov={ss['covered'].mean()*100:.1f}%  "
                    f"pooled_w={pp['width'].mean():.0f}  split_w={ss['width'].mean():.0f}  "
                    f"MAE={pp['abs_res'].mean():.0f}  P95AE={float(pp['abs_res'].quantile(0.95)):.0f}"
                )
            print()

    # Overall summary
    print(f"\n{'='*100}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*100}")
    for seg in ["prevail", "counter"]:
        p_sub = pdf.filter(pl.col("sign_seg") == seg)
        s_sub = sdf.filter(pl.col("sign_seg") == seg)
        if p_sub.height == 0:
            continue
        p_cov = p_sub["covered"].mean() * 100
        s_cov = s_sub["covered"].mean() * 100
        p_w = p_sub["width"].mean()
        s_w = s_sub["width"].mean()
        print(
            f"  {seg:>8s}: n={p_sub.height:>8,}  "
            f"pooled_cov={p_cov:.2f}%  split_cov={s_cov:.2f}%  diff={s_cov-p_cov:+.2f}pp  "
            f"pooled_w={p_w:.0f}  split_w={s_w:.0f}  w_diff={((s_w/p_w)-1)*100:+.1f}%"
        )

    both_p = pdf["covered"].mean() * 100
    both_s = sdf["covered"].mean() * 100
    both_pw = pdf["width"].mean()
    both_sw = sdf["width"].mean()
    print(
        f"\n  {'all':>8s}: n={pdf.height:>8,}  "
        f"pooled_cov={both_p:.2f}%  split_cov={both_s:.2f}%  diff={both_s-both_p:+.2f}pp  "
        f"pooled_w={both_pw:.0f}  split_w={both_sw:.0f}  w_diff={((both_sw/both_pw)-1)*100:+.1f}%"
    )

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
