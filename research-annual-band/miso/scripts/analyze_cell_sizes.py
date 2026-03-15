"""Analyze cell sizes for overfitting assessment.

For each (round, quarter, PY, bin, class, sign) cell, count rows.
Shows whether quantile estimation has enough data per cell,
especially for temporal CV where early folds are data-starved.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/analyze_cell_sizes.py
"""

from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

R1_DATA_DIR = Path("/opt/temp/qianli/annual_research/crossproduct_work")
R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")

MCP_COL = "mcp_mean"
PY_COL = "planning_year"
CLASS_COL = "class_type"
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def add_sign_seg(df: pl.DataFrame, baseline_col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(baseline_col) > 0)
        .then(pl.lit("prevail"))
        .when(pl.col(baseline_col) < 0)
        .then(pl.lit("counter"))
        .otherwise(pl.lit("zero"))
        .alias("sign_seg")
    )


def compute_quantile_boundaries(series: pl.Series, n_bins: int) -> tuple[list[float], list[str]]:
    abs_vals = series.abs()
    quantiles = [i / n_bins for i in range(1, n_bins)]
    cuts = [0.0]
    for q in quantiles:
        cuts.append(round(float(abs_vals.quantile(q)), 1))
    cuts.append(float("inf"))
    labels = [f"q{i+1}" for i in range(n_bins)]
    return cuts, labels


def assign_bins(abs_baseline: pl.Series, boundaries: list[float], labels: list[str]) -> pl.Series:
    exprs = []
    for i, label in enumerate(labels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == 0:
            cond = abs_baseline.lt(hi)
        elif i == len(labels) - 1:
            cond = abs_baseline.ge(lo)
        else:
            cond = abs_baseline.ge(lo) & abs_baseline.lt(hi)
        exprs.append((cond, label))

    result = pl.Series("bin", [""] * len(abs_baseline))
    for cond, label in reversed(exprs):
        result = pl.when(cond).then(pl.lit(label)).otherwise(result)
    return result


def analyze_round(round_num: int, baseline_col: str, loader, pys: list[int], n_bins_list: list[int]):
    print(f"\n{'#'*100}")
    print(f"  ROUND {round_num} — CELL SIZE ANALYSIS")
    print(f"{'#'*100}")

    for quarter in QUARTERS:
        df = loader(quarter)
        df = add_sign_seg(df, baseline_col)
        total = df.height
        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in pys if py in available_pys]

        print(f"\n  {'='*90}")
        print(f"  R{round_num} {quarter.upper()} — {total:,} total rows, PYs={pys_to_use}")
        print(f"  {'='*90}")

        # ── Per-PY row counts ──
        print(f"\n  Per-PY row counts:")
        print(f"  {'PY':<6} {'Total':>10} {'onpeak':>10} {'offpeak':>10} {'prevail':>10} {'counter':>10} {'zero':>8}")
        for py in pys_to_use:
            py_df = df.filter(pl.col(PY_COL) == py)
            n = py_df.height
            n_on = py_df.filter(pl.col(CLASS_COL) == "onpeak").height
            n_off = py_df.filter(pl.col(CLASS_COL) == "offpeak").height
            n_prev = py_df.filter(pl.col("sign_seg") == "prevail").height
            n_ctr = py_df.filter(pl.col("sign_seg") == "counter").height
            n_zero = py_df.filter(pl.col("sign_seg") == "zero").height
            print(f"  {py:<6} {n:>10,} {n_on:>10,} {n_off:>10,} {n_prev:>10,} {n_ctr:>10,} {n_zero:>8,}")

        # ── Temporal fold analysis: for each test PY, what does training look like? ──
        print(f"\n  Temporal fold training set sizes:")
        print(f"  {'Test PY':<8} {'Train PYs':>10} {'Train rows':>12}")
        for test_py in pys_to_use:
            train_pys = [p for p in pys_to_use if p < test_py]
            if not train_pys:
                continue
            train_df = df.filter(pl.col(PY_COL).is_in(train_pys))
            print(f"  {test_py:<8} {len(train_pys):>10} {train_df.height:>12,}")

        # ── Per-bin analysis for different bin counts ──
        for n_bins in n_bins_list:
            print(f"\n  --- {n_bins} BINS ---")

            # Compute boundaries on ALL data (representative)
            boundaries, bin_labels = compute_quantile_boundaries(df[baseline_col], n_bins)
            bounds_str = [f"{b:.0f}" if b != float("inf") else "inf" for b in boundaries]
            print(f"  Boundaries: {bounds_str}")

            # Overall bin × class × sign cell sizes
            abs_bl = df[baseline_col].abs()
            bins_col = assign_bins(abs_bl, boundaries, bin_labels)
            df_with_bins = df.with_columns(bins_col.alias("_bin"))

            print(f"\n  Overall cell sizes (bin × class × sign):")
            print(f"  {'Bin':<5} {'Class':<8} {'Sign':<9} {'Total':>8} | ", end="")
            print(" | ".join(f"PY{py}" for py in pys_to_use))

            all_cells = []
            for b in bin_labels:
                for cls in ["onpeak", "offpeak"]:
                    for sign in ["prevail", "counter"]:
                        mask = (
                            (pl.col("_bin") == b)
                            & (pl.col(CLASS_COL) == cls)
                            & (pl.col("sign_seg") == sign)
                        )
                        cell_df = df_with_bins.filter(mask)
                        n_total = cell_df.height
                        py_counts = []
                        for py in pys_to_use:
                            n_py = cell_df.filter(pl.col(PY_COL) == py).height
                            py_counts.append(n_py)

                        all_cells.append({
                            "bin": b, "class": cls, "sign": sign,
                            "total": n_total, "py_counts": py_counts,
                        })

                        py_str = " | ".join(f"{c:>5}" for c in py_counts)
                        flag = " <<<" if n_total < 500 else ""
                        print(f"  {b:<5} {cls:<8} {sign:<9} {n_total:>8,} | {py_str}{flag}")

            # Summary stats
            totals = [c["total"] for c in all_cells]
            print(f"\n  Cell size summary ({n_bins} bins × 2 classes × 2 signs = {n_bins * 4} cells):")
            print(f"    Min:    {min(totals):>8,}")
            print(f"    P10:    {sorted(totals)[len(totals)//10]:>8,}")
            print(f"    Median: {sorted(totals)[len(totals)//2]:>8,}")
            print(f"    P90:    {sorted(totals)[len(totals)*9//10]:>8,}")
            print(f"    Max:    {max(totals):>8,}")
            print(f"    <500:   {sum(1 for t in totals if t < 500)} cells")
            print(f"    <1000:  {sum(1 for t in totals if t < 1000)} cells")

            # Per-fold training cell sizes (the real overfitting risk)
            print(f"\n  Per-fold TRAINING cell sizes (temporal CV):")
            print(f"  {'Test PY':<8} {'Train PYs':>3} {'Min cell':>9} {'P10':>9} {'Median':>9} {'<500':>6} {'<1000':>6} {'<MIN_CELL':>10}")

            for test_py in pys_to_use:
                train_pys = [p for p in pys_to_use if p < test_py]
                if not train_pys:
                    continue

                train_df = df_with_bins.filter(pl.col(PY_COL).is_in(train_pys))
                fold_cells = []
                for b in bin_labels:
                    for cls in ["onpeak", "offpeak"]:
                        for sign in ["prevail", "counter"]:
                            n = train_df.filter(
                                (pl.col("_bin") == b)
                                & (pl.col(CLASS_COL) == cls)
                                & (pl.col("sign_seg") == sign)
                            ).height
                            fold_cells.append(n)

                s = sorted(fold_cells)
                n_cells = len(s)
                print(
                    f"  {test_py:<8} {len(train_pys):>3}"
                    f" {s[0]:>9,}"
                    f" {s[n_cells//10]:>9,}"
                    f" {s[n_cells//2]:>9,}"
                    f" {sum(1 for x in s if x < 500):>6}"
                    f" {sum(1 for x in s if x < 1000):>6}"
                    f" {sum(1 for x in s if x < 500):>10}"
                )

        del df, df_with_bins
        import gc
        gc.collect()


def main():
    print(f"Cell Size Analysis for Overfitting Assessment")
    print(f"Memory at start: {mem_mb():.0f} MB")

    n_bins_list = [5, 6, 8]

    # R1
    def r1_loader(quarter: str) -> pl.DataFrame:
        parquet_path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
        return (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2019)
                & pl.col("nodal_f0").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .collect()
        )

    analyze_round(1, "nodal_f0", r1_loader, R1_PYS, n_bins_list)
    print(f"\nMemory after R1: {mem_mb():.0f} MB")

    # R2
    def r2_loader(quarter: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 2)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )

    analyze_round(2, "mtm_1st_mean", r2_loader, R2R3_PYS, n_bins_list)
    print(f"\nMemory after R2: {mem_mb():.0f} MB")

    # R3
    def r3_loader(quarter: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 3)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )

    analyze_round(3, "mtm_1st_mean", r3_loader, R2R3_PYS, n_bins_list)
    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
