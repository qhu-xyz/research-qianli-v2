# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""PJM: Buy clearing rate analysis at (round, PY, bin, flow_type) granularity.

All trades are buy. Buy clearing rate = P(MCP <= price).
For a buy bid at the P95 upper band edge, expected clearing rate = 97.5%.

Flow type: prevail (baseline > 0) vs counter (baseline < 0).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/analyze_clearing_rates.py
"""

from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "miso" / "scripts"))

import run_v9_bands
run_v9_bands.CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]
run_v9_bands.MIN_CELL_ROWS = 200

from run_v9_bands import (
    calibrate_asymmetric_per_class,
    apply_asymmetric_bands_per_class_fast,
    compute_quantile_boundaries,
    assign_bins,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path("/opt/temp/qianli/annual_research/pjm_annual_with_mcp.parquet")
PJM_CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]
DEV_PYS = list(range(2017, 2025))
N_BINS = 5
MIN_TRAIN_PYS = 2


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_round(round_num: int) -> pl.DataFrame:
    df = (
        pl.scan_parquet(DATA_PATH)
        .filter(
            (pl.col("round") == round_num)
            & pl.col("class_type").is_in(PJM_CLASSES)
            & pl.col("mtm_1st_mean").is_not_null()
            & pl.col("mcp").is_not_null()
            & pl.col("planning_year").is_in(DEV_PYS)
        )
        .select(["mtm_1st_mean", "mcp", "planning_year", "class_type", "source_id", "sink_id"])
        .collect()
    )
    df = df.with_columns((pl.col("mtm_1st_mean") * 12).alias("baseline"))
    return df


def buy_clear_rate(mcp: np.ndarray, upper: np.ndarray) -> float:
    """Buy clearing rate = P(MCP <= upper)."""
    if len(mcp) == 0:
        return float("nan")
    return float(np.mean(mcp <= upper) * 100)


def analyze_round(round_num: int, df: pl.DataFrame) -> list[dict]:
    """Run temporal CV and compute buy clearing rates per (PY, bin, flow_type)."""
    rows = []

    for test_py in DEV_PYS:
        train = df.filter(pl.col("planning_year") < test_py)
        test = df.filter(pl.col("planning_year") == test_py)
        if train.height == 0 or test.height == 0:
            continue
        train_pys = sorted(train["planning_year"].unique().to_list())
        if len(train_pys) < MIN_TRAIN_PYS:
            continue

        boundaries, bin_labels = compute_quantile_boundaries(train["baseline"], N_BINS)
        bin_pairs = calibrate_asymmetric_per_class(
            train, "baseline", "mcp", "class_type",
            boundaries, bin_labels, COVERAGE_LEVELS,
        )
        banded = apply_asymmetric_bands_per_class_fast(
            test, bin_pairs, "baseline", "class_type", boundaries, bin_labels,
        )

        mcp = banded["mcp"].to_numpy()
        baselines = banded["baseline"].to_numpy()
        upper_p95 = banded["upper_p95"].to_numpy()
        test_bins = assign_bins(banded["baseline"].abs(), boundaries, bin_labels).to_numpy()
        flow_types = np.where(baselines > 0, "prevail", "counter")

        # Full grid: (bin, flow_type)
        for bl in bin_labels:
            for ft in ["prevail", "counter"]:
                mask = (test_bins == bl) & (flow_types == ft)
                n = int(mask.sum())
                if n == 0:
                    continue
                rate = buy_clear_rate(mcp[mask], upper_p95[mask])
                rows.append({
                    "round": round_num,
                    "py": test_py,
                    "bin": bl,
                    "flow_type": ft,
                    "n": n,
                    "buy_at_upper_p95": round(rate, 1),
                    "shortfall": round(rate - 97.5, 1),
                })

        # Also compute aggregates for summary tables
        # Per-PY overall
        rate = buy_clear_rate(mcp, upper_p95)
        rows.append({
            "round": round_num, "py": test_py, "bin": "all", "flow_type": "all",
            "n": len(mcp), "buy_at_upper_p95": round(rate, 1), "shortfall": round(rate - 97.5, 1),
        })
        # Per-PY per flow_type
        for ft in ["prevail", "counter"]:
            mask = flow_types == ft
            n = int(mask.sum())
            if n > 0:
                rate = buy_clear_rate(mcp[mask], upper_p95[mask])
                rows.append({
                    "round": round_num, "py": test_py, "bin": "all", "flow_type": ft,
                    "n": n, "buy_at_upper_p95": round(rate, 1), "shortfall": round(rate - 97.5, 1),
                })
        # Per-PY per bin
        for bl in bin_labels:
            mask = test_bins == bl
            n = int(mask.sum())
            if n > 0:
                rate = buy_clear_rate(mcp[mask], upper_p95[mask])
                rows.append({
                    "round": round_num, "py": test_py, "bin": bl, "flow_type": "all",
                    "n": n, "buy_at_upper_p95": round(rate, 1), "shortfall": round(rate - 97.5, 1),
                })

    return rows


def print_report(all_rows: list[dict]):
    """Print report per CLAUDE.md reporting standard."""

    # 1. Overall per round
    print("\n" + "=" * 90)
    print("  LEVEL 1: Overall per Round (all PYs, all bins, all flow types)")
    print("=" * 90)
    print(f"  {'Round':>5} {'N':>10} {'Buy@Up95':>9} {'Shortfall':>10}")
    for rnd in [1, 2, 3, 4]:
        agg = [r for r in all_rows if r["round"] == rnd and r["bin"] == "all" and r["flow_type"] == "all"]
        total_n = sum(r["n"] for r in agg)
        # Weighted average
        if total_n > 0:
            wavg = sum(r["buy_at_upper_p95"] * r["n"] for r in agg) / total_n
            print(f"  R{rnd:>4} {total_n:>10,} {wavg:>8.1f}% {wavg - 97.5:>+9.1f}pp")

    # 2. Per PY per round
    print("\n" + "=" * 90)
    print("  LEVEL 2: Per PY per Round")
    print("=" * 90)
    for rnd in [1, 2, 3, 4]:
        print(f"\n  --- R{rnd} ---")
        print(f"  {'PY':>6} {'N':>10} {'Buy@Up95':>9} {'Shortfall':>10}")
        pys = sorted(set(r["py"] for r in all_rows if r["round"] == rnd and r["bin"] == "all" and r["flow_type"] == "all"))
        for py in pys:
            r = next((r for r in all_rows if r["round"] == rnd and r["py"] == py and r["bin"] == "all" and r["flow_type"] == "all"), None)
            if r:
                flag = " ***" if r["shortfall"] < -5 else " *" if r["shortfall"] < -3 else ""
                print(f"  {py:>6} {r['n']:>10,} {r['buy_at_upper_p95']:>8.1f}% {r['shortfall']:>+9.1f}pp{flag}")

    # 3. Per bin per round (averaged across PYs)
    print("\n" + "=" * 90)
    print("  LEVEL 3: Per Bin per Round (avg across PYs)")
    print("=" * 90)
    for rnd in [1, 2, 3, 4]:
        print(f"\n  --- R{rnd} ---")
        print(f"  {'Bin':>6} {'N':>10} {'Buy@Up95':>9} {'Shortfall':>10}")
        for bl in [f"q{i+1}" for i in range(N_BINS)]:
            agg = [r for r in all_rows if r["round"] == rnd and r["bin"] == bl and r["flow_type"] == "all"]
            total_n = sum(r["n"] for r in agg)
            if total_n > 0:
                wavg = sum(r["buy_at_upper_p95"] * r["n"] for r in agg) / total_n
                flag = " ***" if wavg - 97.5 < -5 else " *" if wavg - 97.5 < -3 else ""
                print(f"  {bl:>6} {total_n:>10,} {wavg:>8.1f}% {wavg - 97.5:>+9.1f}pp{flag}")

    # 4. Per flow_type per round (averaged across PYs)
    print("\n" + "=" * 90)
    print("  LEVEL 4: Per Flow Type per Round (avg across PYs)")
    print("=" * 90)
    for rnd in [1, 2, 3, 4]:
        print(f"\n  --- R{rnd} ---")
        print(f"  {'Flow':>10} {'N':>10} {'Buy@Up95':>9} {'Shortfall':>10}")
        for ft in ["prevail", "counter"]:
            agg = [r for r in all_rows if r["round"] == rnd and r["bin"] == "all" and r["flow_type"] == ft]
            total_n = sum(r["n"] for r in agg)
            if total_n > 0:
                wavg = sum(r["buy_at_upper_p95"] * r["n"] for r in agg) / total_n
                flag = " ***" if wavg - 97.5 < -5 else " *" if wavg - 97.5 < -3 else ""
                print(f"  {ft:>10} {total_n:>10,} {wavg:>8.1f}% {wavg - 97.5:>+9.1f}pp{flag}")

    # 5. Full grid for flagged cells (shortfall < -3pp)
    print("\n" + "=" * 90)
    print("  LEVEL 5: Flagged Cells (buy@upper_p95 shortfall < -3pp)")
    print("=" * 90)
    flagged = [r for r in all_rows
               if r["bin"] != "all" and r["flow_type"] != "all"
               and r["shortfall"] < -3 and r["n"] >= 100]
    flagged.sort(key=lambda r: r["shortfall"])
    if flagged:
        print(f"  {'Round':>5} {'PY':>6} {'Bin':>4} {'Flow':>10} {'N':>8} {'Buy@Up95':>9} {'Shortfall':>10}")
        for r in flagged:
            sev = "FLAG" if r["shortfall"] < -10 else "CONCERN" if r["shortfall"] < -5 else "WATCH"
            print(f"  R{r['round']:>4} {r['py']:>6} {r['bin']:>4} {r['flow_type']:>10} "
                  f"{r['n']:>8,} {r['buy_at_upper_p95']:>8.1f}% {r['shortfall']:>+9.1f}pp  [{sev}]")
    else:
        print("  None found.")

    # Summary
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    n_flag = len([r for r in flagged if r["shortfall"] < -10])
    n_concern = len([r for r in flagged if -10 <= r["shortfall"] < -5])
    n_watch = len([r for r in flagged if -5 <= r["shortfall"] < -3])
    print(f"  FLAG (< -10pp): {n_flag} cells")
    print(f"  CONCERN (-5 to -10pp): {n_concern} cells")
    print(f"  WATCH (-3 to -5pp): {n_watch} cells")
    if flagged:
        worst = flagged[0]
        print(f"  Worst cell: R{worst['round']} PY{worst['py']} {worst['bin']} {worst['flow_type']} "
              f"= {worst['buy_at_upper_p95']:.1f}% ({worst['shortfall']:+.1f}pp)")


def main():
    t0 = time.time()
    print(f"PJM Buy Clearing Rate Analysis")
    print(f"Metric: P(MCP <= upper_p95), target = 97.5%")
    print(f"Granularity: (round, PY, bin, flow_type)")
    print(f"mem={mem_mb():.0f}MB")

    all_rows = []
    for rnd in [1, 2, 3, 4]:
        print(f"\nLoading R{rnd}...", end=" ", flush=True)
        df = load_round(rnd)
        n_prev = df.filter(pl.col("baseline") > 0).height
        n_ctr = df.filter(pl.col("baseline") <= 0).height
        print(f"{df.height:,} rows (prevail={n_prev:,}, counter={n_ctr:,})")
        all_rows.extend(analyze_round(rnd, df))
        del df; gc.collect()

    print_report(all_rows)

    # Save raw data
    out_path = ROOT / "versions" / "v1_dev" / "clearing_rates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
        f.write("\n")
    print(f"\nSaved: {out_path}")

    elapsed = time.time() - t0
    print(f"Done. elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
