"""Phase 0: Load PJM annual trades via canonical pipeline.

Pipeline (same as MISO):
  1. get_trades_of_given_duration() → bid-level trades
  2. Filter to annual_period_types, break_offpeak, filter classes
  3. merge_cleared_volume(merge_mcp=True) → adds mcp, cleared_volume
  4. get_m2m_mcp_for_trades_all() → adds mtm_1st_mean, mtm_1st_period, etc.
  5. Filter to buy only, period_type='a'
  6. Cache to parquet

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/load_canonical.py submit
    # Or locally (needs enough memory):
    python ... run
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import time
from pathlib import Path
from typing import Any

# Save to NFS-mounted path accessible from Ray workers
DATA_DIR = Path("/opt/temp/qianli/pjm_annual_research")


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_workload() -> None:
    from pbase.config.ray import init_ray
    init_ray()

    import pandas as pd
    import polars as pl

    from pbase.analysis.tools.all_positions import PjmApTools

    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    aptools = PjmApTools()

    print(f"Phase 0: Load PJM Canonical Annual Trades")
    print(f"  annual_period_types: {aptools.tools.annual_period_types}")
    print(f"  classtypes: {aptools.tools.classtypes}")
    print(f"  mem={mem_mb():.0f}MB")

    # Step 1: Load trades
    print("\nStep 1: Loading trades...")
    trades = aptools.get_trades_of_given_duration(
        participant=None, start_month="2017-06", end_month_in="2026-06"
    )
    print(f"  Total: {trades.shape}, mem={mem_mb():.0f}MB")

    # Step 2: Filter to annual period types
    trades = trades[trades["period_type"].isin(aptools.tools.annual_period_types)].copy()
    print(f"  After annual filter: {trades.shape}")
    print(f"  period_types: {sorted(trades['period_type'].unique())}")

    # Step 3: break_offpeak (REQUIRED for PJM)
    before = trades.shape[0]
    before_cls = sorted(trades["class_type"].unique())
    trades = aptools.tools.break_offpeak(trades)
    after_cls = sorted(trades["class_type"].unique())
    print(f"  break_offpeak: {before} -> {trades.shape[0]}, classes: {before_cls} -> {after_cls}")

    # Filter to valid classes
    trades = trades[trades["class_type"].isin(aptools.tools.classtypes)].copy()
    print(f"  After class filter: {trades.shape}")

    # Step 3.5: Filter to obligation only (options lack MCP data for early years)
    if "hedge_type" in trades.columns:
        before_ht = len(trades)
        trades = trades[trades["hedge_type"] == "obligation"].copy()
        print(f"  After obligation filter: {before_ht} -> {len(trades)}")

    # Step 4: Merge cleared volume + MCP + M2M
    # PJM early years (2017-2018) have missing Option MCP for some retired nodes.
    # merge_cleared_volume(merge_mcp=False) asserts mcp exists, so we add dummy then replace.
    print("\nStep 4: Merging...")
    import numpy as np_load
    trades["mcp"] = np_load.nan  # dummy, will be replaced by merge_mcp
    trades = aptools.tools.merge_cleared_volume(trades, merge_mcp=False)
    print(f"  After merge_cleared_volume (no mcp): {trades.shape}, mem={mem_mb():.0f}MB")

    # Now merge MCP with raise_error=False (tolerates missing nodes)
    trades = trades.drop(columns=["mcp"])  # remove dummy
    trades = aptools.tools.merge_mcp(trades, raise_error=False)
    n_mcp_null = trades["mcp"].isnull().sum()
    print(f"  After merge_mcp: {trades.shape}, mcp nulls={n_mcp_null}")

    # Drop paths with null MCP
    if n_mcp_null > 0:
        trades = trades[trades["mcp"].notna()].copy()
        print(f"  After dropping null mcp: {trades.shape}")

    trades = aptools.tools.get_m2m_mcp_for_trades_all(trades, n_jobs=1, rename=True, oriname=False)
    print(f"  After get_m2m_mcp_for_trades_all: {trades.shape}")
    print(f"  Columns: {sorted(trades.columns)}")

    # Step 5: Filter to buy only + period_type='a'
    trades = trades[(trades["trade_type"] == "buy") & (trades["period_type"] == "a")].copy()
    print(f"\n  After buy + period_type=a: {trades.shape}")

    # Derive planning_year
    if "planning_year" not in trades.columns or trades["planning_year"].isna().all():
        # Debug: show what columns have for parsing
        for c in ["market_name", "market_name_key", "market_period"]:
            if c in trades.columns:
                print(f"  {c} sample: {trades[c].dropna().unique()[:5].tolist()}")

        # Try market_name first (format: "24_25 Annual Auction" -> PY 2024)
        if "market_name" in trades.columns:
            def parse_py(mn):
                try:
                    s = str(mn).strip()
                    parts = s.split("_")
                    yr = int(parts[0])
                    return 2000 + yr if yr < 100 else yr
                except Exception:
                    return None
            trades["planning_year"] = trades["market_name"].apply(parse_py)
            n_valid = trades["planning_year"].notna().sum()
            print(f"  Derived planning_year from market_name: {n_valid}/{len(trades)} valid")

        # Fallback: market_name_key
        if ("planning_year" not in trades.columns or trades["planning_year"].isna().all()) and "market_name_key" in trades.columns:
            trades["planning_year"] = trades["market_name_key"].apply(parse_py)
            n_valid = trades["planning_year"].notna().sum()
            print(f"  Derived planning_year from market_name_key: {n_valid}/{len(trades)} valid")

        # Fallback: market_period (timestamp)
        if ("planning_year" not in trades.columns or trades["planning_year"].isna().all()) and "market_period" in trades.columns:
            trades["planning_year"] = pd.to_datetime(trades["market_period"]).dt.year
            n_valid = trades["planning_year"].notna().sum()
            print(f"  Derived planning_year from market_period: {n_valid}/{len(trades)} valid")

    # Diagnostics
    print(f"\nKey columns:")
    for col in ["mcp", "mtm_1st_period", "mtm_1st_mean", "planning_year",
                "round", "class_type", "source_id", "sink_id"]:
        if col in trades.columns:
            s = trades[col]
            nn = s.notna().sum()
            if s.dtype in ["float64", "int64", "float32"]:
                print(f"  {col}: nn={nn}/{len(trades)}, min={s.min():.1f}, med={s.median():.1f}, max={s.max():.1f}")
            else:
                uniqs = sorted(s.dropna().unique())
                print(f"  {col}: nn={nn}/{len(trades)}, unique={len(uniqs)}, vals={uniqs[:15]}")

    # Scale check
    if "mtm_1st_period" in trades.columns and "mtm_1st_mean" in trades.columns:
        valid = trades[trades["mtm_1st_mean"].notna() & (trades["mtm_1st_mean"].abs() > 1)]
        if len(valid) > 0:
            ratio = (valid["mtm_1st_period"] / valid["mtm_1st_mean"]).median()
            print(f"\n  Scale: mtm_1st_period / mtm_1st_mean = {ratio:.2f} (expect ~12 for annual)")

    # Counts per (round, PY, class)
    print(f"\nCounts per (round, planning_year, class_type):")
    if "planning_year" in trades.columns:
        counts = trades.groupby(["round", "planning_year", "class_type"]).size().reset_index(name="n")
        print(counts.to_string())

    # Row granularity check
    path_key = ["source_id", "sink_id", "class_type", "planning_year", "round"]
    valid_cols = [c for c in path_key if c in trades.columns]
    n_total = len(trades)
    n_unique = len(trades.drop_duplicates(subset=valid_cols))
    print(f"\nRow granularity: {n_total} total, {n_unique} unique paths, ratio={n_total/max(n_unique,1):.1f}")

    # Save
    out_path = str(DATA_DIR / "canonical_annual_paths.parquet")
    print(f"\nSaving to {out_path}...")
    df_pl = pl.from_pandas(trades)
    df_pl.write_parquet(out_path)
    print(f"Saved {df_pl.height:,} rows, mem={mem_mb():.0f}MB")

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.0f}s")


def _build_runtime_env() -> dict[str, Any]:
    import pbase
    import pmodel
    from pbase.utils.ray_job import build_runtime_env
    return build_runtime_env(
        py_modules=[
            os.path.dirname(pmodel.__file__),
            os.path.dirname(pbase.__file__),
        ],
        pip=["lightgbm", "polars"],
        working_dir=str(Path(__file__).resolve().parent),
    )


def _submit_and_wait(args: argparse.Namespace) -> None:
    from pbase.utils.ray_job import build_entrypoint, submit_and_wait
    entrypoint = build_entrypoint(f"python {Path(__file__).name} run", args)
    submit_and_wait(args, entrypoint, _build_runtime_env())


def _build_parser() -> argparse.ArgumentParser:
    from pbase.utils.ray_job import add_submit_args
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    parser_submit = subparsers.add_parser("submit")
    add_submit_args(parser_submit)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "submit":
        _submit_and_wait(args)
    elif args.command == "run":
        run_workload()


if __name__ == "__main__":
    main()
