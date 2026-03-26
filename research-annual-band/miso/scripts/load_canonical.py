"""Phase 1: Load MISO annual trades via canonical get_trades_of_given_duration.

Pipeline:
  1. get_trades_of_given_duration() → bid-level trades (16 cols)
  2. merge_cleared_volume(merge_mcp=True) → adds mcp, cleared_volume (+4 cols)
  3. get_m2m_mcp_for_trades_all() → adds mtm_1st_mean, mcp_mean, mtm_now_*, path (+19 cols)
  4. Derive planning_year from market_name
  5. Filter: buy only, annual period types, valid class types
  6. Drop mcp_mean (deprecated, per CLAUDE.md)
  7. Cache to parquet

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/miso/scripts/load_canonical.py run
    # Or via Ray:
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/miso/scripts/load_canonical.py submit
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANNUAL_PTYPES = ["aq1", "aq2", "aq3", "aq4"]


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_workload() -> None:
    from pbase.config.ray import init_ray
    init_ray()

    from pbase.analysis.tools.all_positions import MisoApTools
    aptools = MisoApTools()

    t0 = time.time()
    print(f"Phase 1: Load MISO Canonical Annual Trades")
    print(f"mem={mem_mb():.0f}MB")

    # ── Step 0: Verify constants ─────────────────────────────────────────
    print(f"\n=== Step 0: Constants ===")
    print(f"annual_period_types: {aptools.tools.annual_period_types}")
    print(f"classtypes: {aptools.tools.classtypes}")
    valid_classes = set(aptools.tools.classtypes)

    # ── Step 1: Load all trades ──────────────────────────────────────────
    print(f"\n=== Step 1: Load trades ===")
    trades = aptools.get_trades_of_given_duration(
        participant=None, start_month="2018-06", end_month_in="2026-06"
    )
    print(f"Total trades: {trades.shape}")
    print(f"Columns: {sorted(trades.columns)}")
    print(f"period_types: {sorted(trades['period_type'].unique())}")

    # Filter to annual
    trades = trades[trades["period_type"].isin(ANNUAL_PTYPES)].copy()
    print(f"After annual filter: {trades.shape}")

    # ── Step 1.5: break_offpeak (verify no-op for MISO) ─────────────────
    before_shape = trades.shape
    before_classes = sorted(trades["class_type"].unique())
    trades = aptools.tools.break_offpeak(trades)
    after_classes = sorted(trades["class_type"].unique())
    print(f"\nbreak_offpeak: {before_shape} -> {trades.shape}")
    print(f"  classes: {before_classes} -> {after_classes}")
    assert before_shape == trades.shape, "break_offpeak changed row count for MISO — unexpected"

    # Filter to valid class types
    trades = trades[trades["class_type"].isin(valid_classes)].copy()
    print(f"After class filter: {trades.shape}, classes: {sorted(trades['class_type'].unique())}")

    # Filter to buy only (per user: all trades are buy)
    n_before = len(trades)
    trades = trades[trades["trade_type"] == "buy"].copy()
    n_sell = n_before - len(trades)
    print(f"After buy filter: {trades.shape} (dropped {n_sell} sell trades, {n_sell/n_before*100:.1f}%)")

    print(f"mem={mem_mb():.0f}MB, elapsed={time.time()-t0:.0f}s")

    # ── Step 2: Merge cleared volume + MCP ───────────────────────────────
    print(f"\n=== Step 2: merge_cleared_volume ===")
    t1 = time.time()
    trades = aptools.tools.merge_cleared_volume(trades, merge_mcp=True)
    print(f"After merge: {trades.shape}")
    print(f"mcp: null={trades['mcp'].isnull().sum()}, min={trades['mcp'].min():.2f}, max={trades['mcp'].max():.2f}")
    print(f"cleared_volume: null={trades['cleared_volume'].isnull().sum()}")
    assert trades["mcp"].isnull().sum() == 0, "mcp has nulls after merge_cleared_volume"
    print(f"elapsed={time.time()-t1:.0f}s, mem={mem_mb():.0f}MB")

    # ── Step 3: Merge M2M columns ────────────────────────────────────────
    print(f"\n=== Step 3: get_m2m_mcp_for_trades_all ===")
    t2 = time.time()
    trades = aptools.tools.get_m2m_mcp_for_trades_all(trades, n_jobs=1, rename=True, oriname=False)
    print(f"After m2m: {trades.shape}")
    for col in ["mtm_1st_mean", "mcp_mean", "mtm_2nd_mean", "mtm_3rd_mean"]:
        if col in trades.columns:
            s = trades[col]
            n_null = s.isnull().sum()
            pct = n_null / len(s) * 100
            print(f"  {col}: null={n_null}/{len(s)} ({pct:.1f}%)")
            if col == "mtm_1st_mean":
                # R1 has no prior round → expected nulls. Check per-round.
                for rnd in sorted(trades["round"].unique()):
                    sub = trades[trades["round"] == rnd]
                    rn = sub[col].isnull().sum()
                    print(f"    R{rnd}: null={rn}/{len(sub)} ({rn/len(sub)*100:.1f}%)")
    print(f"elapsed={time.time()-t2:.0f}s, mem={mem_mb():.0f}MB")

    # ── Step 4: Derive planning_year ─────────────────────────────────────
    print(f"\n=== Step 4: Derive planning_year ===")
    trades["planning_year"] = trades["market_name"].str.extract(r"(\d{4})").astype(int)
    pys = sorted(trades["planning_year"].unique())
    print(f"planning_years: {pys}")

    # ── Step 5: MCP scale verification ───────────────────────────────────
    print(f"\n=== Step 5: MCP scale check ===")
    if "mcp_mean" in trades.columns:
        mask = trades["mcp_mean"].notna() & (trades["mcp_mean"].abs() > 0.01)
        ratio = (trades.loc[mask, "mcp"] / trades.loc[mask, "mcp_mean"]).median()
        print(f"mcp / mcp_mean median ratio: {ratio:.3f} (expect ~3.0)")
        assert 2.5 < ratio < 3.5, f"MCP scale ratio {ratio} is not ~3.0 — investigate"
        # Drop mcp_mean per CLAUDE.md
        trades = trades.drop(columns=["mcp_mean"])
        print("Dropped mcp_mean (deprecated)")

    # ── Step 6: Row granularity check ────────────────────────────────────
    print(f"\n=== Step 6: Row granularity ===")
    path_key = ["source_id", "sink_id", "class_type", "planning_year", "round", "period_type"]
    n_total = len(trades)
    n_unique = len(trades.drop_duplicates(subset=path_key))
    ratio = n_total / n_unique
    print(f"Total rows: {n_total:,}, unique paths: {n_unique:,}, ratio: {ratio:.2f}")
    if ratio > 1.05:
        print(f"WARNING: ratio > 1.0 — data has multiple rows per path")
        # Verify mcp is constant within each path group
        var = trades.groupby(path_key)["mcp"].std().max()
        print(f"Max within-group mcp std: {var:.6f}")
        assert var < 0.01, f"mcp is NOT constant within path groups (std={var})"

    # ── Step 7: Exhaustive inspection ────────────────────────────────────
    print(f"\n=== Step 7: Exhaustive inspection ===")
    print(f"Shape: {trades.shape}")
    print(f"Columns ({len(trades.columns)}): {sorted(trades.columns)}")
    print(f"\nDtypes:")
    for col in sorted(trades.columns):
        print(f"  {col}: {trades[col].dtype}")

    nulls = trades.isnull().sum()
    if nulls.any():
        print(f"\nNull counts (non-zero):")
        for col in nulls[nulls > 0].index:
            print(f"  {col}: {nulls[col]} ({nulls[col]/len(trades)*100:.1f}%)")

    for col in ["mcp", "mtm_1st_mean", "bid_price", "cleared_volume"]:
        if col in trades.columns:
            s = trades[col].dropna()
            print(f"\n{col}: min={s.min():.2f}, p25={s.quantile(0.25):.2f}, "
                  f"median={s.median():.2f}, p75={s.quantile(0.75):.2f}, max={s.max():.2f}")

    # Groupby counts
    print(f"\n=== Groupby: round × period_type × planning_year × class_type ===")
    gb = trades.groupby(["round", "period_type", "planning_year", "class_type"]).size().reset_index(name="n")
    print(gb.to_string())

    # ── Step 8: Cache ────────────────────────────────────────────────────
    print(f"\n=== Step 8: Cache ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Path-level (deduplicated)
    if ratio > 1.05:
        trades_paths = trades.drop_duplicates(subset=path_key)
    else:
        trades_paths = trades

    path_out = DATA_DIR / "canonical_annual_paths.parquet"
    trades_paths.to_parquet(path_out, index=False)
    print(f"Saved paths: {path_out} ({len(trades_paths):,} rows, {path_out.stat().st_size/1e6:.1f}MB)")

    # Monthly-level (full, if different)
    monthly_out = DATA_DIR / "canonical_annual_monthly.parquet"
    trades.to_parquet(monthly_out, index=False)
    print(f"Saved monthly: {monthly_out} ({len(trades):,} rows, {monthly_out.stat().st_size/1e6:.1f}MB)")

    elapsed = time.time() - t0
    print(f"\nDone. Total elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


def _build_runtime_env() -> dict[str, Any]:
    import pbase
    from pbase.utils.ray_job import build_runtime_env
    return build_runtime_env(
        py_modules=[os.path.dirname(pbase.__file__)],
        pip=[],
        working_dir=str(Path(__file__).resolve().parent),
    )


def _submit_and_wait(args: argparse.Namespace) -> None:
    from pbase.utils.ray_job import build_entrypoint, submit_and_wait
    entrypoint = build_entrypoint(f"python {Path(__file__).name} run", args)
    submit_and_wait(args, entrypoint, _build_runtime_env())


def _build_parser() -> argparse.ArgumentParser:
    from pbase.utils.ray_job import add_submit_args
    parser = argparse.ArgumentParser(description="Load MISO canonical annual trades")
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
