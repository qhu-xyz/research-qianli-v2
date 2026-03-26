"""
Verify that nodal f0 stitching (sink_node_f0 - source_node_f0) matches the
actual f0 path MCP exactly, for each individual month.

Approach:
  1. Load f0 path data for selected months
  2. Load nodal f0 MCPs via MisoCalculator.get_mcp_df (column 0)
  3. For each path, compute sink_node - source_node
  4. Compare to the path's actual MCP
  5. Report match rate and analyze mismatches
"""

import resource
import gc
import sys
import datetime
import numpy as np
import pandas as pd
import polars as pl

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# ── Ray init ──────────────────────────────────────────────────────────────────
from pbase.config.ray import init_ray
import pmodel
init_ray(address='ray://10.8.0.36:10001', extra_modules=[pmodel])

from pbase.data.m2m.calculator import MisoCalculator

print(f"[init] memory: {mem_mb():.0f} MB")

# ── Config ────────────────────────────────────────────────────────────────────
# Test months spread across different years and seasons
TEST_MONTHS = [
    "2020-01",   # winter, early year
    "2021-07",   # summer
    "2022-03",   # spring
    "2023-06",   # summer
    "2024-09",   # fall, recent
]

TOLERANCE_ABS = 0.015  # $/MW absolute tolerance for "match"

# ── Load f0 path data (all months at once, filtered) ─────────────────────────
print("\n" + "="*80)
print("LOADING F0 PATH DATA")
print("="*80)

# Build proper datetime list for polars is_in
month_timestamps = [datetime.datetime(int(m[:4]), int(m[5:7]), 1) for m in TEST_MONTHS]

# Load only f0 paths for our test months
f0_paths_all = (
    pl.scan_parquet("/opt/temp/qianli/annual_research/f0p_cleared_all.parquet")
    .filter(pl.col("period_type") == "f0")
    .filter(pl.col("split_market_month").is_in(month_timestamps))
    .select(["class_type", "source_id", "sink_id", "mcp", "split_month_mcp",
             "split_market_month", "split_month_hours"])
    .collect()
)

print(f"Loaded {len(f0_paths_all):,} f0 path rows across {len(TEST_MONTHS)} months")
print(f"Months in data: {sorted(f0_paths_all['split_market_month'].unique().to_list())}")
print(f"[after path load] memory: {mem_mb():.0f} MB")

# ── Per-month verification ────────────────────────────────────────────────────
calc = MisoCalculator()

all_results = []

for month_str in TEST_MONTHS:
    print(f"\n{'='*80}")
    print(f"MONTH: {month_str}")
    print(f"{'='*80}")

    month_dt = datetime.datetime(int(month_str[:4]), int(month_str[5:7]), 1)

    # ── 1. Get path data for this month ───────────────────────────────────────
    paths_month = f0_paths_all.filter(
        pl.col("split_market_month") == month_dt
    )

    if len(paths_month) == 0:
        print(f"  WARNING: No path data for {month_str}")
        continue

    # Deduplicate to unique (class_type, source_id, sink_id) combinations
    # Use the first mcp value for each unique path (they should all be the same
    # for the same path in the same month from the same auction)
    unique_paths = (
        paths_month
        .group_by(["class_type", "source_id", "sink_id"])
        .agg([
            pl.col("split_month_mcp").first().alias("path_mcp"),
            pl.col("split_month_hours").first().alias("hours"),
            pl.len().alias("trade_count"),
        ])
    )

    n_total_rows = len(paths_month)
    n_unique_paths = len(unique_paths)
    print(f"  Path data: {n_total_rows:,} total rows -> {n_unique_paths:,} unique paths")

    # Check if there are paths with multiple different MCPs
    multi_mcp = (
        paths_month
        .group_by(["class_type", "source_id", "sink_id"])
        .agg(pl.col("split_month_mcp").n_unique().alias("n_mcps"))
        .filter(pl.col("n_mcps") > 1)
    )
    if len(multi_mcp) > 0:
        print(f"  NOTE: {len(multi_mcp)} paths have multiple MCP values (from different auctions/rounds)")

    # ── 2. Load nodal MCP for this month ──────────────────────────────────────
    try:
        mcp_df, _ = calc.get_mcp_df(market_month=month_str, fillna=True)
    except Exception as e:
        print(f"  ERROR loading MCP: {e}")
        continue

    if mcp_df is None or len(mcp_df) == 0:
        print(f"  WARNING: No nodal MCP data for {month_str}")
        continue

    # Column 0 = f0 monthly forward (most recent monthly auction)
    col0 = mcp_df.columns[0]
    nodal_f0 = mcp_df[[col0]].copy()
    nodal_f0.columns = ["nodal_mcp"]
    nodal_f0 = nodal_f0.reset_index()  # pnode_id, class_type, nodal_mcp

    print(f"  Nodal MCP: {len(nodal_f0):,} (node, class_type) entries")
    print(f"  MCP df has {len(mcp_df.columns)} columns; using column 0 (f0 monthly forward)")

    # ── 3. Convert to pandas for the join ─────────────────────────────────────
    paths_pd = unique_paths.to_pandas()
    # class_type is categorical in polars, convert to string
    paths_pd["class_type"] = paths_pd["class_type"].astype(str)

    # Build lookup dict: (pnode_id, class_type) -> nodal_mcp
    nodal_lookup = {}
    for _, row in nodal_f0.iterrows():
        nodal_lookup[(row["pnode_id"], row["class_type"])] = row["nodal_mcp"]

    # ── 4. Compute stitched MCP and compare ───────────────────────────────────
    source_mcps = []
    sink_mcps = []
    source_found = []
    sink_found = []

    for _, row in paths_pd.iterrows():
        ct = row["class_type"]
        src = row["source_id"]
        snk = row["sink_id"]

        src_val = nodal_lookup.get((src, ct), np.nan)
        snk_val = nodal_lookup.get((snk, ct), np.nan)

        source_mcps.append(src_val)
        sink_mcps.append(snk_val)
        source_found.append(not np.isnan(src_val))
        sink_found.append(not np.isnan(snk_val))

    paths_pd["source_mcp"] = source_mcps
    paths_pd["sink_mcp"] = sink_mcps
    paths_pd["source_found"] = source_found
    paths_pd["sink_found"] = sink_found
    paths_pd["stitched_mcp"] = paths_pd["sink_mcp"] - paths_pd["source_mcp"]
    paths_pd["diff"] = paths_pd["stitched_mcp"] - paths_pd["path_mcp"]
    paths_pd["abs_diff"] = paths_pd["diff"].abs()

    # ── 5. Analyze results ────────────────────────────────────────────────────
    both_found = paths_pd["source_found"] & paths_pd["sink_found"]
    n_both_found = int(both_found.sum())
    n_missing_source = int((~paths_pd["source_found"]).sum())
    n_missing_sink = int((~paths_pd["sink_found"]).sum())
    n_either_missing = int((~both_found).sum())

    print(f"\n  Node lookup results:")
    print(f"    Both nodes found:     {n_both_found:,} / {n_unique_paths:,} ({100*n_both_found/n_unique_paths:.1f}%)")
    print(f"    Missing source node:  {n_missing_source:,}")
    print(f"    Missing sink node:    {n_missing_sink:,}")

    # Show some missing node examples
    if n_missing_source > 0:
        missing_src_nodes = paths_pd[~paths_pd["source_found"]]["source_id"].unique()[:5]
        print(f"    Sample missing source nodes: {list(missing_src_nodes)}")
    if n_missing_sink > 0:
        missing_snk_nodes = paths_pd[~paths_pd["sink_found"]]["sink_id"].unique()[:5]
        print(f"    Sample missing sink nodes: {list(missing_snk_nodes)}")

    if n_both_found == 0:
        print("  SKIP: no paths with both nodes found")
        continue

    matched = paths_pd[both_found].copy()

    exact_match = int((matched["abs_diff"] < TOLERANCE_ABS).sum())
    close_match_01 = int((matched["abs_diff"] < 0.1).sum())
    close_match_1 = int((matched["abs_diff"] < 1.0).sum())

    print(f"\n  Match analysis (where both nodes found):")
    print(f"    Exact match (<{TOLERANCE_ABS}):  {exact_match:,} / {n_both_found:,} ({100*exact_match/n_both_found:.2f}%)")
    print(f"    Close match (<0.1):   {close_match_01:,} / {n_both_found:,} ({100*close_match_01/n_both_found:.2f}%)")
    print(f"    Close match (<1.0):   {close_match_1:,} / {n_both_found:,} ({100*close_match_1/n_both_found:.2f}%)")

    print(f"\n  Difference statistics (stitched - path):")
    print(f"    Mean:   {matched['diff'].mean():.6f}")
    print(f"    Median: {matched['diff'].median():.6f}")
    print(f"    Std:    {matched['diff'].std():.6f}")
    print(f"    Min:    {matched['diff'].min():.6f}")
    print(f"    Max:    {matched['diff'].max():.6f}")
    print(f"    P1:     {matched['diff'].quantile(0.01):.6f}")
    print(f"    P99:    {matched['diff'].quantile(0.99):.6f}")

    # Show sample mismatches if any
    mismatches = matched[matched["abs_diff"] >= TOLERANCE_ABS].sort_values("abs_diff", ascending=False)
    n_mismatches = len(mismatches)
    if n_mismatches > 0:
        print(f"\n  TOP MISMATCHES ({n_mismatches:,} total with abs_diff >= {TOLERANCE_ABS}):")
        top = mismatches.head(15)
        for _, row in top.iterrows():
            print(f"    {row['class_type']:8s} | src={row['source_id']:30s} snk={row['sink_id']:30s} | "
                  f"path={row['path_mcp']:10.4f}  stitched={row['stitched_mcp']:10.4f}  "
                  f"diff={row['diff']:10.4f}")

        # Analyze mismatch patterns
        print(f"\n  Mismatch distribution by class_type:")
        for ct in sorted(mismatches["class_type"].unique()):
            ct_mm = mismatches[mismatches["class_type"] == ct]
            print(f"    {ct}: {len(ct_mm):,} mismatches, mean abs_diff = {ct_mm['abs_diff'].mean():.4f}, "
                  f"max abs_diff = {ct_mm['abs_diff'].max():.4f}")

        # Check: are the mismatched paths ones that had multiple auction entries?
        if len(multi_mcp) > 0:
            print(f"\n  Checking overlap of mismatches with multi-MCP paths...")
            multi_set = set(
                (str(r["class_type"]), r["source_id"], r["sink_id"])
                for r in multi_mcp.iter_rows(named=True)
            )
            n_overlap = sum(
                1 for _, r in mismatches.iterrows()
                if (r["class_type"], r["source_id"], r["sink_id"]) in multi_set
            )
            print(f"    {n_overlap} / {n_mismatches} mismatches are multi-MCP paths")
    else:
        print(f"\n  ALL {n_both_found:,} PATHS MATCH EXACTLY (diff < {TOLERANCE_ABS})!")

    # Store summary
    all_results.append({
        "month": month_str,
        "n_unique_paths": n_unique_paths,
        "n_both_found": n_both_found,
        "n_missing": n_either_missing,
        "exact_match_rate": 100 * exact_match / n_both_found if n_both_found > 0 else 0,
        "mean_abs_diff": float(matched["abs_diff"].mean()),
        "max_abs_diff": float(matched["abs_diff"].max()),
        "n_mismatches": n_mismatches,
    })

    # Cleanup
    del paths_pd, matched, mismatches, nodal_f0, mcp_df
    gc.collect()

    print(f"\n  [after {month_str}] memory: {mem_mb():.0f} MB")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY ACROSS ALL MONTHS")
print(f"{'='*80}")
print(f"{'Month':>10s} | {'Paths':>7s} | {'Found':>7s} | {'Missing':>7s} | {'Match%':>8s} | {'MeanDiff':>10s} | {'MaxDiff':>10s} | {'Mismatches':>10s}")
print("-" * 95)
for r in all_results:
    print(f"{r['month']:>10s} | {r['n_unique_paths']:>7,d} | {r['n_both_found']:>7,d} | {r['n_missing']:>7,d} | "
          f"{r['exact_match_rate']:>7.2f}% | {r['mean_abs_diff']:>10.6f} | {r['max_abs_diff']:>10.4f} | {r['n_mismatches']:>10,d}")

# Overall
if all_results:
    total_found = sum(r["n_both_found"] for r in all_results)
    total_mismatches = sum(r["n_mismatches"] for r in all_results)
    total_paths = sum(r["n_unique_paths"] for r in all_results)
    total_missing = sum(r["n_missing"] for r in all_results)
    overall_match_pct = 100 * (total_found - total_mismatches) / total_found if total_found > 0 else 0
    avg_mean_diff = np.mean([r["mean_abs_diff"] for r in all_results])
    max_max_diff = max(r["max_abs_diff"] for r in all_results)
    print("-" * 95)
    print(f"{'TOTAL':>10s} | {total_paths:>7,d} | {total_found:>7,d} | {total_missing:>7,d} | "
          f"{overall_match_pct:>7.2f}% | {avg_mean_diff:>10.6f} | {max_max_diff:>10.4f} | {total_mismatches:>10,d}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
if all_results:
    if overall_match_pct > 99.9:
        print("RESULT: Nodal f0 stitching (sink - source) matches path MCP with near-perfect accuracy.")
        print(f"        Overall match rate: {overall_match_pct:.2f}% across {total_found:,} paths in {len(all_results)} months.")
    elif overall_match_pct > 95.0:
        print(f"RESULT: Mostly matching. {overall_match_pct:.2f}% of paths match exactly.")
        print(f"        {total_mismatches:,} mismatches out of {total_found:,} paths.")
    else:
        print(f"RESULT: Significant mismatches detected. Only {overall_match_pct:.2f}% match.")
        print(f"        {total_mismatches:,} mismatches out of {total_found:,} paths.")

# ── Ray shutdown ──────────────────────────────────────────────────────────────
del f0_paths_all
gc.collect()

import ray
ray.shutdown()

print(f"\n[final] memory: {mem_mb():.0f} MB")
print("\nDone.")
