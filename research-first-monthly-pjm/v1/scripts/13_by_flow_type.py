"""
Segment annual paths by flow type (positive vs negative annual MCP)
and compute monthly MCP distributions for each group.

Approach:
1. Load annual R4 cleared paths (get source/sink + annual MCP per path)
2. Load f0-f11 node MCPs from June monthly auction
3. Compute path_fx = sink_fx - source_fx for each annual path
4. Split by sign of annual MCP
5. Show distribution for each group
"""
import os
import gc
import resource
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import pandas as pd
import numpy as np
from pbase.data.dataset.ftr.mcp.pjm import PjmMcp
from pbase.data.dataset.ftr.cleared.pjm import PjmClearedFtrs
from pbase.data.const.market.auction_type import AuctionType

mcp_loader = PjmMcp()
cleared_loader = PjmClearedFtrs()
CLASS_TYPE = '24h'
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

YEARS = [2020, 2021, 2022]

for year in YEARS:
    print(f"\n{'#' * 70}")
    print(f"# PY {year}")
    print(f"{'#' * 70}")

    auction_month = f'{year}-06-01'

    # --- Step 1: Load annual R4 node MCPs ---
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)
    ][['pnode_id', 'mcp']].copy()
    annual_node_mcp = annual_r4.set_index('pnode_id')['mcp']
    print(f"Annual R4 nodes: {len(annual_node_mcp)}")

    # --- Step 2: Load monthly f0-f11 node MCPs ---
    monthly_node_mcp = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == CLASS_TYPE][['pnode_id', 'mcp']].copy()
        monthly_node_mcp[fx] = df_24h.set_index('pnode_id')['mcp']

    # --- Step 3: Get annual cleared paths ---
    # Load cleared FTRs from annual auction (use period_type='a' for annual)
    # The annual cleared paths define the universe of paths we care about
    cleared = cleared_loader._load_annual(planning_year=year)
    print(f"Annual cleared raw rows: {len(cleared)}")
    print(f"Columns: {list(cleared.columns)}")

    # Filter to 24h, get unique paths with their IDs
    if 'class_type' in cleared.columns:
        cleared_24h = cleared[cleared['class_type'] == CLASS_TYPE]
    else:
        cleared_24h = cleared
    print(f"After 24h filter: {len(cleared_24h)}")

    # Get unique paths
    path_cols = ['source_name', 'sink_name', 'source_id', 'sink_id']
    if all(c in cleared_24h.columns for c in path_cols):
        paths = cleared_24h[path_cols].drop_duplicates()
    else:
        print(f"  Missing columns! Available: {list(cleared_24h.columns)}")
        continue

    print(f"Unique annual paths (24h): {len(paths)}")

    # --- Step 4: Compute annual path MCP and monthly path MCPs ---
    records = []
    for _, row in paths.iterrows():
        src_id = row['source_id']
        snk_id = row['sink_id']

        # Annual path MCP
        if src_id not in annual_node_mcp.index or snk_id not in annual_node_mcp.index:
            continue
        ann_path_mcp = annual_node_mcp[snk_id] - annual_node_mcp[src_id]

        # Monthly path MCPs
        fx_mcps = []
        valid = True
        for fx in range(12):
            if src_id not in monthly_node_mcp[fx].index or snk_id not in monthly_node_mcp[fx].index:
                valid = False
                break
            fx_mcps.append(monthly_node_mcp[fx][snk_id] - monthly_node_mcp[fx][src_id])

        if not valid:
            continue

        fx_sum = sum(fx_mcps)
        records.append({
            'source': row['source_name'],
            'sink': row['sink_name'],
            'annual_mcp': ann_path_mcp,
            'monthly_sum': fx_sum,
            **{f'mcp_f{fx}': fx_mcps[fx] for fx in range(12)},
        })

    df_paths = pd.DataFrame(records)
    print(f"Paths with valid data: {len(df_paths)}")

    # --- Step 5: Split by annual MCP sign ---
    pos_paths = df_paths[df_paths['annual_mcp'] > 100].copy()  # positive MTM
    neg_paths = df_paths[df_paths['annual_mcp'] < -100].copy()  # negative MTM

    print(f"  Positive annual MCP (>100): {len(pos_paths)} paths")
    print(f"  Negative annual MCP (<-100): {len(neg_paths)} paths")
    print(f"  Near-zero (skipped): {len(df_paths) - len(pos_paths) - len(neg_paths)} paths")

    # --- Step 6: Compute distributions ---
    for label, subset in [('POSITIVE MTM (annual > 0)', pos_paths),
                           ('NEGATIVE MTM (annual < 0)', neg_paths)]:
        if len(subset) == 0:
            print(f"\n  {label}: no paths")
            continue

        # Filter paths with near-zero monthly sum (noisy %)
        subset = subset[subset['monthly_sum'].abs() > 100].copy()

        # Compute pct for each path
        for fx in range(12):
            subset[f'pct_f{fx}'] = subset[f'mcp_f{fx}'] / subset['monthly_sum'] * 100

        print(f"\n  --- {label} ({len(subset)} paths) ---")
        print(f"  {'':5} {'Month':<5} {'Median':>8} {'Mean':>8} {'P25':>8} {'P75':>8}")
        for fx in range(12):
            col = f'pct_f{fx}'
            vals = subset[col]
            print(f"  f{fx:<4} {cal_months[fx]:<5} {vals.median():>7.1f}% {vals.mean():>7.1f}% {vals.quantile(0.25):>7.1f}% {vals.quantile(0.75):>7.1f}%")

        # Also show the "total MCP weighted" approach (sum across all paths)
        total_per_fx = [subset[f'mcp_f{fx}'].sum() for fx in range(12)]
        grand_total = sum(total_per_fx)
        if abs(grand_total) > 1:
            print(f"\n  Aggregate (value-weighted):")
            for fx in range(12):
                pct = total_per_fx[fx] / grand_total * 100
                print(f"    f{fx} ({cal_months[fx]}): {pct:>6.1f}%")

    del annual, cleared, df_paths
    gc.collect()
    print(f"\n  Memory: {mem_mb():.0f} MB")

# ================================================================
# Cross-year summary
# ================================================================
print(f"\n{'#' * 70}")
print("# CROSS-YEAR SUMMARY")
print(f"{'#' * 70}")

# Re-run to collect cross-year data
summary = {}
for year in YEARS:
    auction_month = f'{year}-06-01'

    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)
    ][['pnode_id', 'mcp']].copy()
    annual_node = annual_r4.set_index('pnode_id')['mcp']

    monthly_node = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == CLASS_TYPE][['pnode_id', 'mcp']].copy()
        monthly_node[fx] = df_24h.set_index('pnode_id')['mcp']

    cleared = cleared_loader._load_annual(planning_year=year)
    if 'class_type' in cleared.columns:
        cleared = cleared[cleared['class_type'] == CLASS_TYPE]
    paths = cleared[['source_name', 'sink_name', 'source_id', 'sink_id']].drop_duplicates()

    records = []
    for _, row in paths.iterrows():
        src_id, snk_id = row['source_id'], row['sink_id']
        if src_id not in annual_node.index or snk_id not in annual_node.index:
            continue
        ann_mcp = annual_node[snk_id] - annual_node[src_id]
        fx_mcps = []
        valid = True
        for fx in range(12):
            if src_id not in monthly_node[fx].index or snk_id not in monthly_node[fx].index:
                valid = False
                break
            fx_mcps.append(monthly_node[fx][snk_id] - monthly_node[fx][src_id])
        if not valid:
            continue
        fx_sum = sum(fx_mcps)
        records.append({'annual_mcp': ann_mcp, 'monthly_sum': fx_sum,
                        **{f'mcp_f{fx}': fx_mcps[fx] for fx in range(12)}})

    df_all = pd.DataFrame(records)

    for sign_label, mask in [('pos', df_all['annual_mcp'] > 100),
                              ('neg', df_all['annual_mcp'] < -100)]:
        subset = df_all[mask & (df_all['monthly_sum'].abs() > 100)].copy()
        if len(subset) == 0:
            continue
        # Value-weighted aggregate
        totals = [subset[f'mcp_f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        pcts = [t / grand * 100 for t in totals]
        # Median
        for fx in range(12):
            subset[f'pct_f{fx}'] = subset[f'mcp_f{fx}'] / subset['monthly_sum'] * 100
        medians = [subset[f'pct_f{fx}'].median() for fx in range(12)]
        summary[(year, sign_label)] = {'agg': pcts, 'median': medians, 'n': len(subset)}

    del annual, cleared, df_all
    gc.collect()

# Print cross-year table
for sign_label, title in [('pos', 'POSITIVE MTM paths'), ('neg', 'NEGATIVE MTM paths')]:
    print(f"\n--- {title} (value-weighted aggregate) ---")
    print(f"{'':5} {'Month':<5}", end='')
    for year in YEARS:
        print(f"  PY{year}", end='')
    print(f"  {'3yr avg':>8}")

    for fx in range(12):
        print(f"f{fx:<4} {cal_months[fx]:<5}", end='')
        vals = []
        for year in YEARS:
            key = (year, sign_label)
            if key in summary:
                v = summary[key]['agg'][fx]
                vals.append(v)
                print(f"  {v:>5.1f}%", end='')
            else:
                print(f"  {'N/A':>6}", end='')
        if vals:
            print(f"  {np.mean(vals):>7.1f}%", end='')
        print()

    # Path counts
    print(f"  n: ", end='')
    for year in YEARS:
        key = (year, sign_label)
        if key in summary:
            print(f"  {summary[key]['n']:>5}", end='')
    print()

# Also show median version
for sign_label, title in [('pos', 'POSITIVE MTM paths'), ('neg', 'NEGATIVE MTM paths')]:
    print(f"\n--- {title} (median across paths) ---")
    print(f"{'':5} {'Month':<5}", end='')
    for year in YEARS:
        print(f"  PY{year}", end='')
    print(f"  {'3yr avg':>8}")

    for fx in range(12):
        print(f"f{fx:<4} {cal_months[fx]:<5}", end='')
        vals = []
        for year in YEARS:
            key = (year, sign_label)
            if key in summary:
                v = summary[key]['median'][fx]
                vals.append(v)
                print(f"  {v:>5.1f}%", end='')
            else:
                print(f"  {'N/A':>6}", end='')
        if vals:
            print(f"  {np.mean(vals):>7.1f}%", end='')
        print()

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
