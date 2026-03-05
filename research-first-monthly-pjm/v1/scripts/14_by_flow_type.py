"""
Segment annual paths by flow type (positive vs negative annual MCP)
and compute monthly MCP distributions for each group.

Fixed: class_type='24H' (uppercase) in cleared data, match by node_name.
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

mcp_loader = PjmMcp()
cleared_loader = PjmClearedFtrs()
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

YEARS = [2020, 2021, 2022]
summary = {}

for year in YEARS:
    print(f"\n{'#' * 70}")
    print(f"# PY {year}")
    print(f"{'#' * 70}")

    auction_month = f'{year}-06-01'

    # --- Load annual R4 node MCPs (24h lowercase) ---
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == '24h')
    ][['node_name', 'mcp']].copy()
    annual_node_mcp = annual_r4.drop_duplicates('node_name').set_index('node_name')['mcp']
    print(f"Annual R4 24h nodes: {len(annual_node_mcp)}")

    # --- Load monthly f0-f11 node MCPs ---
    monthly_node_mcp = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == '24h'][['node_name', 'mcp']].copy()
        monthly_node_mcp[fx] = df_24h.drop_duplicates('node_name').set_index('node_name')['mcp']

    # --- Load annual cleared paths (24H uppercase, R4) ---
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_24h = cleared[
        (cleared['class_type'] == '24H') & (cleared['market_round'] == 4)
    ]
    print(f"Annual cleared R4 24H rows: {len(cleared_24h)}")

    # Get unique paths
    paths = cleared_24h[['source_name', 'sink_name']].drop_duplicates()
    print(f"Unique annual paths: {len(paths)}")

    # --- Compute annual and monthly path MCPs ---
    records = []
    skipped = 0
    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']

        # Annual path MCP
        if src not in annual_node_mcp.index or snk not in annual_node_mcp.index:
            skipped += 1
            continue
        ann_path_mcp = annual_node_mcp[snk] - annual_node_mcp[src]

        # Monthly path MCPs
        fx_mcps = []
        valid = True
        for fx in range(12):
            if src not in monthly_node_mcp[fx].index or snk not in monthly_node_mcp[fx].index:
                valid = False
                break
            fx_mcps.append(monthly_node_mcp[fx][snk] - monthly_node_mcp[fx][src])

        if not valid:
            skipped += 1
            continue

        fx_sum = sum(fx_mcps)
        records.append({
            'source': src, 'sink': snk,
            'annual_mcp': ann_path_mcp, 'monthly_sum': fx_sum,
            **{f'mcp_f{fx}': fx_mcps[fx] for fx in range(12)},
        })

    df_paths = pd.DataFrame(records)
    print(f"Paths with valid data: {len(df_paths)} (skipped {skipped})")

    # --- Split by annual MCP sign ---
    pos = df_paths[(df_paths['annual_mcp'] > 100) & (df_paths['monthly_sum'].abs() > 100)].copy()
    neg = df_paths[(df_paths['annual_mcp'] < -100) & (df_paths['monthly_sum'].abs() > 100)].copy()

    print(f"  Positive annual MCP: {len(pos)} paths")
    print(f"  Negative annual MCP: {len(neg)} paths")

    for sign_label, subset, title in [
        ('pos', pos, 'POSITIVE MTM (annual > 0)'),
        ('neg', neg, 'NEGATIVE MTM (annual < 0)'),
    ]:
        if len(subset) == 0:
            continue

        # Compute pct
        for fx in range(12):
            subset[f'pct_f{fx}'] = subset[f'mcp_f{fx}'] / subset['monthly_sum'] * 100

        # Value-weighted aggregate
        totals = [subset[f'mcp_f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        agg_pcts = [t / grand * 100 for t in totals] if abs(grand) > 1 else [0]*12

        # Medians
        medians = [subset[f'pct_f{fx}'].median() for fx in range(12)]

        summary[(year, sign_label)] = {
            'agg': agg_pcts, 'median': medians, 'n': len(subset)
        }

        print(f"\n  --- {title} ({len(subset)} paths) ---")
        print(f"  {'':5} {'Month':<5} {'Agg':>8} {'Median':>8} {'P25':>8} {'P75':>8}")
        for fx in range(12):
            col = f'pct_f{fx}'
            print(f"  f{fx:<4} {cal_months[fx]:<5} {agg_pcts[fx]:>7.1f}% {subset[col].median():>7.1f}% {subset[col].quantile(0.25):>7.1f}% {subset[col].quantile(0.75):>7.1f}%")

    del annual, cleared, df_paths
    gc.collect()
    print(f"\n  Memory: {mem_mb():.0f} MB")

# ================================================================
# Cross-year summary
# ================================================================
print(f"\n{'#' * 70}")
print("# CROSS-YEAR SUMMARY")
print(f"{'#' * 70}")

for sign_label, title in [('pos', 'POSITIVE MTM paths (annual > 0)'),
                            ('neg', 'NEGATIVE MTM paths (annual < 0)')]:
    print(f"\n--- {title} ---")

    # Value-weighted
    print(f"\n  Value-weighted aggregate:")
    print(f"  {'':5} {'Month':<5}", end='')
    for year in YEARS:
        print(f"  PY{year}", end='')
    print(f"  {'3yr avg':>8}")

    for fx in range(12):
        print(f"  f{fx:<4} {cal_months[fx]:<5}", end='')
        vals = []
        for year in YEARS:
            key = (year, sign_label)
            if key in summary:
                v = summary[key]['agg'][fx]
                vals.append(v)
                print(f"  {v:>5.1f}%", end='')
            else:
                print(f"    N/A", end='')
        if vals:
            print(f"  {np.mean(vals):>7.1f}%", end='')
        print()

    counts = [summary.get((y, sign_label), {}).get('n', 0) for y in YEARS]
    print(f"  n paths: {counts}")

    # Median
    print(f"\n  Median across paths:")
    print(f"  {'':5} {'Month':<5}", end='')
    for year in YEARS:
        print(f"  PY{year}", end='')
    print(f"  {'3yr avg':>8}")

    for fx in range(12):
        print(f"  f{fx:<4} {cal_months[fx]:<5}", end='')
        vals = []
        for year in YEARS:
            key = (year, sign_label)
            if key in summary:
                v = summary[key]['median'][fx]
                vals.append(v)
                print(f"  {v:>5.1f}%", end='')
            else:
                print(f"    N/A", end='')
        if vals:
            print(f"  {np.mean(vals):>7.1f}%", end='')
        print()

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
