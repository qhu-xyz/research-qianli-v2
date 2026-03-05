"""
FINAL ANALYSIS: PJM June Monthly Auction MCP Distribution by Flow Type.

Fixes from audit:
- Use ALL annual rounds (1-4) for path universe (not just R4)
- Filter to Obligation hedge_type only (Options have different pricing)
- Use annual R4 node MCPs for computing annual path MCP (R4 = cumulative final)
- Verify obligation MCPs match sink_node - source_node

Output: percentage distribution of monthly MCPs for positive and negative MTM paths.
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

    # --- Load annual R4 obligation node MCPs ---
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == '24h')
    ][['node_name', 'mcp']].copy()
    annual_node = annual_r4.drop_duplicates('node_name').set_index('node_name')['mcp']
    print(f"Annual R4 24h nodes: {len(annual_node)}")

    # --- Load monthly f0-f11 obligation node MCPs ---
    monthly_node = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == '24h'][['node_name', 'mcp']].copy()
        monthly_node[fx] = df_24h.drop_duplicates('node_name').set_index('node_name')['mcp']

    # --- Load annual cleared paths: ALL rounds, Obligation only, 24H ---
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'] == '24H') &
        (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()
    print(f"Annual cleared Obligation 24H paths (all rounds): {len(paths)}")

    # --- Quick sanity: verify cleared.mcp matches node computation for obligations ---
    sample = cleared_filt.head(50)
    match_count = 0
    for _, row in sample.iterrows():
        s, k = row['source_name'], row['sink_name']
        if s in annual_node.index and k in annual_node.index:
            computed = annual_node[k] - annual_node[s]
            # Note: cleared.mcp is for the specific round, not necessarily R4
            # So we only check R4 cleared trades
            if row['market_round'] == 4:
                if abs(row['mcp'] - computed) < 0.1:
                    match_count += 1
    r4_count = (sample['market_round'] == 4).sum()
    print(f"  Sanity: {match_count}/{r4_count} R4 obligation MCPs match node computation")

    # --- Compute path MCPs ---
    records = []
    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']

        if src not in annual_node.index or snk not in annual_node.index:
            continue
        ann_path = annual_node[snk] - annual_node[src]

        fx_vals = []
        valid = True
        for fx in range(12):
            if src not in monthly_node[fx].index or snk not in monthly_node[fx].index:
                valid = False
                break
            fx_vals.append(monthly_node[fx][snk] - monthly_node[fx][src])
        if not valid:
            continue

        records.append({
            'source': src, 'sink': snk,
            'annual_mcp': ann_path, 'monthly_sum': sum(fx_vals),
            **{f'mcp_f{fx}': fx_vals[fx] for fx in range(12)},
        })

    df_paths = pd.DataFrame(records)
    print(f"Paths with valid data: {len(df_paths)}")

    # --- Split by annual MCP sign ---
    for sign_label, mask, title in [
        ('pos', df_paths['annual_mcp'] > 100, 'POSITIVE MTM (annual > 0)'),
        ('neg', df_paths['annual_mcp'] < -100, 'NEGATIVE MTM (annual < 0)'),
    ]:
        subset = df_paths[mask & (df_paths['monthly_sum'].abs() > 100)].copy()
        if len(subset) == 0:
            continue

        for fx in range(12):
            subset[f'pct_f{fx}'] = subset[f'mcp_f{fx}'] / subset['monthly_sum'] * 100

        # Value-weighted aggregate
        totals = [subset[f'mcp_f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        agg_pcts = [t / grand * 100 for t in totals]
        medians = [subset[f'pct_f{fx}'].median() for fx in range(12)]

        summary[(year, sign_label)] = {
            'agg': agg_pcts, 'median': medians, 'n': len(subset)
        }

        print(f"\n  --- {title} ({len(subset)} paths) ---")
        print(f"  {'':5} {'Month':<5} {'Agg':>8} {'Median':>8} {'P25':>8} {'P75':>8}")
        for fx in range(12):
            col = f'pct_f{fx}'
            print(f"  f{fx:<4} {cal_months[fx]:<5} {agg_pcts[fx]:>7.1f}% {subset[col].median():>7.1f}% {subset[col].quantile(0.25):>7.1f}% {subset[col].quantile(0.75):>7.1f}%")
        print(f"  {'Sum':>11} {sum(agg_pcts):>7.1f}%")

    del annual, cleared, df_paths
    gc.collect()
    print(f"\n  Memory: {mem_mb():.0f} MB")

# ================================================================
# Cross-year summary
# ================================================================
print(f"\n{'#' * 70}")
print("# CROSS-YEAR SUMMARY")
print(f"{'#' * 70}")

for sign_label, title in [('pos', 'POSITIVE MTM (annual > 0)'),
                            ('neg', 'NEGATIVE MTM (annual < 0)')]:
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    # Value-weighted
    print(f"\n  Value-weighted aggregate:")
    print(f"  {'':5} {'Month':<5}", end='')
    for year in YEARS:
        print(f"  PY{year}", end='')
    print(f"  {'3yr avg':>8}")

    year_aggs = {}
    for year in YEARS:
        key = (year, sign_label)
        if key in summary:
            year_aggs[year] = summary[key]['agg']

    for fx in range(12):
        print(f"  f{fx:<4} {cal_months[fx]:<5}", end='')
        vals = []
        for year in YEARS:
            if year in year_aggs:
                v = year_aggs[year][fx]
                vals.append(v)
                print(f"  {v:>5.1f}%", end='')
            else:
                print(f"    N/A", end='')
        if vals:
            print(f"  {np.mean(vals):>7.1f}%", end='')
        print()

    print(f"  n: ", end='')
    for year in YEARS:
        key = (year, sign_label)
        n = summary.get(key, {}).get('n', 0)
        print(f"  {n:>5}", end='')
    print()

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
