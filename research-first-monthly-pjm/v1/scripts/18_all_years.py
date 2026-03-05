"""
Full analysis: PY 2017-2025 (9 years).
Same methodology as script 16, extended to all available years.
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

YEARS = list(range(2017, 2026))
summary = {}

for year in YEARS:
    print(f"\n--- PY {year} ---")
    auction_month = f'{year}-06-01'

    # Annual R4 node MCPs
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == '24h')
    ][['node_name', 'mcp']].copy()
    annual_node = annual_r4.drop_duplicates('node_name').set_index('node_name')['mcp']

    # Monthly f0-f11 node MCPs
    monthly_node = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == '24h'][['node_name', 'mcp']].copy()
        monthly_node[fx] = df_24h.drop_duplicates('node_name').set_index('node_name')['mcp']

    # Annual cleared Obligation 24H paths (all rounds)
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'] == '24H') & (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()

    # Compute path MCPs
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
            'annual_mcp': ann_path, 'monthly_sum': sum(fx_vals),
            **{f'mcp_f{fx}': fx_vals[fx] for fx in range(12)},
        })

    df_paths = pd.DataFrame(records)

    for sign_label, mask in [
        ('pos', df_paths['annual_mcp'] > 100),
        ('neg', df_paths['annual_mcp'] < -100),
    ]:
        subset = df_paths[mask & (df_paths['monthly_sum'].abs() > 100)].copy()
        if len(subset) == 0:
            continue

        # Value-weighted aggregate
        totals = [subset[f'mcp_f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        agg_pcts = [t / grand * 100 for t in totals]

        # Median
        for fx in range(12):
            subset[f'pct_f{fx}'] = subset[f'mcp_f{fx}'] / subset['monthly_sum'] * 100
        medians = [subset[f'pct_f{fx}'].median() for fx in range(12)]

        summary[(year, sign_label)] = {
            'agg': agg_pcts, 'median': medians, 'n': len(subset)
        }

    n_pos = summary.get((year, 'pos'), {}).get('n', 0)
    n_neg = summary.get((year, 'neg'), {}).get('n', 0)
    print(f"  Paths: {len(df_paths)} total, {n_pos} pos, {n_neg} neg")

    del annual, cleared, df_paths
    gc.collect()
    print(f"  Memory: {mem_mb():.0f} MB")

# ================================================================
# CROSS-YEAR SUMMARY
# ================================================================
print(f"\n{'#' * 70}")
print("# CROSS-YEAR SUMMARY (PY 2017-2025)")
print(f"{'#' * 70}")

for sign_label, title in [('pos', 'POSITIVE MTM (annual > 0)'),
                            ('neg', 'NEGATIVE MTM (annual < 0)')]:
    print(f"\n{'=' * 70}")
    print(f"{title} — Value-weighted aggregate")
    print(f"{'=' * 70}")

    # Print year-by-year table
    print(f"\n  {'':5} {'Mon':<4}", end='')
    for year in YEARS:
        print(f" {str(year)[-2:]:>5}", end='')
    print(f"  {'Avg':>6} {'Std':>5}")

    all_year_aggs = {}
    for year in YEARS:
        key = (year, sign_label)
        if key in summary:
            all_year_aggs[year] = summary[key]['agg']

    for fx in range(12):
        print(f"  f{fx:<4} {cal_months[fx]:<4}", end='')
        vals = []
        for year in YEARS:
            if year in all_year_aggs:
                v = all_year_aggs[year][fx]
                vals.append(v)
                print(f" {v:>4.1f}%", end='')
            else:
                print(f"   N/A", end='')
        if vals:
            print(f"  {np.mean(vals):>5.1f}% {np.std(vals):>4.1f}%", end='')
        print()

    # Path counts
    print(f"  n:", end='')
    for year in YEARS:
        n = summary.get((year, sign_label), {}).get('n', 0)
        print(f" {n:>5}", end='')
    print()

    # Also show median version
    print(f"\n  Median across paths:")
    print(f"  {'':5} {'Mon':<4}", end='')
    for year in YEARS:
        print(f" {str(year)[-2:]:>5}", end='')
    print(f"  {'Avg':>6} {'Std':>5}")

    all_year_meds = {}
    for year in YEARS:
        key = (year, sign_label)
        if key in summary:
            all_year_meds[year] = summary[key]['median']

    for fx in range(12):
        print(f"  f{fx:<4} {cal_months[fx]:<4}", end='')
        vals = []
        for year in YEARS:
            if year in all_year_meds:
                v = all_year_meds[year][fx]
                vals.append(v)
                print(f" {v:>4.1f}%", end='')
            else:
                print(f"   N/A", end='')
        if vals:
            print(f"  {np.mean(vals):>5.1f}% {np.std(vals):>4.1f}%", end='')
        print()

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
