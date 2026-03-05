"""
Full analysis: PJM June monthly auction MCP distribution (f0-f11).

For each year (2020-2022), load all f0-f11 MCPs from the June monthly auction,
compute % distribution, and compare to annual MCP.
Focus on 24h class_type for path-level analysis.
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

mcp_loader = PjmMcp()

YEARS = [2020, 2021, 2022]
CLASS_TYPE = '24h'

results = {}

for year in YEARS:
    print(f"\n{'='*60}")
    print(f"Processing PY {year} (Jun {year} - May {year+1})")
    print(f"{'='*60}")

    # Load annual auction, last round (round 4), 24h
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)
    ][['node_name', 'pnode_id', 'mcp']].copy()
    annual_r4 = annual_r4.rename(columns={'mcp': 'annual_mcp'})
    print(f"  Annual R4 24h: {len(annual_r4)} nodes")

    # Load all f0-f11 for June monthly auction
    auction_month = f'{year}-06-01'
    monthly_all = []
    for fx in range(12):
        pt = f'f{fx}'
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=pt,
        )
        df_24h = df[df['class_type'] == CLASS_TYPE][['node_name', 'pnode_id', 'mcp']].copy()
        df_24h = df_24h.rename(columns={'mcp': f'mcp_{pt}'})
        monthly_all.append(df_24h)
        print(f"    {pt}: {len(df_24h)} nodes, median MCP={df_24h[f'mcp_{pt}'].median():.1f}")

    # Merge all f0-f11 on node_name + pnode_id
    merged = monthly_all[0]
    for i in range(1, 12):
        merged = merged.merge(monthly_all[i], on=['node_name', 'pnode_id'], how='inner')

    print(f"  Merged monthly: {len(merged)} nodes (inner join)")

    # Compute sum of f0-f11
    fx_cols = [f'mcp_f{i}' for i in range(12)]
    merged['sum_monthly'] = merged[fx_cols].sum(axis=1)

    # Compute percentage for each fx
    for fx in range(12):
        col = f'mcp_f{fx}'
        merged[f'pct_f{fx}'] = merged[col] / merged['sum_monthly']

    # Merge with annual
    merged = merged.merge(annual_r4, on=['node_name', 'pnode_id'], how='inner')
    merged['ratio_monthly_to_annual'] = merged['sum_monthly'] / merged['annual_mcp']

    print(f"  After merge with annual: {len(merged)} nodes")

    # Filter out nodes where annual_mcp is too small (noise)
    # and where sum_monthly is near zero (division issues)
    mask = (merged['annual_mcp'].abs() > 100) & (merged['sum_monthly'].abs() > 100)
    filtered = merged[mask].copy()
    print(f"  After filtering (|annual|>100 & |sum|>100): {len(filtered)} nodes")

    results[year] = filtered

    # Print summary statistics
    pct_cols = [f'pct_f{i}' for i in range(12)]
    print(f"\n  --- Percentage Distribution (median across nodes) ---")
    for fx in range(12):
        col = f'pct_f{fx}'
        med = filtered[col].median() * 100
        mean = filtered[col].mean() * 100
        print(f"    f{fx}: median={med:.1f}%, mean={mean:.1f}%")

    print(f"\n  --- Monthly/Annual ratio ---")
    print(f"    Median: {filtered['ratio_monthly_to_annual'].median():.4f}")
    print(f"    Mean: {filtered['ratio_monthly_to_annual'].mean():.4f}")
    print(f"    Std: {filtered['ratio_monthly_to_annual'].std():.4f}")

    del annual, monthly_all
    gc.collect()
    print(f"  Memory: {mem_mb():.0f} MB")

# --- Cross-year summary ---
print(f"\n{'='*60}")
print("CROSS-YEAR SUMMARY")
print(f"{'='*60}")

# Map f0-f11 to calendar months for June auction
# f0=Jun, f1=Jul, f2=Aug, f3=Sep, f4=Oct, f5=Nov, f6=Dec, f7=Jan, f8=Feb, f9=Mar, f10=Apr, f11=May
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

print(f"\n{'Period':<8} {'CalMonth':<10}", end='')
for year in YEARS:
    print(f"  PY{year}_med  PY{year}_mean", end='')
print()

all_medians = []
for fx in range(12):
    col = f'pct_f{fx}'
    print(f"f{fx:<7} {cal_months[fx]:<10}", end='')
    year_medians = []
    for year in YEARS:
        med = results[year][col].median() * 100
        mean = results[year][col].mean() * 100
        year_medians.append(med)
        print(f"  {med:>8.1f}%  {mean:>8.1f}%", end='')
    all_medians.append(year_medians)
    print()

# Average across years
print(f"\n--- Average median % across 2020-2022 ---")
avg_pcts = []
for fx in range(12):
    avg = np.mean(all_medians[fx])
    avg_pcts.append(avg)
    print(f"  f{fx} ({cal_months[fx]}): {avg:.1f}%")
print(f"  Sum: {sum(avg_pcts):.1f}%")

# Also check: is the distribution consistent when we look at positive-MCP-only paths?
print(f"\n--- Positive-sum paths only (MCP sum > 0) ---")
for year in YEARS:
    pos = results[year][results[year]['sum_monthly'] > 0]
    print(f"\n  PY{year}: {len(pos)} positive-sum paths")
    for fx in range(12):
        col = f'pct_f{fx}'
        med = pos[col].median() * 100
        print(f"    f{fx} ({cal_months[fx]}): {med:.1f}%")

# Save results to parquet for later use
for year in YEARS:
    path = f'/home/xyz/workspace/research-qianli-v2/research-first-monthly-pjm/mcp_distribution_{year}.parquet'
    results[year].to_parquet(path)
    print(f"\nSaved: {path}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
