"""
Step 4: Verify findings and refine the analysis.

1. Verify that node-level % distribution = path-level % distribution
2. Look at the distribution weighted by absolute MCP (bigger nodes matter more)
3. Check if the negative winter pattern is consistent
4. Produce a final recommended distribution
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
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# --- Approach 1: System-wide average MCP distribution ---
# Instead of computing pct per node then averaging, compute the TOTAL MCP across all nodes
# for each fx, then compute percentages. This gives us the "system average" distribution.
print("="*60)
print("APPROACH 1: System-wide total MCP distribution")
print("="*60)

for year in YEARS:
    print(f"\nPY {year}:")
    auction_month = f'{year}-06-01'

    fx_totals = []
    for fx in range(12):
        pt = f'f{fx}'
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=pt,
        )
        df_24h = df[df['class_type'] == CLASS_TYPE]
        total = df_24h['mcp'].sum()
        median = df_24h['mcp'].median()
        fx_totals.append(total)

    total_sum = sum(fx_totals)
    print(f"  {'Period':<6} {'CalMonth':<6} {'Total MCP':>14} {'Pct':>8}")
    for fx in range(12):
        pct = fx_totals[fx] / total_sum * 100
        print(f"  f{fx:<5} {cal_months[fx]:<6} {fx_totals[fx]:>14,.0f} {pct:>7.1f}%")
    print(f"  {'Sum':<6} {'':6} {total_sum:>14,.0f} {100.0:>7.1f}%")

gc.collect()
print(f"\nMemory: {mem_mb():.0f} MB")

# --- Approach 2: Verify path-level equivalence ---
# Take a few paths (source-sink pairs) and compute path MCP distribution
print(f"\n{'='*60}")
print("APPROACH 2: Path-level verification (sample paths)")
print(f"{'='*60}")

year = 2020
auction_month = '2020-06-01'

# Load all f0-f11 for 2020
monthly_dfs = {}
for fx in range(12):
    pt = f'f{fx}'
    df = mcp_loader.load_data(
        auction_month=auction_month,
        market_round=1,
        period_type=pt,
    )
    df_24h = df[df['class_type'] == CLASS_TYPE][['node_name', 'mcp']].copy()
    df_24h = df_24h.rename(columns={'mcp': f'mcp_f{fx}'})
    monthly_dfs[fx] = df_24h

# Build node-level table
node_df = monthly_dfs[0]
for fx in range(1, 12):
    node_df = node_df.merge(monthly_dfs[fx], on='node_name', how='inner')

# Pick some sample paths
sample_paths = [
    ('GREENSBURG 138 KV TR1', 'SEWARD   138 KV  TR1'),
    ('02ANGOLA138 KV  TR1', 'BAGLEY  34 KV   230-2LD'),
]

for source, sink in sample_paths:
    src = node_df[node_df['node_name'] == source]
    snk = node_df[node_df['node_name'] == sink]
    if len(src) == 0 or len(snk) == 0:
        print(f"\n  Path {source} -> {sink}: node not found, skipping")
        continue

    print(f"\n  Path: {source} -> {sink}")
    path_vals = []
    for fx in range(12):
        col = f'mcp_f{fx}'
        path_mcp = snk[col].values[0] - src[col].values[0]
        path_vals.append(path_mcp)

    path_sum = sum(path_vals)
    print(f"  {'Period':<6} {'CalMonth':<6} {'Path MCP':>10} {'Pct':>8}")
    for fx in range(12):
        pct = path_vals[fx] / path_sum * 100 if path_sum != 0 else 0
        print(f"  f{fx:<5} {cal_months[fx]:<6} {path_vals[fx]:>10.1f} {pct:>7.1f}%")
    print(f"  {'Sum':<6} {'':6} {path_sum:>10.1f}")

del monthly_dfs, node_df
gc.collect()

# --- Approach 3: Robust median distribution using MCP-hourly if available ---
# Actually, let's just use the node-level median approach but handle the sign issue.
# The key insight: negative MCPs in winter are real (congestion flows reverse in winter).
# For percentage distribution, we should use ALL nodes including negative ones.

# --- Final: Compute robust distribution ---
print(f"\n{'='*60}")
print("FINAL: Robust percentage distribution")
print(f"{'='*60}")

# Use system-wide approach: sum all node MCPs for each fx, then compute %
all_year_pcts = []
for year in YEARS:
    auction_month = f'{year}-06-01'
    fx_totals = []
    for fx in range(12):
        pt = f'f{fx}'
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=pt,
        )
        df_24h = df[df['class_type'] == CLASS_TYPE]
        fx_totals.append(df_24h['mcp'].sum())

    total = sum(fx_totals)
    pcts = [t / total * 100 for t in fx_totals]
    all_year_pcts.append(pcts)

# Average across years
avg_pcts = [np.mean([all_year_pcts[y][fx] for y in range(len(YEARS))]) for fx in range(12)]

print(f"\n{'Period':<6} {'CalMonth':<6}", end='')
for year in YEARS:
    print(f"  PY{year}", end='')
print(f"  {'Average':>8}")

for fx in range(12):
    print(f"f{fx:<5} {cal_months[fx]:<6}", end='')
    for y in range(len(YEARS)):
        print(f"  {all_year_pcts[y][fx]:>6.1f}%", end='')
    print(f"  {avg_pcts[fx]:>7.1f}%")

print(f"\n{'Sum':<13}", end='')
for y in range(len(YEARS)):
    print(f"  {sum(all_year_pcts[y]):>6.1f}%", end='')
print(f"  {sum(avg_pcts):>7.1f}%")

# Normalized to sum to exactly 100%
norm_pcts = [p / sum(avg_pcts) * 100 for p in avg_pcts]
print(f"\n--- Normalized distribution (sums to 100%) ---")
for fx in range(12):
    print(f"  f{fx} ({cal_months[fx]}): {norm_pcts[fx]:.2f}%")
print(f"  Sum: {sum(norm_pcts):.2f}%")

# Compare to flat 1/12 = 8.33%
print(f"\n--- Difference from flat (8.33%) ---")
for fx in range(12):
    diff = norm_pcts[fx] - 100/12
    print(f"  f{fx} ({cal_months[fx]}): {norm_pcts[fx]:.2f}% (diff: {diff:+.2f}%)")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
