"""
AUDIT: Verify the final analysis (script 14) is correct.

Checks:
1. Does cleared.mcp == node_mcp[sink] - node_mcp[source] for annual paths?
   (sanity check that we're computing path MCPs correctly)
2. Does the annual cleared data's market_round match what we think?
3. Are the pct computations correct for a few sample paths?
4. Does the value-weighted aggregate actually sum to 100%?
5. Reconcile with the hub-to-hub results from script 12.
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

year = 2020

# ================================================================
# CHECK 1: cleared.mcp == node_mcp[sink] - node_mcp[source]?
# ================================================================
print("=" * 70)
print("CHECK 1: Does cleared.mcp match sink_node_mcp - source_node_mcp?")
print("=" * 70)

# Load annual cleared
cleared = cleared_loader._load_annual(planning_year=year)
cleared_r4_24h = cleared[(cleared['market_round'] == 4) & (cleared['class_type'] == '24H')]
print(f"Cleared R4 24H rows: {len(cleared_r4_24h)}")
print(f"Columns: {list(cleared_r4_24h.columns)}")
print(f"\nSample cleared rows:")
print(cleared_r4_24h[['source_name', 'sink_name', 'mcp', 'market_round', 'hedge_type']].head(10).to_string())

# Load annual R4 node MCPs
annual = mcp_loader._load_annual(planning_year=year)
annual_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == '24h')]
node_mcp = annual_r4.drop_duplicates('node_name').set_index('node_name')['mcp']

# For each cleared path, compare cleared.mcp vs computed path mcp
mismatches = 0
matches = 0
errors = []
for _, row in cleared_r4_24h.head(100).iterrows():
    src, snk = row['source_name'], row['sink_name']
    cleared_mcp = row['mcp']

    if src in node_mcp.index and snk in node_mcp.index:
        computed_mcp = node_mcp[snk] - node_mcp[src]
        diff = abs(cleared_mcp - computed_mcp)
        if diff > 0.1:
            mismatches += 1
            errors.append((src, snk, cleared_mcp, computed_mcp, diff))
        else:
            matches += 1

print(f"\nFirst 100 rows: {matches} match, {mismatches} mismatch")
if errors:
    print(f"Sample mismatches:")
    for src, snk, c, n, d in errors[:5]:
        print(f"  {src} -> {snk}: cleared={c:.2f}, computed={n:.2f}, diff={d:.2f}")

# ================================================================
# CHECK 2: market_round values in cleared data
# ================================================================
print(f"\n{'=' * 70}")
print("CHECK 2: market_round values in cleared data")
print(f"{'=' * 70}")
print(f"Unique market_round: {cleared['market_round'].unique()}")
print(f"Counts per round:")
for r in sorted(cleared['market_round'].unique()):
    print(f"  Round {r}: {len(cleared[cleared['market_round'] == r])} rows")

# ================================================================
# CHECK 3: Trace a specific path end-to-end
# ================================================================
print(f"\n{'=' * 70}")
print("CHECK 3: Trace a specific path end-to-end")
print(f"{'=' * 70}")

# Pick WESTERN HUB -> DOMINION HUB
src_name = 'WESTERN HUB'
snk_name = 'DOMINION HUB'

# Annual node MCPs
ann_src = node_mcp.get(src_name, None)
ann_snk = node_mcp.get(snk_name, None)
print(f"Annual R4 node MCPs:")
print(f"  {src_name}: {ann_src:.2f}")
print(f"  {snk_name}: {ann_snk:.2f}")
print(f"  Path MCP (sink - source): {ann_snk - ann_src:.2f}")

# Check if this path appears in cleared
cleared_path = cleared_r4_24h[
    (cleared_r4_24h['source_name'] == src_name) &
    (cleared_r4_24h['sink_name'] == snk_name)
]
if len(cleared_path) > 0:
    print(f"\n  Found in cleared data: {len(cleared_path)} trades")
    print(f"  cleared.mcp: {cleared_path['mcp'].values}")
else:
    print(f"\n  NOT found in cleared data (this path was not traded in annual R4)")
    # This is OK — we use cleared paths to define the universe

# Monthly f0-f11
print(f"\nMonthly f0-f11 path MCPs:")
monthly_node = {}
for fx in range(12):
    df = mcp_loader.load_data(
        auction_month='2020-06-01', market_round=1, period_type=f'f{fx}',
    )
    df_24h = df[df['class_type'] == '24h']
    monthly_node[fx] = df_24h.drop_duplicates('node_name').set_index('node_name')['mcp']

fx_mcps = []
for fx in range(12):
    src_v = monthly_node[fx].get(src_name, None)
    snk_v = monthly_node[fx].get(snk_name, None)
    path_v = snk_v - src_v if src_v is not None and snk_v is not None else None
    fx_mcps.append(path_v)
    print(f"  f{fx} ({cal_months[fx]}): src={src_v:.1f}, snk={snk_v:.1f}, path={path_v:.1f}")

fx_sum = sum(fx_mcps)
print(f"\n  Sum(f0..f11): {fx_sum:.1f}")
print(f"  Annual path MCP: {ann_snk - ann_src:.1f}")
print(f"  Ratio: {fx_sum / (ann_snk - ann_src):.4f}")

print(f"\n  Percentage distribution:")
for fx in range(12):
    pct = fx_mcps[fx] / fx_sum * 100
    print(f"    f{fx} ({cal_months[fx]}): {fx_mcps[fx]:.1f} = {pct:.1f}%")
print(f"  Sum of pcts: {sum(m/fx_sum*100 for m in fx_mcps):.1f}%")

# ================================================================
# CHECK 4: Value-weighted aggregate sums to 100%?
# ================================================================
print(f"\n{'=' * 70}")
print("CHECK 4: Verify aggregate computation")
print(f"{'=' * 70}")

# Rebuild the positive MTM paths for PY2020 and verify
paths = cleared_r4_24h[['source_name', 'sink_name']].drop_duplicates()
records = []
for _, row in paths.iterrows():
    src, snk = row['source_name'], row['sink_name']
    if src not in node_mcp.index or snk not in node_mcp.index:
        continue
    ann_path = node_mcp[snk] - node_mcp[src]

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
        'annual_mcp': ann_path,
        'monthly_sum': sum(fx_vals),
        **{f'mcp_f{fx}': fx_vals[fx] for fx in range(12)},
    })

df_all = pd.DataFrame(records)
pos = df_all[(df_all['annual_mcp'] > 100) & (df_all['monthly_sum'].abs() > 100)]

print(f"Positive MTM paths: {len(pos)}")
totals = [pos[f'mcp_f{fx}'].sum() for fx in range(12)]
grand = sum(totals)
pcts = [t / grand * 100 for t in totals]
print(f"Sum of percentages: {sum(pcts):.6f}%")
print(f"Grand total: {grand:.2f}")

print(f"\nValue-weighted aggregate for PY2020 positive paths:")
for fx in range(12):
    print(f"  f{fx} ({cal_months[fx]}): total={totals[fx]:>12.1f}, pct={pcts[fx]:>6.1f}%")

# ================================================================
# CHECK 5: Reconcile with hub-to-hub results from script 12
# ================================================================
print(f"\n{'=' * 70}")
print("CHECK 5: Reconcile with hub-to-hub WESTERN HUB -> DOMINION HUB")
print(f"{'=' * 70}")

# From script 12, PY2020:
# WESTERN HUB -> DOMINION HUB: 5.8%, 5.1%, 5.5%, 8.6%, 9.5%, 6.4%, 7.4%, 15.5%, 13.3%, 6.4%, 7.9%, 8.7%
script12_pcts = [5.8, 5.1, 5.5, 8.6, 9.5, 6.4, 7.4, 15.5, 13.3, 6.4, 7.9, 8.7]

# From our traced path above
my_pcts = [m / fx_sum * 100 for m in fx_mcps]

print(f"{'f#':<5} {'Month':<5} {'Script12':>10} {'ThisAudit':>10} {'Diff':>8}")
for fx in range(12):
    diff = my_pcts[fx] - script12_pcts[fx]
    print(f"f{fx:<4} {cal_months[fx]:<5} {script12_pcts[fx]:>9.1f}% {my_pcts[fx]:>9.1f}% {diff:>7.2f}%")

# ================================================================
# CHECK 6: Are we using the right annual round for splitting?
# ================================================================
print(f"\n{'=' * 70}")
print("CHECK 6: Does it matter which annual round we use?")
print(f"{'=' * 70}")

# The annual auction has 4 rounds. We used R4 cleared paths to define the universe.
# But does the choice of round affect which paths are in the universe?
for r in [1, 2, 3, 4]:
    cr = cleared[(cleared['market_round'] == r) & (cleared['class_type'] == '24H')]
    n_paths = len(cr[['source_name', 'sink_name']].drop_duplicates())
    print(f"  Round {r}: {len(cr)} trades, {n_paths} unique paths")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
