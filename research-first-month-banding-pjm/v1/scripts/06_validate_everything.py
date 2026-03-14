"""
Comprehensive validation of MCP distribution analysis.

Validates:
1. f0-f11 to calendar month mapping
2. MCP units (total $ vs $/MWh)
3. Annual round numbering (is R4 really the last round?)
4. Whether system-wide sum is a valid approach
5. Path-level distributions for real cleared paths
6. Whether negative winter MCPs are real
7. Cross-check: monthly auction date vs planning year alignment
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
from pbase.data.const.period.pjm import PjmPeriodType

mcp_loader = PjmMcp()
CLASS_TYPE = '24h'

# ================================================================
# VALIDATION 1: f0-f11 to calendar month mapping
# ================================================================
print("=" * 60)
print("VALIDATION 1: f0-f11 to calendar month mapping")
print("=" * 60)

# Use PjmPeriodType.model_term_to_mkt_term to verify
auction_month = pd.Timestamp('2020-06-01')
print(f"Auction month: {auction_month}")
for fx in range(12):
    pt = f'f{fx}'
    market_name, period_name = PjmPeriodType.model_term_to_mkt_term(auction_month, pt)
    print(f"  {pt} -> market_name='{market_name}', period_name='{period_name}'")

# Also check via the loaded data's market_period column
print(f"\nVerify from loaded data:")
for fx in [0, 5, 11]:
    pt = f'f{fx}'
    df = mcp_loader.load_data(
        auction_month='2020-06-01',
        market_round=1,
        period_type=pt,
    )
    mp = df['market_period'].iloc[0]
    print(f"  {pt}: market_period = '{mp}'")

# ================================================================
# VALIDATION 2: MCP units — verify by checking mcp / hours
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 2: MCP units")
print(f"{'='*60}")

# Load f0 (June 2020) and check if values make sense as total $
df_f0 = mcp_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f0',
)
df_f0_24h = df_f0[df_f0['class_type'] == CLASS_TYPE]

print(f"f0 (June 2020), 24h class:")
print(f"  Count: {len(df_f0_24h)}")
print(f"  MCP stats: min={df_f0_24h['mcp'].min():.2f}, max={df_f0_24h['mcp'].max():.2f}")
print(f"  MCP stats: mean={df_f0_24h['mcp'].mean():.2f}, median={df_f0_24h['mcp'].median():.2f}")
print(f"  Hours in June 2020: 720")
print(f"  If total $: median $/MWh = {df_f0_24h['mcp'].median() / 720:.4f}")
print(f"  If $/MWh already: median total $ = {df_f0_24h['mcp'].median() * 720:.2f}")

# Compare f0 onpeak vs 24h to check if hours are baked in
df_f0_onpk = df_f0[df_f0['class_type'] == 'onpeak']
sample_node = '02ANGOLA138 KV  TR1'
v_24h = df_f0_24h[df_f0_24h['node_name'] == sample_node]['mcp'].values[0]
v_onpk = df_f0_onpk[df_f0_onpk['node_name'] == sample_node]['mcp'].values[0]
print(f"\n  Sample node '{sample_node}':")
print(f"    24h MCP  = {v_24h:.2f}")
print(f"    onpeak MCP = {v_onpk:.2f}")
print(f"    Ratio 24h/onpeak = {v_24h/v_onpk:.4f}")
print(f"    (If total $, ratio should reflect hours: 720/~350 ≈ 2.06)")
print(f"    (If $/MWh, ratio should be ~1 or less)")

# Also load the ANNUAL data to compare
annual = mcp_loader._load_annual(planning_year=2020)
annual_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)]
ann_node = annual_r4[annual_r4['node_name'] == sample_node]['mcp'].values[0]
print(f"\n  Annual R4 24h MCP for same node: {ann_node:.2f}")
print(f"    If total $: $/MWh = {ann_node / 8760:.4f} (annual has ~8760 hours)")
print(f"    Sum(f0..f11) for this node (from script 02): ~13274")
print(f"    Annual / 12 = {ann_node / 12:.2f}")

del df_f0, annual
gc.collect()

# ================================================================
# VALIDATION 3: Annual round numbering
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 3: Annual auction rounds")
print(f"{'='*60}")

annual = mcp_loader._load_annual(planning_year=2020)
# Check round values and see if MCP changes across rounds for same node
for r in [1, 2, 3, 4]:
    ann_r = annual[(annual['market_round'] == r) & (annual['class_type'] == CLASS_TYPE)]
    node_val = ann_r[ann_r['node_name'] == sample_node]['mcp'].values
    if len(node_val) > 0:
        print(f"  Round {r}: {len(ann_r)} nodes, sample node MCP = {node_val[0]:.2f}")
    else:
        print(f"  Round {r}: {len(ann_r)} nodes, sample node NOT FOUND")

# Check auction dates for each round
print(f"\n  Auction dates by round:")
for r in [1, 2, 3, 4]:
    ann_r = annual[(annual['market_round'] == r)]
    dates = ann_r['auction_date'].unique()
    print(f"  Round {r}: {dates}")

# Also check market_name to confirm it's the same planning year
print(f"\n  Market names: {annual['market_name'].unique()}")

del annual
gc.collect()

# ================================================================
# VALIDATION 4: System-wide sum vs node-level median
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 4: System-wide sum vs other approaches")
print(f"{'='*60}")

# The issue: summing all node MCPs is dominated by high-value nodes.
# Let's check how many nodes are positive vs negative for each fx.
year = 2020
monthly_dfs = {}
for fx in range(12):
    df = mcp_loader.load_data(
        auction_month=f'{year}-06-01',
        market_round=1,
        period_type=f'f{fx}',
    )
    df_24h = df[df['class_type'] == CLASS_TYPE][['node_name', 'mcp']].copy()
    df_24h.columns = ['node_name', f'mcp_f{fx}']
    monthly_dfs[fx] = df_24h

# Merge all
merged = monthly_dfs[0]
for fx in range(1, 12):
    merged = merged.merge(monthly_dfs[fx], on='node_name', how='inner')

fx_cols = [f'mcp_f{fx}' for fx in range(12)]
merged['sum_mcp'] = merged[fx_cols].sum(axis=1)

print(f"PY {year}, {len(merged)} nodes total")
print(f"\n  Positive/negative MCP counts per period:")
for fx in range(12):
    col = f'mcp_f{fx}'
    pos = (merged[col] > 0).sum()
    neg = (merged[col] < 0).sum()
    zero = (merged[col] == 0).sum()
    print(f"  f{fx}: pos={pos}, neg={neg}, zero={zero}")

print(f"\n  Sum_mcp distribution:")
print(f"    positive: {(merged['sum_mcp'] > 0).sum()}")
print(f"    negative: {(merged['sum_mcp'] < 0).sum()}")
print(f"    near-zero (|sum|<100): {(merged['sum_mcp'].abs() < 100).sum()}")

# Approach A: System-wide sum
sys_pcts = []
for fx in range(12):
    sys_pcts.append(merged[f'mcp_f{fx}'].sum() / merged['sum_mcp'].sum() * 100)

# Approach B: Node-level median (filtered |sum| > 100)
filt = merged[merged['sum_mcp'].abs() > 100].copy()
for fx in range(12):
    filt[f'pct_f{fx}'] = filt[f'mcp_f{fx}'] / filt['sum_mcp'] * 100
med_pcts = [filt[f'pct_f{fx}'].median() for fx in range(12)]

# Approach C: Absolute-value-weighted median
# Weight each node by |sum_mcp| to give more influence to high-value nodes
filt['abs_sum'] = filt['sum_mcp'].abs()
wt_pcts = []
for fx in range(12):
    # Weighted average
    wt = np.average(filt[f'pct_f{fx}'], weights=filt['abs_sum'])
    wt_pcts.append(wt)

# Approach D: Only positive-sum nodes, median
pos_filt = filt[filt['sum_mcp'] > 100].copy()
pos_med_pcts = [pos_filt[f'pct_f{fx}'].median() for fx in range(12)]

cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
print(f"\n  {'f#':<5} {'Month':<5} {'SysSum':>8} {'Median':>8} {'WtAvg':>8} {'Pos Med':>8}")
for fx in range(12):
    print(f"  f{fx:<4} {cal_months[fx]:<5} {sys_pcts[fx]:>7.1f}% {med_pcts[fx]:>7.1f}% {wt_pcts[fx]:>7.1f}% {pos_med_pcts[fx]:>7.1f}%")
print(f"  {'Sum':<11} {sum(sys_pcts):>7.1f}% {sum(med_pcts):>7.1f}% {sum(wt_pcts):>7.1f}% {sum(pos_med_pcts):>7.1f}%")

# ================================================================
# VALIDATION 5: Path-level distributions for actual cleared paths
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 5: Path-level distributions (real paths)")
print(f"{'='*60}")

# Get some cleared paths from the June 2020 monthly auction
from pbase.data.dataset.ftr.cleared.pjm import PjmClearedFtrs
from pbase.data.const.market.auction_type import AuctionType

cleared_loader = PjmClearedFtrs()
cleared = cleared_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f0',
)
print(f"Cleared FTRs for June 2020 f0: {len(cleared)} rows")
print(f"Columns: {list(cleared.columns)}")
print(f"Sample:\n{cleared.head(5).to_string()}")

# Get unique paths (source_name, sink_name)
if 'source_name' in cleared.columns and 'sink_name' in cleared.columns:
    path_cols = ['source_name', 'sink_name']
elif 'source' in cleared.columns and 'sink' in cleared.columns:
    path_cols = ['source', 'sink']
else:
    # Check what columns contain source/sink info
    src_cols = [c for c in cleared.columns if 'source' in c.lower() or 'src' in c.lower()]
    snk_cols = [c for c in cleared.columns if 'sink' in c.lower() or 'snk' in c.lower()]
    print(f"Source-like columns: {src_cols}")
    print(f"Sink-like columns: {snk_cols}")
    path_cols = None

if path_cols:
    # Take top 10 paths by volume
    if 'cleared_volume' in cleared.columns:
        vol_col = 'cleared_volume'
    elif 'volume' in cleared.columns:
        vol_col = 'volume'
    else:
        vol_col = None

    if vol_col:
        cleared_24h = cleared[cleared['class_type'] == CLASS_TYPE] if 'class_type' in cleared.columns else cleared
        top_paths = cleared_24h.nlargest(20, vol_col)[path_cols].drop_duplicates().head(10)
        print(f"\nTop cleared paths:")
        print(top_paths.to_string())

        # For each path, compute the distribution
        # Build node lookup
        node_lookup = {}
        for fx in range(12):
            node_lookup[fx] = monthly_dfs[fx].set_index('node_name')[f'mcp_f{fx}']

        print(f"\nPath-level MCP distributions:")
        for _, row in top_paths.iterrows():
            src = row[path_cols[0]]
            snk = row[path_cols[1]]

            path_mcps = []
            valid = True
            for fx in range(12):
                if src in node_lookup[fx].index and snk in node_lookup[fx].index:
                    path_mcp = node_lookup[fx][snk] - node_lookup[fx][src]
                    path_mcps.append(path_mcp)
                else:
                    valid = False
                    break

            if not valid or abs(sum(path_mcps)) < 10:
                continue

            path_sum = sum(path_mcps)
            print(f"\n  {src} -> {snk} (sum={path_sum:.0f})")
            for fx in range(12):
                pct = path_mcps[fx] / path_sum * 100
                print(f"    f{fx} ({cal_months[fx]}): {path_mcps[fx]:>8.1f} = {pct:>6.1f}%")

# ================================================================
# VALIDATION 6: Sanity check on negative winter MCPs
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 6: Negative winter MCP deep dive")
print(f"{'='*60}")

for fx in [7, 8]:  # Jan, Feb
    col = f'mcp_f{fx}'
    vals = merged[col]
    print(f"\n  f{fx} ({cal_months[fx]}):")
    print(f"    Nodes with MCP > 0: {(vals > 0).sum()} ({(vals > 0).mean()*100:.1f}%)")
    print(f"    Nodes with MCP < 0: {(vals < 0).sum()} ({(vals < 0).mean()*100:.1f}%)")
    print(f"    Nodes with MCP = 0: {(vals == 0).sum()}")
    print(f"    Mean: {vals.mean():.2f}, Median: {vals.median():.2f}")
    print(f"    Sum positive: {vals[vals > 0].sum():,.0f}")
    print(f"    Sum negative: {vals[vals < 0].sum():,.0f}")
    print(f"    Net: {vals.sum():,.0f}")

# Compare with f4 (Oct) which should be strongly positive
for fx in [4]:
    col = f'mcp_f{fx}'
    vals = merged[col]
    print(f"\n  f{fx} ({cal_months[fx]}):")
    print(f"    Nodes with MCP > 0: {(vals > 0).sum()} ({(vals > 0).mean()*100:.1f}%)")
    print(f"    Nodes with MCP < 0: {(vals < 0).sum()} ({(vals < 0).mean()*100:.1f}%)")
    print(f"    Mean: {vals.mean():.2f}, Median: {vals.median():.2f}")
    print(f"    Sum positive: {vals[vals > 0].sum():,.0f}")
    print(f"    Sum negative: {vals[vals < 0].sum():,.0f}")

# ================================================================
# VALIDATION 7: Check if annual MCP is also total $ over 12 months
# ================================================================
print(f"\n{'='*60}")
print("VALIDATION 7: Annual MCP unit verification")
print(f"{'='*60}")

annual = mcp_loader._load_annual(planning_year=2020)
annual_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)]

# For the sample node, compare annual MCP vs sum of monthly MCPs
ann_val = annual_r4[annual_r4['node_name'] == sample_node]['mcp'].values[0]

# Also check if annual has onpeak
annual_r4_onpk = annual[(annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')]
ann_onpk = annual_r4_onpk[annual_r4_onpk['node_name'] == sample_node]['mcp'].values[0]

print(f"  Sample node: {sample_node}")
print(f"  Annual R4 24h:    {ann_val:.2f}")
print(f"  Annual R4 onpeak: {ann_onpk:.2f}")
print(f"  Ratio 24h/onpeak: {ann_val/ann_onpk:.4f}")
print(f"  (If total $: ratio ≈ 8760/~4400 ≈ 2.0)")
print(f"  (24h includes ALL hours; onpeak ≈ half)")

# Sum of monthly for this node
monthly_sum = merged[merged['node_name'] == sample_node]['sum_mcp'].values[0]
print(f"\n  Sum(f0..f11) monthly: {monthly_sum:.2f}")
print(f"  Annual:               {ann_val:.2f}")
print(f"  Monthly/Annual ratio: {monthly_sum/ann_val:.4f}")
print(f"  (These are from DIFFERENT auctions, so ratio ≠ 1 is expected)")

# Final: also check annual R4 statistics
print(f"\n  Annual R4 24h stats across all nodes:")
print(f"    Count: {len(annual_r4)}")
print(f"    Mean: {annual_r4['mcp'].mean():.2f}")
print(f"    Median: {annual_r4['mcp'].median():.2f}")
print(f"    Min: {annual_r4['mcp'].min():.2f}")
print(f"    Max: {annual_r4['mcp'].max():.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
