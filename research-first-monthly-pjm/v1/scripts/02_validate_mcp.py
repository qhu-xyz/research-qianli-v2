"""Step 2: Validate MCP values — understand units and relationship between annual vs monthly."""
import os
import resource
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import pandas as pd
from pbase.data.dataset.ftr.mcp.pjm import PjmMcp

mcp_loader = PjmMcp()

# Focus on 24h class_type for simplicity
# Pick a specific node to trace values

# Load annual round 4 for PY 2020
print("=== Loading Annual PY2020 (all rounds) ===")
annual = mcp_loader._load_annual(planning_year=2020)
# Filter to round 4 only, 24h class
annual_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == '24h')].copy()
print(f"Annual R4 24h shape: {annual_r4.shape}")

# Pick a sample node
sample_node = '02ANGOLA138 KV  TR1'
ann_node = annual_r4[annual_r4['node_name'] == sample_node]
print(f"\nAnnual R4 for {sample_node}:")
print(ann_node[['node_name', 'class_type', 'mcp', 'market_round', 'period_type']].to_string())

# Load all f0-f11 for June 2020 monthly auction, 24h
print(f"\n=== Loading Monthly June 2020, f0-f11 ===")
monthly_dfs = {}
for fx in range(12):
    pt = f'f{fx}'
    df = mcp_loader.load_data(
        auction_month='2020-06-01',
        market_round=1,
        period_type=pt,
    )
    df_24h = df[df['class_type'] == '24h'].copy()
    monthly_dfs[pt] = df_24h
    node_val = df_24h[df_24h['node_name'] == sample_node]['mcp'].values
    print(f"  {pt}: {node_val[0] if len(node_val) > 0 else 'N/A'}")

# Sum f0-f11 for the sample node
total_monthly = sum(
    monthly_dfs[f'f{fx}'][monthly_dfs[f'f{fx}']['node_name'] == sample_node]['mcp'].values[0]
    for fx in range(12)
)
ann_val = ann_node['mcp'].values[0]
print(f"\nSample node: {sample_node}")
print(f"  Annual R4 MCP (24h): {ann_val:.2f}")
print(f"  Sum of f0-f11 MCPs (24h): {total_monthly:.2f}")
print(f"  Ratio sum/annual: {total_monthly/ann_val:.4f}")

# Check a few more nodes
print("\n=== Checking multiple nodes ===")
# Get top nodes by absolute MCP from annual
top_nodes = annual_r4.nlargest(20, 'mcp')['node_name'].unique()[:5]
for node in top_nodes:
    ann_v = annual_r4[annual_r4['node_name'] == node]['mcp'].values
    if len(ann_v) == 0:
        continue
    ann_v = ann_v[0]
    monthly_sum = 0
    for fx in range(12):
        vals = monthly_dfs[f'f{fx}'][monthly_dfs[f'f{fx}']['node_name'] == node]['mcp'].values
        if len(vals) > 0:
            monthly_sum += vals[0]
    ratio = monthly_sum / ann_v if ann_v != 0 else float('inf')
    print(f"  {node}: annual={ann_v:.2f}, sum_monthly={monthly_sum:.2f}, ratio={ratio:.4f}")

# Show percentage distribution for sample node
print(f"\n=== Percentage Distribution for {sample_node} (24h) ===")
for fx in range(12):
    pt = f'f{fx}'
    val = monthly_dfs[pt][monthly_dfs[pt]['node_name'] == sample_node]['mcp'].values[0]
    pct = val / total_monthly * 100 if total_monthly != 0 else 0
    print(f"  {pt}: MCP={val:.2f}, pct={pct:.1f}%")

print(f"\nMemory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
