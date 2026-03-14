"""
Step 7: Address nodal replacement + path-level validation.

Key questions:
1. Are node names consistent across f0-f11 within the same June auction?
2. Do we need nodal replacement mapping when matching annual to monthly?
3. What do path-level distributions look like for real cleared paths?
4. Should we use pnode_id instead of node_name for matching?
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
CLASS_TYPE = '24h'
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# ================================================================
# CHECK 1: Node consistency across f0-f11 within same auction
# ================================================================
print("=" * 60)
print("CHECK 1: Node consistency across f0-f11")
print("=" * 60)

for year in [2020, 2021, 2022]:
    auction_month = f'{year}-06-01'
    all_nodesets = []
    all_id_sets = []
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == CLASS_TYPE]
        all_nodesets.append(set(df_24h['node_name']))
        all_id_sets.append(set(df_24h['pnode_id']))

    # Check intersection vs union
    name_intersection = set.intersection(*all_nodesets)
    name_union = set.union(*all_nodesets)
    id_intersection = set.intersection(*all_id_sets)
    id_union = set.union(*all_id_sets)

    print(f"\nPY {year}:")
    print(f"  By node_name: intersection={len(name_intersection)}, union={len(name_union)}, diff={len(name_union)-len(name_intersection)}")
    print(f"  By pnode_id:  intersection={len(id_intersection)}, union={len(id_union)}, diff={len(id_union)-len(id_intersection)}")

    if len(name_union) != len(name_intersection):
        # Find which nodes are missing in which fx
        for fx in range(12):
            missing = name_union - all_nodesets[fx]
            if missing:
                print(f"    f{fx} missing {len(missing)} nodes: {list(missing)[:5]}...")
    else:
        print(f"  -> All nodes consistent across f0-f11 (no nodal replacement issues within auction)")

gc.collect()

# ================================================================
# CHECK 2: Node consistency between annual R4 and June monthly
# ================================================================
print(f"\n{'='*60}")
print("CHECK 2: Annual R4 vs June monthly node matching")
print(f"{'='*60}")

for year in [2020, 2021, 2022]:
    # Annual
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)]
    ann_names = set(annual_r4['node_name'])
    ann_ids = set(annual_r4['pnode_id'])

    # Monthly f0
    df_f0 = mcp_loader.load_data(
        auction_month=f'{year}-06-01',
        market_round=1,
        period_type='f0',
    )
    df_f0_24h = df_f0[df_f0['class_type'] == CLASS_TYPE]
    mon_names = set(df_f0_24h['node_name'])
    mon_ids = set(df_f0_24h['pnode_id'])

    print(f"\nPY {year}:")
    print(f"  Annual R4: {len(ann_names)} nodes / {len(ann_ids)} pnode_ids")
    print(f"  Monthly f0: {len(mon_names)} nodes / {len(mon_ids)} pnode_ids")

    # By name
    only_ann = ann_names - mon_names
    only_mon = mon_names - ann_names
    both = ann_names & mon_names
    print(f"  By name: shared={len(both)}, only_annual={len(only_ann)}, only_monthly={len(only_mon)}")

    # By id
    only_ann_id = ann_ids - mon_ids
    only_mon_id = mon_ids - ann_ids
    both_id = ann_ids & mon_ids
    print(f"  By id:   shared={len(both_id)}, only_annual={len(only_ann_id)}, only_monthly={len(only_mon_id)}")

    # Check if name mismatches can be resolved by pnode_id
    if only_ann:
        # Nodes in annual but not monthly — check if their pnode_ids exist in monthly under diff names
        ann_only_df = annual_r4[annual_r4['node_name'].isin(only_ann)][['node_name', 'pnode_id']]
        ann_only_ids = set(ann_only_df['pnode_id'])
        resolved_by_id = ann_only_ids & mon_ids
        print(f"  Annual-only nodes resolvable by pnode_id: {len(resolved_by_id)} of {len(only_ann)}")
        if resolved_by_id:
            # Show some examples
            for pid in list(resolved_by_id)[:3]:
                ann_name = annual_r4[annual_r4['pnode_id'] == pid]['node_name'].values[0]
                mon_name = df_f0_24h[df_f0_24h['pnode_id'] == pid]['node_name'].values[0]
                print(f"    pnode_id={pid}: annual='{ann_name}', monthly='{mon_name}'")

    del annual, df_f0
    gc.collect()

# ================================================================
# CHECK 3: Path-level distributions for cleared paths
# ================================================================
print(f"\n{'='*60}")
print("CHECK 3: Path-level distributions for cleared paths")
print(f"{'='*60}")

year = 2020
auction_month = f'{year}-06-01'

# Load node MCP data for all f0-f11 using pnode_id as key (more reliable than name)
print(f"\nLoading node MCPs by pnode_id for PY {year}...")
node_mcp = {}
for fx in range(12):
    df = mcp_loader.load_data(
        auction_month=auction_month,
        market_round=1,
        period_type=f'f{fx}',
    )
    df_24h = df[df['class_type'] == CLASS_TYPE][['pnode_id', 'mcp']].copy()
    # Verify no duplicate pnode_ids
    dupes = df_24h['pnode_id'].duplicated().sum()
    if dupes > 0:
        print(f"  WARNING: f{fx} has {dupes} duplicate pnode_ids!")
    node_mcp[fx] = df_24h.set_index('pnode_id')['mcp']

# Load cleared paths
cleared = cleared_loader.load_data(
    auction_month=auction_month,
    market_round=1,
    period_type='f0',
)
cleared_24h = cleared[cleared['class_type'] == CLASS_TYPE]

# Get unique paths with their IDs
paths = cleared_24h[['source_name', 'sink_name', 'source_id', 'sink_id']].drop_duplicates()
print(f"Unique cleared paths (24h): {len(paths)}")

# Compute path-level distribution for each path
path_distributions = []
for _, row in paths.iterrows():
    src_id = row['source_id']
    snk_id = row['sink_id']

    # Check both nodes exist in all fx
    mcps = []
    valid = True
    for fx in range(12):
        if src_id in node_mcp[fx].index and snk_id in node_mcp[fx].index:
            path_mcp = node_mcp[fx][snk_id] - node_mcp[fx][src_id]
            mcps.append(path_mcp)
        else:
            valid = False
            break

    if not valid:
        continue

    total = sum(mcps)
    if abs(total) < 10:  # Skip near-zero paths
        continue

    pcts = [m / total * 100 for m in mcps]
    path_distributions.append({
        'source': row['source_name'],
        'sink': row['sink_name'],
        'total': total,
        **{f'pct_f{fx}': pcts[fx] for fx in range(12)},
        **{f'mcp_f{fx}': mcps[fx] for fx in range(12)},
    })

pdf = pd.DataFrame(path_distributions)
print(f"Paths with valid distributions: {len(pdf)}")
print(f"  (positive total: {(pdf['total'] > 0).sum()}, negative: {(pdf['total'] < 0).sum()})")

# Show aggregate path-level distribution
print(f"\n--- Path-level distribution (median across cleared paths) ---")
print(f"{'f#':<5} {'Month':<5} {'Median':>8} {'Mean':>8} {'P25':>8} {'P75':>8}")
for fx in range(12):
    col = f'pct_f{fx}'
    vals = pdf[col]
    print(f"f{fx:<4} {cal_months[fx]:<5} {vals.median():>7.1f}% {vals.mean():>7.1f}% {vals.quantile(0.25):>7.1f}% {vals.quantile(0.75):>7.1f}%")

# Positive-path-only
pos_pdf = pdf[pdf['total'] > 0]
print(f"\n--- Positive-path-only distribution (median) ---")
print(f"({len(pos_pdf)} paths)")
print(f"{'f#':<5} {'Month':<5} {'Median':>8} {'P25':>8} {'P75':>8}")
for fx in range(12):
    col = f'pct_f{fx}'
    vals = pos_pdf[col]
    print(f"f{fx:<4} {cal_months[fx]:<5} {vals.median():>7.1f}% {vals.quantile(0.25):>7.1f}% {vals.quantile(0.75):>7.1f}%")

# Show some sample paths
print(f"\n--- Sample individual path distributions ---")
sample_paths = pdf.nlargest(5, 'total')
for _, row in sample_paths.iterrows():
    print(f"\n  {row['source']} -> {row['sink']} (total={row['total']:.0f})")
    for fx in range(12):
        print(f"    f{fx} ({cal_months[fx]}): MCP={row[f'mcp_f{fx}']:>8.1f}, pct={row[f'pct_f{fx}']:>6.1f}%")

# ================================================================
# CHECK 4: Do same analysis for PY 2021 and 2022
# ================================================================
print(f"\n{'='*60}")
print("CHECK 4: Path-level distributions for PY 2021, 2022")
print(f"{'='*60}")

for year in [2021, 2022]:
    auction_month = f'{year}-06-01'
    print(f"\n--- PY {year} ---")

    # Load node MCPs
    node_mcp = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == CLASS_TYPE][['pnode_id', 'mcp']].copy()
        node_mcp[fx] = df_24h.set_index('pnode_id')['mcp']

    # Load cleared paths
    cleared = cleared_loader.load_data(
        auction_month=auction_month,
        market_round=1,
        period_type='f0',
    )
    cleared_24h = cleared[cleared['class_type'] == CLASS_TYPE]
    paths = cleared_24h[['source_name', 'sink_name', 'source_id', 'sink_id']].drop_duplicates()

    path_dists = []
    for _, row in paths.iterrows():
        src_id, snk_id = row['source_id'], row['sink_id']
        mcps = []
        valid = True
        for fx in range(12):
            if src_id in node_mcp[fx].index and snk_id in node_mcp[fx].index:
                mcps.append(node_mcp[fx][snk_id] - node_mcp[fx][src_id])
            else:
                valid = False
                break
        if not valid or abs(sum(mcps)) < 10:
            continue
        total = sum(mcps)
        pcts = [m / total * 100 for m in mcps]
        path_dists.append({f'pct_f{fx}': pcts[fx] for fx in range(12)} | {'total': total})

    pdf2 = pd.DataFrame(path_dists)
    pos_pdf2 = pdf2[pdf2['total'] > 0]

    print(f"  Total paths: {len(pdf2)}, positive: {len(pos_pdf2)}")
    print(f"  Positive-path median distribution:")
    for fx in range(12):
        col = f'pct_f{fx}'
        print(f"    f{fx} ({cal_months[fx]}): {pos_pdf2[col].median():>7.1f}%")

    del node_mcp, cleared
    gc.collect()

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
