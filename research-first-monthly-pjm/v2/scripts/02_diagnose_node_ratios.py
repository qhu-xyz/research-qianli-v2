"""
V2 diagnostic: Why are node-level seasonal factors so noisy?

Hypotheses:
1. annual_mcp (from R4) and sum(monthly_mcps) (from June R1) differ substantially
   -> ratios fx/annual don't sum to 1, creating systematic bias
2. The relationship between annual and monthly MCPs is not multiplicative
   -> Maybe additive offsets (monthly = annual/12 + seasonal_delta) work better
3. Only certain nodes have stable ratios — node-type filtering may help

Also test: use fx/sum_monthly instead of fx/annual as the ratio denominator.
"""
import os
import gc
import resource
import numpy as np
import pandas as pd

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

from pbase.data.dataset.ftr.mcp.pjm import PjmMcp

mcp_loader = PjmMcp()

YEARS = list(range(2017, 2026))
CAL_MONTHS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# Load all node data
print("Loading data...")
annual_nodes = {}
monthly_nodes = {}

for year in YEARS:
    auction_month = f'{year}-06-01'
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_nodes[year] = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    monthly_nodes[year] = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_nodes[year][fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    print(f"  PY{year}: loaded, mem={mem_mb():.0f} MB")
    del annual
    gc.collect()

# ================================================================
# Diagnostic 1: How different are annual vs sum(monthly)?
# ================================================================
print(f"\n{'=' * 70}")
print("DIAGNOSTIC 1: annual vs sum(monthly) at node level")
print(f"{'=' * 70}")

for year in YEARS:
    ann = annual_nodes[year]
    # Sum monthly MCPs for each node
    common_nodes = set(ann.index)
    for fx in range(12):
        common_nodes &= set(monthly_nodes[year][fx].index)

    sums = pd.Series(0.0, index=list(common_nodes))
    for fx in range(12):
        sums += monthly_nodes[year][fx].reindex(sums.index).fillna(0)

    ann_aligned = ann.reindex(sums.index)
    ratio = sums / ann_aligned
    ratio_clean = ratio[ann_aligned.abs() > 50]

    print(f"  PY{year}: {len(ratio_clean)} nodes")
    print(f"    sum/annual ratio: mean={ratio_clean.mean():.4f}, "
          f"median={ratio_clean.median():.4f}, "
          f"std={ratio_clean.std():.4f}, "
          f"p10={ratio_clean.quantile(0.10):.4f}, "
          f"p90={ratio_clean.quantile(0.90):.4f}")

# ================================================================
# Diagnostic 2: Compare ratio approaches
# ================================================================
print(f"\n{'=' * 70}")
print("DIAGNOSTIC 2: fx/annual vs fx/sum_monthly ratio stability")
print(f"{'=' * 70}")

# For a sample of nodes, compute both ratio types across years and compare std
node_sample = list(set.intersection(*[set(annual_nodes[y].index) for y in YEARS]))[:3000]

ratio_vs_annual_stds = []
ratio_vs_sum_stds = []
delta_stds = []

for node in node_sample:
    ann_vals = [annual_nodes[y][node] for y in YEARS if abs(annual_nodes[y][node]) > 50]
    if len(ann_vals) < 5:
        continue

    year_ratio_a = []  # fx_mcp / annual_mcp
    year_ratio_s = []  # fx_mcp / sum_monthly_mcp
    year_delta = []    # fx_mcp - annual/12  (additive)

    for y in YEARS:
        ann = annual_nodes[y][node]
        if abs(ann) < 50:
            continue
        fx_vals = []
        valid = True
        for fx in range(12):
            if node not in monthly_nodes[y][fx].index:
                valid = False
                break
            fx_vals.append(monthly_nodes[y][fx][node])
        if not valid:
            continue

        monthly_sum = sum(fx_vals)
        if abs(monthly_sum) < 50:
            continue

        year_ratio_a.append([fv / ann for fv in fx_vals])
        year_ratio_s.append([fv / monthly_sum for fv in fx_vals])
        year_delta.append([fv - ann / 12.0 for fv in fx_vals])

    if len(year_ratio_a) >= 5:
        ratio_vs_annual_stds.append(np.std(year_ratio_a, axis=0))
        ratio_vs_sum_stds.append(np.std(year_ratio_s, axis=0))
        delta_stds.append(np.std(year_delta, axis=0))

ra = np.array(ratio_vs_annual_stds)
rs = np.array(ratio_vs_sum_stds)
ds = np.array(delta_stds)

print(f"\n  Nodes with >=5 years: {len(ra)}")
print(f"\n  {'fx':<4} {'Mon':<4} {'std(fx/ann)':>12} {'std(fx/sum)':>12} {'std(delta)':>12}")
for fx in range(12):
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {ra[:, fx].mean():>12.4f} {rs[:, fx].mean():>12.4f} {ds[:, fx].mean():>12.1f}")

print(f"\n  Overall mean:")
print(f"    ratio vs annual: {ra.mean():.4f}")
print(f"    ratio vs sum:    {rs.mean():.4f}")
print(f"    delta (abs):     {np.abs(ds).mean():.1f}")

# ================================================================
# Diagnostic 3: Additive approach - what do seasonal deltas look like?
# ================================================================
print(f"\n{'=' * 70}")
print("DIAGNOSTIC 3: Additive seasonal deltas")
print(f"{'=' * 70}")

# delta_fx = fx_mcp - annual_mcp/12
# If deltas are more stable than ratios, we should use:
#   pred_fx_path = (annual_path / 12) + avg_delta_sink[fx] - avg_delta_src[fx]

# Average delta across all nodes and years
delta_by_fx = {fx: [] for fx in range(12)}
for y in YEARS:
    ann = annual_nodes[y]
    for node in ann.index:
        if abs(ann[node]) < 50:
            continue
        valid = True
        for fx in range(12):
            if node not in monthly_nodes[y][fx].index:
                valid = False
                break
        if not valid:
            continue
        for fx in range(12):
            delta_by_fx[fx].append(monthly_nodes[y][fx][node] - ann[node] / 12.0)

print(f"  Average node-level delta (fx_mcp - annual/12):")
print(f"  {'fx':<4} {'Mon':<4} {'mean':>10} {'median':>10} {'std':>10} {'p10':>10} {'p90':>10}")
for fx in range(12):
    vals = np.array(delta_by_fx[fx])
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {vals.mean():>10.1f} {np.median(vals):>10.1f} "
          f"{vals.std():>10.1f} {np.percentile(vals, 10):>10.1f} {np.percentile(vals, 90):>10.1f}")

# ================================================================
# Diagnostic 4: Cross-node correlation in ratios
# ================================================================
print(f"\n{'=' * 70}")
print("DIAGNOSTIC 4: What fraction of ratio variance is year-specific (market-wide)?")
print(f"{'=' * 70}")

# If most variance is year-specific (all nodes shift together), then using
# THIS year's early monthly data could help more than historical averages
for fx in [0, 3, 6, 7]:  # Jun, Sep, Dec, Jan
    year_means = []
    for y in YEARS:
        ann = annual_nodes[y]
        ratios = []
        for node in ann.index:
            if abs(ann[node]) < 50:
                continue
            if node not in monthly_nodes[y][fx].index:
                continue
            ratios.append(monthly_nodes[y][fx][node] / ann[node])
        year_means.append(np.mean(ratios) if ratios else np.nan)

    print(f"\n  f{fx} ({CAL_MONTHS[fx]}):")
    for i, y in enumerate(YEARS):
        if not np.isnan(year_means[i]):
            print(f"    PY{y}: mean ratio = {year_means[i]:.4f} (vs flat 1/12 = {1/12:.4f})")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
