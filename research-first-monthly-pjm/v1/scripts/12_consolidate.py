"""
CONSOLIDATION: Final summary of hub-to-hub path MCP distributions.

Key question: Is there a universal distribution, or is it path-dependent?
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
CLASS_TYPE = '24h'
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

HUBS = [
    'WESTERN HUB', 'EASTERN HUB', 'AEP-DAYTON HUB', 'CHICAGO HUB',
    'DOMINION HUB', 'N ILLINOIS HUB', 'NEW JERSEY HUB', 'OHIO HUB',
    'WEST INT HUB', 'AEP GEN HUB',
]

YEARS = [2020, 2021, 2022]

PATH_PAIRS = [
    ('WESTERN HUB', 'DOMINION HUB'),
    ('WESTERN HUB', 'EASTERN HUB'),
    ('WESTERN HUB', 'NEW JERSEY HUB'),
    ('WESTERN HUB', 'AEP-DAYTON HUB'),
    ('CHICAGO HUB', 'AEP-DAYTON HUB'),
    ('N ILLINOIS HUB', 'EASTERN HUB'),
    ('AEP GEN HUB', 'DOMINION HUB'),
    ('AEP-DAYTON HUB', 'EASTERN HUB'),
    ('OHIO HUB', 'NEW JERSEY HUB'),
]

# Collect all data
all_hub_mcps = {}  # {year: {hub: [mcp_f0, ..., mcp_f11]}}

for year in YEARS:
    auction_month = f'{year}-06-01'
    hub_mcps = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        df_24h = df[df['class_type'] == CLASS_TYPE]
        for hub in HUBS:
            row = df_24h[df_24h['node_name'] == hub]
            if len(row) > 0:
                hub_mcps.setdefault(hub, []).append(row['mcp'].values[0])
    all_hub_mcps[year] = hub_mcps

# ================================================================
# SECTION 1: Node-level distributions for "well-behaved" hubs
# ================================================================
print("=" * 70)
print("SECTION 1: Node-level MCP distributions for major hubs")
print("=" * 70)
print("(pct = node_mcp_fx / sum(node_mcp_f0..f11))")

# Pick stable hubs (not CHICAGO/N_ILLINOIS which have near-zero sums)
stable_hubs = ['WESTERN HUB', 'DOMINION HUB', 'WEST INT HUB', 'AEP-DAYTON HUB',
               'OHIO HUB', 'AEP GEN HUB']

print(f"\n{'Hub':<20} {'Year':>4}", end='')
for fx in range(12):
    print(f" {cal_months[fx]:>5}", end='')
print()

for hub in stable_hubs:
    for year in YEARS:
        mcps = all_hub_mcps[year].get(hub, [])
        if len(mcps) != 12:
            continue
        s = sum(mcps)
        if abs(s) < 1:
            continue
        pcts = [m / s * 100 for m in mcps]
        print(f"{hub:<20} {year:>4}", end='')
        for p in pcts:
            print(f" {p:>4.1f}%", end='')
        print()

# Average across years for each hub
print(f"\n{'--- 3yr average ---':<20} {'':>4}", end='')
for fx in range(12):
    print(f" {cal_months[fx]:>5}", end='')
print()

hub_avgs = {}
for hub in stable_hubs:
    year_pcts = []
    for year in YEARS:
        mcps = all_hub_mcps[year].get(hub, [])
        if len(mcps) != 12:
            continue
        s = sum(mcps)
        if abs(s) < 1:
            continue
        year_pcts.append([m / s * 100 for m in mcps])

    if year_pcts:
        avg = [np.mean([yp[fx] for yp in year_pcts]) for fx in range(12)]
        hub_avgs[hub] = avg
        print(f"{hub:<20} {'avg':>4}", end='')
        for a in avg:
            print(f" {a:>4.1f}%", end='')
        print()

# ================================================================
# SECTION 2: Hub-to-hub PATH distributions, cross-year
# ================================================================
print(f"\n{'=' * 70}")
print("SECTION 2: Hub-to-hub PATH distributions (3yr avg)")
print("=" * 70)
print("(pct = path_mcp_fx / sum(path_mcp_f0..f11), path = sink - source)")

path_avgs = {}
for src, snk in PATH_PAIRS:
    year_pcts = []
    for year in YEARS:
        src_mcps = all_hub_mcps[year].get(src, [])
        snk_mcps = all_hub_mcps[year].get(snk, [])
        if len(src_mcps) != 12 or len(snk_mcps) != 12:
            continue
        path_mcps = [snk_mcps[fx] - src_mcps[fx] for fx in range(12)]
        s = sum(path_mcps)
        if abs(s) < 10:
            continue
        # Note: if s < 0, a negative path, the percentages flip sign
        # For consistency, always show how $1 of annual MCP distributes
        pcts = [m / s * 100 for m in path_mcps]
        year_pcts.append(pcts)

    if not year_pcts:
        continue

    avg = [np.mean([yp[fx] for yp in year_pcts]) for fx in range(12)]
    path_avgs[(src, snk)] = avg

    label = f"{src} -> {snk}"
    print(f"\n{label}")
    for i, year in enumerate(YEARS):
        if i < len(year_pcts):
            print(f"  {year}: ", end='')
            for p in year_pcts[i]:
                print(f" {p:>5.1f}%", end='')
            print()
    print(f"  avg: ", end='')
    for a in avg:
        print(f" {a:>5.1f}%", end='')
    print()

# ================================================================
# SECTION 3: Is there a universal pattern?
# ================================================================
print(f"\n{'=' * 70}")
print("SECTION 3: Comparing all path averages side by side")
print("=" * 70)

print(f"\n{'Path':<35}", end='')
for fx in range(12):
    print(f" {cal_months[fx]:>5}", end='')
print()

for (src, snk), avg in path_avgs.items():
    label = f"{src[:15]}->{snk[:15]}"
    print(f"{label:<35}", end='')
    for a in avg:
        print(f" {a:>4.1f}%", end='')
    print()

# Grand average across all paths
if path_avgs:
    all_avgs = list(path_avgs.values())
    grand_avg = [np.mean([a[fx] for a in all_avgs]) for fx in range(12)]
    print(f"\n{'GRAND AVG (all paths)':<35}", end='')
    for a in grand_avg:
        print(f" {a:>4.1f}%", end='')
    print(f"  sum={sum(grand_avg):.1f}%")

    print(f"{'Flat (1/12)':<35}", end='')
    for _ in range(12):
        print(f"  8.3%", end='')
    print()

# Standard deviation across paths
if path_avgs:
    stds = [np.std([a[fx] for a in all_avgs]) for fx in range(12)]
    print(f"\n{'Std across paths':<35}", end='')
    for s in stds:
        print(f" {s:>4.1f}%", end='')
    print()
    print(f"\n  -> Large std = distribution is NOT universal, it's path-dependent")

# ================================================================
# SECTION 4: Grouping paths by behavior
# ================================================================
print(f"\n{'=' * 70}")
print("SECTION 4: Grouping paths by seasonal behavior")
print("=" * 70)

for (src, snk), avg in path_avgs.items():
    # Classify: summer (f0-f2), fall (f3-f5), winter (f6-f8), spring (f9-f11)
    summer = sum(avg[0:3])
    fall = sum(avg[3:6])
    winter = sum(avg[6:9])
    spring = sum(avg[9:12])

    peak_season = max([('summer', summer), ('fall', fall), ('winter', winter), ('spring', spring)],
                      key=lambda x: x[1])
    label = f"{src[:15]}->{snk[:15]}"
    print(f"  {label:<35} summer={summer:>5.1f}% fall={fall:>5.1f}% winter={winter:>5.1f}% spring={spring:>5.1f}% -> peak: {peak_season[0]}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
