"""
CLEAN ANALYSIS: Trace MCP values for PJM hubs and hub-to-hub paths.

Strategy:
  1. Load node-level MCPs for well-known hubs (both annual R4 and monthly f0-f11)
  2. Show raw values side by side — NO aggregation yet
  3. Compute path MCPs for hub-to-hub pairs
  4. Show percentage distributions for these paths
  5. Only THEN scale to broader universe
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

for year in YEARS:
    print(f"\n{'#' * 70}")
    print(f"# PY {year} (Jun {year} - May {year+1})")
    print(f"{'#' * 70}")

    auction_month = f'{year}-06-01'

    # --- Load annual R4 ---
    annual = mcp_loader._load_annual(planning_year=year)
    annual_r4 = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == CLASS_TYPE)
    ].copy()

    # --- Load all f0-f11 ---
    monthly = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month,
            market_round=1,
            period_type=f'f{fx}',
        )
        monthly[fx] = df[df['class_type'] == CLASS_TYPE].copy()

    # ================================================================
    # Part A: Hub-level node MCPs
    # ================================================================
    print(f"\n--- Part A: Node-level MCPs for hubs ---")
    print(f"{'Hub':<20} {'Annual R4':>10}", end='')
    for fx in range(12):
        print(f" {cal_months[fx]:>7}", end='')
    print(f" {'Sum(f)':>8} {'Sum/Ann':>8}")

    for hub in HUBS:
        # Annual
        ann_row = annual_r4[annual_r4['node_name'] == hub]
        if len(ann_row) == 0:
            continue
        ann_mcp = ann_row['mcp'].values[0]

        # Monthly f0-f11
        fx_mcps = []
        for fx in range(12):
            m_row = monthly[fx][monthly[fx]['node_name'] == hub]
            if len(m_row) == 0:
                fx_mcps.append(None)
            else:
                fx_mcps.append(m_row['mcp'].values[0])

        if any(v is None for v in fx_mcps):
            continue

        fx_sum = sum(fx_mcps)
        ratio = fx_sum / ann_mcp if ann_mcp != 0 else float('inf')

        print(f"{hub:<20} {ann_mcp:>10.1f}", end='')
        for v in fx_mcps:
            print(f" {v:>7.1f}", end='')
        print(f" {fx_sum:>8.1f} {ratio:>8.3f}")

    # ================================================================
    # Part B: Hub-level percentage distributions
    # ================================================================
    print(f"\n--- Part B: Percentage distributions (node MCP_fx / sum(f0..f11)) ---")
    print(f"{'Hub':<20}", end='')
    for fx in range(12):
        print(f" {cal_months[fx]:>7}", end='')
    print()

    for hub in HUBS:
        fx_mcps = []
        for fx in range(12):
            m_row = monthly[fx][monthly[fx]['node_name'] == hub]
            if len(m_row) == 0:
                break
            fx_mcps.append(m_row['mcp'].values[0])
        if len(fx_mcps) != 12:
            continue

        fx_sum = sum(fx_mcps)
        if abs(fx_sum) < 1:
            continue

        print(f"{hub:<20}", end='')
        for v in fx_mcps:
            pct = v / fx_sum * 100
            print(f" {pct:>6.1f}%", end='')
        print()

    # ================================================================
    # Part C: Hub-to-hub PATH distributions
    # ================================================================
    print(f"\n--- Part C: Hub-to-hub PATH distributions ---")
    # Build all hub pairs
    hub_mcps = {}
    for hub in HUBS:
        mcps = []
        for fx in range(12):
            m_row = monthly[fx][monthly[fx]['node_name'] == hub]
            if len(m_row) == 0:
                break
            mcps.append(m_row['mcp'].values[0])
        if len(mcps) == 12:
            hub_mcps[hub] = mcps

    # Compute path MCPs for selected pairs
    path_pairs = [
        ('WESTERN HUB', 'EASTERN HUB'),
        ('WESTERN HUB', 'AEP-DAYTON HUB'),
        ('WESTERN HUB', 'DOMINION HUB'),
        ('AEP-DAYTON HUB', 'EASTERN HUB'),
        ('CHICAGO HUB', 'AEP-DAYTON HUB'),
        ('N ILLINOIS HUB', 'EASTERN HUB'),
        ('OHIO HUB', 'NEW JERSEY HUB'),
        ('WESTERN HUB', 'NEW JERSEY HUB'),
        ('AEP GEN HUB', 'DOMINION HUB'),
    ]

    print(f"{'Path':<45}", end='')
    for fx in range(12):
        print(f" {cal_months[fx]:>7}", end='')
    print(f" {'Sum':>8}")

    for src, snk in path_pairs:
        if src not in hub_mcps or snk not in hub_mcps:
            continue
        path_mcps = [hub_mcps[snk][fx] - hub_mcps[src][fx] for fx in range(12)]
        path_sum = sum(path_mcps)

        label = f"{src[:15]} -> {snk[:15]}"
        print(f"\n{label:<45}", end='')
        for v in path_mcps:
            print(f" {v:>7.1f}", end='')
        print(f" {path_sum:>8.1f}")

        # Percentages
        if abs(path_sum) > 1:
            print(f"{'  (% of sum)':<45}", end='')
            for v in path_mcps:
                pct = v / path_sum * 100
                print(f" {pct:>6.1f}%", end='')
            print()

    del annual, monthly
    gc.collect()
    print(f"\nMemory: {mem_mb():.0f} MB")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
