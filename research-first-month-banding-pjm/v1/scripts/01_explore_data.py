"""Step 1: Explore PJM MCP data structure and validate we can load it."""
import os
import resource

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"Memory at start: {mem_mb():.0f} MB")

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import pandas as pd
from pbase.data.dataset.ftr.mcp.pjm import PjmMcp
from pbase.data.const.data import ClassType

mcp_loader = PjmMcp()

# --- Test 1: Load annual auction data for planning year 2020 ---
# PJM annual auction_month is April, planning year 2020 means Jun 2020 - May 2021
print("\n=== Annual Auction MCP for PY 2020 (round 4) ===")
annual_2020 = mcp_loader._load_annual(planning_year=2020)
print(f"Shape: {annual_2020.shape}")
print(f"Columns: {list(annual_2020.columns)}")
print(f"Head:\n{annual_2020.head(10)}")
print(f"Dtypes:\n{annual_2020.dtypes}")

# Check what rounds are available
if 'auction_round' in annual_2020.columns:
    print(f"\nRounds available: {sorted(annual_2020['auction_round'].unique())}")
elif 'market_round' in annual_2020.columns:
    print(f"\nRounds available: {sorted(annual_2020['market_round'].unique())}")

print(f"\nMemory: {mem_mb():.0f} MB")

# --- Test 2: Load June monthly auction, f0, round 1 for 2020 ---
print("\n=== Monthly Auction MCP: auction_month=2020-06, f0, round 1 ===")
monthly_f0 = mcp_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f0',
)
print(f"Shape: {monthly_f0.shape}")
print(f"Columns: {list(monthly_f0.columns)}")
print(f"Head:\n{monthly_f0.head(10)}")

# --- Test 3: Load June monthly auction, f11, round 1 for 2020 ---
print("\n=== Monthly Auction MCP: auction_month=2020-06, f11, round 1 ===")
monthly_f11 = mcp_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f11',
)
print(f"Shape: {monthly_f11.shape}")
print(f"Head:\n{monthly_f11.head(10)}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
