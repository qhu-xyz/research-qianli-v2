"""
Step 5: Analyze MCP in $/MWh terms (hourly MCP) for better comparison.

The raw MCP is total $ over the contract period. To get a fair monthly comparison,
we need $/MWh (hourly MCP) since each month has different hours.

Also look at the problem from the perspective the user actually cares about:
Given annual MCP for a path, what fraction settles in each month?
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

# Check if hourly MCP is available in the data
print("=== Checking for hourly MCP column ===")
df = mcp_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f0',
)
print(f"Columns: {list(df.columns)}")
print(f"Sample:\n{df[df['class_type'] == CLASS_TYPE].head(3).to_string()}")

# Check if mcp_hourly exists
if 'mcp_hourly' in df.columns:
    print("\nmcp_hourly column found!")
else:
    print("\nmcp_hourly NOT found — need to compute from total MCP / hours")

# Let's compute hours per month for 24h class type
# For PJM: 24h means all hours of the month
from calendar import monthrange

def hours_in_month(year, month):
    """Number of hours in a given month (ignoring DST for simplicity)."""
    return monthrange(year, month)[1] * 24

# For June 2020 planning year (Jun 2020 - May 2021)
print("\n=== Hours per month for PY 2020-06 to 2021-05 ===")
for fx in range(12):
    # f0=Jun, f1=Jul, ..., f6=Dec, f7=Jan(next), ...
    if fx <= 6:  # Jun-Dec of first year
        y, m = 2020, 6 + fx
    else:  # Jan-May of next year
        y, m = 2021, fx - 6
    hrs = hours_in_month(y, m)
    print(f"  f{fx} ({cal_months[fx]} {y}): {hrs} hours")

del df
gc.collect()

# --- Now let's look at hourly MCP distributions ---
# This tells us $/MWh per month, which is the fair comparison
print(f"\n{'='*60}")
print("HOURLY MCP ANALYSIS ($/MWh per month)")
print(f"{'='*60}")

for year in YEARS:
    print(f"\nPY {year}:")
    auction_month = f'{year}-06-01'

    fx_total_mcp = []
    fx_hours = []
    fx_hourly = []

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

        # Calculate hours for this month
        if fx <= 6:
            y, m = year, 6 + fx
        else:
            y, m = year + 1, fx - 6
        hrs = hours_in_month(y, m)

        fx_total_mcp.append(total)
        fx_hours.append(hrs)
        fx_hourly.append(median / hrs)  # Median hourly MCP

    # For distribution, we want: what fraction of ANNUAL MCP-HOURS comes from each month?
    # annual_mcp = sum over 12 months of (hourly_mcp × hours_in_month)
    # So the weight of each month = (hourly_mcp × hours) / sum(hourly_mcp × hours)
    # Which is exactly = total_mcp / sum(total_mcp) = what we already computed

    # But the USER wants: given the annual's total MCP, how to split it into months
    # The answer is exactly the total_mcp distribution we already have.

    total_sum = sum(fx_total_mcp)
    print(f"  {'f#':<5} {'Month':<5} {'Hours':>6} {'Median $/MWh':>14} {'Total':>14} {'Pct':>7}")
    for fx in range(12):
        pct = fx_total_mcp[fx] / total_sum * 100
        print(f"  f{fx:<4} {cal_months[fx]:<5} {fx_hours[fx]:>6} {fx_hourly[fx]:>14.2f} {fx_total_mcp[fx]:>14,.0f} {pct:>6.1f}%")

del df
gc.collect()

# --- Key question: Is the TOTAL MCP approach the right one? ---
# The user wants to SPLIT an annual MCP for a path into monthly pieces.
# Annual MCP for path = sum of (hourly_mcp_source × hours) - sum of (hourly_mcp_sink × hours)
#                     = sum over months of path_monthly_mcp
# So yes, the fraction = path_monthly_mcp / path_annual_mcp
#
# But at NODE level, every node sees the same "auction clearing" distribution.
# The distribution should be CONSISTENT across nodes for a given auction.
# Let's verify: for a given month/fx, does every node have the same % distribution?

print(f"\n{'='*60}")
print("NODE-LEVEL CONSISTENCY CHECK")
print("Does every node see the same % distribution?")
print(f"{'='*60}")

year = 2020
auction_month = '2020-06-01'

# Load all f0-f11 into a single DataFrame
all_fx = []
for fx in range(12):
    df = mcp_loader.load_data(
        auction_month=auction_month,
        market_round=1,
        period_type=f'f{fx}',
    )
    df_24h = df[df['class_type'] == CLASS_TYPE][['node_name', 'mcp']].copy()
    df_24h.columns = ['node_name', f'mcp_f{fx}']
    all_fx.append(df_24h)

merged = all_fx[0]
for i in range(1, 12):
    merged = merged.merge(all_fx[i], on='node_name', how='inner')

fx_cols = [f'mcp_f{fx}' for fx in range(12)]
merged['sum_mcp'] = merged[fx_cols].sum(axis=1)

# Filter out near-zero sum
merged = merged[merged['sum_mcp'].abs() > 100]

for fx in range(12):
    merged[f'pct_f{fx}'] = merged[f'mcp_f{fx}'] / merged['sum_mcp'] * 100

# Show distribution of pct values across nodes
print(f"\nPY 2020 — Distribution of node-level percentages:")
print(f"  {'f#':<5} {'Month':<5} {'Median':>8} {'Mean':>8} {'Std':>8} {'P10':>8} {'P90':>8}")
for fx in range(12):
    col = f'pct_f{fx}'
    vals = merged[col]
    print(f"  f{fx:<4} {cal_months[fx]:<5} {vals.median():>7.1f}% {vals.mean():>7.1f}% {vals.std():>7.1f}% {vals.quantile(0.1):>7.1f}% {vals.quantile(0.9):>7.1f}%")

print(f"\n  Key insight: Large std = distribution varies A LOT by node/path!")
print(f"  This means a single % distribution is an approximation.")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
