"""
Extend analysis to more years: 2018, 2019, 2023, 2024, 2025.
First check which years have data available.
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

# Check which years have data
print("=" * 70)
print("Checking data availability by year")
print("=" * 70)

for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    # Check annual
    try:
        annual = mcp_loader._load_annual(planning_year=year)
        ann_r4 = annual[(annual['market_round'] == 4) & (annual['class_type'] == '24h')]
        ann_count = len(ann_r4)
    except Exception as e:
        ann_count = f"ERROR: {e}"

    # Check June monthly f0
    try:
        df = mcp_loader.load_data(
            auction_month=f'{year}-06-01', market_round=1, period_type='f0',
        )
        mon_count = len(df[df['class_type'] == '24h'])
    except Exception as e:
        mon_count = f"ERROR: {e}"

    # Check cleared
    try:
        cleared = cleared_loader._load_annual(planning_year=year)
        cl_count = len(cleared[(cleared['class_type'] == '24H') & (cleared['hedge_type'] == 'Obligation')])
    except Exception as e:
        cl_count = f"ERROR: {e}"

    print(f"  PY {year}: annual_R4={ann_count}, monthly_f0={mon_count}, cleared_oblig={cl_count}")

    del annual, df, cleared
    gc.collect()

print(f"\nMemory: {mem_mb():.0f} MB")
import ray
ray.shutdown()
