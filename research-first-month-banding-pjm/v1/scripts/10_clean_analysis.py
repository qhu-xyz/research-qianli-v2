"""
CLEAN RESTART: PJM June Monthly Auction MCP Distribution

Approach: Start from a single node, trace every number, verify manually.
Then scale up only after the single-node case makes complete sense.

Step 1: Pick one well-known hub node (WESTERN HUB) and trace its MCPs.
Step 2: Pick one well-known path (WESTERN HUB -> AEP-DAYTON HUB) and trace.
Step 3: Then scale to all paths.
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

# ================================================================
# Step 0: Find hub node names in the data
# ================================================================
print("=" * 70)
print("Step 0: Find hub/aggregate node names")
print("=" * 70)

df_sample = mcp_loader.load_data(
    auction_month='2020-06-01',
    market_round=1,
    period_type='f0',
)
all_names = df_sample['node_name'].unique()
hub_names = [n for n in all_names if 'HUB' in str(n).upper() or 'ZONE' in str(n).upper() or 'AGGREGATE' in str(n).upper()]
print(f"Total nodes: {len(all_names)}")
print(f"Hub/Zone/Aggregate nodes ({len(hub_names)}):")
for h in sorted(hub_names):
    print(f"  {h}")

del df_sample
gc.collect()
