"""Quick debug: check annual cleared data structure."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

from pbase.data.dataset.ftr.cleared.pjm import PjmClearedFtrs
cleared_loader = PjmClearedFtrs()

cleared = cleared_loader._load_annual(planning_year=2020)
print(f"Shape: {cleared.shape}")
print(f"Columns: {list(cleared.columns)}")
print(f"\nclass_type values: {cleared['class_type'].unique()}")
print(f"\nSample rows:")
print(cleared.head(3).to_string())

# Check how to get source/sink node IDs
# The MCP loader uses pnode_id, but cleared uses node names
# We need to map names to IDs using the MCP data
from pbase.data.dataset.ftr.mcp.pjm import PjmMcp
mcp_loader = PjmMcp()
df = mcp_loader.load_data(auction_month='2020-06-01', market_round=1, period_type='f0')
print(f"\nMCP columns: {list(df.columns)}")
print(f"MCP sample:")
print(df.head(3)[['node_name', 'pnode_id', 'class_type']].to_string())

import ray
ray.shutdown()
