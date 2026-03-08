"""Test ground truth fetching for one quarter. Requires Ray."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

from ml.data_loader import load_v61_group
from ml.ground_truth import get_ground_truth

v61 = load_v61_group("2024-06", "aq1")
result = get_ground_truth("2024-06", "aq1", v61, cache=True)
print(f"\nResult: {len(result)} rows")
n_binding = len(result.filter(result["realized_shadow_price"] > 0))
print(f"Binding: {n_binding}/{len(result)} ({100*n_binding/len(result):.1f}%)")
print("\nTop 10 binding constraints:")
print(result.filter(result["realized_shadow_price"] > 0)
      .sort("realized_shadow_price", descending=True)
      .select(["branch_name", "realized_shadow_price"])
      .head(10))

import ray
ray.shutdown()
