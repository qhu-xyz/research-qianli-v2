"""Pre-fetch and cache realized DA shadow prices for all 28 quarters."""
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import ray
from ml.config import PLANNING_YEARS, AQ_ROUNDS
from ml.data_loader import load_v61_group
from ml.ground_truth import get_ground_truth

for year in PLANNING_YEARS:
    for aq in AQ_ROUNDS:
        try:
            v61 = load_v61_group(year, aq)
            get_ground_truth(year, aq, v61, cache=True)
        except Exception as e:
            print(f"[cache] ERROR {year}/{aq}: {e}")

ray.shutdown()
print("[cache] Done. All ground truth cached.")
