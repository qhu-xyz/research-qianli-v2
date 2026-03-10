import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append("/home/xyz/workspace/research-spice-shadow-price-pred/src")

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.data_loader import PjmDataLoader
from shadow_price_prediction.iso_configs import PJM_ISO_CONFIG


def inspect_scaler():
    print("Loading training data for PJM (2023-07 to 2024-07)...")

    config = PredictionConfig(iso=PJM_ISO_CONFIG)
    loader = PjmDataLoader(config)

    start_date = pd.Timestamp("2023-07-01")
    end_date = start_date + pd.DateOffset(months=12)

    # Load 'f0' data primarily as that's usually the one with binding constraints
    # 'f1' is also relevant.
    # We will mimic the 'short_term' group (f0)

    try:
        data = loader.load_training_data(
            train_start=start_date,
            train_end=end_date,
            required_period_types={"f0", "f1", "f2", "q1", "q2", "q3", "q4"},
            branch_name="FOS-IRNT       1",
            verbose=False,
        )

        if data is None or data.empty:
            print("No data loaded.")
            return

        print(f"Loaded {len(data)} samples.")

        # Calculate forecast horizon to filter for 'f0' group logic
        # For f0, horizon is 0.
        # But let's just look at specific feature distribution across ALL loaded data first
        # because the Default Scaler is trained on the group data.

        # Filter for horizon=0 (f0)
        group_data = data[data["forecast_horizon"] == 0].copy()

        print(f"Loaded {len(data)} samples.")

        # Check statistics for each horizon group in PJM config
        from shadow_price_prediction.iso_configs import PJM_HORIZON_GROUPS

        for group in PJM_HORIZON_GROUPS:
            print(f"\nAnalyzing Group: {group.name} (Horizon {group.min_horizon}-{group.max_horizon})...")

            group_data = data[
                (data["forecast_horizon"] >= group.min_horizon) & (data["forecast_horizon"] <= group.max_horizon)
            ]

            if len(group_data) == 0:
                print("  No samples found.")
                continue

            print(f"  Samples: {len(group_data)}")

            feat_name = "prob_exceed_100"
            values = group_data[feat_name].values

            v_max = np.max(values)

            print(f"  Max: {v_max}")
            if v_max > 0:
                print(f"  Scale Factor (1/Max): {1.0 / v_max:.2f}")
                print(f"  0.05 Scales To: {0.05 / v_max:.4f}")
            else:
                print("  Max is 0 (Constant feature)")

            if abs((0.05 / v_max if v_max > 0 else 0) - 12) < 2.0:
                print("  !!! CANDIDATE MATCH FOUND !!!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_scaler()
