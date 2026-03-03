import pandas as pd

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.iso_configs import PJM_HORIZON_GROUPS, PJM_ISO_CONFIG
from shadow_price_prediction.pipeline import ShadowPricePipeline

# Setup Config
config = PredictionConfig()
config.iso = PJM_ISO_CONFIG
config.horizon_groups = PJM_HORIZON_GROUPS
config.class_type = "onpeak"

# Initialize Pipeline
pipeline = ShadowPricePipeline(config)

# Define Test Period (same as in user notebook)
auction_month = pd.Timestamp("2025-11-01")
market_month = pd.Timestamp("2025-11-01")

print(f"Testing label loading for Auction: {auction_month}, Market: {market_month}")

# Load Test Data
# accessing the internal method to test directly
period_type = pipeline.data_loader.get_period_type(auction_month, market_month)
print(f"Period Type: {period_type}")

try:
    test_data = pipeline.data_loader.load_test_data_for_period(
        auction_month=auction_month, market_month=market_month, period_type=period_type, verbose=True
    )

    if test_data is not None:
        print(f"Test data loaded: {len(test_data)} rows")
        if "label" in test_data.columns:
            non_null_labels = test_data["label"].notna().sum()
            print(f"Label column present. Non-null count: {non_null_labels}")
            if non_null_labels > 0:
                print("SUCCESS: Labels are loaded!")
                print(test_data["label"].head())
            else:
                print("WARNING: Label column exists but all are NaN (might be no actual data available)")
        else:
            print("FAILURE: 'label' column NOT found in test data.")
    else:
        print("FAILURE: No test data loaded.")

except Exception as e:
    print(f"Error during loading: {e}")
    import traceback

    traceback.print_exc()
