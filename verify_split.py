from unittest.mock import patch

import pandas as pd

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.pipeline import ShadowPricePipeline

# Mock config
config = PredictionConfig()

# Create Pipeline
pipeline = ShadowPricePipeline(config)

# Mock _process_auction_month to return dummy data
# Signature: results_per_outage, final_results, auction_month, market_month
# Return: list[tuple], models, detector, train_data, auction_month
dummy_auc_month = pd.Timestamp("2025-01-01")
dummy_train_data = pd.DataFrame({"col": [1, 2, 3]})
dummy_result = (
    [],  # results list
    None,  # models
    None,  # detector
    dummy_train_data,
    dummy_auc_month,
)

# Patch the module-level function used by parallel execution or the method if it was a method (it's a standalone function in pipeline.py)
# We need to patch 'shadow_price_prediction.pipeline._process_auction_month'
with patch("shadow_price_prediction.pipeline._process_auction_month", return_value=dummy_result) as mock_process:
    # Run pipeline with dummy inputs
    # use_parallel=False to avoid Ray complexity and use direct function call (which we patched? No, parallel_equal_pool calls it)
    # If use_parallel=False, it calls _process_auction_month directly.

    test_periods = [(dummy_auc_month, pd.Timestamp("2025-02-01"))]

    # call run
    results = pipeline.run(test_periods=test_periods, verbose=True, use_parallel=False, train_only=False)

    # Check results[3] which should be the train_data dict
    train_data_result = results[3]

    print("\n--- Verification ---")
    print(f"Result type: {type(train_data_result)}")
    if isinstance(train_data_result, dict):
        print("SUCCESS: Return value is a dictionary.")
        if dummy_auc_month in train_data_result:
            print(f"SUCCESS: Dictionary contains key {dummy_auc_month}.")
            print(f"Data shape: {train_data_result[dummy_auc_month].shape}")
        else:
            print(f"FAILURE: Dictionary missing key {dummy_auc_month}.")
    else:
        print(f"FAILURE: Return value is not a dictionary. Got {type(train_data_result)}")

    # Check pipeline attribute
    print(f"\nPipeline attribute type: {type(pipeline.train_data)}")
    if isinstance(pipeline.train_data, dict):
        print("SUCCESS: pipeline.train_data is a dictionary.")
    else:
        print("FAILURE: pipeline.train_data is not a dictionary.")
