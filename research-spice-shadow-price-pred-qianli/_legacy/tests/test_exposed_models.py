"""Test that trained models are exposed in pipeline after run."""

import pandas as pd

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.pipeline import ShadowPricePipeline

# Define a simple test period
test_periods = [(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01"))]

# Initialize pipeline
config = PredictionConfig()
pipeline = ShadowPricePipeline(config)

print("Running pipeline...")
# Run sequentially to avoid Ray complexity in test
pipeline.run(test_periods=test_periods, verbose=True, use_parallel=False, n_jobs=1)

print("\nChecking exposed models...")
auction_month = pd.Timestamp("2025-06-01")

if auction_month in pipeline.trained_models:
    print(f"✓ Found trained models for {auction_month}")
    models = pipeline.trained_models[auction_month]

    # Check if default ensemble is populated
    if hasattr(models, "clf_default_ensemble_f0") and len(models.clf_default_ensemble_f0) > 0:
        print(f"✓ Default ensemble f0 has {len(models.clf_default_ensemble_f0)} models")

        # Check feature importance access
        xgb_model = models.clf_default_ensemble_f0[0][0]
        if hasattr(xgb_model, "feature_importances_"):
            print("✓ Can access feature importances")
        else:
            print("❌ Model does not have feature_importances_")
    else:
        print("❌ Default ensemble f0 is empty")
else:
    print(f"❌ No trained models found for {auction_month}")
    print(f"Keys found: {list(pipeline.trained_models.keys())}")
