import pandas as pd

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.iso_configs import PJM_HORIZON_GROUPS, PJM_ISO_CONFIG
from shadow_price_prediction.pipeline import ShadowPricePipeline

# Setup Config
config = PredictionConfig()
config.iso = PJM_ISO_CONFIG
config.horizon_groups = PJM_HORIZON_GROUPS
config.class_type = "onpeak"
config.training.test_unbind_rule = None  # Disable unbind rule to ensure we get results

# Initialize Pipeline
pipeline = ShadowPricePipeline(config)

# Define Test Period (same as in user notebook)
auction_month = pd.Timestamp("2025-11-01")
market_month = pd.Timestamp("2025-11-01")
test_periods = [(auction_month, market_month)]

print(f"Testing label loading and final_results for Auction: {auction_month}, Market: {market_month}")

try:
    # Run full pipeline for prediction
    # We need models to predict, but we can try to run with what we have or just check if we can reach the aggregation step
    # Since we don't want to train models, we can assume models are not trained and it might fail at prediction.
    # To verify `_create_results_dataframes` we need `predict` to finish.

    # Actually, we can just instantiate Predictor and call _create_results_dataframes manually with dummy data?
    # Or just run pipeline in 'predict_only=False' (it trains) which takes time.
    # The previous verify_fix.py only checked data loader.

    # Let's inspect `predict` method output.
    # If we don't have trained models, we can't run predict easily.
    # However, user's notebook was running pipeline.run(), which trains and predicts.

    # Let's modify pipeline.run to return quickly or check `_create_results_dataframes` via unit test approach.

    from shadow_price_prediction.anomaly_detection import AnomalyDetector
    from shadow_price_prediction.models import ShadowPriceModels
    from shadow_price_prediction.prediction import Predictor

    models = ShadowPriceModels(config)
    anomaly_detector = AnomalyDetector(config)
    predictor = Predictor(config, models, anomaly_detector)

    # Create dummy test data with labels
    test_data = pd.DataFrame(
        {
            "constraint_id": ["c1", "c2"],
            "flow_direction": [1, 1],
            "branch_name": ["b1", "b2"],
            "outage_date": [pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-01")],
            "auction_month": [auction_month, auction_month],
            "market_month": [market_month, market_month],
            "label": [10.0, 20.0],
            "hist_da": [5.0, 5.0],
        }
    )
    # Add dummy features
    for f in config.features.all_features:
        test_data[f] = 0.5

    import numpy as np

    y_pred_binary = np.array([1, 1])
    y_pred_proba = np.array([0.9, 0.8])
    y_pred_proba_scaled = np.array([0.9, 0.8])
    y_pred_threshold = np.array([0.5, 0.5])
    y_pred_shadow_price = np.array([12.0, 18.0])
    model_used = ["test", "test"]

    results_per_outage, final_results = predictor._create_results_dataframes(
        test_data,
        y_pred_binary,
        y_pred_proba,
        y_pred_proba_scaled,
        y_pred_threshold,
        y_pred_shadow_price,
        model_used,
        verbose=True,
    )

    print("Final Results Columns:", final_results.columns.tolist())

    if "da_label" in final_results.columns:
        print("SUCCESS: 'da_label' column found in final_results!")
        print(final_results[["da_label", "actual_shadow_price"]])
    else:
        print("FAILURE: 'da_label' column NOT found.")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
