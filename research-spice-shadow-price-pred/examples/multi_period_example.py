"""
Example: Multi-Period Shadow Price Prediction

This example demonstrates how to use the updated framework to predict
shadow prices for different period types (f0, f1, f2, f3, q2, q3, q4).
"""

import pandas as pd

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.pipeline import ShadowPricePipeline

# Example 1: Predicting for multiple period types
# ================================================

# Define test periods as (auction_month, market_month) pairs
test_periods = [
    # f0: Prompt month (auction = market)
    (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01")),
    # f1: One month ahead
    (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")),
    # f2: Two months ahead
    (pd.Timestamp("2025-08-01"), pd.Timestamp("2025-10-01")),
    # q2: Quarterly (Sep-Nov), testing September
    (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-01")),
    # q3: Quarterly (Dec-Feb), testing December
    (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-01")),
]

# Initialize configuration
config = PredictionConfig()

# Initialize pipeline
pipeline = ShadowPricePipeline(config)

# Run predictions
print("Running multi-period predictions...")
results_per_outage, final_results, metrics = pipeline.run(
    test_periods=test_periods, verbose=True, use_parallel=False, n_jobs=1
)

# Analyze results by forecast horizon
print("\n" + "=" * 80)
print("Results by Forecast Horizon")
print("=" * 80)

# Add horizon info to results
results_per_outage["horizon"] = (
    results_per_outage["market_month"].dt.year - results_per_outage["auction_month"].dt.year
) * 12 + (results_per_outage["market_month"].dt.month - results_per_outage["auction_month"].dt.month)

# Group by horizon
for horizon in sorted(results_per_outage["horizon"].unique()):
    horizon_data = results_per_outage[results_per_outage["horizon"] == horizon]

    print(f"\nHorizon {horizon} months:")
    print(f"  Samples: {len(horizon_data):,}")
    print(f"  F1 Score: {metrics.get('monthly', {}).get('F1', 'N/A')}")
    print(f"  MAE: {metrics.get('monthly', {}).get('MAE', 'N/A'):.2f}")
    print(f"  RMSE: {metrics.get('monthly', {}).get('RMSE', 'N/A'):.2f}")

# Example 2: Analyzing feature importance
# ========================================

print("\n" + "=" * 80)
print("Feature Analysis")
print("=" * 80)

# Check if forecast_horizon is being used
if "forecast_horizon" in pipeline.models.default_classifiers[0].model.feature_names_in_:
    # For XGBoost models
    feature_importance = pipeline.models.default_classifiers[0].model.feature_importances_
    feature_names = pipeline.models.default_classifiers[0].model.feature_names_in_

    # Find horizon importance
    horizon_idx = list(feature_names).index("forecast_horizon")
    print(f"\nForecast Horizon Importance: {feature_importance[horizon_idx]:.4f}")

    # Top 5 features
    top_features = sorted(zip(feature_names, feature_importance, strict=False), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

# Save results
print("\n" + "=" * 80)
print("Saving Results")
print("=" * 80)

results_per_outage.to_parquet("multi_period_results_per_outage.parquet")
final_results.to_parquet("multi_period_final_results.parquet")

print("✓ Results saved to:")
print("  - multi_period_results_per_outage.parquet")
print("  - multi_period_final_results.parquet")
