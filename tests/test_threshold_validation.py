"""Test that default threshold must be set during training."""

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.models import ShadowPriceModels

# Create a model without training
config = PredictionConfig()
models = ShadowPriceModels(config)

print("Testing that default threshold validation works...")
print(f"Initial threshold value: {models.optimal_threshold_default_f1}")

try:
    # Try to get classifier ensemble for a branch that doesn't exist (would use default)
    ensemble, threshold, scaler = models.get_classifier_ensemble("test_branch", horizon=1)
    print("ERROR: Should have raised ValueError but didn't!")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}")
except Exception as e:
    print(f"ERROR: Raised wrong exception type: {type(e).__name__}: {e}")

print("\nTest complete!")
