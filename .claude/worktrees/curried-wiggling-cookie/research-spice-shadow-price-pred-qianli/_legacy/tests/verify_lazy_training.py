import os
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.models import ShadowPriceModels


def verify_lazy_training():
    print("Verifying Lazy Training Optimization...")

    # Create Dummy Data
    n_samples = 20
    data = pd.DataFrame(
        {
            "branch_name": ["B1"] * n_samples * 4,
            "forecast_horizon": ([0] * n_samples) + ([1] * n_samples) + ([2] * n_samples) + ([4] * n_samples),
            "label": np.random.randn(n_samples * 4) * 100,
            "auction_month": [1] * n_samples * 4,
        }
    )

    config = PredictionConfig()
    # Add features
    for col in config.features.all_features:
        if col != "forecast_horizon":
            data[col] = np.random.rand(n_samples * 4)

    # Case 1: Only f0 (Horizon 0)
    print("\nCase 1: Testing f0 only (Horizon 0)")
    config.test_periods = [(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"))]  # h=0
    models = ShadowPriceModels(config)
    models.train_classifiers(data, test_branches={"B1"}, verbose=False)

    assert len(models.clf_ensembles_f0) > 0, "f0 should be trained"
    assert len(models.clf_ensembles_f1) == 0, "f1 should NOT be trained"
    assert len(models.clf_ensembles_medium) == 0, "medium should NOT be trained"
    assert len(models.clf_ensembles_long) == 0, "long should NOT be trained"
    print("✓ Correctly trained only f0")

    # Case 2: Only Medium (Horizon 2)
    print("\nCase 2: Testing medium only (Horizon 2)")
    config.test_periods = [(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01"))]  # h=2
    models = ShadowPriceModels(config)
    models.train_classifiers(data, test_branches={"B1"}, verbose=False)

    assert len(models.clf_ensembles_f0) == 0, "f0 should NOT be trained"
    assert len(models.clf_ensembles_f1) == 0, "f1 should NOT be trained"
    assert len(models.clf_ensembles_medium) > 0, "medium should be trained"
    assert len(models.clf_ensembles_long) == 0, "long should NOT be trained"
    print("✓ Correctly trained only medium")

    # Case 3: f0 and Long (Horizon 0 and 4)
    print("\nCase 3: Testing f0 and long (Horizon 0 and 4)")
    config.test_periods = [
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")),  # h=0
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-05-01")),  # h=4
    ]
    models = ShadowPriceModels(config)
    models.train_classifiers(data, test_branches={"B1"}, verbose=False)

    assert len(models.clf_ensembles_f0) > 0, "f0 should be trained"
    assert len(models.clf_ensembles_f1) == 0, "f1 should NOT be trained"
    assert len(models.clf_ensembles_medium) == 0, "medium should NOT be trained"
    assert len(models.clf_ensembles_long) > 0, "long should be trained"
    print("✓ Correctly trained f0 and long")

    print("\nAll lazy training checks passed!")


if __name__ == "__main__":
    verify_lazy_training()
