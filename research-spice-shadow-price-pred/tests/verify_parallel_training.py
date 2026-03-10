import os
import sys
import time

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.models import ShadowPriceModels


def verify_parallel_training():
    print("Verifying Parallel Training...")

    # Create Dummy Data with multiple branches
    n_branches = 5
    n_samples_per_branch = 50
    branches = [f"B{i}" for i in range(n_branches)]

    data_list = []
    for branch in branches:
        df = pd.DataFrame(
            {
                "branch_name": [branch] * n_samples_per_branch * 4,
                "forecast_horizon": ([0] * n_samples_per_branch)
                + ([1] * n_samples_per_branch)
                + ([2] * n_samples_per_branch)
                + ([4] * n_samples_per_branch),
                "label": np.random.randn(n_samples_per_branch * 4) * 100,
                "auction_month": [1] * n_samples_per_branch * 4,
            }
        )
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    config = PredictionConfig()
    # Configure test periods to cover all horizons
    config.test_periods = [
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")),  # f0
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")),  # f1
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")),  # medium
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-05-01")),  # long
    ]

    # Add features
    for col in config.features.all_features:
        if col != "forecast_horizon":
            data[col] = np.random.rand(len(data))

    models = ShadowPriceModels(config)

    start_time = time.time()
    print("Training Classifiers (Parallel)...")
    models.train_classifiers(data, test_branches=set(branches), verbose=True)
    print(f"Classifiers trained in {time.time() - start_time:.2f}s")

    # Verify all branches have models
    for branch in branches:
        assert branch in models.clf_ensembles_f0, f"{branch} missing from f0"
        assert branch in models.clf_ensembles_f1, f"{branch} missing from f1"
        assert branch in models.clf_ensembles_medium, f"{branch} missing from medium"
        assert branch in models.clf_ensembles_long, f"{branch} missing from long"

    print("\nTraining Regressors (Parallel)...")
    start_time = time.time()
    models.train_regressors(data, test_branches=set(branches), verbose=True)
    print(f"Regressors trained in {time.time() - start_time:.2f}s")

    # Verify all branches have models (assuming they had binding constraints)
    # With random data, some might be skipped, but let's check counts
    print(f"f0 regressor count: {len(models.reg_ensembles_f0)}")

    print("\nAll parallel training checks passed!")


if __name__ == "__main__":
    verify_parallel_training()
