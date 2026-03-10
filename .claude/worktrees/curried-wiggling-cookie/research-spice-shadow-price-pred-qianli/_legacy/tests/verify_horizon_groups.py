import os
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.models import ShadowPriceModels


def verify_horizon_groups():
    print("Verifying Horizon Groups...")

    # 1. Setup Config
    config = PredictionConfig()
    models = ShadowPriceModels(config)

    # 2. Create Dummy Data
    # Create data for f0 (h=0), f1 (h=1), medium (h=2), long (h=4)
    # Branch 'B1' for all
    n_samples = 50
    data = pd.DataFrame(
        {
            "branch_name": ["B1"] * n_samples * 4,
            "forecast_horizon": ([0] * n_samples) + ([1] * n_samples) + ([2] * n_samples) + ([4] * n_samples),
            "label": np.random.randn(n_samples * 4) * 100,
            "auction_month": [1] * n_samples * 4,
        }
    )

    # Add features
    for col in config.features.all_features:
        if col != "forecast_horizon":
            data[col] = np.random.rand(n_samples * 4)

    # 3. Train Classifiers
    print("Training Classifiers...")
    models.train_classifiers(data, test_branches={"B1"}, verbose=True)

    # 4. Verify Storage
    print("\nVerifying Storage:")
    print(f"f0 ensemble size: {len(models.clf_ensembles_f0)}")
    print(f"f1 ensemble size: {len(models.clf_ensembles_f1)}")
    print(f"medium ensemble size: {len(models.clf_ensembles_medium)}")
    print(f"long ensemble size: {len(models.clf_ensembles_long)}")

    assert "B1" in models.clf_ensembles_f0, "B1 missing from f0"
    assert "B1" in models.clf_ensembles_f1, "B1 missing from f1"
    assert "B1" in models.clf_ensembles_medium, "B1 missing from medium"
    assert "B1" in models.clf_ensembles_long, "B1 missing from long"

    # 5. Verify Retrieval
    print("\nVerifying Retrieval:")

    # f0
    ens_f0, _, _ = models.get_classifier_ensemble("B1", 0)
    assert ens_f0 == models.clf_ensembles_f0["B1"], "Retrieval mismatch for f0"
    print("f0 retrieval correct")

    # f1
    ens_f1, _, _ = models.get_classifier_ensemble("B1", 1)
    assert ens_f1 == models.clf_ensembles_f1["B1"], "Retrieval mismatch for f1"
    print("f1 retrieval correct")

    # medium
    ens_med, _, _ = models.get_classifier_ensemble("B1", 2)
    assert ens_med == models.clf_ensembles_medium["B1"], "Retrieval mismatch for medium"
    print("medium retrieval correct")

    # long
    ens_long, _, _ = models.get_classifier_ensemble("B1", 4)
    assert ens_long == models.clf_ensembles_long["B1"], "Retrieval mismatch for long"
    print("long retrieval correct")

    print("\nAll checks passed!")


if __name__ == "__main__":
    verify_horizon_groups()
