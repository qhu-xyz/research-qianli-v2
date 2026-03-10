import os
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.config import PredictionConfig
from shadow_price_prediction.models import ShadowPriceModels
from shadow_price_prediction.prediction import Predictor


def verify_prediction_fix():
    print("Verifying Prediction Fix...")

    # 1. Setup Config & Models
    config = PredictionConfig()
    # Configure test periods to cover all horizons to ensure models are initialized
    config.test_periods = [
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")),  # f0
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")),  # f1
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01")),  # medium
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-05-01")),  # long
    ]
    models = ShadowPriceModels(config)

    # Manually populate some dummy models to avoid training
    # This is just to ensure get_classifier_ensemble returns something valid
    # But wait, get_classifier_ensemble returns default if empty.
    # However, we want to test the "has_branch_model" logic.
    # So we need to populate branch ensembles.

    models.clf_ensembles_f0["B1"] = [("dummy", 1.0)]
    models.clf_ensembles_f1["B1"] = [("dummy", 1.0)]
    models.clf_ensembles_medium["B1"] = [("dummy", 1.0)]
    models.clf_ensembles_long["B1"] = [("dummy", 1.0)]

    models.reg_ensembles_f0["B1"] = [("dummy", 1.0)]
    models.reg_ensembles_f1["B1"] = [("dummy", 1.0)]
    models.reg_ensembles_medium["B1"] = [("dummy", 1.0)]
    models.reg_ensembles_long["B1"] = [("dummy", 1.0)]

    from shadow_price_prediction.prediction import AnomalyDetector

    anomaly_detector = AnomalyDetector(config)
    predictor = Predictor(models, config, anomaly_detector)

    # 2. Create Dummy Test Data
    # We need data for B1 at different horizons
    # Predictor expects multi-index or columns?
    # It iterates unique branches.
    # It expects 'branch_name', 'forecast_horizon', 'auction_month', 'market_month'

    n_samples = 10
    data = pd.DataFrame(
        {
            "branch_name": ["B1"] * n_samples * 4,
            "forecast_horizon": ([0] * n_samples) + ([1] * n_samples) + ([2] * n_samples) + ([4] * n_samples),
            "auction_month": [1] * n_samples * 4,
            "market_month": [1] * n_samples * 4,  # Dummy
            "outage_date": [pd.Timestamp("2025-01-01")] * n_samples * 4,
            "constraint_id": range(n_samples * 4),
            "flow_direction": ["Positive"] * n_samples * 4,
        }
    )

    # Add features
    for col in config.features.all_features:
        if col not in data.columns:
            data[col] = np.random.rand(len(data))

    # Set index as expected by Predictor?
    # Predictor.predict takes test_data.
    # It doesn't enforce index, but uses columns.

    # 3. Run Prediction
    # We need to mock predict_ensemble because we have dummy models
    # Or we can just catch the error inside predict_ensemble if we can't mock it easily.
    # But the AttributeError happened BEFORE prediction, at "has_branch_model = ..."

    # Let's try to run it and see if it crashes with AttributeError.
    # It will likely crash with "dummy object has no predict" inside predict_ensemble,
    # but that means it passed the AttributeError check!

    print("Running prediction...")
    try:
        predictor.predict(data, verbose=True)
    except AttributeError as e:
        if "clf_ensembles_short" in str(e) or "reg_ensembles_short" in str(e):
            print(f"❌ FAILED: AttributeError detected: {e}")
            sys.exit(1)
        else:
            print(f"✓ Caught expected error (due to dummy models): {e}")
            # This is fine, we just wanted to pass the attribute check
    except Exception as e:
        print(f"✓ Caught expected error (due to dummy models): {e}")

    print("✓ No legacy attribute error detected!")


if __name__ == "__main__":
    verify_prediction_fix()
