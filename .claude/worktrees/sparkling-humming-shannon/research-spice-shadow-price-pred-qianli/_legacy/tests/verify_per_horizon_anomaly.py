"""
Verification script for per-horizon never-binding detection.
Tests that branches are correctly identified as never-binding only for the appropriate horizons.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.anomaly_detection import AnomalyDetector
from shadow_price_prediction.config import PredictionConfig


def verify_per_horizon_anomaly():
    print("=" * 80)
    print("Verifying Per-Horizon Never-Binding Detection")
    print("=" * 80)

    config = PredictionConfig()
    anomaly_detector = AnomalyDetector(config)

    # Create synthetic training data
    # Branch A: never-binding at f0, but binding at f1
    # Branch B: never-binding at both f0 and f1
    # Branch C: binding at both f0 and f1

    n_samples = 50
    data_list = []

    # Branch A - f0 (never-binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["A"] * n_samples,
                "forecast_horizon": [0] * n_samples,
                "label": [0] * n_samples,  # Never binding
                "prob_exceed_100": np.random.rand(n_samples) * 0.5,  # Low flow
                "auction_month": [1] * n_samples,
            }
        )
    )

    # Branch A - f1 (binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["A"] * n_samples,
                "forecast_horizon": [1] * n_samples,
                "label": np.random.rand(n_samples) * 100,  # Has binding samples
                "prob_exceed_100": np.random.rand(n_samples) * 0.5,
                "auction_month": [1] * n_samples,
            }
        )
    )

    # Branch B - f0 (never-binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["B"] * n_samples,
                "forecast_horizon": [0] * n_samples,
                "label": [0] * n_samples,  # Never binding
                "prob_exceed_100": np.random.rand(n_samples) * 0.3,
                "auction_month": [1] * n_samples,
            }
        )
    )

    # Branch B - f1 (never-binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["B"] * n_samples,
                "forecast_horizon": [1] * n_samples,
                "label": [0] * n_samples,  # Never binding
                "prob_exceed_100": np.random.rand(n_samples) * 0.3,
                "auction_month": [1] * n_samples,
            }
        )
    )

    # Branch C - f0 (binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["C"] * n_samples,
                "forecast_horizon": [0] * n_samples,
                "label": np.random.rand(n_samples) * 100,  # Has binding samples
                "prob_exceed_100": np.random.rand(n_samples) * 0.7,
                "auction_month": [1] * n_samples,
            }
        )
    )

    # Branch C - f1 (binding)
    data_list.append(
        pd.DataFrame(
            {
                "branch_name": ["C"] * n_samples,
                "forecast_horizon": [1] * n_samples,
                "label": np.random.rand(n_samples) * 100,  # Has binding samples
                "prob_exceed_100": np.random.rand(n_samples) * 0.7,
                "auction_month": [1] * n_samples,
            }
        )
    )

    train_data = pd.concat(data_list, ignore_index=True)

    # Characterize never-binding branches for f0
    print("\n" + "=" * 80)
    print("Characterizing f0...")
    print("=" * 80)
    f0_data = train_data[train_data["forecast_horizon"] == 0]
    anomaly_detector.characterize_never_binding_branches(f0_data, "f0", verbose=True)

    # Characterize never-binding branches for f1
    print("\n" + "=" * 80)
    print("Characterizing f1...")
    print("=" * 80)
    f1_data = train_data[train_data["forecast_horizon"] == 1]
    anomaly_detector.characterize_never_binding_branches(f1_data, "f1", verbose=True)

    # Verify results
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)

    # Expected:
    # f0: A and B should be never-binding (C is binding)
    # f1: Only B should be never-binding (A and C are binding)

    f0_never_binding = anomaly_detector.never_binding_branches["f0"]
    f1_never_binding = anomaly_detector.never_binding_branches["f1"]

    print(f"\nf0 never-binding branches: {sorted(f0_never_binding)}")
    print(f"f1 never-binding branches: {sorted(f1_never_binding)}")

    # Assertions
    errors = []

    if "A" not in f0_never_binding:
        errors.append("✗ Branch A should be never-binding at f0")
    else:
        print("\n✓ Branch A is never-binding at f0")

    if "A" in f1_never_binding:
        errors.append("✗ Branch A should NOT be never-binding at f1")
    else:
        print("✓ Branch A is binding at f1")

    if "B" not in f0_never_binding:
        errors.append("✗ Branch B should be never-binding at f0")
    else:
        print("✓ Branch B is never-binding at f0")

    if "B" not in f1_never_binding:
        errors.append("✗ Branch B should be never-binding at f1")
    else:
        print("✓ Branch B is never-binding at f1")

    if "C" in f0_never_binding:
        errors.append("✗ Branch C should NOT be never-binding at f0")
    else:
        print("✓ Branch C is binding at f0")

    if "C" in f1_never_binding:
        errors.append("✗ Branch C should NOT be never-binding at f1")
    else:
        print("✓ Branch C is binding at f1")

    # Test anomaly detection with horizon group
    print("\n" + "=" * 80)
    print("Testing Anomaly Detection")
    print("=" * 80)

    # Create a test sample for Branch A at f0 (never-binding)
    test_sample_A = pd.Series(
        {
            "prob_exceed_100": 10.0  # Very high flow (anomaly)
        }
    )

    # Should detect anomaly at f0 (where A is never-binding)
    is_anomaly_f0, conf_f0, reason_f0 = anomaly_detector.detect_flow_anomaly(test_sample_A, "A", "f0")

    # Should NOT detect anomaly at f1 (where A is binding)
    is_anomaly_f1, conf_f1, reason_f1 = anomaly_detector.detect_flow_anomaly(test_sample_A, "A", "f1")

    print("\nBranch A anomaly detection:")
    print(f"  f0: is_anomaly={is_anomaly_f0}, confidence={conf_f0:.2f}")
    print(f"      {reason_f0}")
    print(f"  f1: is_anomaly={is_anomaly_f1}, confidence={conf_f1:.2f}")
    print(f"      {reason_f1}")

    if is_anomaly_f0:
        print("\n✓ Correctly detected anomaly at f0 for Branch A")
    else:
        errors.append("✗ Should detect anomaly at f0 for Branch A")

    if not is_anomaly_f1:
        print("✓ Correctly did NOT detect anomaly at f1 for Branch A (no stats)")
    else:
        errors.append("✗ Should NOT detect anomaly at f1 for Branch A")

    # Final result
    print("\n" + "=" * 80)
    if errors:
        print("VERIFICATION FAILED:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)


if __name__ == "__main__":
    verify_per_horizon_anomaly()
