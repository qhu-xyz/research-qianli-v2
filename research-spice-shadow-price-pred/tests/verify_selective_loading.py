"""
Verification script for selective data loading optimization.
Tests that only required period types are loaded based on test_periods.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.data_loader import DataLoader


def verify_selective_loading():
    print("=" * 80)
    print("Verifying Selective Data Loading")
    print("=" * 80)

    # Test 1: f0 only
    test_periods_f0 = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"))]
    required = DataLoader.get_required_period_types(test_periods_f0)
    print("\nTest 1: f0 only")
    print(f"  Test periods: {test_periods_f0}")
    print(f"  Required period types: {sorted(required)}")
    assert required == {"f0"}, f"Expected {{'f0'}}, got {required}"
    print("  ✓ Passed")

    # Test 2: f0 and f1
    test_periods_f0_f1 = [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")),  # f0
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")),  # f1
    ]
    required = DataLoader.get_required_period_types(test_periods_f0_f1)
    print("\nTest 2: f0 and f1")
    print(f"  Test periods: {len(test_periods_f0_f1)} periods")
    print(f"  Required period types: {sorted(required)}")
    assert required == {"f0", "f1"}, f"Expected {{'f0', 'f1'}}, got {required}"
    print("  ✓ Passed")

    # Test 3: Medium term (f2-f3)
    test_periods_medium = [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")),  # f2
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")),  # f3
    ]
    required = DataLoader.get_required_period_types(test_periods_medium)
    print("\nTest 3: Medium term (f2, f3)")
    print(f"  Test periods: {len(test_periods_medium)} periods")
    print(f"  Required period types: {sorted(required)}")
    assert required == {"f2", "f3"}, f"Expected {{'f2', 'f3'}}, got {required}"
    print("  ✓ Passed")

    # Test 4: Long term (quarters)
    test_periods_long = [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"))  # horizon 5 -> quarters
    ]
    required = DataLoader.get_required_period_types(test_periods_long)
    print("\nTest 4: Long term (quarters)")
    print(f"  Test periods: {test_periods_long}")
    print(f"  Required period types: {sorted(required)}")
    assert "q2" in required and "q3" in required and "q4" in required, f"Expected quarters in {required}"
    print("  ✓ Passed")

    # Test 5: Mixed horizons
    test_periods_mixed = [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")),  # f0
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")),  # f1
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")),  # quarters
    ]
    required = DataLoader.get_required_period_types(test_periods_mixed)
    print("\nTest 5: Mixed horizons (f0, f1, quarters)")
    print(f"  Test periods: {len(test_periods_mixed)} periods")
    print(f"  Required period types: {sorted(required)}")
    assert "f0" in required and "f1" in required, f"Expected f0 and f1 in {required}"
    assert "q2" in required or "q3" in required or "q4" in required, f"Expected at least one quarter in {required}"
    print("  ✓ Passed")

    # Test 6: None (load all)
    required = DataLoader.get_required_period_types(None)
    print("\nTest 6: None (should load all period types)")
    print(f"  Required period types: {sorted(required)}")
    assert required == {"f0", "f1", "f2", "f3", "q2", "q3", "q4"}, f"Expected all period types, got {required}"
    print("  ✓ Passed")

    # Test 7: Empty list (load all)
    required = DataLoader.get_required_period_types([])
    print("\nTest 7: Empty list (should load all period types)")
    print(f"  Required period types: {sorted(required)}")
    assert required == {"f0", "f1", "f2", "f3", "q2", "q3", "q4"}, f"Expected all period types, got {required}"
    print("  ✓ Passed")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    verify_selective_loading()
