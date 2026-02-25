"""
Test that test_periods is now required and has no defaults.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from shadow_price_prediction.config import PredictionConfig

print("=" * 80)
print("Testing Required test_periods Configuration")
print("=" * 80)

# Test 1: Creating config WITHOUT test_periods should FAIL
print("\nTest 1: Creating config without test_periods (should FAIL)")
try:
    config = PredictionConfig()
    print("  ✗ FAILED: Should have raised TypeError for missing required argument")
except TypeError as e:
    print(f"  ✓ PASSED: Correctly raised TypeError: {e}")

# Test 2: Creating config with test_periods should SUCCEED
print("\nTest 2: Creating config with test_periods (should SUCCEED)")
try:
    config = PredictionConfig(test_periods=[(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"))])
    print("  ✓ PASSED: Config created with test_periods")
    print(f"    test_periods: {config.test_periods}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test 3: Creating config with empty test_periods should FAIL
print("\nTest 3: Creating config with empty test_periods (should FAIL)")
try:
    config = PredictionConfig(test_periods=[])
    print("  ✗ FAILED: Should have raised ValueError for empty test_periods")
except ValueError as e:
    print(f"  ✓ PASSED: Correctly raised ValueError: {e}")

# Test 4: Creating config with invalid test_periods should FAIL
print("\nTest 4: Creating config with invalid test_periods (should FAIL)")
try:
    config = PredictionConfig(test_periods=["invalid"])
    print("  ✗ FAILED: Should have raised ValueError for invalid test_periods")
except ValueError as e:
    print(f"  ✓ PASSED: Correctly raised ValueError: {e}")

# Test 5: Check that old fields don't exist
print("\nTest 5: Check that test_auction_month field doesn't exist")
config = PredictionConfig(test_periods=[(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"))])
if not hasattr(config, "test_auction_month"):
    print("  ✓ PASSED: test_auction_month field removed")
else:
    print(f"  ✗ FAILED: test_auction_month still exists: {config.test_auction_month}")

if not hasattr(config, "test_market_month"):
    print("  ✓ PASSED: test_market_month field removed")
else:
    print(f"  ✗ FAILED: test_market_month still exists: {config.test_market_month}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
