"""
Diagnostic script to verify AUCTION_SCHEDULE is loaded correctly
and check what training data would be loaded.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd

from shadow_price_prediction.config import AUCTION_SCHEDULE

print("=" * 80)
print("AUCTION_SCHEDULE Diagnostic")
print("=" * 80)

print("\nCurrent AUCTION_SCHEDULE configuration:")
for month, periods in sorted(AUCTION_SCHEDULE.items()):
    print(f"  Month {month:2d}: {periods}")

print("\n" + "=" * 80)
print("Training Data Analysis for Auction Month July 2025 (month 7)")
print("=" * 80)

# Simulate what would be loaded for July 2025 (month 7)
auction_month = pd.Timestamp("2025-07-01")
available_periods = AUCTION_SCHEDULE.get(auction_month.month, [])

print(f"\nAuction Month: {auction_month.strftime('%Y-%m')}")
print(f"Month number: {auction_month.month}")
print(f"Available periods from AUCTION_SCHEDULE: {available_periods}")

if len(available_periods) == 1 and available_periods[0] == "f0":
    print("\n⚠️  WARNING: Only 'f0' is configured!")
    print("   This means training data will ONLY include f0 periods.")
elif len(available_periods) > 1:
    print("\n✓ Multiple periods configured")
    print(f"   Training data will include periods: {', '.join(available_periods)}")

print("\n" + "=" * 80)
