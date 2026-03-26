# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Integration test: Annual band generator save + inference modes.

Exercises the full pipeline:
  1. SAVE MODE — save_annual_training_data() for multiple months
  2. INFERENCE MODE — generate_annual_bands() loads training data, calibrates, applies bands

Uses a temp directory so nothing touches the production training store.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_annual_integration.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PERIOD_TYPE = "aq1"
CLASS_TYPE = "offpeak"
# Simulate 6 historical auction months
HISTORICAL_MONTHS = ["2023-01", "2023-04", "2023-07", "2024-01", "2024-04", "2024-07"]
INFERENCE_MONTH = "2025-01"


def make_synthetic_trades(auction_month: str, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic annual trades with realistic columns."""
    rng = np.random.default_rng(seed + hash(auction_month) % 10000)
    mtm = rng.normal(0, 500, n)
    mcp = mtm + rng.normal(0, 150, n)  # MCP = baseline + noise
    return pd.DataFrame({
        "source_id": [f"NODE_{i % 50}" for i in range(n)],
        "sink_id": [f"NODE_{(i + 25) % 50}" for i in range(n)],
        "path": [f"NODE_{i % 50},NODE_{(i + 25) % 50}" for i in range(n)],
        "period_type": PERIOD_TYPE,
        "class_type": CLASS_TYPE,
        "mtm_1st_mean": mtm,
        "mtm_2nd_mean": mtm * 0.95 + rng.normal(0, 10, n),
        "mtm_3rd_mean": mtm * 0.90 + rng.normal(0, 20, n),
        "mcp_mean": mcp,
        "trade_type": "buy",
        "auction_month": pd.Timestamp(auction_month + "-01"),
    })


def test_save_mode(tmp_dir: Path) -> None:
    """Test 1: Save training data for multiple historical months."""
    from pmodel.base.ftr24.v1.annual_band_generator import save_annual_training_data

    logger.info("=" * 70)
    logger.info("TEST 1: SAVE MODE")
    logger.info("=" * 70)

    save_path_template = str(tmp_dir / "auction_month={auction_month}/period_type={period_type}/class_type={class_type}")

    for month in HISTORICAL_MONTHS:
        trades = make_synthetic_trades(month, n=2000)
        try:
            save_annual_training_data(
                trades=trades,
                auction_month=month,
                period_type=PERIOD_TYPE,
                class_type=CLASS_TYPE,
                save_path=save_path_template,
            )
        except ValueError as e:
            # Expected — save_annual_training_data always raises to halt
            logger.info(f"  [OK] {month}: {e}")

    # Verify files were created
    parquet_files = list(tmp_dir.glob("auction_month=*/period_type=*/class_type=*/training.parquet"))
    logger.info(f"\n  Created {len(parquet_files)} parquet files:")
    for f in sorted(parquet_files):
        df = pd.read_parquet(f)
        month_part = f.parent.parent.parent.name
        logger.info(f"    {month_part}: {len(df)} rows, cols={list(df.columns)}")

    assert len(parquet_files) == len(HISTORICAL_MONTHS), (
        f"Expected {len(HISTORICAL_MONTHS)} files, got {len(parquet_files)}"
    )
    logger.info("\n  SAVE MODE: PASS ✓")


def test_inference_mode(tmp_dir: Path) -> None:
    """Test 2: Load training data and generate bands."""
    from pmodel.base.ftr24.v1.annual_band_generator import (
        BAND_ORDER,
        generate_annual_bands,
        load_annual_training_data,
    )

    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: INFERENCE MODE")
    logger.info("=" * 70)

    save_path_template = str(tmp_dir / "auction_month={auction_month}/period_type={period_type}/class_type={class_type}")

    # 2a. Test load_annual_training_data
    logger.info("\n--- 2a. load_annual_training_data ---")
    train_df = load_annual_training_data(
        auction_month=INFERENCE_MONTH,
        period_type=PERIOD_TYPE,
        class_type=CLASS_TYPE,
        base_path=save_path_template,
    )
    logger.info(f"  Loaded {len(train_df):,} training rows (from months < {INFERENCE_MONTH})")
    assert len(train_df) > 0, "No training data loaded!"
    assert "mtm_1st_mean" in train_df.columns
    assert "mcp_mean" in train_df.columns
    logger.info(f"  Columns: {list(train_df.columns)}")
    logger.info(f"  mtm_1st_mean stats: mean={train_df['mtm_1st_mean'].mean():.1f}, std={train_df['mtm_1st_mean'].std():.1f}")

    # 2b. Test generate_annual_bands (full pipeline)
    logger.info("\n--- 2b. generate_annual_bands ---")
    inference_trades = make_synthetic_trades(INFERENCE_MONTH, n=500, seed=999)
    # Remove mcp_mean from inference trades (wouldn't have it in production)
    inference_trades = inference_trades.drop(columns=["mcp_mean"])

    result = generate_annual_bands(
        trades=inference_trades,
        auction_month=INFERENCE_MONTH,
        period_type=PERIOD_TYPE,
        class_type=CLASS_TYPE,
        treat_all_as_buy=True,
        base_path=save_path_template,
    )

    logger.info(f"  Result: {len(result)} rows, {len(result.columns)} columns")

    # Check band columns
    for band in BAND_ORDER:
        assert band in result.columns, f"Missing band column: {band}"
    n_band_nan = result["upper_50"].isna().sum()
    logger.info(f"  Band NaN count (upper_50): {n_band_nan}/{len(result)}")

    # Check bid_price columns
    for j in range(1, 11):
        assert f"bid_price_{j}" in result.columns, f"Missing bid_price_{j}"
        assert f"clearing_prob_{j}" in result.columns, f"Missing clearing_prob_{j}"

    n_bid_nan = result["bid_price_1"].isna().sum()
    logger.info(f"  bid_price_1 NaN count: {n_bid_nan}/{len(result)}")

    # Verify descending sort (treat_all_as_buy=True)
    violations = 0
    for i in range(len(result)):
        bids = [result.iloc[i][f"bid_price_{j}"] for j in range(1, 11)]
        if any(pd.isna(b) for b in bids):
            continue
        for k in range(len(bids) - 1):
            if bids[k] < bids[k + 1]:
                violations += 1
                break
    logger.info(f"  Sort violations (should be 0): {violations}")
    assert violations == 0, f"{violations} rows have non-descending bid prices"

    # Print sample
    logger.info("\n  Sample output (first 3 rows):")
    bid_cols = [f"bid_price_{j}" for j in range(1, 6)] + [f"clearing_prob_{j}" for j in range(1, 6)]
    logger.info(f"\n{result[bid_cols].head(3).to_string()}")

    # Band width statistics
    logger.info("\n  Band width statistics:")
    for clabel in ["10", "50", "99"]:
        widths = result[f"upper_{clabel}"] - result[f"lower_{clabel}"]
        logger.info(f"    {clabel}% band: mean={widths.mean():.1f}, median={widths.median():.1f}, std={widths.std():.1f}")

    logger.info("\n  INFERENCE MODE: PASS ✓")


def test_empty_training_data(tmp_dir: Path) -> None:
    """Test 3: Inference with no training data (edge case)."""
    from pmodel.base.ftr24.v1.annual_band_generator import generate_annual_bands

    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: EMPTY TRAINING DATA (edge case)")
    logger.info("=" * 70)

    empty_dir = tmp_dir / "empty_store"
    empty_dir.mkdir(parents=True, exist_ok=True)
    empty_path = str(empty_dir / "auction_month={auction_month}/period_type={period_type}/class_type={class_type}")

    trades = make_synthetic_trades("2025-01", n=50, seed=777)
    trades = trades.drop(columns=["mcp_mean"])

    result = generate_annual_bands(
        trades=trades,
        auction_month="2025-01",
        period_type=PERIOD_TYPE,
        class_type=CLASS_TYPE,
        treat_all_as_buy=True,
        base_path=empty_path,
    )

    # Should return trades with NaN bands (graceful degradation)
    assert len(result) == 50
    assert result["bid_price_1"].isna().all(), "Expected NaN bid prices when no training data"
    logger.info(f"  Result: {len(result)} rows, all bid_price_1 NaN as expected")
    logger.info("\n  EMPTY TRAINING DATA: PASS ✓")


def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="annual_band_test_"))
    logger.info(f"Temp directory: {tmp_dir}")

    try:
        test_save_mode(tmp_dir)
        test_inference_mode(tmp_dir)
        test_empty_training_data(tmp_dir)

        logger.info("\n" + "=" * 70)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 70)
    finally:
        shutil.rmtree(tmp_dir)
        logger.info(f"\nCleaned up {tmp_dir}")


if __name__ == "__main__":
    main()
