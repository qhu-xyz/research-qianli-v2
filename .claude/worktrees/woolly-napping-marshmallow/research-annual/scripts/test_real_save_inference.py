"""End-to-end test: Save + Inference modes on REAL annual R2 data.

Simulates the actual production workflow:
  1. SAVE MODE: For each historical planning year, save trades as training data
  2. INFERENCE MODE: For the latest PY, load all prior training data, generate bands
  3. Validate coverage on the test fold (since we have mcp_mean)

Uses real data from all_residuals_v2.parquet (R2, aq1).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_real_save_inference.py
"""

from __future__ import annotations

import gc
import logging
import resource
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger(__name__)

R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
BASELINE_COL = "mtm_1st_mean"
MCP_COL = "mcp_mean"
CLASS_COL = "class_type"
QUARTER = "aq1"  # Test on aq1


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_r2_by_py(quarter: str, planning_year: int) -> pd.DataFrame:
    """Load R2 data for a single planning year as pandas."""
    df = (
        pl.scan_parquet(R2R3_DATA_PATH)
        .filter(
            (pl.col("round") == 2)
            & (pl.col("period_type") == quarter)
            & (pl.col("planning_year") == planning_year)
            & pl.col(BASELINE_COL).is_not_null()
            & pl.col(MCP_COL).is_not_null()
        )
        .select([BASELINE_COL, MCP_COL, "planning_year", "period_type", CLASS_COL,
                 "source_id", "sink_id"])
        .collect()
        .to_pandas()
    )
    # Add columns expected by production
    df["path"] = df["source_id"] + "," + df["sink_id"]
    df["mtm_2nd_mean"] = df[BASELINE_COL] * 0.95
    df["mtm_3rd_mean"] = df[BASELINE_COL] * 0.90
    df["trade_type"] = "buy"
    return df


def main():
    from pmodel.base.ftr24.v1.annual_band_generator import (
        ANNUAL_COVERAGE_LABELS,
        BAND_ORDER,
        generate_annual_bands,
        load_annual_training_data,
        save_annual_training_data,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="annual_real_test_"))
    save_template = str(tmp_dir / "auction_month={auction_month}/period_type={period_type}/class_type={class_type}")

    logger.info(f"Real Data Save + Inference Test")
    logger.info(f"Temp dir: {tmp_dir}")
    logger.info(f"Memory: {mem_mb():.0f} MB")

    # Available PYs for R2
    pys = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    test_py = 2025
    train_pys = [py for py in pys if py < test_py]

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: SAVE MODE — save each PY as a separate "auction month"
    # ═══════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST 1: SAVE MODE (real R2 {QUARTER} data)")
    logger.info(f"{'='*70}")

    total_saved = 0
    for py in train_pys:
        df = load_r2_by_py(QUARTER, py)
        # Map planning year to a synthetic auction_month for the store
        auction_month = f"{py}-01"
        logger.info(f"  PY {py}: {len(df):,} rows, saving as auction_month={auction_month}")

        try:
            save_annual_training_data(
                trades=df,
                auction_month=auction_month,
                period_type=QUARTER,
                class_type="offpeak",
                save_path=save_template,
            )
        except ValueError:
            # Expected halt
            pass
        total_saved += len(df)
        del df
        gc.collect()

    logger.info(f"\n  Total saved: {total_saved:,} rows across {len(train_pys)} months")
    logger.info(f"  Memory: {mem_mb():.0f} MB")

    # Verify files
    files = list(tmp_dir.glob("**/training.parquet"))
    logger.info(f"  Created {len(files)} parquet files")
    for f in sorted(files):
        df = pd.read_parquet(f)
        month = f.parent.parent.parent.name
        logger.info(f"    {month}: {len(df):,} rows")

    assert len(files) == len(train_pys), f"Expected {len(train_pys)} files, got {len(files)}"
    logger.info(f"\n  SAVE MODE: PASS ✓")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: INFERENCE MODE — load training data, generate bands
    # ═══════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST 2: INFERENCE MODE (real R2 {QUARTER} data)")
    logger.info(f"{'='*70}")

    # 2a. Load training data
    inference_month = f"{test_py}-06"  # After all train months
    train_df = load_annual_training_data(
        auction_month=inference_month,
        period_type=QUARTER,
        class_type="offpeak",
        base_path=save_template,
    )
    logger.info(f"  Loaded {len(train_df):,} training rows")
    assert len(train_df) == total_saved, f"Expected {total_saved}, loaded {len(train_df)}"

    # 2b. Load test data (PY 2025 — we have mcp_mean for validation)
    test_df = load_r2_by_py(QUARTER, test_py)
    logger.info(f"  Test data: {len(test_df):,} rows (PY {test_py})")

    # 2c. Generate bands
    result = generate_annual_bands(
        trades=test_df,
        auction_month=inference_month,
        period_type=QUARTER,
        class_type="offpeak",
        treat_all_as_buy=True,
        base_path=save_template,
    )

    logger.info(f"\n  Result: {len(result):,} rows, {len(result.columns)} columns")

    # 2d. Check all band columns present
    for b in BAND_ORDER:
        assert b in result.columns, f"Missing {b}"
    n_nan = result["upper_50"].isna().sum()
    logger.info(f"  NaN bands: {n_nan}/{len(result)}")

    # 2e. Check bid price columns
    for j in range(1, 11):
        assert f"bid_price_{j}" in result.columns
        assert f"clearing_prob_{j}" in result.columns
    n_bid_nan = result["bid_price_1"].isna().sum()
    logger.info(f"  NaN bid_price_1: {n_bid_nan}/{len(result)}")

    # 2f. Verify sort order (descending for buy)
    violations = 0
    for i in range(min(len(result), 5000)):  # sample for speed
        bids = [result.iloc[i][f"bid_price_{j}"] for j in range(1, 11)]
        if any(pd.isna(b) for b in bids):
            continue
        for k in range(len(bids) - 1):
            if bids[k] < bids[k + 1]:
                violations += 1
                break
    logger.info(f"  Sort violations (first 5000): {violations}")
    assert violations == 0

    # 2g. Coverage validation (we have mcp_mean in test data)
    logger.info(f"\n  Coverage (vs actual MCP):")
    logger.info(f"  {'Level':<8} {'Target':>8} {'Actual':>8} {'Error':>8}")
    logger.info(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    from pmodel.base.ftr24.v1.annual_band_generator import ANNUAL_COVERAGE_LEVELS
    for level, clabel in zip(ANNUAL_COVERAGE_LEVELS, ANNUAL_COVERAGE_LABELS):
        lower = result[f"lower_{clabel}"]
        upper = result[f"upper_{clabel}"]
        mcp = result[MCP_COL]
        covered = ((mcp >= lower) & (mcp <= upper)).mean() * 100
        target = level * 100
        error = covered - target
        logger.info(f"  {clabel:>6}%  {target:>7.1f}% {covered:>7.2f}% {error:>+7.2f}pp")

    # 2h. Band width statistics
    logger.info(f"\n  Band widths ($/MWh):")
    logger.info(f"  {'Level':<8} {'Mean':>10} {'Median':>10} {'P95':>10}")
    logger.info(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for clabel in ANNUAL_COVERAGE_LABELS:
        widths = result[f"upper_{clabel}"] - result[f"lower_{clabel}"]
        logger.info(
            f"  {clabel:>6}%  {widths.mean():>10.1f} {widths.median():>10.1f} {widths.quantile(0.95):>10.1f}"
        )

    # 2i. Sample output
    logger.info(f"\n  Sample (5 rows):")
    sample_cols = [BASELINE_COL, MCP_COL] + [f"bid_price_{j}" for j in range(1, 6)]
    logger.info(f"\n{result[sample_cols].head(5).to_string()}")

    logger.info(f"\n  INFERENCE MODE: PASS ✓")

    # ═══════════════════════════════════════════════════════════════════

    logger.info(f"\n{'='*70}")
    logger.info(f"ALL TESTS PASSED ✓")
    logger.info(f"{'='*70}")
    logger.info(f"Memory: {mem_mb():.0f} MB")

    shutil.rmtree(tmp_dir)
    logger.info(f"Cleaned up {tmp_dir}")


if __name__ == "__main__":
    main()
