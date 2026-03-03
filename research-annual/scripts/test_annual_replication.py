"""Replication test: Compare production annual_band_generator (pandas) vs
research run_v9_bands (polars) on the same real data.

Loads R2 data, runs both calibration pipelines, and compares:
  1. Bin boundaries (should be identical)
  2. Calibrated quantile pairs (lo, hi) per cell (should match within rounding)
  3. Applied bands on a test fold (should match within rounding)
  4. Coverage at each level (should be identical)

This validates the pandas port faithfully replicates the polars research code.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_annual_replication.py
"""

from __future__ import annotations

import logging
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)8s] %(message)s")
logger = logging.getLogger(__name__)

# ── Data paths ──────────────────────────────────────────────────────────────

R2R3_DATA_PATH = Path("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
BASELINE_COL = "mtm_1st_mean"
MCP_COL = "mcp_mean"
CLASS_COL = "class_type"

# Production coverage levels (5 levels matching BAND_ORDER)
PROD_COVERAGE_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.99]
PROD_COVERAGE_LABELS = ["10", "30", "50", "70", "99"]

# Research coverage levels (7 levels used in run_v9_bands.py)
RESEARCH_COVERAGE_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
RESEARCH_COVERAGE_LABELS = ["p10", "p30", "p50", "p70", "p80", "p90", "p95"]

# Shared levels (intersection) for comparison
SHARED_LEVELS = [0.10, 0.30, 0.50, 0.70]
SHARED_PROD_LABELS = ["10", "30", "50", "70"]
SHARED_RESEARCH_LABELS = ["p10", "p30", "p50", "p70"]

N_BINS = 5
MIN_CELL_ROWS = 500


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_r2_data(quarter: str) -> pl.DataFrame:
    """Load R2 data for a specific quarter (same as run_v9_bands.py)."""
    return (
        pl.scan_parquet(R2R3_DATA_PATH)
        .filter(
            (pl.col("round") == 2)
            & (pl.col("period_type") == quarter)
            & (pl.col("planning_year") >= 2019)
            & pl.col(BASELINE_COL).is_not_null()
            & pl.col(MCP_COL).is_not_null()
        )
        .select([BASELINE_COL, MCP_COL, "planning_year", "period_type", CLASS_COL, "source_id", "sink_id"])
        .collect()
    )


def test_replication_on_quarter(quarter: str, test_py: int = 2025) -> dict:
    """Run both pipelines on one quarter, compare outputs."""
    logger.info(f"\n{'='*70}")
    logger.info(f"REPLICATION TEST: {quarter.upper()} R2, test_py={test_py}")
    logger.info(f"{'='*70}")

    # Load data
    df_pl = load_r2_data(quarter)
    logger.info(f"Loaded {df_pl.height:,} rows for {quarter}")

    # Split train/test (temporal: train < test_py)
    train_pl = df_pl.filter(pl.col("planning_year") < test_py)
    test_pl = df_pl.filter(pl.col("planning_year") == test_py)
    logger.info(f"Train: {train_pl.height:,} rows, Test: {test_pl.height:,} rows")

    if train_pl.height == 0 or test_pl.height == 0:
        logger.warning(f"Skipping {quarter} — no train or test data")
        return {"skipped": True}

    # Convert to pandas
    train_pd = train_pl.to_pandas()
    test_pd = test_pl.to_pandas()

    # ── Step 1: Compare bin boundaries ─────────────────────────────────

    # Research (polars)
    from scripts.run_v9_bands import compute_quantile_boundaries as research_compute_boundaries

    research_boundaries, research_labels = research_compute_boundaries(train_pl[BASELINE_COL], N_BINS)

    # Production (pandas)
    from pmodel.base.ftr24.v1.annual_band_generator import (
        compute_quantile_boundaries as prod_compute_boundaries,
    )

    prod_boundaries, prod_labels = prod_compute_boundaries(train_pd[BASELINE_COL], N_BINS)

    logger.info(f"\n--- Bin Boundaries ---")
    logger.info(f"  Research: {research_boundaries}")
    logger.info(f"  Production: {prod_boundaries}")
    logger.info(f"  Labels match: {research_labels == prod_labels}")

    boundaries_match = all(
        abs(a - b) < 0.2 if not (a == float("inf") and b == float("inf")) else True
        for a, b in zip(research_boundaries, prod_boundaries)
    )
    logger.info(f"  Boundaries match (within 0.2): {boundaries_match}")

    # ── Step 2: Compare calibrated quantile pairs ──────────────────────

    # Research (polars)
    from scripts.run_v9_bands import (
        add_sign_seg as research_add_sign_seg,
        calibrate_asymmetric_per_class_sign as research_calibrate,
    )

    train_pl_with_sign = research_add_sign_seg(train_pl, BASELINE_COL)
    research_pairs = research_calibrate(
        train_pl_with_sign, BASELINE_COL, MCP_COL, CLASS_COL,
        research_boundaries, research_labels,
        RESEARCH_COVERAGE_LEVELS,
    )

    # Production (pandas)
    from pmodel.base.ftr24.v1.annual_band_generator import (
        calibrate_asymmetric_per_class_sign as prod_calibrate,
    )

    prod_pairs = prod_calibrate(
        train_pd,
        baseline_col=BASELINE_COL,
        mcp_col=MCP_COL,
        class_col=CLASS_COL,
        boundaries=prod_boundaries,
        labels=prod_labels,
        coverage_levels=SHARED_LEVELS,
        coverage_labels=SHARED_PROD_LABELS,
    )

    logger.info(f"\n--- Calibrated Quantile Pairs (shared levels) ---")
    n_compared = 0
    n_match = 0
    max_diff = 0.0
    for bin_label in prod_labels:
        r_cell = research_pairs.get(bin_label, {})
        p_cell = prod_pairs.get(bin_label, {})
        for cls in ["onpeak", "offpeak"]:
            for seg in ["prevail", "counter"]:
                r_key = (cls, seg)
                r_data = r_cell.get(r_key, {})
                p_data = p_cell.get(r_key, {})
                for prod_label, res_label in zip(SHARED_PROD_LABELS, SHARED_RESEARCH_LABELS):
                    r_pair = r_data.get(res_label)
                    p_pair = p_data.get(prod_label)
                    if r_pair is not None and p_pair is not None:
                        n_compared += 1
                        diff_lo = abs(r_pair[0] - p_pair[0])
                        diff_hi = abs(r_pair[1] - p_pair[1])
                        diff = max(diff_lo, diff_hi)
                        max_diff = max(max_diff, diff)
                        if diff <= 0.2:
                            n_match += 1
                        else:
                            logger.warning(
                                f"  MISMATCH {bin_label}/{cls}/{seg}/{prod_label}: "
                                f"research={r_pair}, prod={p_pair}, diff={diff:.1f}"
                            )

    logger.info(f"  Compared {n_compared} (bin, class, sign, level) cells")
    logger.info(f"  Match within 0.2: {n_match}/{n_compared}")
    logger.info(f"  Max difference: {max_diff:.1f}")

    # ── Step 3: Compare applied bands on test fold ─────────────────────

    # Research (polars)
    from scripts.run_v9_bands import (
        apply_asymmetric_bands_per_class_sign_fast as research_apply_bands,
    )

    test_pl_with_sign = research_add_sign_seg(test_pl, BASELINE_COL)
    research_banded = research_apply_bands(
        test_pl_with_sign, research_pairs, BASELINE_COL, CLASS_COL,
        research_boundaries, research_labels,
    )

    # Production (pandas)
    from pmodel.base.ftr24.v1.annual_band_generator import (
        apply_asymmetric_bands as prod_apply_bands,
    )

    prod_banded = prod_apply_bands(
        test_pd,
        bin_pairs=prod_pairs,
        baseline_col=BASELINE_COL,
        class_col=CLASS_COL,
        boundaries=prod_boundaries,
        labels=prod_labels,
        coverage_labels=SHARED_PROD_LABELS,
    )

    logger.info(f"\n--- Applied Bands Comparison (test fold) ---")
    logger.info(f"  Research rows: {research_banded.height}, Production rows: {len(prod_banded)}")

    band_diffs = {}
    for prod_label, res_label in zip(SHARED_PROD_LABELS, SHARED_RESEARCH_LABELS):
        for side in ["lower", "upper"]:
            prod_col = f"{side}_{prod_label}"
            res_col = f"{side}_{res_label}"
            if prod_col in prod_banded.columns and res_col in research_banded.columns:
                prod_vals = prod_banded[prod_col].values
                res_vals = research_banded[res_col].to_numpy()
                diffs = np.abs(prod_vals - res_vals)
                valid = ~np.isnan(diffs)
                if valid.sum() > 0:
                    mean_diff = np.nanmean(diffs)
                    max_d = np.nanmax(diffs)
                    band_diffs[f"{side}_{prod_label}"] = {"mean": mean_diff, "max": max_d}
                    status = "OK" if max_d < 1.0 else "WARN"
                    logger.info(f"  {side}_{prod_label}: mean_diff={mean_diff:.2f}, max_diff={max_d:.2f} [{status}]")

    # ── Step 4: Compare coverage ───────────────────────────────────────

    logger.info(f"\n--- Coverage Comparison ---")
    logger.info(f"  {'Level':<8} {'Research':>10} {'Production':>10} {'Diff':>8}")
    logger.info(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    coverage_diffs = {}
    for level, prod_label, res_label in zip(SHARED_LEVELS, SHARED_PROD_LABELS, SHARED_RESEARCH_LABELS):
        # Research coverage
        r_lower = research_banded[f"lower_{res_label}"]
        r_upper = research_banded[f"upper_{res_label}"]
        r_mcp = research_banded[MCP_COL]
        r_cov = float(((r_mcp >= r_lower) & (r_mcp <= r_upper)).mean()) * 100

        # Production coverage
        p_lower = prod_banded[f"lower_{prod_label}"]
        p_upper = prod_banded[f"upper_{prod_label}"]
        p_mcp = prod_banded[MCP_COL]
        p_cov = ((p_mcp >= p_lower) & (p_mcp <= p_upper)).mean() * 100

        diff = p_cov - r_cov
        target = level * 100
        coverage_diffs[prod_label] = {"research": r_cov, "production": p_cov, "diff": diff, "target": target}
        logger.info(f"  {prod_label:>6}%  {r_cov:>9.2f}% {p_cov:>9.2f}% {diff:>+7.2f}pp  (target: {target:.0f}%)")

    # ── Step 5: Test bid price generation ──────────────────────────────
    # Need all 5 production levels (including 99%) for bid prices.
    # Re-calibrate with full production levels.

    from pmodel.base.ftr24.v1.annual_band_generator import apply_annual_bid_prices

    prod_pairs_full = prod_calibrate(
        train_pd,
        baseline_col=BASELINE_COL,
        mcp_col=MCP_COL,
        class_col=CLASS_COL,
        boundaries=prod_boundaries,
        labels=prod_labels,
        coverage_levels=PROD_COVERAGE_LEVELS,
        coverage_labels=PROD_COVERAGE_LABELS,
    )
    prod_banded_full = prod_apply_bands(
        test_pd,
        bin_pairs=prod_pairs_full,
        baseline_col=BASELINE_COL,
        class_col=CLASS_COL,
        boundaries=prod_boundaries,
        labels=prod_labels,
        coverage_labels=PROD_COVERAGE_LABELS,
    )
    prod_with_bids = apply_annual_bid_prices(prod_banded_full, treat_all_as_buy=True)

    n_with_bids = prod_with_bids["bid_price_1"].notna().sum()
    logger.info(f"\n--- Bid Price Generation ---")
    logger.info(f"  Rows with bid prices: {n_with_bids}/{len(prod_with_bids)}")

    # Check sort order (descending for buy)
    violations = 0
    for i in range(len(prod_with_bids)):
        bids = [prod_with_bids.iloc[i][f"bid_price_{j}"] for j in range(1, 11)]
        if any(pd.isna(b) for b in bids):
            continue
        for k in range(len(bids) - 1):
            if bids[k] < bids[k + 1]:
                violations += 1
                break
    logger.info(f"  Sort violations: {violations}")

    # Print sample
    logger.info(f"\n  Sample (first 3 rows):")
    sample_cols = ["mtm_1st_mean"] + [f"bid_price_{j}" for j in range(1, 6)]
    logger.info(f"\n{prod_with_bids[sample_cols].head(3).to_string()}")

    return {
        "quarter": quarter,
        "boundaries_match": boundaries_match,
        "calibration_cells_compared": n_compared,
        "calibration_match_rate": n_match / n_compared if n_compared > 0 else 0,
        "max_calibration_diff": max_diff,
        "band_diffs": band_diffs,
        "coverage_diffs": coverage_diffs,
        "n_with_bids": n_with_bids,
        "sort_violations": violations,
    }


def main():
    logger.info(f"Annual Band Generator Replication Test")
    logger.info(f"Comparing production (pandas) vs research (polars) on real R2 data")
    logger.info(f"Memory: {mem_mb():.0f} MB")

    # Add research scripts to path
    research_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(research_dir))

    results = {}
    for quarter in ["aq1", "aq2", "aq3", "aq4"]:
        results[quarter] = test_replication_on_quarter(quarter)

    # ── Summary ────────────────────────────────────────────────────────

    logger.info(f"\n{'='*70}")
    logger.info(f"REPLICATION SUMMARY")
    logger.info(f"{'='*70}")

    all_pass = True
    for q, r in results.items():
        if r.get("skipped"):
            logger.info(f"  {q}: SKIPPED")
            continue
        boundaries_ok = r["boundaries_match"]
        cal_ok = r["calibration_match_rate"] >= 0.95
        bids_ok = r["sort_violations"] == 0
        status = "PASS" if (boundaries_ok and cal_ok and bids_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(
            f"  {q}: {status} — boundaries={boundaries_ok}, "
            f"calibration={r['calibration_match_rate']:.0%} match, "
            f"max_diff={r['max_calibration_diff']:.1f}, "
            f"bids={r['n_with_bids']}, sort_violations={r['sort_violations']}"
        )

    if all_pass:
        logger.info(f"\n  ALL QUARTERS PASS ✓")
    else:
        logger.info(f"\n  SOME QUARTERS FAILED — investigate above")

    logger.info(f"\nMemory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
