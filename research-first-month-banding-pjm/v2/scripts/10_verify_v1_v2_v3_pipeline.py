"""Verify PJM v1/v2/v3 band pipeline: save_training_data + generate_bands → halt.

Tests that all three band versions can:
1. save_training_data() — writes parquet with required columns
2. generate_bands() — produces bid_price_1..10, clearing_prob_1..10, baseline
3. Cross-version: v2==v3 on non-June, v2!=v3 on June, v1!=v2

Run from pmodel venv:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-first-monthly-pjm/v2/scripts/10_verify_v1_v2_v3_pipeline.py
"""

import gc
import logging
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_pipeline")

# ---- Constants ----
TRAINING2_BASE = Path("/opt/temp/qianli/pjm_mcp_pred_training2")
TRAINING1_BASE = Path("/opt/temp/qianli/pjm_mcp_pred_training")

BID_PRICE_COLS = [f"bid_price_{i}" for i in range(1, 11)]
CLEARING_PROB_COLS = [f"clearing_prob_{i}" for i in range(1, 11)]
BAND_COLS = BID_PRICE_COLS + CLEARING_PROB_COLS + ["baseline"]

PTYPE = "f0"
CLASS_TYPE = "onpeak"
MIN_PRIOR_MONTHS = 6


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def find_test_months() -> tuple[str, str]:
    """Find a June and a non-June month with enough prior training data."""
    pattern = f"auction_month=*/period_type={PTYPE}/class_type={CLASS_TYPE}/training.parquet"
    months = sorted(
        p.parts[-4].replace("auction_month=", "")
        for p in TRAINING2_BASE.glob(pattern)
    )
    logger.info(f"Available months: {len(months)} total")

    # Need MIN_PRIOR_MONTHS+ prior months, so eligible = months[MIN_PRIOR_MONTHS:]
    eligible = months[MIN_PRIOR_MONTHS:]
    if not eligible:
        raise RuntimeError(f"Not enough months (need {MIN_PRIOR_MONTHS}+ prior). Found: {len(months)}")

    june_months = [m for m in eligible if m.endswith("-06")]
    non_june_months = [m for m in eligible if not m.endswith("-06")]

    if not june_months:
        raise RuntimeError(f"No eligible June months found. Eligible: {eligible}")
    if not non_june_months:
        raise RuntimeError(f"No eligible non-June months found. Eligible: {eligible}")

    # Pick the most recent of each
    june = june_months[-1]
    non_june = non_june_months[-1]
    logger.info(f"Test months: June={june}, non-June={non_june}")
    return june, non_june


def load_pool_data(month: str) -> pd.DataFrame:
    """Load training2 pool data for one month."""
    path = (
        TRAINING2_BASE
        / f"auction_month={month}"
        / f"period_type={PTYPE}"
        / f"class_type={CLASS_TYPE}"
        / "training.parquet"
    )
    df = pd.read_parquet(path)
    if "auction_month" not in df.columns:
        df["auction_month"] = month
    if "period_type" not in df.columns:
        df["period_type"] = PTYPE
    if "trade_type" not in df.columns:
        df["trade_type"] = "buy"
    logger.info(f"Loaded {len(df):,} rows for {PTYPE}/{month} ({mem_mb():.0f} MB)")
    return df


# ---- Test 1: save_training_data ----

def test_save_training_data(df: pd.DataFrame, month: str) -> bool:
    """Test that save_training_data() writes a valid parquet."""
    from pmodel.base.ftr24.v1.band_generator import save_training_data

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = save_training_data(
            trades=df,
            auction_month=month,
            period_type=PTYPE,
            class_type=CLASS_TYPE,
            training_base=tmpdir,
        )
        parquet_path = save_path / "training.parquet"
        if not parquet_path.exists():
            logger.error(f"FAIL: save_training_data did not create {parquet_path}")
            return False

        saved = pd.read_parquet(parquet_path)
        required = ["source_id", "sink_id", "path", "period_type", "class_type",
                     "mtm_1st_mean", "mtm_2nd_mean", "mtm_3rd_mean", "mcp_mean"]
        missing = [c for c in required if c not in saved.columns]
        if missing:
            logger.error(f"FAIL: saved parquet missing columns: {missing}")
            return False

        logger.info(f"PASS: save_training_data — {len(saved)} rows, {len(saved.columns)} cols")
        return True


# ---- Test 2: generate_bands for one version ----

def test_generate_bands(
    df: pd.DataFrame,
    band_version: str,
    label: str,
) -> pd.DataFrame | None:
    """Run generate_bands for a single version. Returns result or None on failure."""
    from pmodel.base.ftr24.v1.band_generator import generate_bands

    t0 = time.time()
    logger.info(f"--- generate_bands(band_version='{band_version}') for {label} ---")

    result = generate_bands(
        df=df.copy(),
        use_ray=False,
        class_type=CLASS_TYPE,
        treat_all_as_buy=True,
        band_version=band_version,
    )
    elapsed = time.time() - t0

    present = [c for c in BAND_COLS if c in result.columns]
    missing = [c for c in BAND_COLS if c not in result.columns]

    if missing:
        logger.error(f"FAIL: {band_version}/{label} — missing band cols: {missing}")
        return None

    # Validate no all-NaN band columns
    all_nan_cols = [c for c in BAND_COLS if result[c].isna().all()]
    if all_nan_cols:
        logger.error(f"FAIL: {band_version}/{label} — all-NaN cols: {all_nan_cols}")
        return None

    # Basic sanity: bid prices should be sorted descending per row
    bp = result[BID_PRICE_COLS].values
    valid_rows = ~np.isnan(bp).any(axis=1)
    if valid_rows.sum() > 0:
        sorted_ok = (np.diff(bp[valid_rows], axis=1) <= 1e-9).all()
        if not sorted_ok:
            logger.warning(f"WARN: {band_version}/{label} — some bid prices not sorted descending")

    # Clearing probs should be in [0, 1]
    cp = result[CLEARING_PROB_COLS].values
    valid_cp = ~np.isnan(cp).any(axis=1)
    if valid_cp.sum() > 0:
        cp_valid = cp[valid_cp]
        if cp_valid.min() < -0.01 or cp_valid.max() > 1.01:
            logger.warning(
                f"WARN: {band_version}/{label} — clearing probs out of [0,1]: "
                f"min={cp_valid.min():.4f}, max={cp_valid.max():.4f}"
            )

    nan_pct = result[BAND_COLS].isna().mean().mean() * 100
    logger.info(
        f"PASS: {band_version}/{label} — {len(result):,} rows, "
        f"21/21 band cols, {nan_pct:.1f}% NaN, {elapsed:.1f}s"
    )
    return result


# ---- Test 3: Cross-version comparison ----

def compare_versions(
    r1: pd.DataFrame,
    r2: pd.DataFrame,
    v1_label: str,
    v2_label: str,
    expect_equal: bool,
    month: str,
) -> bool:
    """Compare two band results. Returns True if expectation met."""
    # Align on index
    common_idx = r1.index.intersection(r2.index)
    if len(common_idx) == 0:
        logger.error(f"FAIL: no common rows between {v1_label} and {v2_label}")
        return False

    a = r1.loc[common_idx, BAND_COLS]
    b = r2.loc[common_idx, BAND_COLS]

    # Use baseline difference as the primary signal
    baseline_diff = (a["baseline"] - b["baseline"]).abs()
    max_diff = baseline_diff.max()
    mean_diff = baseline_diff.mean()
    same = max_diff < 1e-6

    if expect_equal and not same:
        logger.error(
            f"FAIL: {v1_label} vs {v2_label} on {month} — expected EQUAL but "
            f"max_baseline_diff={max_diff:.6f}, mean={mean_diff:.6f}"
        )
        return False
    elif not expect_equal and same:
        logger.error(
            f"FAIL: {v1_label} vs {v2_label} on {month} — expected DIFFERENT but "
            f"baselines are identical (max_diff={max_diff:.6f})"
        )
        return False

    status = "EQUAL" if same else f"DIFFERENT (max_base_diff={max_diff:.4f}, mean={mean_diff:.4f})"
    expectation = "as expected" if True else "UNEXPECTED"
    logger.info(f"PASS: {v1_label} vs {v2_label} on {month}: {status} ({expectation})")
    return True


# ---- Main ----

def main():
    t_start = time.time()
    logger.info(f"=== PJM v1/v2/v3 Pipeline Verification ===")
    logger.info(f"Memory: {mem_mb():.0f} MB")

    # Find test months
    june_month, non_june_month = find_test_months()

    results = {}  # (version, month) -> DataFrame
    passes = []
    fails = []

    def record(name: str, ok: bool):
        (passes if ok else fails).append(name)

    # ---- Phase 1: save_training_data (use June month) ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: save_training_data")
    logger.info("=" * 60)

    june_df = load_pool_data(june_month)
    record("save_training_data", test_save_training_data(june_df, june_month))

    # ---- Phase 2: generate_bands — June month (v1, v2, v3) ----
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: generate_bands — June ({june_month})")
    logger.info("=" * 60)

    for version in ["v1", "v2", "v3"]:
        result = test_generate_bands(june_df, version, f"june-{june_month}")
        key = (version, june_month)
        results[key] = result
        record(f"bands_{version}_june", result is not None)
        gc.collect()

    del june_df
    gc.collect()
    logger.info(f"Memory after June phase: {mem_mb():.0f} MB")

    # ---- Phase 3: generate_bands — non-June month (v2, v3 only) ----
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 3: generate_bands — non-June ({non_june_month})")
    logger.info("=" * 60)

    non_june_df = load_pool_data(non_june_month)

    for version in ["v2", "v3"]:
        result = test_generate_bands(non_june_df, version, f"nonjune-{non_june_month}")
        key = (version, non_june_month)
        results[key] = result
        record(f"bands_{version}_nonjune", result is not None)
        gc.collect()

    del non_june_df
    gc.collect()
    logger.info(f"Memory after non-June phase: {mem_mb():.0f} MB")

    # ---- Phase 4: Cross-version comparisons ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Cross-version comparisons")
    logger.info("=" * 60)

    # v1 vs v2 on June: should be DIFFERENT (v2 uses different baseline)
    r_v1_june = results.get(("v1", june_month))
    r_v2_june = results.get(("v2", june_month))
    r_v3_june = results.get(("v3", june_month))
    r_v2_nonjune = results.get(("v2", non_june_month))
    r_v3_nonjune = results.get(("v3", non_june_month))

    if r_v1_june is not None and r_v2_june is not None:
        record(
            "v1_vs_v2_june_different",
            compare_versions(r_v1_june, r_v2_june, "v1", "v2",
                             expect_equal=False, month=june_month),
        )

    # v2 vs v3 on June: should be DIFFERENT (v3 has seasonal adjustment)
    if r_v2_june is not None and r_v3_june is not None:
        record(
            "v2_vs_v3_june_different",
            compare_versions(r_v2_june, r_v3_june, "v2", "v3",
                             expect_equal=False, month=june_month),
        )

    # v2 vs v3 on non-June: should be EQUAL (seasonal only affects June)
    if r_v2_nonjune is not None and r_v3_nonjune is not None:
        record(
            "v2_vs_v3_nonjune_equal",
            compare_versions(r_v2_nonjune, r_v3_nonjune, "v2", "v3",
                             expect_equal=True, month=non_june_month),
        )

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name in passes:
        logger.info(f"  PASS  {name}")
    for name in fails:
        logger.info(f"  FAIL  {name}")

    total = len(passes) + len(fails)
    logger.info(f"\n  {len(passes)}/{total} passed, {len(fails)} failed")
    logger.info(f"  Total elapsed: {time.time() - t_start:.1f}s")
    logger.info(f"  Peak memory: {mem_mb():.0f} MB")

    if fails:
        logger.error("VERIFICATION FAILED")
        sys.exit(1)
    else:
        logger.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
