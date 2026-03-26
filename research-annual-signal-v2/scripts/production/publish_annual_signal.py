"""Publish annual signal for a planning year and round.

Generates constraints + SF parquets for all (aq, class_type) combinations.
Validates against V6.1 for overlapping constraints.
Signal name includes round:
  - TEST.Signal.MISO.SPICE_ANNUAL_V7.1B.R{round}
  - TEST.Signal.MISO.SPICE_ANNUAL_V7.2B.R{round}

Usage:
    PYTHONPATH=. uv run python scripts/publish_annual_signal.py --py 2025-06 --market-round 1
    PYTHONPATH=. uv run python scripts/publish_annual_signal.py --py 2025-06 --market-round 2 --aq aq1
    PYTHONPATH=. uv run python scripts/publish_annual_signal.py --py 2024-06 --market-round 1 --validate-only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd
import polars as pl

# Add psignal/pbase source paths
sys.path.insert(0, "/home/xyz/workspace/psignal/src")
sys.path.insert(0, "/home/xyz/workspace/pbase/src")

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")

from ml.config import AQ_QUARTERS, CLASS_TYPES
from ml.signal_publisher import publish_signal, publish_signal_72b

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_PREFIXES = {
    "7.1b": "TEST.Signal.MISO.SPICE_ANNUAL_V7.1B",
    "7.2b": "TEST.Signal.MISO.SPICE_ANNUAL_V7.2B",
}
V61_CONSTRAINTS_PATH = "/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1"
V61_SF_PATH = "/opt/data/xyz-dataset/signal_data/miso/sf/Signal.MISO.SPICE_ANNUAL_V6.1"

# Default tier config: (200, 400) = 5 × 200 = 1,000
DEFAULT_TIER_SIZES = [200, 200, 200, 200, 200]


def validate_against_v61(
    constraints_df: pd.DataFrame,
    planning_year: str,
    aq_quarter: str,
    class_type: str,
) -> dict:
    """Compare published constraints against V6.1 for overlapping CIDs."""
    v61_path = f"{V61_CONSTRAINTS_PATH}/{planning_year}/{aq_quarter}/{class_type}/"
    try:
        v61 = pl.read_parquet(v61_path).to_pandas()
    except FileNotFoundError:
        logger.warning("V6.1 not found at %s — skipping validation", v61_path)
        return {"status": "skipped", "reason": "V6.1 not found"}

    our_cids = set(constraints_df["constraint_id"].tolist())
    v61_cids = set(v61["constraint_id"].tolist())
    overlap = our_cids & v61_cids

    results = {
        "v61_count": len(v61_cids),
        "our_count": len(our_cids),
        "overlap": len(overlap),
        "overlap_pct": len(overlap) / len(v61_cids) * 100 if v61_cids else 0,
    }

    if not overlap:
        results["status"] = "no_overlap"
        return results

    # Merge on overlap
    merged = v61.merge(constraints_df, on="constraint_id", how="inner", suffixes=("_v61", "_v70"))

    # Check metadata columns that should match
    for col in ["bus_key", "equipment", "branch_name"]:
        c_v61 = f"{col}_v61" if f"{col}_v61" in merged.columns else col
        c_v70 = f"{col}_v70" if f"{col}_v70" in merged.columns else col
        if c_v61 in merged.columns and c_v70 in merged.columns:
            match = (merged[c_v61].fillna("").str.strip() == merged[c_v70].fillna("").str.strip()).sum()
            results[f"{col}_match"] = f"{match}/{len(merged)} ({match / len(merged) * 100:.1f}%)"

    # Check flow_direction
    if "flow_direction_v61" in merged.columns and "flow_direction_v70" in merged.columns:
        fd_match = (merged["flow_direction_v61"] == merged["flow_direction_v70"]).sum()
        results["flow_direction_match"] = f"{fd_match}/{len(merged)} ({fd_match / len(merged) * 100:.1f}%)"

    # Schema check
    expected_cols = set(v61.columns)
    our_cols = set(constraints_df.columns)
    results["schema_match"] = expected_cols == our_cols
    results["missing_cols"] = list(expected_cols - our_cols)
    results["extra_cols"] = list(our_cols - expected_cols)

    results["status"] = "validated"
    return results


def save_signal(
    constraints_df: pd.DataFrame,
    sf_df: pd.DataFrame,
    planning_year: str,
    aq_quarter: str,
    class_type: str,
    signal_name: str,
    dry_run: bool = False,
) -> None:
    """Save constraints + SF parquets to the output path."""
    cstr_root = f"/opt/data/xyz-dataset/signal_data/miso/constraints/{signal_name}"
    sf_root = f"/opt/data/xyz-dataset/signal_data/miso/sf/{signal_name}"
    cstr_dir = f"{cstr_root}/{planning_year}/{aq_quarter}/{class_type}"
    sf_dir = f"{sf_root}/{planning_year}/{aq_quarter}/{class_type}"

    if dry_run:
        logger.info("DRY RUN: would save %d constraints to %s", len(constraints_df), cstr_dir)
        logger.info("DRY RUN: would save SF %s to %s", sf_df.shape, sf_dir)
        return

    os.makedirs(cstr_dir, exist_ok=True)
    os.makedirs(sf_dir, exist_ok=True)

    # Set index for constraints (matching V6.1 format)
    cstr_out = constraints_df.set_index("__index_level_0__")
    cstr_out.to_parquet(f"{cstr_dir}/signal.parquet")

    # SF: pnode_id must be the index (matching DA_ANNUAL_2YR.V3.R1 format)
    sf_out = sf_df.set_index("pnode_id")
    sf_out.to_parquet(f"{sf_dir}/signal.parquet")

    logger.info("Saved: %s (%d constraints, %s SF)", cstr_dir, len(cstr_out), sf_df.shape)


def main():
    parser = argparse.ArgumentParser(description="Publish annual signal (V7.1B / V7.2B)")
    parser.add_argument("--py", required=True, help="Planning year (e.g., 2025-06)")
    parser.add_argument("--aq", default=None, help="Single quarter (e.g., aq1)")
    parser.add_argument("--class-type", default=None, choices=CLASS_TYPES)
    parser.add_argument("--market-round", type=int, required=True, help="Auction round (explicit; no default)")
    parser.add_argument("--scoring-mode", default="7.1b", choices=["7.1b", "7.2b"])
    parser.add_argument("--validate-only", action="store_true", help="Validate without saving")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be saved")
    parser.add_argument("--tier-sizes", default=None, help="Comma-separated tier sizes (default: 200,200,200,200,200)")
    parser.add_argument("--signal-name", default=None, help="Signal name override (default: mode-specific .R{round})")
    args = parser.parse_args()

    # Init Ray
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    planning_year = args.py
    market_round = args.market_round
    signal_prefix = DEFAULT_SIGNAL_PREFIXES[args.scoring_mode]
    signal_name = args.signal_name or f"{signal_prefix}.R{market_round}"
    quarters = [args.aq] if args.aq else AQ_QUARTERS
    class_types = [args.class_type] if args.class_type else CLASS_TYPES
    tier_sizes = [int(x) for x in args.tier_sizes.split(",")] if args.tier_sizes else DEFAULT_TIER_SIZES
    publish_fn = publish_signal_72b if args.scoring_mode == "7.2b" else publish_signal

    logger.info("Publishing %s for %s", signal_name, planning_year)
    logger.info(
        "Mode: %s, Round: R%s, Quarters: %s, Classes: %s, Tiers: %s",
        args.scoring_mode, market_round, quarters, class_types, tier_sizes,
    )

    t0 = time.time()
    all_results = []

    for aq in quarters:
        for ct in class_types:
            t1 = time.time()
            logger.info("\n%s %s/%s/%s %s", "=" * 20, planning_year, aq, ct, "=" * 20)

            try:
                constraints_df, sf_df = publish_fn(
                    planning_year=planning_year,
                    aq_quarter=aq,
                    class_type=ct,
                    tier_sizes=tier_sizes,
                    market_round=args.market_round,
                )

                # Validate
                val = validate_against_v61(constraints_df, planning_year, aq, ct)
                logger.info("Validation: %s", val)

                # Save
                if not args.validate_only:
                    save_signal(constraints_df, sf_df, planning_year, aq, ct,
                                signal_name=signal_name, dry_run=args.dry_run)

                all_results.append({
                    "quarter": aq, "class": ct,
                    "constraints": len(constraints_df),
                    "sf_pnodes": len(sf_df),
                    "sf_constraints": len(sf_df.columns) - 1,
                    "validation": val.get("status", "unknown"),
                    "overlap_pct": val.get("overlap_pct", 0),
                    "time": f"{time.time() - t1:.1f}s",
                })

            except Exception as e:
                logger.error("FAILED %s/%s/%s: %s", planning_year, aq, ct, e, exc_info=True)
                all_results.append({"quarter": aq, "class": ct, "error": str(e)})

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  {signal_name} Publication Summary — {planning_year} R{market_round}")
    print(f"{'=' * 80}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['quarter']}/{r['class']}: ERROR — {r['error']}")
        else:
            print(f"  {r['quarter']}/{r['class']}: {r['constraints']} constraints, "
                  f"SF {r['sf_pnodes']}×{r['sf_constraints']}, "
                  f"V6.1 overlap {r['overlap_pct']:.0f}%, {r['time']}")

    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
