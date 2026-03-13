"""Universe threshold calibration — run on Day 1.

Calibration is BRANCH-LEVEL throughout:
  - right_tail_max is computed per cid, then aggregated to max per branch
  - realized SP is summed per branch (using annual bridge only, no monthly fallback)
  - the 95% SP capture target is relative to annual-bridge-mapped branch SP,
    NOT total quarter DA SP
  - the elbow threshold is chosen by branch rank

The resulting threshold is later applied at the CID level in load_collapsed()
(a cid is "active" if its right_tail_max >= threshold). This is consistent
because branch_rtm = max(cid_rtm) >= threshold implies at least one active cid.

Produces:
  registry/threshold_calibration/
    threshold.json           — {threshold, rationale, date}
    calibration_data.parquet — branch-level threshold sweep table
                               (rank, right_tail_max, cumulative_sp, pct_sp)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `ml` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from ml.config import (
    DENSITY_PATH, RIGHT_TAIL_BINS, REGISTRY_DIR,
    get_market_months,
)
from ml.bridge import load_bridge_partition
from ml.realized_da import load_quarter


def compute_right_tail_max(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Compute right_tail_max per cid for one (PY, quarter)."""
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        path = (
            f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round=1/"
        )
        if not Path(path).exists():
            continue
        df = pl.read_parquet(path).select(["constraint_id"] + RIGHT_TAIL_BINS)
        frames.append(df)

    assert len(frames) > 0, f"No density data for {planning_year}/{aq_quarter}"
    raw = pl.concat(frames, how="diagonal")

    # Per row: right_tail = max(bin_80, bin_90, bin_100, bin_110)
    raw = raw.with_columns(
        pl.max_horizontal([pl.col(b) for b in RIGHT_TAIL_BINS]).alias("right_tail")
    )

    # Per cid: right_tail_max = max across all outage_dates
    return raw.group_by("constraint_id").agg(
        pl.col("right_tail").max().alias("right_tail_max")
    )


def run_calibration():
    out_dir = REGISTRY_DIR / "threshold_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary: 2024-06/aq1
    print("Computing right_tail_max for 2024-06/aq1...")
    rtm_2024 = compute_right_tail_max("2024-06", "aq1")
    print(f"  Total cids: {len(rtm_2024)}")

    # Get binding SP for 2024-06/aq1 (combined onpeak+offpeak, summed per cid)
    mm_2024 = get_market_months("2024-06", "aq1")
    da_2024 = load_quarter(mm_2024)

    # Bridge mapping: cid -> branch (annual bridge only, no monthly fallback)
    bridge = load_bridge_partition(auction_type="annual", auction_month="2024-06", period_type="aq1")

    # Work at BRANCH level to avoid fan-out (multiple cids per branch)
    # 1. Map density cids to branches, keep max right_tail_max per branch
    cid_with_branch = rtm_2024.join(bridge, on="constraint_id", how="inner")
    branch_rtm = cid_with_branch.group_by("branch_name").agg(
        pl.col("right_tail_max").max()
    )

    # 2. Map DA cids to branches, sum realized_sp per branch
    da_mapped = da_2024.join(bridge, on="constraint_id", how="inner")
    branch_sp = da_mapped.group_by("branch_name").agg(pl.col("realized_sp").sum())

    # 3. Join branch-level: right_tail_max + realized_sp
    branch_df = branch_rtm.join(branch_sp, on="branch_name", how="left").with_columns(
        pl.col("realized_sp").fill_null(0.0)
    )

    # Sort by right_tail_max descending, compute cumulative SP (no fan-out)
    sorted_df = branch_df.sort("right_tail_max", descending=True)
    total_sp = sorted_df["realized_sp"].sum()

    print(f"  Total branches: {len(sorted_df)}")
    print(f"  Total binding SP: {total_sp:.1f}")
    print(f"  Branches with SP > 0: {sorted_df.filter(pl.col('realized_sp') > 0).height}")

    # Compute cumulative SP capture at each threshold
    rtm_values = sorted_df["right_tail_max"].to_list()
    sp_values = sorted_df["realized_sp"].to_list()

    cumsum = 0.0
    thresholds = []
    for i, (rtm, sp) in enumerate(zip(rtm_values, sp_values)):
        cumsum += sp
        thresholds.append({
            "rank": i + 1,
            "right_tail_max": rtm,
            "cumulative_sp": cumsum,
            "pct_sp": cumsum / total_sp if total_sp > 0 else 0,
        })

    # Print coverage at key thresholds
    print("\n  Coverage at selected thresholds:")
    for pct in [0.80, 0.85, 0.90, 0.95, 0.99]:
        idx = next(
            (i for i, t in enumerate(thresholds) if t["pct_sp"] >= pct),
            len(thresholds) - 1,
        )
        print(
            f"    {pct*100:.0f}% SP: threshold={thresholds[idx]['right_tail_max']:.6f}, "
            f"branches={thresholds[idx]['rank']}"
        )

    # Find elbow: where cumulative SP capture > 95% of total
    target_pct = 0.95
    elbow_idx = next(
        (i for i, t in enumerate(thresholds) if t["pct_sp"] >= target_pct),
        len(thresholds) - 1,
    )
    elbow_threshold = thresholds[elbow_idx]["right_tail_max"]
    elbow_branch_count = thresholds[elbow_idx]["rank"]

    print(f"\n  Elbow at {target_pct*100}% SP capture:")
    print(f"    threshold = {elbow_threshold:.6f}")
    print(f"    branch count = {elbow_branch_count}")

    # Cross-check with 2023-06/aq1 (also at branch level)
    print("\nCross-checking with 2023-06/aq1...")
    rtm_2023 = compute_right_tail_max("2023-06", "aq1")
    bridge_2023 = load_bridge_partition(auction_type="annual", auction_month="2023-06", period_type="aq1")
    cid_br_2023 = rtm_2023.join(bridge_2023, on="constraint_id", how="inner")
    branch_rtm_2023 = cid_br_2023.group_by("branch_name").agg(pl.col("right_tail_max").max())
    branches_2023 = branch_rtm_2023.filter(pl.col("right_tail_max") >= elbow_threshold)
    print(f"  2023-06/aq1 filtered branches: {len(branches_2023)}")
    ratio = len(branches_2023) / elbow_branch_count if elbow_branch_count > 0 else 0
    print(f"  Ratio: {ratio:.2f} (should be 0.80-1.20)")
    assert 0.5 <= ratio <= 2.0, f"Cross-check failed: ratio {ratio:.2f} outside tolerance"

    # Save artifact
    artifact = {
        "threshold": elbow_threshold,
        "calibration_py": "2024-06",
        "calibration_quarter": "aq1",
        "branch_count_at_threshold": elbow_branch_count,
        "sp_capture_pct": thresholds[elbow_idx]["pct_sp"],
        "cross_check_py": "2023-06",
        "cross_check_branches": len(branches_2023),
        "cross_check_ratio": ratio,
        "date": "2026-03-12",
    }
    with open(out_dir / "threshold.json", "w") as f:
        json.dump(artifact, f, indent=2)

    # Save calibration data
    cal_data = pl.DataFrame(thresholds)
    cal_data.write_parquet(str(out_dir / "calibration_data.parquet"))

    print(f"\nSaved to {out_dir}/")
    print(f"\n*** ACTION REQUIRED: Update UNIVERSE_THRESHOLD in ml/config.py to {elbow_threshold} ***")


if __name__ == "__main__":
    run_calibration()
