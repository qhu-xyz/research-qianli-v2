"""Density + limits loading -> universe filter -> Level 1+2 collapse -> branch features.

Row unit: branch_name (not constraint_id).
Two-level collapse:
  Level 1: mean across outage_dates per cid per bin
  Level 2: max + min across cids per bin per branch
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import (
    DENSITY_PATH, LIMIT_PATH, SELECTED_BINS, RIGHT_TAIL_BINS,
    UNIVERSE_THRESHOLD, COLLAPSED_CACHE_DIR,
    get_market_months,
)
from ml.bridge import map_cids_to_branches

logger = logging.getLogger(__name__)


def _cid_mapping_cache_path(planning_year: str, aq_quarter: str) -> Path:
    """Cache path for branch↔CID mapping."""
    threshold_tag = f"{UNIVERSE_THRESHOLD:.6e}".replace(".", "p").replace("+", "")
    return COLLAPSED_CACHE_DIR / f"{planning_year}_{aq_quarter}_cid_map_t{threshold_tag}.parquet"


def load_cid_mapping(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Load branch↔CID mapping for one (PY, quarter).

    Returns DataFrame with columns: constraint_id, branch_name, is_active.
    Cached alongside the branch-level cache in COLLAPSED_CACHE_DIR.
    """
    cache_path = _cid_mapping_cache_path(planning_year, aq_quarter)
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    # Force load_collapsed to run (which caches the CID mapping)
    load_collapsed(planning_year, aq_quarter)

    assert cache_path.exists(), f"CID mapping not cached after load_collapsed: {cache_path}"
    return pl.read_parquet(cache_path)


def load_raw_density(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Load raw density distribution for one (PY, quarter).

    Reads partition-specific paths (NOT hive scan — Trap 21).
    Returns all rows with constraint_id, outage_date, and bin columns.
    """
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        path = (
            f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round=1/"
        )
        if not Path(path).exists():
            logger.warning("Density partition missing: %s", path)
            continue
        df = pl.read_parquet(path)
        frames.append(df)

    assert len(frames) > 0, f"No density data for {planning_year}/{aq_quarter}"
    return pl.concat(frames, how="diagonal")


def compute_right_tail_max(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Compute right_tail_max per cid BEFORE any filtering.

    right_tail = max(bin_80, bin_90, bin_100, bin_110) per row
    right_tail_max = max(right_tail) across all outage_dates per cid
    """
    raw = load_raw_density(planning_year, aq_quarter)

    # Per row: right_tail
    raw = raw.with_columns(
        pl.max_horizontal([pl.col(b) for b in RIGHT_TAIL_BINS]).alias("right_tail")
    )

    # Per cid: right_tail_max
    return raw.group_by("constraint_id").agg(
        pl.col("right_tail").max().alias("right_tail_max")
    )


def _level1_collapse(raw: pl.DataFrame, bin_cols: list[str]) -> pl.DataFrame:
    """Level 1: mean across outage_dates (and months) per cid per bin."""
    return raw.group_by("constraint_id").agg(
        [pl.col(b).mean().alias(b) for b in bin_cols]
    )


def _level2_collapse(
    cid_df: pl.DataFrame,
    bin_cols: list[str],
) -> pl.DataFrame:
    """Level 2: max + min across cids per branch per bin.

    Input must have branch_name column (from bridge join).
    """
    agg_exprs = []
    for b in bin_cols:
        col_name = f"bin_{b}"
        agg_exprs.append(pl.col(b).max().alias(f"{col_name}_cid_max"))
        agg_exprs.append(pl.col(b).min().alias(f"{col_name}_cid_min"))

    return cid_df.group_by("branch_name").agg(agg_exprs)


def _load_limits(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Load constraint limits, Level 1 aggregate (mean per cid)."""
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        path = (
            f"{LIMIT_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round=1/"
        )
        if not Path(path).exists():
            continue
        df = pl.read_parquet(path)
        if "constraint_limit" in df.columns:
            frames.append(df.select(["constraint_id", "constraint_limit"]))
        elif "limit" in df.columns:
            frames.append(
                df.select(["constraint_id", "limit"]).rename({"limit": "constraint_limit"})
            )

    if not frames:
        logger.warning("No limit data for %s/%s", planning_year, aq_quarter)
        return pl.DataFrame(schema={"constraint_id": pl.Utf8, "constraint_limit": pl.Float64})

    raw = pl.concat(frames, how="diagonal")

    # Level 1: mean(limit) across dates/months per cid
    return raw.group_by("constraint_id").agg(
        pl.col("constraint_limit").mean().alias("constraint_limit")
    )


def _limit_level2(cid_limits: pl.DataFrame) -> pl.DataFrame:
    """Level 2 for limits: min/mean/max/std per branch."""
    return cid_limits.group_by("branch_name").agg(
        pl.col("constraint_limit").min().alias("limit_min"),
        pl.col("constraint_limit").mean().alias("limit_mean"),
        pl.col("constraint_limit").max().alias("limit_max"),
        pl.col("constraint_limit").std().alias("limit_std"),
    )


def load_collapsed(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Full pipeline: density -> universe filter -> Level 1+2 -> branch features.

    Universe filter: threshold is applied at CID level (cid is "active" if its
    right_tail_max >= UNIVERSE_THRESHOLD). Branches with at least 1 active cid
    are kept. The threshold was calibrated at branch level (see calibrate_threshold.py)
    but is consistent: branch_rtm = max(cid_rtm) >= threshold iff >=1 active cid.

    Returns branch-level DataFrame with:
    - bin_{X}_cid_max, bin_{X}_cid_min for each selected bin
    - limit_min, limit_mean, limit_max, limit_std
    - count_cids, count_active_cids
    - branch_name (unique per row)
    """
    # Check cache — key includes threshold to avoid stale data after recalibration
    threshold_tag = f"{UNIVERSE_THRESHOLD:.6e}".replace(".", "p").replace("+", "")
    cache_path = COLLAPSED_CACHE_DIR / f"{planning_year}_{aq_quarter}_t{threshold_tag}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    # Step 1: Load raw density
    raw = load_raw_density(planning_year, aq_quarter)

    # Step 2: Compute right_tail and is_active BEFORE filtering
    raw = raw.with_columns(
        pl.max_horizontal([pl.col(b) for b in RIGHT_TAIL_BINS]).alias("right_tail")
    )
    cid_rtm = raw.group_by("constraint_id").agg(
        pl.col("right_tail").max().alias("right_tail_max")
    )
    cid_rtm = cid_rtm.with_columns(
        (pl.col("right_tail_max") >= UNIVERSE_THRESHOLD).alias("is_active")
    )

    # Step 3: Map ALL cids -> branch via shared map_cids_to_branches()
    # This enforces the same ambiguity rule (detect+log+drop) as GT and history_features.
    cid_with_branch, bridge_diag = map_cids_to_branches(
        cid_df=cid_rtm,
        auction_type="annual",
        auction_month=planning_year,
        period_type=aq_quarter,
    )
    if bridge_diag["ambiguous_cids"] > 0:
        logger.info(
            "Dropped %d ambiguous cids from density mapping for %s/%s",
            bridge_diag["ambiguous_cids"], planning_year, aq_quarter,
        )

    # Cache CID mapping for downstream use (Phase 4b constraint propagation)
    cid_map_path = _cid_mapping_cache_path(planning_year, aq_quarter)
    if not cid_map_path.exists():
        cid_with_branch.select(
            ["constraint_id", "branch_name", "is_active"]
        ).write_parquet(str(cid_map_path))

    # Step 4: Compute count_cids (total) and count_active_cids per branch
    branch_counts = cid_with_branch.group_by("branch_name").agg(
        pl.len().alias("count_cids"),
        pl.col("is_active").sum().cast(pl.Int64).alias("count_active_cids"),
    )

    # Step 5: Filter to branches with at least 1 active cid
    active_branches = branch_counts.filter(pl.col("count_active_cids") >= 1)
    active_branch_names = active_branches["branch_name"].to_list()

    # Step 6: Level 1 collapse (mean across outage_dates per cid per bin)
    l1 = _level1_collapse(raw, SELECTED_BINS)

    # Step 7: Attach branch_name to cid-level data (only active branches)
    active_cids = cid_with_branch.filter(
        pl.col("branch_name").is_in(active_branch_names)
    ).select(["constraint_id", "branch_name"])
    l1_with_branch = l1.join(active_cids, on="constraint_id", how="inner")

    # Step 8: Level 2 collapse (max+min across cids per branch per bin)
    l2 = _level2_collapse(l1_with_branch, SELECTED_BINS)

    # Step 9: Limits
    cid_limits = _load_limits(planning_year, aq_quarter)
    if len(cid_limits) > 0:
        cid_limits_with_branch = cid_limits.join(active_cids, on="constraint_id", how="inner")
        limit_l2 = _limit_level2(cid_limits_with_branch)
    else:
        limit_l2 = pl.DataFrame({
            "branch_name": active_branch_names,
            "limit_min": [0.0] * len(active_branch_names),
            "limit_mean": [0.0] * len(active_branch_names),
            "limit_max": [0.0] * len(active_branch_names),
            "limit_std": [0.0] * len(active_branch_names),
        })

    # Step 10: Join everything at branch level
    result = (
        l2
        .join(limit_l2, on="branch_name", how="left")
        .join(active_branches.select(["branch_name", "count_cids", "count_active_cids"]),
              on="branch_name", how="left")
    )

    # Fill nulls for limits (branches without limit data)
    for col in ["limit_min", "limit_mean", "limit_max", "limit_std"]:
        if col in result.columns:
            result = result.with_columns(pl.col(col).fill_null(0.0))

    assert result["branch_name"].n_unique() == len(result), "Duplicate branch_names in output"

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(cache_path))
    logger.info(
        "Collapsed %s/%s: %d branches (%d active cids)",
        planning_year, aq_quarter, len(result),
        int(result["count_active_cids"].sum()),
    )

    return result
