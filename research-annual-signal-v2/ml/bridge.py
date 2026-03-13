"""Bridge table loading and cid-to-branch mapping.

This is the SINGLE SOURCE OF TRUTH for bridge loading. All consumers
(data_loader, ground_truth, history_features) use map_cids_to_branches().
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import BRIDGE_PATH

logger = logging.getLogger(__name__)


def load_bridge_partition(
    auction_type: str,
    auction_month: str,
    period_type: str,
) -> pl.DataFrame:
    """Load bridge for BOTH class types and UNION them.

    Applies convention < 10 filter. Returns unique (constraint_id, branch_name).
    RAISES FileNotFoundError if NEITHER class_type partition exists.
    Logs warning if only one class_type is found.
    """
    frames: list[pl.DataFrame] = []
    missing: list[str] = []

    for ctype in ["onpeak", "offpeak"]:
        part_path = (
            f"{BRIDGE_PATH}/spice_version=v6/auction_type={auction_type}"
            f"/auction_month={auction_month}/market_round=1"
            f"/period_type={period_type}/class_type={ctype}/"
        )
        if not Path(part_path).exists():
            missing.append(ctype)
            continue
        df = (
            pl.read_parquet(part_path)
            .filter(
                (pl.col("convention") < 10) & pl.col("branch_name").is_not_null()
            )
            .select(["constraint_id", "branch_name"])
            .unique()
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No bridge partition found for {auction_type}/{auction_month}/"
            f"{period_type} in either onpeak or offpeak"
        )
    if missing:
        logger.warning(
            "Bridge partition missing for class_type=%s (%s/%s/%s). Using %s only.",
            missing, auction_type, auction_month, period_type,
            "offpeak" if "onpeak" in missing else "onpeak",
        )

    return pl.concat(frames).unique()


def map_cids_to_branches(
    cid_df: pl.DataFrame,
    auction_type: str,
    auction_month: str,
    period_type: str,
) -> tuple[pl.DataFrame, dict]:
    """Map constraint_ids to branch_names via bridge table.

    Handles:
    - Both-ctype UNION (onpeak + offpeak)
    - Convention < 10 filter
    - Ambiguous cid detection (cids mapping to multiple branch_names after union)
    - Logging ambiguous cid count
    - Dropping ambiguous cids

    Returns:
        (mapped_df with branch_name column, diagnostics dict)
    """
    bridge = load_bridge_partition(auction_type, auction_month, period_type)

    # Detect ambiguous cids: cids that map to >1 branch_name
    # Scope to input cids only — global bridge ambiguity is irrelevant to the caller.
    input_cids = cid_df["constraint_id"].unique()
    bridge_for_input = bridge.filter(pl.col("constraint_id").is_in(input_cids.to_list()))
    cid_branch_counts = bridge_for_input.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambiguous_cids = cid_branch_counts.filter(pl.col("n_branches") > 1)["constraint_id"]
    n_ambiguous = len(ambiguous_cids)

    # Compute ambiguous SP if cid_df has a realized_sp column
    ambiguous_sp = 0.0
    if n_ambiguous > 0:
        if "realized_sp" in cid_df.columns:
            ambiguous_sp = float(
                cid_df.filter(pl.col("constraint_id").is_in(ambiguous_cids.to_list()))
                ["realized_sp"].sum()
            )
        logger.warning(
            "Found %d ambiguous cids (SP=%.1f) mapping to >1 branch in %s/%s/%s. Dropping.",
            n_ambiguous, ambiguous_sp, auction_type, auction_month, period_type,
        )
        bridge = bridge.filter(~pl.col("constraint_id").is_in(ambiguous_cids.to_list()))

    # Now bridge should have at most 1 branch per cid
    bridge_unique = bridge.unique(subset=["constraint_id"])

    # Inner join: keep only cids that have a bridge mapping
    assert "constraint_id" in cid_df.columns, "cid_df must have constraint_id column"
    mapped = cid_df.join(bridge_unique, on="constraint_id", how="inner")

    diagnostics = {
        "ambiguous_cids": n_ambiguous,
        "ambiguous_sp": ambiguous_sp,
        "total_bridge_cids": len(bridge_unique),
        "mapped_cids": len(mapped),
        "unmapped_cids": len(cid_df) - len(mapped),
    }

    return mapped, diagnostics
