# ml/branch_mapping.py
"""PJM constraint_id → branch_name mapping via constraint_info.

The naive join (constraint_id.split(":")[0] → DA monitored_facility) captures
only ~46% of DA binding value. The branch-level join via constraint_info
captures 96-99%.

Reference: research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805

How it works:
  1. constraint_info maps each constraint_id to a branch_name
  2. constraint_id.split(":")[0] → monitored_facility → .upper() → match_str
  3. DA monitored_facility (uppercased) matches match_str → branch_name
  4. Interface fallback: prefix-match for interface contingencies
  5. Aggregate realized_sp by branch_name
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_CI_BASE


def load_constraint_info(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load constraint_info for an auction month.

    constraint_info is stored only under class_type=onpeak (by design —
    it's physical topology, class-invariant).
    """
    path = (
        Path(SPICE6_CI_BASE)
        / f"auction_month={auction_month}"
        / "market_round=1"
        / f"period_type={period_type}"
        / "class_type=onpeak"
    )
    if not path.exists():
        return pl.DataFrame()

    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame()

    dfs = [pl.read_parquet(str(f)) for f in parquet_files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    if "constraint_id" in df.columns:
        df = df.with_columns(pl.col("constraint_id").cast(pl.String))

    return df


def build_branch_map(ci: pl.DataFrame) -> pl.DataFrame:
    """Build branch mapping from constraint_info.

    Returns DataFrame with columns: constraint_id, branch_name, match_str, type.
    match_str = monitored_facility.upper() (extracted from constraint_id).
    """
    if len(ci) == 0:
        return pl.DataFrame(schema={
            "constraint_id": pl.String,
            "branch_name": pl.String,
            "match_str": pl.String,
            "type": pl.String,
        })

    result = ci.select([
        pl.col("constraint_id"),
        pl.col("branch_name"),
        pl.col("type") if "type" in ci.columns else pl.lit("branch_constraint").alias("type"),
    ]).unique()

    # Build match_str: monitored_facility = constraint_id.split(":")[0], uppercased
    result = result.with_columns(
        pl.col("constraint_id")
        .str.split(":")
        .list.first()
        .str.to_uppercase()
        .alias("match_str")
    )

    return result


def map_da_to_branches(
    da_df: pl.DataFrame,
    branch_map: pl.DataFrame,
) -> pl.DataFrame:
    """Map DA shadow prices to branch_names and aggregate.

    Parameters
    ----------
    da_df : pl.DataFrame
        Raw DA data with columns: monitored_facility, shadow_price.
    branch_map : pl.DataFrame
        From build_branch_map(), with columns: branch_name, match_str, type.

    Returns
    -------
    pl.DataFrame
        Columns: branch_name, realized_sp (sum of |shadow_price| per branch).
    """
    if len(da_df) == 0 or len(branch_map) == 0:
        return pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })

    # Uppercase DA monitored_facility for matching
    da = da_df.with_columns(
        pl.col("monitored_facility").str.to_uppercase().alias("match_str")
    )

    # Separate interface and non-interface entries
    non_interface = branch_map.filter(pl.col("type") != "interface")
    interface_map = branch_map.filter(pl.col("type") == "interface")

    # Step 1: Direct match on match_str (non-interface)
    direct_map = non_interface.select(["match_str", "branch_name"]).unique(subset=["match_str"])
    da = da.join(direct_map, on="match_str", how="left")

    # Step 2: Interface fallback — prefix match for unmatched DA rows
    if len(interface_map) > 0:
        unmatched = da.filter(pl.col("branch_name").is_null())
        if len(unmatched) > 0:
            interface_strs = interface_map.select(["match_str", "branch_name"]).unique(subset=["match_str"])
            interface_prefixes = interface_strs["match_str"].to_list()
            interface_branch_names = interface_strs["branch_name"].to_list()

            unmatched_strs = unmatched["match_str"].unique().to_list()
            prefix_map = {}
            for ums in unmatched_strs:
                first_word = ums.split(" ")[0] if " " in ums else ums
                for ip, ibn in zip(interface_prefixes, interface_branch_names):
                    if first_word == ip or ums.startswith(ip + " "):
                        prefix_map[ums] = ibn
                        break

            if prefix_map:
                prefix_df = pl.DataFrame({
                    "match_str": list(prefix_map.keys()),
                    "interface_branch": list(prefix_map.values()),
                })
                da = da.join(prefix_df, on="match_str", how="left")
                da = da.with_columns(
                    pl.col("branch_name").fill_null(pl.col("interface_branch"))
                )
                if "interface_branch" in da.columns:
                    da = da.drop("interface_branch")

    # Aggregate: sum |shadow_price| by branch_name
    result = (
        da.filter(pl.col("branch_name").is_not_null())
        .group_by("branch_name")
        .agg(pl.col("shadow_price").abs().sum().alias("realized_sp"))
    )

    return result
