"""Bridge table loading and cid-to-branch mapping.

This is the SINGLE SOURCE OF TRUTH for bridge loading. All consumers
(data_loader, ground_truth, history_features) use map_cids_to_branches().

For DA CID → branch resolution, use map_cids_to_branches_with_supplement()
which adds supplement key fallback after the bridge + monthly fallback.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

from ml.config import BRIDGE_PATH

SUPPLEMENT_PATH = "/opt/data/xyz-dataset/modeling_data/miso/MISO_DA_SHADOW_PRICE_SUPPLEMENT.parquet"

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


# ── Supplement key matching ──────────────────────────────────────────────


def _normalize_branch(s: str) -> str:
    """Collapse whitespace for branch name comparison."""
    return re.sub(r"\s+", " ", s.strip()) if s else ""


_supplement_cache: dict[str, pl.DataFrame] = {}


def load_supplement_keys(market_months: list[str]) -> pl.DataFrame:
    """Load supplement keys for the given market months (cached).

    Returns DataFrame with: constraint_id, key1, key2, key3, device_type
    One row per unique constraint_id.
    """
    cache_key = ",".join(sorted(market_months))
    if cache_key in _supplement_cache:
        return _supplement_cache[cache_key]

    year_months = [(int(m[:4]), int(m[5:7])) for m in market_months]
    frames = []
    cols = ["constraint_id", "key1", "key2", "key3", "device_type", "year", "month"]
    for y, m in year_months:
        part = (
            pl.scan_parquet(SUPPLEMENT_PATH)
            .select(cols)
            .filter((pl.col("year") == y) & (pl.col("month") == m))
            .collect()
        )
        if len(part) > 0:
            frames.append(part)

    if not frames:
        result = pl.DataFrame(schema={c: pl.Utf8 for c in cols[:5]})
    else:
        result = pl.concat(frames).unique(subset=["constraint_id"]).select(cols[:5])

    _supplement_cache[cache_key] = result
    return result


def supplement_match_unmapped(
    unmapped_cids: list[str],
    supp: pl.DataFrame,
    spice_branch_set: set[str],
) -> dict[str, str]:
    """Match unmapped CIDs to SPICE branches via supplement keys.

    Rules:
      XF (transformer): branch = key1 + " " + key3
      LN (line/other):  branch = key2 + " " + key3

    Returns: {constraint_id: branch_name} for recovered CIDs only.
    """
    spice_norm = {_normalize_branch(b): b for b in spice_branch_set}
    recovered: dict[str, str] = {}

    for cid in unmapped_cids:
        row = supp.filter(pl.col("constraint_id") == cid)
        if len(row) == 0:
            continue
        dt = row["device_type"][0]
        k1 = row["key1"][0] or ""
        k2 = row["key2"][0] or ""
        k3 = row["key3"][0] or ""

        if dt == "XF":
            candidate = _normalize_branch(f"{k1} {k3}")
        else:
            candidate = _normalize_branch(f"{k2} {k3}")

        if candidate and candidate in spice_norm:
            recovered[cid] = spice_norm[candidate]

    return recovered


def map_cids_to_branches_with_supplement(
    cid_df: pl.DataFrame,
    auction_type: str,
    auction_month: str,
    period_type: str,
    market_months: list[str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Map CIDs to branches: bridge first, then supplement key fallback.

    1. Try annual bridge (existing map_cids_to_branches)
    2. For still-unmapped CIDs, load supplement keys
    3. Construct branch: XF -> key1+key3, LN -> key2+key3
    4. Match against SPICE branches from the same bridge
    5. Return combined mapping with provenance

    Args:
        market_months: needed for supplement key loading. If None, supplement
            fallback is skipped (behaves like map_cids_to_branches).
    """
    # Step 1: standard bridge mapping
    mapped, diag = map_cids_to_branches(
        cid_df, auction_type, auction_month, period_type,
    )

    supplement_recovered_cids = 0
    supplement_recovered_sp = 0.0
    supplement_no_entry = 0

    if market_months is None or diag["unmapped_cids"] == 0:
        diag["supplement_recovered_cids"] = 0
        diag["supplement_recovered_sp"] = 0.0
        diag["supplement_no_entry"] = 0
        return mapped, diag

    # Step 2: identify unmapped CIDs
    mapped_cid_set = set(mapped["constraint_id"].to_list())
    all_cids = cid_df["constraint_id"].unique().to_list()
    unmapped = [c for c in all_cids if c not in mapped_cid_set]

    if not unmapped:
        diag["supplement_recovered_cids"] = 0
        diag["supplement_recovered_sp"] = 0.0
        diag["supplement_no_entry"] = 0
        return mapped, diag

    # Step 3: load supplement keys
    supp = load_supplement_keys(market_months)

    # Step 4: get SPICE branch set from the same bridge
    bridge = load_bridge_partition(auction_type, auction_month, period_type)
    spice_branches = set(bridge["branch_name"].to_list())

    # Step 5: match
    recovered = supplement_match_unmapped(unmapped, supp, spice_branches)

    if recovered:
        # Build recovery rows and append to mapped
        recovery_rows = []
        for cid, branch in recovered.items():
            cid_row = cid_df.filter(pl.col("constraint_id") == cid)
            if len(cid_row) > 0:
                row = cid_row.with_columns(pl.lit(branch).alias("branch_name"))
                recovery_rows.append(row)

        if recovery_rows:
            recovery_df = pl.concat(recovery_rows)
            mapped = pl.concat([mapped, recovery_df], how="diagonal")

            supplement_recovered_cids = len(recovered)
            if "realized_sp" in cid_df.columns:
                supplement_recovered_sp = float(
                    cid_df.filter(
                        pl.col("constraint_id").is_in(list(recovered.keys()))
                    )["realized_sp"].sum()
                )

    # Count no-entry
    supp_cids = set(supp["constraint_id"].to_list())
    supplement_no_entry = sum(1 for c in unmapped if c not in supp_cids and c not in recovered)

    diag["supplement_recovered_cids"] = supplement_recovered_cids
    diag["supplement_recovered_sp"] = supplement_recovered_sp
    diag["supplement_no_entry"] = supplement_no_entry
    diag["unmapped_cids"] = diag["unmapped_cids"] - supplement_recovered_cids

    logger.info(
        "Supplement recovery: %d CIDs recovered (SP=%.0f), %d no entry, %d still unmapped",
        supplement_recovered_cids, supplement_recovered_sp,
        supplement_no_entry, diag["unmapped_cids"],
    )

    return mapped, diag
