"""Ground truth: realized DA shadow prices mapped to branches via annual + monthly fallback.

Returns only POSITIVE-BINDING branches (realized_shadow_price > 0, label_tier in {1,2,3}).
Non-binding branches (label_tier=0) are added in features.py when GT is left-joined
onto the branch universe from data_loader.
"""
from __future__ import annotations

import logging

import polars as pl

from ml.config import get_market_months
from ml.markets.miso.bridge import map_cids_to_branches, map_cids_to_branches_with_supplement
from ml.markets.miso.realized_da import load_quarter_per_ctype

logger = logging.getLogger(__name__)


def _try_monthly_bridge(
    unmapped_cids: pl.DataFrame,
    market_month: str,
    market_round: int,
) -> tuple[pl.DataFrame, int, dict]:
    """Try to map unmapped cids via monthly f0 bridge for one market_month.

    Uses map_cids_to_branches() for consistent ambiguity handling (detect+log+drop).
    Returns (mapped_df with constraint_id + branch_name, n_recovered, diagnostics).
    """
    try:
        mapped, diag = map_cids_to_branches(
            cid_df=unmapped_cids,
            auction_type="monthly",
            auction_month=market_month,
            period_type="f0",
            market_round=market_round,
        )
    except FileNotFoundError:
        return pl.DataFrame(schema={"constraint_id": pl.Utf8, "branch_name": pl.Utf8}), 0, {}

    result = mapped.select(["constraint_id", "branch_name"])
    return result, result.height, diag


def build_ground_truth(
    planning_year: str,
    aq_quarter: str,
    market_round: int,
) -> tuple[pl.DataFrame, dict]:
    """Build branch-level GT for one (PY, quarter).

    Pipeline:
      1. Load DA per ctype (onpeak + offpeak separately)
      2. Combine into total DA per cid
      3. Map cids to branches via annual bridge
      4. Monthly fallback for unmapped cids
      5. Aggregate to branch level (SUM)
      6. Assign tiered labels (tertiles of positive SP)

    Returns:
      (gt_df, diagnostics) where gt_df has ONLY positive-binding branches.
    """
    market_months = get_market_months(planning_year, aq_quarter)

    # Step 1: Load DA per ctype
    onpeak_da, offpeak_da = load_quarter_per_ctype(market_months)

    # Step 2: Combined DA per cid
    combined_frames = []
    if len(onpeak_da) > 0:
        combined_frames.append(onpeak_da.rename({"realized_sp": "total_sp"}))
    if len(offpeak_da) > 0:
        combined_frames.append(offpeak_da.rename({"realized_sp": "total_sp"}))

    combined_da = pl.concat(combined_frames).group_by("constraint_id").agg(
        pl.col("total_sp").sum()
    )

    # Count ALL DA cids (including zero-SP) and total SP (including zero-SP cids).
    # These are raw DA coverage counts, not positive-binding-only counts.
    total_da_cids = len(combined_da)
    total_da_sp = float(combined_da["total_sp"].sum())

    # Step 3: Map via annual bridge
    annual_mapped, annual_diag = map_cids_to_branches(
        cid_df=combined_da,
        auction_type="annual",
        auction_month=planning_year,
        period_type=aq_quarter,
        market_round=market_round,
    )

    annual_mapped_cids_set = set(annual_mapped["constraint_id"].to_list())
    annual_mapped_sp = float(
        combined_da.filter(
            pl.col("constraint_id").is_in(list(annual_mapped_cids_set))
        )["total_sp"].sum()
    )

    # Step 4: Monthly fallback for unmapped cids
    unmapped = combined_da.filter(
        ~pl.col("constraint_id").is_in(list(annual_mapped_cids_set))
    )

    monthly_mapped_frames = []
    monthly_recovery_detail = {}
    for mm in market_months:
        if len(unmapped) == 0:
            break
        recovered, n_recovered, _monthly_diag = _try_monthly_bridge(
            unmapped.select(["constraint_id"]), mm, market_round=1
        )
        monthly_recovery_detail[mm] = n_recovered
        if n_recovered > 0:
            monthly_mapped_frames.append(recovered)
            recovered_cids = set(recovered["constraint_id"].to_list())
            unmapped = unmapped.filter(
                ~pl.col("constraint_id").is_in(list(recovered_cids))
            )

    # Combine all mappings: annual + monthly
    all_mappings = [annual_mapped.select(["constraint_id", "branch_name"])]
    if monthly_mapped_frames:
        all_mappings.extend(monthly_mapped_frames)
    all_cid_branch = pl.concat(all_mappings)

    monthly_recovered_cids = sum(monthly_recovery_detail.values())
    monthly_recovered_sp = float(
        combined_da.filter(
            pl.col("constraint_id").is_in(
                [cid for frame in monthly_mapped_frames for cid in frame["constraint_id"].to_list()]
            ) if monthly_mapped_frames else pl.lit(False)
        )["total_sp"].sum()
    ) if monthly_mapped_frames else 0.0

    # Step 4b: Supplement key matching for still-unmapped cids
    supplement_recovered_cids = 0
    supplement_recovered_sp = 0.0
    supplement_no_entry = 0

    if len(unmapped) > 0:
        from ml.markets.miso.bridge import load_supplement_keys, supplement_match_unmapped, load_bridge_partition

        supp = load_supplement_keys(market_months)
        bridge = load_bridge_partition("annual", planning_year, aq_quarter, market_round=market_round)
        spice_branches = set(bridge["branch_name"].to_list())
        unmapped_cid_list = unmapped["constraint_id"].to_list()

        recovered_map = supplement_match_unmapped(unmapped_cid_list, supp, spice_branches)

        if recovered_map:
            supp_frames = []
            for cid, branch in recovered_map.items():
                supp_frames.append(pl.DataFrame({
                    "constraint_id": [cid],
                    "branch_name": [branch],
                }))
            supp_mapped = pl.concat(supp_frames)
            all_cid_branch = pl.concat([all_cid_branch, supp_mapped])

            supplement_recovered_cids = len(recovered_map)
            supplement_recovered_sp = float(
                unmapped.filter(
                    pl.col("constraint_id").is_in(list(recovered_map.keys()))
                )["total_sp"].sum()
            )
            unmapped = unmapped.filter(
                ~pl.col("constraint_id").is_in(list(recovered_map.keys()))
            )

        supp_cid_set = set(supp["constraint_id"].to_list())
        supplement_no_entry = sum(
            1 for c in unmapped_cid_list
            if c not in supp_cid_set and c not in recovered_map
        )

    still_unmapped_cids = len(unmapped)
    still_unmapped_sp = float(unmapped["total_sp"].sum()) if len(unmapped) > 0 else 0.0

    # Step 5: Join DA onto mapping, aggregate to branch level
    # Also build per-ctype columns
    cid_with_branch = combined_da.join(all_cid_branch, on="constraint_id", how="inner")

    # Per-ctype SP at cid level
    onpeak_with_branch = onpeak_da.join(all_cid_branch, on="constraint_id", how="inner")
    offpeak_with_branch = offpeak_da.join(all_cid_branch, on="constraint_id", how="inner")

    # Aggregate to branch
    branch_total = cid_with_branch.group_by("branch_name").agg(
        pl.col("total_sp").sum().alias("realized_shadow_price")
    )
    branch_onpeak = onpeak_with_branch.group_by("branch_name").agg(
        pl.col("realized_sp").sum().alias("onpeak_sp")
    )
    branch_offpeak = offpeak_with_branch.group_by("branch_name").agg(
        pl.col("realized_sp").sum().alias("offpeak_sp")
    )

    gt = (
        branch_total
        .join(branch_onpeak, on="branch_name", how="left")
        .join(branch_offpeak, on="branch_name", how="left")
        .with_columns(
            pl.col("onpeak_sp").fill_null(0.0),
            pl.col("offpeak_sp").fill_null(0.0),
        )
    )

    # Step 5b: Filter to positive-binding only
    gt = gt.filter(pl.col("realized_shadow_price") > 0)

    # Step 6: Tiered labels — tertiles of positive SP
    gt = gt.sort("realized_shadow_price")
    n = len(gt)
    third = n // 3

    # Assign tiers: bottom third = 1, middle = 2, top = 3
    gt = gt.with_row_index("_idx")
    gt = gt.with_columns(
        pl.when(pl.col("_idx") < third)
        .then(pl.lit(1))
        .when(pl.col("_idx") < 2 * third)
        .then(pl.lit(2))
        .otherwise(pl.lit(3))
        .alias("label_tier")
    ).drop("_idx")

    assert gt["branch_name"].n_unique() == len(gt), "Duplicate branch_names in GT"

    # Per-class total DA SP (cross-universe denominator for Abs_SP@K).
    # These include ALL DA cids for each class, whether or not they map to branches.
    onpeak_total_da_sp = float(onpeak_da["realized_sp"].sum()) if len(onpeak_da) > 0 else 0.0
    offpeak_total_da_sp = float(offpeak_da["realized_sp"].sum()) if len(offpeak_da) > 0 else 0.0

    # Diagnostics use explicit naming to avoid confusion about what is counted.
    # "total_da_cids" includes zero-SP cids; "total_da_sp" includes SP from zero-SP cids (=0).
    # "annual_mapped_cids" is count of cids successfully mapped (including zero-SP ones).
    diagnostics = {
        "total_da_cids": total_da_cids,           # all DA cids (incl zero-SP)
        "annual_mapped_cids": len(annual_mapped),  # mapped by annual bridge (incl zero-SP)
        "monthly_recovered_cids": monthly_recovered_cids,  # recovered by monthly fallback
        "still_unmapped_cids": still_unmapped_cids,
        "total_da_sp": total_da_sp,                # combined onpeak+offpeak total DA SP
        "onpeak_total_da_sp": onpeak_total_da_sp,  # onpeak-only total DA SP (cross-universe)
        "offpeak_total_da_sp": offpeak_total_da_sp, # offpeak-only total DA SP (cross-universe)
        "annual_mapped_sp": annual_mapped_sp,
        "monthly_recovered_sp": monthly_recovered_sp,
        "still_unmapped_sp": still_unmapped_sp,
        "annual_ambiguous_cids": annual_diag["ambiguous_cids"],
        "annual_ambiguous_sp": annual_diag["ambiguous_sp"],
        "monthly_recovery_detail": monthly_recovery_detail,
        "supplement_recovered_cids": supplement_recovered_cids,
        "supplement_recovered_sp": supplement_recovered_sp,
        "supplement_no_entry": supplement_no_entry,
    }

    logger.info(
        "GT %s/%s: %d positive-binding branches, %.1f total SP, "
        "%d annual mapped, %d monthly recovered, %d still unmapped",
        planning_year, aq_quarter, len(gt), total_da_sp,
        len(annual_mapped), monthly_recovered_cids, still_unmapped_cids,
    )

    return gt, diagnostics
