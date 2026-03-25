"""NB (New Binder) detection — branches that bind in target quarter but had no
recent historical binding.

Reuses monthly_binding_table from history_features (no duplicate DA scans).
"""
from __future__ import annotations

import logging

import polars as pl

from ml.config import get_history_cutoff_month, BF_FLOOR_MONTH
from ml.markets.miso.history_features import _generate_month_range

logger = logging.getLogger(__name__)


def compute_nb_flags(
    universe_branches: list[str],
    planning_year: str,
    aq_quarter: str,
    gt_df: pl.DataFrame,
    monthly_binding_table: pl.DataFrame,
    market_round: int,
) -> pl.DataFrame:
    """Compute NB flags for all universe branches.

    Combined-ctype NB (gate + monitoring):
      is_nb_N: NO binding in EITHER ctype for last N months, AND branch binds in target quarter.
      Windows: N = 6, 12, 24.

    Per-ctype NB12 (monitoring only):
      nb_onpeak_12: no onpeak binding for 12 months AND onpeak_sp > 0 in target.
      nb_offpeak_12: no offpeak binding for 12 months AND offpeak_sp > 0 in target.

    Args:
        universe_branches: list of branch_name strings
        planning_year: eval PY (e.g., "2024-06")
        aq_quarter: quarter (e.g., "aq1")
        gt_df: from build_ground_truth() — has realized_shadow_price, onpeak_sp, offpeak_sp
        monthly_binding_table: from build_monthly_binding_table() — full monthly table
    """
    # Filter binding table to universe
    binding_in_universe = monthly_binding_table.filter(
        pl.col("branch_name").is_in(universe_branches)
    )

    # Calendar months for NB windows include the partial cutoff month.
    # A branch binding on April 10 should be non-NB for R2 (cutoff April 21).
    from ml.config import get_history_cutoff_date
    cutoff_month = get_history_cutoff_month(planning_year, market_round=market_round)
    cutoff_date = get_history_cutoff_date(planning_year, market_round=market_round)
    cutoff_date_month = f"{cutoff_date.year}-{cutoff_date.month:02d}"
    binding_table_end = max(cutoff_month, cutoff_date_month)
    all_calendar_months = _generate_month_range(BF_FLOOR_MONTH, binding_table_end)
    all_calendar_months_desc = list(reversed(all_calendar_months))

    # Start with universe
    result = pl.DataFrame({"branch_name": universe_branches})

    # Get target binding from GT
    gt_binding = gt_df.select([
        "branch_name", "realized_shadow_price", "onpeak_sp", "offpeak_sp",
    ])
    result = result.join(gt_binding, on="branch_name", how="left").with_columns(
        pl.col("realized_shadow_price").fill_null(0.0),
        pl.col("onpeak_sp").fill_null(0.0),
        pl.col("offpeak_sp").fill_null(0.0),
    )

    # Combined-ctype NB at windows 6, 12, 24
    for window in [6, 12, 24]:
        window_months = all_calendar_months_desc[:window]
        flag_name = f"is_nb_{window}"

        if len(window_months) == 0:
            # No history at all — all binding branches are NB
            result = result.with_columns(
                (pl.col("realized_shadow_price") > 0).alias(flag_name)
            )
            continue

        window_data = binding_in_universe.filter(
            pl.col("month").is_in(window_months)
        )

        # Count combined_bound months per branch
        branch_bound = window_data.group_by("branch_name").agg(
            pl.col("combined_bound").sum().alias("_n_bound")
        )

        result = result.join(branch_bound, on="branch_name", how="left").with_columns(
            pl.col("_n_bound").fill_null(0).alias("_n_bound")
        )

        # NB = no binding in window AND binds in target quarter
        result = result.with_columns(
            ((pl.col("_n_bound") == 0) & (pl.col("realized_shadow_price") > 0)).alias(flag_name)
        ).drop("_n_bound")

    # Per-ctype NB12 (monitoring only)
    window_months_12 = all_calendar_months_desc[:12]

    for _ctype, bound_col, sp_col, flag_name in [
        ("onpeak", "onpeak_bound", "onpeak_sp", "nb_onpeak_12"),
        ("offpeak", "offpeak_bound", "offpeak_sp", "nb_offpeak_12"),
    ]:
        if len(window_months_12) == 0:
            result = result.with_columns(
                (pl.col(sp_col) > 0).alias(flag_name)
            )
            continue

        window_data = binding_in_universe.filter(
            pl.col("month").is_in(window_months_12)
        )

        branch_ctype_bound = window_data.group_by("branch_name").agg(
            pl.col(bound_col).sum().alias("_n_ctype_bound")
        )

        result = result.join(branch_ctype_bound, on="branch_name", how="left").with_columns(
            pl.col("_n_ctype_bound").fill_null(0).alias("_n_ctype_bound")
        )

        result = result.with_columns(
            ((pl.col("_n_ctype_bound") == 0) & (pl.col(sp_col) > 0)).alias(flag_name)
        ).drop("_n_ctype_bound")

    # Drop intermediate GT columns (only keep NB flags)
    result = result.select([
        "branch_name",
        "is_nb_6", "is_nb_12", "is_nb_24",
        "nb_onpeak_12", "nb_offpeak_12",
    ])

    n_nb12 = int(result["is_nb_12"].sum())
    logger.info(
        "NB detection %s/%s: %d NB12 out of %d universe branches",
        planning_year, aq_quarter, n_nb12, len(result),
    )

    return result
