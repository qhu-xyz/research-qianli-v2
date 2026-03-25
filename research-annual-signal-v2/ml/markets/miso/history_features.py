"""History features: BF (binding frequency) + da_rank_value from monthly binding table.

Builds one monthly branch-binding table per eval PY (single DA scan), then derives
BF windows and da_rank_value from it.

Bridge rule: always uses eval PY's annual bridge for ALL historical months.
Monthly fallback for unmapped cids uses month M's f0 bridge.
"""
from __future__ import annotations

import logging
from datetime import date

import polars as pl

from ml.core.calendars import get_history_cutoff_month
from ml.markets.miso.config import (
    BF_FLOOR_MONTH, BF_WINDOWS_ONPEAK, BF_WINDOWS_OFFPEAK, BF_WINDOWS_COMBINED,
)
from ml.markets.miso.bridge import (
    map_cids_to_branches,
    load_supplement_keys,
    supplement_match_unmapped,
    load_bridge_partition,
)
from ml.markets.miso.realized_da import load_month

logger = logging.getLogger(__name__)


def _generate_month_range(floor_month: str, cutoff_month: str) -> list[str]:
    """Generate YYYY-MM strings from floor_month through cutoff_month inclusive."""
    fy, fm = int(floor_month[:4]), int(floor_month[5:7])
    cy, cm = int(cutoff_month[:4]), int(cutoff_month[5:7])
    months = []
    y, m = fy, fm
    while (y, m) <= (cy, cm):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def build_monthly_binding_table(
    eval_py: str,
    aq_quarter: str,
    cutoff_month: str,
    floor_month: str,
    market_round: int,
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Build monthly branch-binding table for one eval PY and quarter.

    For each month in [floor_month, cutoff_month]:
      - Load onpeak + offpeak DA
      - Map cids -> branches via eval PY's annual bridge for the given quarter
        + monthly fallback
      - Per branch per month: bound flags + SP columns

    If cutoff_date is provided and daily cache is available, the cutoff month
    is loaded from daily files (both ctypes) with date filtering. Full months
    before cutoff use the fast monthly cache.

    Bridge is quarter-sensitive: aq1/aq2/aq3/aq4 may have different constraint universes.

    Returns DataFrame with columns:
      month, branch_name, onpeak_bound, offpeak_bound, combined_bound,
      onpeak_sp, offpeak_sp, combined_sp
    """
    from ml.markets.miso.realized_da import load_month_daily, has_daily_cache

    months = _generate_month_range(floor_month, cutoff_month)
    assert len(months) > 0, f"No months in range [{floor_month}, {cutoff_month}]"

    use_daily = has_daily_cache() and cutoff_date is not None

    all_rows: list[pl.DataFrame] = []

    # Pre-load supplement keys for ALL months (avoid per-month parquet scan)
    supp_all = load_supplement_keys(months)
    bridge_for_supp = load_bridge_partition("annual", eval_py, aq_quarter, market_round=market_round)
    spice_branches_set = set(bridge_for_supp["branch_name"].to_list())

    for month in months:
        # Determine if this month needs daily cutoff filtering.
        # A month needs daily filtering if cutoff_date falls within it.
        is_cutoff_month = False
        if use_daily and cutoff_date is not None:
            y, m = int(month[:4]), int(month[5:7])
            import calendar as _cal
            month_last_day = _cal.monthrange(y, m)[1]
            month_first = date(y, m, 1)
            month_last = date(y, m, month_last_day)
            is_cutoff_month = month_first <= cutoff_date <= month_last

        # Load DA for this month — both ctypes
        if is_cutoff_month:
            # Partial month: load from daily cache with cutoff for BOTH ctypes
            onpeak_da = load_month_daily(month, "onpeak", cutoff_date=cutoff_date)
            offpeak_da = load_month_daily(month, "offpeak", cutoff_date=cutoff_date)
            if len(onpeak_da) == 0 and len(offpeak_da) == 0:
                continue
        else:
            # Full month: use fast monthly cache
            try:
                onpeak_da = load_month(month, "onpeak")
            except FileNotFoundError:
                continue
            try:
                offpeak_da = load_month(month, "offpeak")
            except FileNotFoundError:
                offpeak_da = pl.DataFrame(schema={"constraint_id": pl.Utf8, "realized_sp": pl.Float64})

        # Combine all cids from both ctypes for bridge mapping
        all_cids = pl.concat([
            onpeak_da.select("constraint_id"),
            offpeak_da.select("constraint_id"),
        ]).unique()

        # Map via eval PY's annual bridge for this quarter and round
        mapped, _ = map_cids_to_branches(
            cid_df=all_cids,
            auction_type="annual",
            auction_month=eval_py,
            period_type=aq_quarter,
            market_round=market_round,
        )
        mapped_cids_set = set(mapped["constraint_id"].to_list())

        # Monthly fallback for unmapped cids
        unmapped_cids = all_cids.filter(
            ~pl.col("constraint_id").is_in(list(mapped_cids_set))
        )
        if len(unmapped_cids) > 0:
            try:
                monthly_mapped, _ = map_cids_to_branches(
                    cid_df=unmapped_cids,
                    auction_type="monthly",
                    auction_month=month,
                    period_type="f0",
                    market_round=1,  # monthly auctions have 1 round
                )
                if len(monthly_mapped) > 0:
                    mapped = pl.concat([
                        mapped.select(["constraint_id", "branch_name"]),
                        monthly_mapped.select(["constraint_id", "branch_name"]),
                    ])
            except FileNotFoundError:
                pass

        # Supplement key fallback for still-unmapped cids
        mapped_cids_after = set(mapped["constraint_id"].to_list())
        still_unmapped = all_cids.filter(
            ~pl.col("constraint_id").is_in(list(mapped_cids_after))
        )
        if len(still_unmapped) > 0:
            recovered_map = supplement_match_unmapped(
                still_unmapped["constraint_id"].to_list(), supp_all, spice_branches_set,
            )
            if recovered_map:
                supp_frames = [
                    pl.DataFrame({"constraint_id": [cid], "branch_name": [branch]})
                    for cid, branch in recovered_map.items()
                ]
                mapped = pl.concat([
                    mapped.select(["constraint_id", "branch_name"]),
                    pl.concat(supp_frames),
                ])

        cid_to_branch = mapped.select(["constraint_id", "branch_name"])

        # Map DA to branches
        on_with_branch = onpeak_da.join(cid_to_branch, on="constraint_id", how="inner")
        off_with_branch = offpeak_da.join(cid_to_branch, on="constraint_id", how="inner")

        # Aggregate to branch level
        branch_on = on_with_branch.group_by("branch_name").agg(
            pl.col("realized_sp").sum().alias("onpeak_sp")
        )
        branch_off = off_with_branch.group_by("branch_name").agg(
            pl.col("realized_sp").sum().alias("offpeak_sp")
        )

        # All branches that appear in either ctype
        all_branches = pl.concat([
            branch_on.select("branch_name"),
            branch_off.select("branch_name"),
        ]).unique()

        month_df = (
            all_branches
            .join(branch_on, on="branch_name", how="left")
            .join(branch_off, on="branch_name", how="left")
            .with_columns(
                pl.col("onpeak_sp").fill_null(0.0),
                pl.col("offpeak_sp").fill_null(0.0),
            )
            .with_columns(
                (pl.col("onpeak_sp") + pl.col("offpeak_sp")).alias("combined_sp"),
                (pl.col("onpeak_sp") > 0).alias("onpeak_bound"),
                (pl.col("offpeak_sp") > 0).alias("offpeak_bound"),
            )
            .with_columns(
                (pl.col("onpeak_bound") | pl.col("offpeak_bound")).alias("combined_bound"),
            )
            .with_columns(pl.lit(month).alias("month"))
        )
        all_rows.append(month_df)

    assert len(all_rows) > 0, f"No DA data found in [{floor_month}, {cutoff_month}]"
    return pl.concat(all_rows)


def compute_history_features(
    eval_py: str,
    aq_quarter: str,
    universe_branches: list[str],
    market_round: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute BF features + da_rank_value for branches in universe.

    Returns (hist_df, monthly_binding_table):
      - hist_df: one row per universe branch with 8 features + has_hist_da
      - monthly_binding_table: full monthly table (needed by nb_detection)
    """
    from ml.core.calendars import get_history_cutoff_date
    cutoff_month_full = get_history_cutoff_month(eval_py, market_round=market_round)
    cutoff_date = get_history_cutoff_date(eval_py, market_round=market_round)

    # The binding table range must include the cutoff_date's month so that
    # daily cache can load partial-month data. E.g. R1 cutoff_date=April 7:
    # cutoff_month_full="2025-03" but we need April in the range for daily loading.
    cutoff_date_month = f"{cutoff_date.year}-{cutoff_date.month:02d}"
    binding_table_end = max(cutoff_month_full, cutoff_date_month)

    binding_table = build_monthly_binding_table(
        eval_py=eval_py,
        aq_quarter=aq_quarter,
        cutoff_month=binding_table_end,
        floor_month=BF_FLOOR_MONTH,
        cutoff_date=cutoff_date,
        market_round=market_round,
    )

    # Filter binding table to universe branches only
    binding_in_universe = binding_table.filter(
        pl.col("branch_name").is_in(universe_branches)
    )

    # Calendar months for BF/NB/recency windows include the partial cutoff month.
    # E.g. R1: April (partial, 5 days) counts as 1 month in BF_12.
    # This is correct: if a branch binds April 5 (before R1 close), that's
    # observable information and should count in all window features.
    all_calendar_months = _generate_month_range(BF_FLOOR_MONTH, binding_table_end)
    all_calendar_months_desc = list(reversed(all_calendar_months))

    # Start with universe as base
    universe_df = pl.DataFrame({"branch_name": universe_branches})

    # BF features: count(bound in last N calendar months) / N
    bf_features = universe_df.clone()

    for window, col_prefix, bound_col in [
        *[(w, "bf", "onpeak_bound") for w in BF_WINDOWS_ONPEAK],
        *[(w, "bfo", "offpeak_bound") for w in BF_WINDOWS_OFFPEAK],
        *[(w, "bf_combined", "combined_bound") for w in BF_WINDOWS_COMBINED],
    ]:
        feature_name = f"{col_prefix}_{window}"
        window_months = all_calendar_months_desc[:window]

        if len(window_months) == 0:
            bf_features = bf_features.with_columns(pl.lit(0.0).alias(feature_name))
            continue

        window_data = binding_in_universe.filter(
            pl.col("month").is_in(window_months)
        )

        # Count bound months per branch
        branch_counts = window_data.group_by("branch_name").agg(
            pl.col(bound_col).sum().alias("_bound_count")
        )

        bf_features = bf_features.join(branch_counts, on="branch_name", how="left").with_columns(
            (pl.col("_bound_count").fill_null(0).cast(pl.Float64) / window).alias(feature_name)
        ).drop("_bound_count")

    # ── Phase 8: strength BF, windowed SPDA, trend, recency ────────────

    # Strength BF: sum(SP in last 12 months) — magnitude, not frequency
    # Only 12mo to avoid collinearity with windowed SPDA at 6/24
    for sp_col, prefix in [("onpeak_sp", "bfsp"), ("offpeak_sp", "bfsp_off")]:
        window_months_12 = all_calendar_months_desc[:12]
        feature_name = f"{prefix}_12"
        if len(window_months_12) == 0:
            bf_features = bf_features.with_columns(pl.lit(0.0).alias(feature_name))
            continue
        window_data = binding_in_universe.filter(pl.col("month").is_in(window_months_12))
        branch_sp = window_data.group_by("branch_name").agg(
            pl.col(sp_col).sum().alias("_sp_sum")
        )
        bf_features = bf_features.join(branch_sp, on="branch_name", how="left").with_columns(
            pl.col("_sp_sum").fill_null(0.0).alias(feature_name)
        ).drop("_sp_sum")

    # Windowed SPDA: cumulative SP over 6/24 months (not all-time)
    for spda_window in [6, 24]:
        spda_months = all_calendar_months_desc[:spda_window]
        if len(spda_months) == 0:
            for prefix in ["spda", "spda_off"]:
                bf_features = bf_features.with_columns(pl.lit(0.0).alias(f"{prefix}_{spda_window}"))
            continue
        spda_data = binding_in_universe.filter(pl.col("month").is_in(spda_months))
        for sp_col, prefix in [("onpeak_sp", "spda"), ("offpeak_sp", "spda_off")]:
            feature_name = f"{prefix}_{spda_window}"
            branch_sp = spda_data.group_by("branch_name").agg(
                pl.col(sp_col).sum().alias("_sp_sum")
            )
            bf_features = bf_features.join(branch_sp, on="branch_name", how="left").with_columns(
                pl.col("_sp_sum").fill_null(0.0).alias(feature_name)
            ).drop("_sp_sum")

    # BF trend: short vs long window
    for freq_prefix in ["bf", "bfo"]:
        short_col, long_col = f"{freq_prefix}_6", f"{freq_prefix}_12"
        if short_col in bf_features.columns and long_col in bf_features.columns:
            bf_features = bf_features.with_columns(
                (pl.col(short_col) - pl.col(long_col)).alias(f"{freq_prefix}_trend_6_12"),
                pl.when(pl.col(long_col) > 0)
                .then(pl.col(short_col) / pl.col(long_col))
                .otherwise(pl.when(pl.col(short_col) > 0).then(2.0).otherwise(0.0))
                .alias(f"{freq_prefix}_accel_6_12"),
            )

    # Recency: months since last bind (0 = most recent month, higher = older)
    month_to_recency = {m: i for i, m in enumerate(all_calendar_months_desc)}
    for bound_col, prefix in [("onpeak_bound", "recency"), ("offpeak_bound", "recency_off")]:
        last_bind = binding_in_universe.filter(pl.col(bound_col)).group_by("branch_name").agg(
            pl.col("month").max().alias("_last_month")
        )
        bf_features = bf_features.join(last_bind, on="branch_name", how="left")
        sentinel_recency = float(len(all_calendar_months_desc))
        bf_features = bf_features.with_columns(
            pl.col("_last_month")
            .replace_strict(month_to_recency, default=None)
            .fill_null(sentinel_recency)
            .cast(pl.Float64)
            .alias(f"{prefix}_months_since")
        ).drop("_last_month")

    # Recent max SP: highest single-month SP in last 12 months
    window_12 = all_calendar_months_desc[:12]
    if len(window_12) > 0:
        window_data_12 = binding_in_universe.filter(pl.col("month").is_in(window_12))
        for sp_col, prefix in [("onpeak_sp", "recent_max_sp"), ("offpeak_sp", "recent_max_sp_off")]:
            branch_max = window_data_12.group_by("branch_name").agg(
                pl.col(sp_col).max().alias("_max_sp")
            )
            bf_features = bf_features.join(branch_max, on="branch_name", how="left").with_columns(
                pl.col("_max_sp").fill_null(0.0).alias(prefix)
            ).drop("_max_sp")
    else:
        bf_features = bf_features.with_columns(
            pl.lit(0.0).alias("recent_max_sp"),
            pl.lit(0.0).alias("recent_max_sp_off"),
        )

    # da_rank_value: dense rank descending of cumulative_sp
    cumulative_sp = binding_in_universe.group_by("branch_name").agg(
        pl.col("combined_sp").sum().alias("cumulative_sp")
    )

    bf_features = bf_features.join(cumulative_sp, on="branch_name", how="left").with_columns(
        pl.col("cumulative_sp").fill_null(0.0)
    )

    # has_hist_da flag
    bf_features = bf_features.with_columns(
        (pl.col("cumulative_sp") > 0).alias("has_hist_da")
    )

    # Dense rank descending for positive-SP branches (ties get same rank)
    positive = bf_features.filter(pl.col("cumulative_sp") > 0)
    n_positive = len(positive)

    if n_positive > 0:
        positive = positive.with_columns(
            pl.col("cumulative_sp").rank(method="dense", descending=True).cast(pl.Float64).alias("da_rank_value")
        )
        n_distinct_ranks = int(positive["da_rank_value"].max())
        # Sentinel for zero-history branches: one past the last dense rank
        sentinel = float(n_distinct_ranks + 1)
        bf_features = bf_features.join(
            positive.select(["branch_name", "da_rank_value"]),
            on="branch_name",
            how="left",
        ).with_columns(
            pl.col("da_rank_value").fill_null(sentinel)
        )
    else:
        bf_features = bf_features.with_columns(
            pl.lit(1.0).alias("da_rank_value")
        )

    # Rename cumulative_sp -> shadow_price_da (raw historical SP, legitimate feature)
    hist_df = bf_features.rename({"cumulative_sp": "shadow_price_da"})

    logger.info(
        "History features %s/%s: %d branches, %d with history, %d months scanned",
        eval_py, aq_quarter, len(hist_df),
        int(hist_df["has_hist_da"].sum()), len(all_calendar_months_desc),
    )

    return hist_df, binding_table
