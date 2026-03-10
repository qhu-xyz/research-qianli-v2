"""As-of historical feature builders for stage6.

Stage6 keeps the safe row boundary from `v10e-lag1` but rebuilds the
historical realized-DA features so that the most recent prior month can
contribute partially through `look_back_days`.

For the first implementation, only the `binding_freq_*` family is rebuilt.
Older prior months still use the stage5 full-month cache. Only the immediately
preceding month is truncated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ml.config import REALIZED_DA_CACHE, REALIZED_DA_DAILY_CACHE
from ml.realized_da import load_realized_da
from ml.realized_da_daily import clamp_day, cutoff_date, load_realized_da_daily


LOOKBACK_WINDOWS: tuple[int, ...] = (1, 3, 6, 12, 15)


def prev_month(month: str) -> str:
    """Return the previous month as YYYY-MM."""
    year, mo = map(int, month.split("-"))
    mo -= 1
    if mo == 0:
        year -= 1
        mo = 12
    return f"{year:04d}-{mo:02d}"


def month_sequence_before(target_month: str, count: int) -> list[str]:
    """Return the last `count` months before `target_month`, oldest first."""
    months: list[str] = []
    current = target_month
    for _ in range(count):
        current = prev_month(current)
        months.append(current)
    months.reverse()
    return months


def full_month_binding_set(
    month: str,
    cache_dir: str = REALIZED_DA_CACHE,
) -> set[str]:
    """Binding set using the stage5 full-month semantics."""
    df = load_realized_da(month, cache_dir=cache_dir)
    return set(df.filter(pl.col("realized_sp") > 0)["constraint_id"].to_list())


def partial_month_realized_sp(
    month: str,
    look_back_days: int,
    cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> pl.DataFrame:
    """Partial-month realized SP preserving monthly netting semantics.

    This aggregates:

        abs(sum(shadow_price_net_day through cutoff))

    which matches the stage5 monthly definition when `look_back_days` reaches
    the full month length.
    """
    daily = load_realized_da_daily(month, cache_dir=cache_dir)
    cutoff = cutoff_date(month, look_back_days)
    partial = daily.filter(pl.col("market_date") <= cutoff)
    if len(partial) == 0:
        return pl.DataFrame(
            schema={"constraint_id": pl.String, "realized_sp_partial": pl.Float64}
        )
    return (
        partial
        .group_by("constraint_id")
        .agg(pl.col("shadow_price_net_day").sum().abs().alias("realized_sp_partial"))
        .select(pl.col("constraint_id").cast(pl.String), pl.col("realized_sp_partial").cast(pl.Float64))
    )


def partial_month_binding_set(
    month: str,
    look_back_days: int,
    cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> set[str]:
    """Binding set for a partial month using exact stage5 netting semantics."""
    partial = partial_month_realized_sp(month, look_back_days, cache_dir=cache_dir)
    return set(partial.filter(pl.col("realized_sp_partial") > 0)["constraint_id"].to_list())


def compute_binding_freq_asof(
    constraint_ids: list[str],
    target_month: str,
    lookback_months: int,
    look_back_days: int = 12,
    monthly_cache_dir: str = REALIZED_DA_CACHE,
    daily_cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> np.ndarray:
    """Binding frequency with a partial most-recent month.

    For target month `M`, months before `M` are considered. All months older than
    `M-1` use the full-month cache. The immediately previous month `M-1` uses the
    partial-month cutoff defined by `look_back_days`.
    """
    prior = month_sequence_before(target_month, lookback_months)
    if not prior:
        return np.zeros(len(constraint_ids), dtype=np.float64)

    partial_month = prev_month(target_month)
    partial_set = partial_month_binding_set(partial_month, look_back_days, cache_dir=daily_cache_dir)

    full_sets: dict[str, set[str]] = {}
    for month in prior:
        if month == partial_month:
            continue
        full_sets[month] = full_month_binding_set(month, cache_dir=monthly_cache_dir)

    freq = np.zeros(len(constraint_ids), dtype=np.float64)
    for month in prior:
        current_set = partial_set if month == partial_month else full_sets[month]
        for i, cid in enumerate(constraint_ids):
            if cid in current_set:
                freq[i] += 1.0
    return freq / len(prior)


def add_binding_freq_columns_asof(
    df: pl.DataFrame,
    target_month: str,
    look_back_days: int = 12,
    monthly_cache_dir: str = REALIZED_DA_CACHE,
    daily_cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> pl.DataFrame:
    """Add `binding_freq_*` columns using stage6 partial-month semantics."""
    cids = df["constraint_id"].to_list()
    out = df
    for window in LOOKBACK_WINDOWS:
        values = compute_binding_freq_asof(
            cids,
            target_month,
            window,
            look_back_days=look_back_days,
            monthly_cache_dir=monthly_cache_dir,
            daily_cache_dir=daily_cache_dir,
        )
        out = out.with_columns(pl.Series(f"binding_freq_{window}", values))
    return out


@dataclass
class MonthCollapseCheck:
    """Comparison result for `look_back_days=31` equivalence."""

    month: str
    look_back_days: int
    partial_count: int
    full_count: int
    missing_in_partial: int
    extra_in_partial: int

    @property
    def matches(self) -> bool:
        return self.missing_in_partial == 0 and self.extra_in_partial == 0


def verify_month_collapse(
    month: str,
    look_back_days: int = 31,
    monthly_cache_dir: str = REALIZED_DA_CACHE,
    daily_cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> MonthCollapseCheck:
    """Verify that a full-length partial month matches the full-month cache.

    This verifies the *historical feature rebuild* only. It does not imply the
    overall stage6 dataset equals the old leaky dataset, because stage6 still
    keeps the safe row boundary and does not reintroduce the `M-1` training row.
    """
    partial = partial_month_binding_set(month, look_back_days, cache_dir=daily_cache_dir)
    full = full_month_binding_set(month, cache_dir=monthly_cache_dir)
    return MonthCollapseCheck(
        month=month,
        look_back_days=look_back_days,
        partial_count=len(partial),
        full_count=len(full),
        missing_in_partial=len(full - partial),
        extra_in_partial=len(partial - full),
    )
