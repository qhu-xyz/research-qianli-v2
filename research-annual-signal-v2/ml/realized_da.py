"""Realized DA cache loading — single source of truth for DA data access.

Cache layout in data/realized_da/:
  {YYYY-MM}.parquet         — onpeak (constraint_id, realized_sp)
  {YYYY-MM}_offpeak.parquet — offpeak

Cache layout in data/realized_da_daily/:
  {YYYY-MM-DD}_{peak_type}.parquet — daily (constraint_id, realized_sp)
  .done_{YYYY-MM}_{peak_type}      — sentinel marking month as fetched
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

from ml.config import DA_CACHE_DIR

DA_DAILY_CACHE_DIR = DA_CACHE_DIR.parent / "realized_da_daily"


def _cache_path(month: str, peak_type: str) -> str:
    suffix = "_offpeak" if peak_type == "offpeak" else ""
    return str(DA_CACHE_DIR / f"{month}{suffix}.parquet")


_VALID_PEAK_TYPES = ("onpeak", "offpeak")


def load_month(month: str, peak_type: str) -> pl.DataFrame:
    """Load cached DA for one month+ctype.

    Returns DataFrame with columns: constraint_id (Utf8), realized_sp (Float64).
    realized_sp = abs(sum(shadow_price)) per constraint_id, already netted within month+ctype.
    """
    assert peak_type in _VALID_PEAK_TYPES, (
        f"Invalid peak_type '{peak_type}', must be one of {_VALID_PEAK_TYPES}"
    )
    path = _cache_path(month, peak_type)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DA cache not found: {path}. Run scripts/fetch_realized_da.py first.")
    df = pl.read_parquet(path)
    assert "constraint_id" in df.columns, f"Missing constraint_id in {path}"
    assert "realized_sp" in df.columns, f"Missing realized_sp in {path}"
    return df


def load_quarter(market_months: list[str]) -> pl.DataFrame:
    """Load combined onpeak+offpeak DA for a quarter (3 months).

    Aggregates: sum(realized_sp) per constraint_id across months and both ctypes.
    These are nonneg values being summed — no re-netting needed.
    """
    frames: list[pl.DataFrame] = []
    for month in market_months:
        for peak_type in ["onpeak", "offpeak"]:
            df = load_month(month, peak_type)
            frames.append(df)

    combined = pl.concat(frames)
    return combined.group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )


def load_quarter_per_ctype(
    market_months: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load quarter DA split by ctype — for per-ctype GT monitoring.

    Returns (onpeak_df, offpeak_df), each with (constraint_id, realized_sp).
    """
    onpeak_frames: list[pl.DataFrame] = []
    offpeak_frames: list[pl.DataFrame] = []

    for month in market_months:
        onpeak_frames.append(load_month(month, "onpeak"))
        offpeak_frames.append(load_month(month, "offpeak"))

    onpeak = pl.concat(onpeak_frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
    offpeak = pl.concat(offpeak_frames).group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
    return onpeak, offpeak


# ── Daily cache ──────────────────────────────────────────────────────────


def _daily_cache_path(trade_date: date, peak_type: str) -> str:
    return str(DA_DAILY_CACHE_DIR / f"{trade_date}_{peak_type}.parquet")


def has_daily_cache() -> bool:
    """Check if any daily cache files exist."""
    return DA_DAILY_CACHE_DIR.exists() and any(DA_DAILY_CACHE_DIR.glob("*.parquet"))


def load_day(trade_date: date, peak_type: str) -> pl.DataFrame:
    """Load cached daily DA for one date+ctype.

    Returns DataFrame with columns: constraint_id (Utf8), realized_sp (Float64).
    """
    assert peak_type in _VALID_PEAK_TYPES
    path = _daily_cache_path(trade_date, peak_type)
    if not os.path.exists(path):
        return pl.DataFrame(schema={"constraint_id": pl.Utf8, "realized_sp": pl.Float64})
    return pl.read_parquet(path)


def load_month_daily(
    month: str,
    peak_type: str,
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Load daily DA for one month+ctype, filtered by cutoff_date.

    If cutoff_date is provided, only includes days strictly before cutoff_date.
    Aggregates daily files into one month-level DataFrame per constraint_id.

    Returns: (constraint_id, realized_sp) — same schema as load_month().
    """
    assert peak_type in _VALID_PEAK_TYPES
    year, mon = int(month[:4]), int(month[5:7])

    import calendar
    n_days = calendar.monthrange(year, mon)[1]

    frames = []
    for day in range(1, n_days + 1):
        d = date(year, mon, day)
        if cutoff_date is not None and d >= cutoff_date:
            break
        df = load_day(d, peak_type)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return pl.DataFrame(schema={"constraint_id": pl.Utf8, "realized_sp": pl.Float64})

    combined = pl.concat(frames)
    return combined.group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )


def load_months_with_cutoff(
    months: list[str],
    peak_type: str,
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Load DA for multiple months, respecting cutoff_date.

    For months entirely before cutoff: loads from monthly cache (fast).
    For the cutoff month: loads from daily cache with date filter.
    For months after cutoff: skipped.

    Falls back to monthly cache if daily cache is not available.
    Returns: (constraint_id, realized_sp).
    """
    use_daily = has_daily_cache() and cutoff_date is not None
    frames = []

    for month in months:
        year, mon = int(month[:4]), int(month[5:7])
        month_first = date(year, mon, 1)

        if cutoff_date is not None and month_first >= cutoff_date:
            # Entire month is at or after cutoff — skip
            continue

        import calendar
        month_last = date(year, mon, calendar.monthrange(year, mon)[1])

        if cutoff_date is not None and cutoff_date <= month_last and use_daily:
            # Cutoff falls within this month — use daily cache
            frames.append(load_month_daily(month, peak_type, cutoff_date=cutoff_date))
        else:
            # Entire month is before cutoff — use monthly cache (fast)
            frames.append(load_month(month, peak_type))

    if not frames:
        return pl.DataFrame(schema={"constraint_id": pl.Utf8, "realized_sp": pl.Float64})

    combined = pl.concat(frames)
    return combined.group_by("constraint_id").agg(
        pl.col("realized_sp").sum()
    )
