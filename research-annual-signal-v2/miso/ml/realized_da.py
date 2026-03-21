"""Realized DA cache loading — single source of truth for DA data access.

Cache layout in data/realized_da/:
  {YYYY-MM}.parquet         — onpeak (constraint_id, realized_sp)
  {YYYY-MM}_offpeak.parquet — offpeak
"""
from __future__ import annotations

import os

import polars as pl

from ml.config import DA_CACHE_DIR


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
