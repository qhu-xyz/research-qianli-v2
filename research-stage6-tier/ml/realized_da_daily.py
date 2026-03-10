"""Daily realized-DA cache for partial-month historical features.

The key requirement for stage6 is preserving the monthly aggregation semantics:

    monthly_realized_sp = abs(sum(shadow_price over the month))

To recover partial months safely, the cache must preserve signed daily netting.
Caching only daily abs values would be insufficient because it would destroy
cross-day cancellation and make `look_back_days=31` fail to reproduce the
full-month aggregate exactly.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import REALIZED_DA_DAILY_CACHE


def month_start(month: str) -> pd.Timestamp:
    """Return the first timestamp of a YYYY-MM month."""
    return pd.Timestamp(f"{month}-01", tz="US/Central")


def month_end_exclusive(month: str) -> pd.Timestamp:
    """Return the exclusive upper bound for a YYYY-MM month."""
    return month_start(month) + pd.offsets.MonthBegin(1)


def clamp_day(month: str, look_back_days: int) -> int:
    """Clamp the requested day to the actual month length.

    A fixed `look_back_days=31` should mean "through month end" for shorter
    months, not an invalid timestamp.
    """
    if look_back_days < 1:
        raise ValueError(f"look_back_days must be >= 1, got {look_back_days}")
    last_day = month_end_exclusive(month).day
    # month_end_exclusive lands on next-month day 1; subtract one day instead.
    last_day = int((month_end_exclusive(month) - pd.Timedelta(days=1)).day)
    return min(look_back_days, last_day)


def cutoff_date(month: str, look_back_days: int) -> date:
    """Return the inclusive cutoff date for the month."""
    return date.fromisoformat(f"{month}-{clamp_day(month, look_back_days):02d}")


def load_realized_da_daily(
    month: str,
    cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> pl.DataFrame:
    """Read daily realized-DA cache for one month."""
    p = Path(cache_dir) / f"{month}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No cached daily realized DA for {month}: {p}")
    df = pl.read_parquet(str(p))
    return df.select(
        pl.col("market_date").cast(pl.Date),
        pl.col("constraint_id").cast(pl.String),
        pl.col("shadow_price_net_day").cast(pl.Float64),
        pl.col("realized_sp_day").cast(pl.Float64),
    )


def fetch_and_cache_daily_month(
    month: str,
    cache_dir: str = REALIZED_DA_DAILY_CACHE,
) -> Path:
    """Fetch realized DA for one month and cache daily signed net sums.

    Requires Ray to be initialized before calling.
    """
    from pbase.analysis.tools.all_positions import MisoApTools

    st = month_start(month)
    et = month_end_exclusive(month)

    aptools = MisoApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st,
        et_ex=et,
        peak_type="onpeak",
    )

    if da_shadow is None or len(da_shadow) == 0:
        df = pl.DataFrame(
            schema={
                "market_date": pl.Date,
                "constraint_id": pl.String,
                "shadow_price_net_day": pl.Float64,
                "realized_sp_day": pl.Float64,
            }
        )
    else:
        raw = pl.from_pandas(da_shadow.reset_index())
        ts_col = "index" if "index" in raw.columns else raw.columns[0]
        raw = raw.with_columns(pl.col(ts_col).dt.date().alias("market_date"))
        df = (
            raw
            .group_by(["market_date", "constraint_id"])
            .agg(pl.col("shadow_price").sum().alias("shadow_price_net_day"))
            .with_columns(pl.col("shadow_price_net_day").abs().alias("realized_sp_day"))
            .select("market_date", "constraint_id", "shadow_price_net_day", "realized_sp_day")
        )

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{month}.parquet"
    df.write_parquet(str(out_path))
    return out_path
