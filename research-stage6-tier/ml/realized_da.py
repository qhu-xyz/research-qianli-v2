"""Realized DA shadow price loader and fetcher.

load_realized_da   -- read a cached month from parquet
fetch_and_cache_month -- fetch from MISO market data via Ray, cache to parquet

REQUIRES RAY for fetch_and_cache_month. Call init_ray() before using it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import REALIZED_DA_CACHE


def load_realized_da(
    month: str,
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Read cached realized DA shadow prices for a month.

    Parameters
    ----------
    month : str
        Month string like "2022-06".
    cache_dir : str
        Directory containing cached parquet files.

    Returns
    -------
    pl.DataFrame
        Columns: [constraint_id (String), realized_sp (Float64)]

    Raises
    ------
    FileNotFoundError
        If the cached parquet does not exist.
    """
    p = Path(cache_dir) / f"{month}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No cached realized DA for {month}: {p}")
    df = pl.read_parquet(str(p))
    return df.select(
        pl.col("constraint_id").cast(pl.String),
        pl.col("realized_sp").cast(pl.Float64),
    )


def fetch_and_cache_month(
    month: str,
    cache_dir: str = REALIZED_DA_CACHE,
) -> Path:
    """Fetch realized DA shadow prices for one month and cache to parquet.

    Requires Ray to be initialized before calling.

    Aggregation: abs(sum(shadow_price)) per constraint_id
    (net shadow price first, then take absolute value).

    Parameters
    ----------
    month : str
        Month string like "2022-06".
    cache_dir : str
        Directory to write cached parquet.

    Returns
    -------
    Path
        Path to the written parquet file.
    """
    from pbase.analysis.tools.all_positions import MisoApTools

    year, mo = month.split("-")
    st = pd.Timestamp(f"{year}-{mo}-01", tz="US/Central")
    et = st + pd.offsets.MonthBegin(1)

    aptools = MisoApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st, et_ex=et, peak_type="onpeak",
    )

    if da_shadow is None or len(da_shadow) == 0:
        df = pl.DataFrame(
            {"constraint_id": pl.Series([], dtype=pl.String),
             "realized_sp": pl.Series([], dtype=pl.Float64)},
        )
    else:
        da_pl = pl.from_pandas(da_shadow.reset_index())
        # net then abs: abs(sum(shadow_price)) per constraint_id
        df = (
            da_pl
            .group_by("constraint_id")
            .agg(pl.col("shadow_price").sum().abs().alias("realized_sp"))
        )
        df = df.with_columns(
            pl.col("constraint_id").cast(pl.String),
            pl.col("realized_sp").cast(pl.Float64),
        )

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{month}.parquet"
    df.write_parquet(str(out_path))
    return out_path
