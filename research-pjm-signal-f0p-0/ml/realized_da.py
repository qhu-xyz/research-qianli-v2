# ml/realized_da.py
"""PJM realized DA shadow price loader and fetcher.

Key difference from MISO: PJM realized DA is aggregated by branch_name
(via branch mapping from constraint_info), not by constraint_id.

load_realized_da    -- read a cached month from parquet
fetch_and_cache_month -- fetch from PJM API via Ray, map to branches, cache
_fetch_raw_da       -- raw DA fetch (no branch mapping)

REQUIRES RAY for fetch_and_cache_month.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import REALIZED_DA_CACHE


def _cache_path(month: str, peak_type: str, cache_dir: str) -> Path:
    """Return cache file path."""
    if peak_type == "onpeak":
        return Path(cache_dir) / f"{month}.parquet"
    return Path(cache_dir) / f"{month}_{peak_type}.parquet"


def load_realized_da(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Read cached realized DA shadow prices for a month.

    Returns
    -------
    pl.DataFrame
        Columns: [branch_name (String), realized_sp (Float64)]
    """
    p = _cache_path(month, peak_type, cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"No cached realized DA for {month}/{peak_type}: {p}")
    df = pl.read_parquet(str(p))
    return df.select(
        pl.col("branch_name").cast(pl.String),
        pl.col("realized_sp").cast(pl.Float64),
    )


def _fetch_raw_da(month: str, peak_type: str) -> pl.DataFrame:
    """Fetch raw PJM DA shadow prices for one month.

    PJM uses US/Eastern timezone (unlike MISO which uses US/Central).

    Returns polars DataFrame with columns: monitored_facility, shadow_price.
    """
    from pbase.analysis.tools.all_positions import PjmApTools

    st = pd.Timestamp(f"{month}-01", tz="US/Eastern")
    et = st + pd.offsets.MonthBegin(1)

    aptools = PjmApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st, et_ex=et, peak_type=peak_type,
    )

    if da_shadow is None or len(da_shadow) == 0:
        return pl.DataFrame(schema={
            "monitored_facility": pl.String,
            "shadow_price": pl.Float64,
        })

    return pl.from_pandas(da_shadow.reset_index()).select([
        pl.col("monitored_facility").cast(pl.String),
        pl.col("shadow_price").cast(pl.Float64),
    ])


def fetch_and_cache_month(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
    period_type: str = "f0",
) -> Path:
    """Fetch realized DA, map to branches via constraint_info, and cache.

    The branch mapping uses constraint_info for the given month. If
    constraint_info is unavailable, falls back to a nearby month.
    If no constraint_info found at all, raises ValueError (fail closed).

    NOTE on period_type: constraint_info is period-type invariant (verified
    in preflight). Safe to use period_type="f0" default.

    Returns path to cached parquet with columns: branch_name, realized_sp.
    """
    from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches

    out_path = _cache_path(month, peak_type, cache_dir)
    if out_path.exists():
        return out_path

    # Fetch raw DA
    raw_da = _fetch_raw_da(month, peak_type)

    if len(raw_da) == 0:
        df = pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })
    else:
        # Load constraint_info for branch mapping
        ci = load_constraint_info(month, period_type=period_type)
        if len(ci) == 0:
            for offset in [1, -1, 2, -2]:
                alt = (pd.Timestamp(month) + pd.DateOffset(months=offset)).strftime("%Y-%m")
                ci = load_constraint_info(alt, period_type=period_type)
                if len(ci) > 0:
                    print(f"[realized_da] Using constraint_info from {alt} for {month}")
                    break

        if len(ci) == 0:
            raise ValueError(
                f"[realized_da] FATAL: no constraint_info for {month} or neighbors. "
                f"Cannot build branch mapping — refusing to fall back to naive join "
                f"(captures only ~46% of DA value). Fix constraint_info availability "
                f"or manually populate the cache for this month."
            )
        else:
            bmap = build_branch_map(ci)
            df = map_da_to_branches(raw_da, bmap)

    # Write atomically
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.write_parquet(str(tmp))
    tmp.rename(out_path)

    print(f"[realized_da] Cached {month}/{peak_type}: {len(df)} branches")
    return out_path
