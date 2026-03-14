"""Data loading for LTR ranking pipeline.

Loads V6.2B signal data for constraint universe and ground truth.
Feature enrichment from raw spice6 is handled by features.py.
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load V6.2B signal data for a single month.

    Parameters
    ----------
    auction_month : str
        Auction month in YYYY-MM format.
    period_type : str
        Period type (f0, f1, etc.).
    class_type : str
        onpeak or offpeak.

    Returns
    -------
    pl.DataFrame
        V6.2B data with constraint_id, flow_direction, rank, tier,
        shadow_price_da, and all V6.2B feature columns.
    """
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    # Drop index column if present
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")
    return df


def load_train_val_test(
    eval_month: str,
    train_months: int = 6,
    val_months: int = 2,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/val/test splits for a single evaluation month.

    For eval_month M with train=6, val=2:
    - Train: months M-8 through M-3 (6 months)
    - Val: months M-2, M-1 (2 months)
    - Test: month M (the target month)

    Each month's data comes from V6.2B with an added 'query_month' column
    for XGBoost query groups.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, val_df, test_df) with query_month column added.
    """
    import pandas as pd

    eval_ts = pd.Timestamp(eval_month)
    total_lookback = train_months + val_months

    # Generate month strings for train and val
    train_month_strs = []
    for i in range(total_lookback, val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    val_month_strs = []
    for i in range(val_months, 0, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        val_month_strs.append(m)

    print(f"[data_loader] eval={eval_month} train={train_month_strs} val={val_month_strs}")
    print(f"[data_loader] mem: {mem_mb():.0f} MB")

    def _load_months(month_strs: list[str]) -> pl.DataFrame:
        dfs = []
        for m in month_strs:
            try:
                df = load_v62b_month(m, period_type, class_type)
                df = df.with_columns(pl.lit(m).alias("query_month"))
                dfs.append(df)
            except FileNotFoundError:
                print(f"[data_loader] WARNING: skipping {m} (not found)")
        if not dfs:
            raise ValueError(f"No data found for months: {month_strs}")
        return pl.concat(dfs)

    train_df = _load_months(train_month_strs)
    val_df = _load_months(val_month_strs)
    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    print(f"[data_loader] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"mem: {mem_mb():.0f} MB")

    return train_df, val_df, test_df
