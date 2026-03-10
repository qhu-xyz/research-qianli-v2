"""Data loading for LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 density features,
and joins realized DA shadow prices as ground truth.

KEY CHANGE from stage4: ground truth = realized_sp (from cached parquet),
NOT shadow_price_da (which is a historical 60-month lookback feature).
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.realized_da import load_realized_da
from ml.spice6_loader import load_spice6_density


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# In-memory cache: {(month, ptype, ctype): DataFrame}
# Adjacent eval months share 7/8 training months — caching cuts NFS reads by ~87%.
_MONTH_CACHE: dict[tuple[str, str, str], pl.DataFrame] = {}


def clear_month_cache() -> None:
    """Clear the month data cache (call between unrelated benchmark runs)."""
    _MONTH_CACHE.clear()


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density + realized DA ground truth.

    Parameters
    ----------
    auction_month : str
        Auction month in YYYY-MM format.
    period_type : str
        Period type (f0, f1, etc.).
    class_type : str
        onpeak or offpeak.
    cache_dir : str or None
        Override realized DA cache directory. If None, uses ml.config.REALIZED_DA_CACHE.

    Returns
    -------
    pl.DataFrame
        V6.2B data enriched with spice6 density features and realized_sp column.
    """
    cache_key = (auction_month, period_type, class_type)
    if cache_key in _MONTH_CACHE:
        print(f"[data_loader] cache hit: {auction_month}")
        return _MONTH_CACHE[cache_key]

    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(
            spice6,
            on=["constraint_id", "flow_direction"],
            how="left",
        )
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6 enrichment: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth
    # For fN, ground truth = realized DA for delivery_month (= auction_month + N)
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)
    da_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    realized = load_realized_da(gt_month, peak_type=class_type, **da_kwargs)
    df = df.join(realized, on="constraint_id", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    n_binding = len(df.filter(pl.col("realized_sp") > 0))
    print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding for {auction_month} "
          f"(gt_month={gt_month})")

    # NO _add_engineered_features() — those 37 features are useless

    _MONTH_CACHE[cache_key] = df
    return df


def load_train_val_test(
    eval_month: str,
    train_months: int = 8,
    val_months: int = 0,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """Load train/val/test splits for a single evaluation month.

    For eval_month M with train=8, val=0:
    - Train: months M-8 through M-1 (8 months)
    - Val: None (val_months=0)
    - Test: month M (the target month)

    Each month's data is loaded via load_v62b_month which joins its OWN
    realized DA ground truth.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]
        (train_df, val_df, test_df) with query_month column added.
        val_df is None when val_months=0.
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
    val_df = _load_months(val_month_strs) if val_month_strs else None
    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    val_len = len(val_df) if val_df is not None else 0
    print(f"[data_loader] train={len(train_df)} val={val_len} test={len(test_df)} "
          f"mem: {mem_mb():.0f} MB")

    return train_df, val_df, test_df
