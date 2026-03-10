# ml/data_loader.py
"""Data loading for PJM LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 density features,
and joins realized DA shadow prices as ground truth.

KEY PJM DIFFERENCE: ground truth joins on branch_name (not constraint_id).
V6.2B parquet already has a branch_name column.
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


_MONTH_CACHE: dict[tuple[str, str, str, str | None], pl.DataFrame] = {}


def clear_month_cache() -> None:
    _MONTH_CACHE.clear()


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density + realized DA.

    PJM-specific: realized DA is joined on branch_name, not constraint_id.
    V6.2B parquet already contains a branch_name column.
    """
    cache_key = (auction_month, period_type, class_type, cache_dir)
    if cache_key in _MONTH_CACHE:
        return _MONTH_CACHE[cache_key]

    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Ensure constraint_id and branch_name are strings
    df = df.with_columns(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
    )

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(spice6, on=["constraint_id", "flow_direction"], how="left")
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth on branch_name
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)

    # Map class_type to peak_type for DA fetch
    peak_type = class_type  # PJM uses same names: onpeak, dailyoffpeak, wkndonpeak
    da_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        realized = load_realized_da(gt_month, peak_type=peak_type, **da_kwargs)
        # Join on branch_name (PJM-specific)
        df = df.join(realized, on="branch_name", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
        n_binding = len(df.filter(pl.col("realized_sp") > 0))
        print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding for {auction_month} "
              f"(gt_month={gt_month})")
    except FileNotFoundError:
        print(f"[data_loader] WARNING: no realized DA for {gt_month}/{peak_type}")
        df = df.with_columns(pl.lit(0.0).alias("realized_sp"))

    _MONTH_CACHE[cache_key] = df
    return df
