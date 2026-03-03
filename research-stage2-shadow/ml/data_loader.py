"""Data loading for stage-2 shadow pipeline.

Dispatches between synthetic smoke data (SMOKE_TEST=true) and real data.
Real data loading will be implemented when we run actual benchmarks.
"""
from __future__ import annotations

import os
import resource

import numpy as np
import polars as pl

from ml.config import PipelineConfig


def mem_mb() -> float:
    """Current process RSS in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_data(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str,
    period_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train/val DataFrames for the given auction month.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration (features are read from cfg.regressor.features).
    auction_month : str
        Auction month in YYYY-MM format.
    class_type : str
        Class type, e.g. "peak" or "offpeak".
    period_type : str
        Period type, e.g. "monthly" or "daily".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (train_df, val_df) polars DataFrames.
    """
    if os.environ.get("SMOKE_TEST", "").lower() == "true":
        return _load_smoke(cfg)
    return _load_real(cfg, auction_month, class_type, period_type)


def _load_smoke(cfg: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic data for testing.

    Returns 80-row train and 20-row val DataFrames with:
    - All regressor feature columns (Float64)
    - actual_shadow_price: ~90% zeros, ~10% positive (exponential, scale=200)
    - constraint_id, branch_name: string metadata columns
    """
    rng = np.random.RandomState(42)
    n_total = 100
    n_train = 80
    features = cfg.regressor.features

    # Feature data: uniform [0, 1) for all features
    feature_data: dict[str, np.ndarray] = {}
    for feat in features:
        feature_data[feat] = rng.uniform(0.0, 1.0, size=n_total)

    # Target: ~90% zeros, ~10% positive from exponential(scale=200)
    is_binding = rng.random(n_total) < 0.10
    shadow_prices = np.where(
        is_binding,
        rng.exponential(scale=200.0, size=n_total),
        0.0,
    )

    # Metadata
    constraint_ids = [f"constraint_{i:04d}" for i in range(n_total)]
    branch_names = [f"branch_{rng.randint(1, 20):03d}" for _ in range(n_total)]

    # Build polars DataFrame
    data: dict[str, list | np.ndarray] = {}
    for feat in features:
        data[feat] = feature_data[feat]
    data["actual_shadow_price"] = shadow_prices
    data["constraint_id"] = constraint_ids
    data["branch_name"] = branch_names

    df = pl.DataFrame(data)

    # Cast feature columns to Float64 explicitly
    cast_exprs = [pl.col(feat).cast(pl.Float64) for feat in features]
    cast_exprs.append(pl.col("actual_shadow_price").cast(pl.Float64))
    df = df.with_columns(cast_exprs)

    train_df = df.slice(0, n_train)
    val_df = df.slice(n_train, n_total - n_train)

    return train_df, val_df


def _load_real(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str,
    period_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load real data for benchmarking.

    Not yet implemented -- will be added when we run real benchmarks.
    """
    raise NotImplementedError(
        "Loading real data is not yet implemented. "
        "Set SMOKE_TEST=true for synthetic data, or implement "
        "_load_real() when real benchmark data is available."
    )
