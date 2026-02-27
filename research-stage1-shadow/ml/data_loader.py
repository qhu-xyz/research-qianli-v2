"""Data loading for shadow price classification pipeline.

Two modes:
  - SMOKE_TEST=true: synthetic data (no Ray, no pbase)
  - SMOKE_TEST=false: real data via Ray + pbase (requires cluster)
"""

import os
import resource

import numpy as np
import polars as pl

from ml.config import FeatureConfig, PipelineConfig


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_data(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and validation DataFrames.

    Returns (train_df, val_df) as polars DataFrames with 14 feature columns
    plus actual_shadow_price, constraint_id, and auction_month.
    """
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"
    if smoke:
        return _load_smoke(config)
    else:
        return _load_real(config)


def _load_smoke(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic data for smoke testing."""
    rng = np.random.RandomState(42)
    fc = FeatureConfig()
    n = 100

    binding = (rng.random(n) < 0.07).astype(bool)

    data = {}
    for feat in fc.features:
        data[feat] = rng.randn(n).tolist()

    data["actual_shadow_price"] = np.where(
        binding, rng.lognormal(3, 1.5, size=n), 0.0
    ).tolist()
    data["constraint_id"] = [f"C{i:04d}" for i in range(n)]
    data["auction_month"] = [config.auction_month or "2021-07"] * n

    df = pl.DataFrame(data)

    # Split 80/20 train/val
    split = int(n * 0.8)
    return df[:split], df[split:]


def _load_real(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load real data via Ray + pbase.

    Requires Ray cluster and pbase data access.
    """
    print(f"[data_loader] mem before Ray init: {mem_mb():.0f} MB")

    from pbase.config.ray import init_ray
    import pmodel
    import ml as shadow_ml

    init_ray(
        address="ray://10.8.0.36:10001", extra_modules=[pmodel, shadow_ml]
    )

    # Real data loading would go here -- placeholder for now
    raise NotImplementedError(
        "Real data loading requires Ray cluster and pbase data access"
    )
