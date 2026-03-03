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

    Returns (train_df, val_df) as polars DataFrames with feature columns
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

    # Source-loader features need specialized synthetic generators
    source_loader_features = {
        "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
        "is_interface", "constraint_limit",
        "density_mean", "density_variance", "density_entropy",
        "tail_concentration", "prob_band_95_100", "prob_band_100_105",
        "hist_da_max_season",
    }

    data = {}
    for feat in fc.features:
        if feat not in source_loader_features:
            data[feat] = rng.randn(n).tolist()

    # Shift factor features (positive values)
    for feat in ["sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac"]:
        data[feat] = np.abs(rng.randn(n)).tolist()

    # Constraint metadata
    data["is_interface"] = (rng.random(n) < 0.3).astype(float).tolist()
    data["constraint_limit"] = np.log1p(rng.uniform(100, 2000, n)).tolist()

    # Distribution shape features
    data["density_mean"] = (rng.uniform(0.5, 1.2, n) * np.where(binding, 1.05, 0.85)).tolist()
    data["density_variance"] = np.abs(rng.randn(n) * 0.1).tolist()
    data["density_entropy"] = rng.uniform(1.0, 5.0, n).tolist()

    # Near-boundary band features
    data["tail_concentration"] = np.where(binding, rng.uniform(0.3, 0.9, n), rng.uniform(0.01, 0.3, n)).tolist()
    data["prob_band_95_100"] = np.abs(rng.randn(n) * 0.05).tolist()
    data["prob_band_100_105"] = np.abs(rng.randn(n) * 0.03).tolist()

    # Historical enrichment
    data["hist_da_max_season"] = np.where(binding, rng.lognormal(2, 1, n), rng.exponential(0.5, n)).tolist()

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
    """Load real data via source repo's MisoDataLoader + Ray.

    Requires Ray cluster and pbase data access.
    """
    import gc
    import sys

    import pandas as pd

    print(f"[data_loader] mem before imports: {mem_mb():.0f} MB")

    # Import source repo's loader
    src_path = "/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from shadow_price_prediction.data_loader import MisoDataLoader
    from shadow_price_prediction.config import PredictionConfig

    # Init Ray (skip if already initialized — benchmark runner manages lifecycle)
    import ray
    we_inited_ray = not ray.is_initialized()
    if we_inited_ray:
        os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
        from pbase.config.ray import init_ray
        import pmodel
        import ml as shadow_ml
        init_ray(extra_modules=[pmodel, shadow_ml])
    print(f"[data_loader] mem after Ray init: {mem_mb():.0f} MB")

    # Create source config
    pred_config = PredictionConfig()
    pred_config.class_type = config.class_type

    loader = MisoDataLoader(pred_config)

    # Compute training window: load [M-lookback, M] where M = auction_month.
    # Lookback must account for forecast horizon: for ptype fN, the last N
    # months' market targets fall at or beyond train_end and get clipped by
    # the source loader's future-month guard (market_month >= train_end).
    # Extending lookback by the horizon pushes val months far enough back
    # that their market months are before train_end.
    auction_month = pd.Timestamp(config.auction_month)
    ptype = config.period_type or "f0"
    import re
    m = re.match(r"f(\d+)", ptype)
    horizon = int(m.group(1)) if m else 3
    lookback = config.train_months + config.val_months + horizon
    train_start = auction_month - pd.DateOffset(months=lookback)
    train_end = auction_month

    required_ptypes = {config.period_type}

    print(f"[data_loader] loading {train_start} to {train_end}, ptypes={required_ptypes}")
    train_data_pd = loader.load_training_data(
        train_start=train_start,
        train_end=train_end,
        required_period_types=required_ptypes,
    )
    print(f"[data_loader] loaded {len(train_data_pd)} rows, mem: {mem_mb():.0f} MB")

    # Rename label → actual_shadow_price (source repo's DA shadow price label)
    if "label" in train_data_pd.columns and "actual_shadow_price" not in train_data_pd.columns:
        train_data_pd = train_data_pd.rename(columns={"label": "actual_shadow_price"})

    # Diagnostic: verify new feature columns are available
    new_cols = [
        "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
        "is_interface", "constraint_limit",
        "density_mean", "density_variance", "density_entropy",
        "tail_concentration", "prob_band_95_100", "prob_band_100_105",
        "hist_da_max_season",
    ]
    available = [c for c in new_cols if c in train_data_pd.columns]
    missing = [c for c in new_cols if c not in train_data_pd.columns]
    print(f"[data_loader] new feature columns available: {available}")
    if missing:
        print(f"[data_loader] WARNING: missing columns: {missing}")

    # Convert to polars
    train_data = pl.from_pandas(train_data_pd)
    del train_data_pd
    gc.collect()

    # Split: first train_months for fit, last val_months for val
    val_boundary = train_start + pd.DateOffset(months=config.train_months)
    val_boundary_str = val_boundary.strftime("%Y-%m")

    # auction_month column may be Timestamp or string; normalize to string for comparison
    if "auction_month" in train_data.columns:
        if train_data["auction_month"].dtype != pl.Utf8:
            train_data = train_data.with_columns(
                pl.col("auction_month").cast(pl.Utf8).str.slice(0, 7).alias("auction_month")
            )
        fit_df = train_data.filter(pl.col("auction_month") < val_boundary_str)
        val_df = train_data.filter(pl.col("auction_month") >= val_boundary_str)
    else:
        # Fallback: split by row proportion
        split = int(len(train_data) * config.train_months / (config.train_months + config.val_months))
        fit_df = train_data[:split]
        val_df = train_data[split:]

    del train_data
    gc.collect()
    print(f"[data_loader] fit: {fit_df.shape}, val: {val_df.shape}, mem: {mem_mb():.0f} MB")

    if we_inited_ray:
        ray.shutdown()
        print(f"[data_loader] Ray shutdown, mem: {mem_mb():.0f} MB")

    return fit_df, val_df
