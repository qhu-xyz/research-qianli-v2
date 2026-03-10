"""Data loading for tier classification pipeline.

Dispatches between synthetic smoke data (SMOKE_TEST=true) and real data.

load_data       — train/val split for model training.
load_test_data  — target month test data for evaluation.
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
        Pipeline configuration (features are read from cfg.tier.features).
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
    - All tier feature columns (Float64)
    - actual_shadow_price: mix of tiers including negatives
    - constraint_id, branch_name: string metadata columns
    """
    rng = np.random.RandomState(42)
    n_total = 100
    n_train = 80
    # All tier features
    features = sorted(set(cfg.tier.features))

    # Feature data: uniform [0, 1) for all features
    feature_data: dict[str, np.ndarray] = {}
    for feat in features:
        feature_data[feat] = rng.uniform(0.0, 1.0, size=n_total)

    # Target: mix of tiers — some negative, some zero-100, some 100-1000, some 1000-3000, some 3000+
    tier_draws = rng.choice(5, size=n_total, p=[0.05, 0.10, 0.15, 0.30, 0.40])
    shadow_prices = np.zeros(n_total)
    for i, tier in enumerate(tier_draws):
        if tier == 0:  # [3000, inf)
            shadow_prices[i] = rng.uniform(3000, 8000)
        elif tier == 1:  # [1000, 3000)
            shadow_prices[i] = rng.uniform(1000, 3000)
        elif tier == 2:  # [100, 1000)
            shadow_prices[i] = rng.uniform(100, 1000)
        elif tier == 3:  # [0, 100)
            shadow_prices[i] = rng.uniform(0, 100)
        else:  # (-inf, 0)
            shadow_prices[i] = rng.uniform(-100, 0)

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
    """Load real data via source repo's MisoDataLoader + Ray.

    Requires Ray cluster and pbase data access. If called from
    benchmark.py (which manages Ray lifecycle), Ray will already be
    initialized and this function skips init/shutdown.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration (train_months, val_months read from here).
    auction_month : str
        Auction month in YYYY-MM format.
    class_type : str
        "onpeak" or "offpeak".
    period_type : str
        Period type, e.g. "f0", "f1".

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (fit_df, val_df) polars DataFrames.
    """
    import gc
    import re
    import sys

    import pandas as pd

    print(f"[data_loader] mem before imports: {mem_mb():.0f} MB")

    # Import source repo's loader
    src_path = "/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from shadow_price_prediction.data_loader import MisoDataLoader
    from shadow_price_prediction.config import PredictionConfig

    # Init Ray if not already initialized (benchmark.py manages lifecycle)
    import ray
    we_inited_ray = not ray.is_initialized()
    if we_inited_ray:
        os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
        from pbase.config.ray import init_ray
        import pmodel
        import ml as shadow_ml
        init_ray(extra_modules=[pmodel, shadow_ml])
    print(f"[data_loader] mem after Ray check: {mem_mb():.0f} MB")

    # Create source config
    pred_config = PredictionConfig()
    pred_config.class_type = class_type

    loader = MisoDataLoader(pred_config)

    # Compute training window: lookback accounts for forecast horizon
    # so val months' market months fall before train_end.
    auction_ts = pd.Timestamp(auction_month)
    m = re.match(r"f(\d+)", period_type)
    horizon = int(m.group(1)) if m else 3
    lookback = cfg.train_months + cfg.val_months + horizon
    train_start = auction_ts - pd.DateOffset(months=lookback)
    train_end = auction_ts

    required_ptypes = {period_type}

    print(f"[data_loader] loading {train_start} to {train_end}, ptypes={required_ptypes}")
    train_data_pd = loader.load_training_data(
        train_start=train_start,
        train_end=train_end,
        required_period_types=required_ptypes,
    )
    print(f"[data_loader] loaded {len(train_data_pd)} rows, mem: {mem_mb():.0f} MB")

    # Rename label → actual_shadow_price (source repo uses "label" for DA shadow price)
    if "label" in train_data_pd.columns and "actual_shadow_price" not in train_data_pd.columns:
        train_data_pd = train_data_pd.rename(columns={"label": "actual_shadow_price"})

    # Diagnostic: verify tier feature columns are available
    all_features = list(cfg.tier.features)
    available = [c for c in all_features if c in train_data_pd.columns]
    missing = [c for c in all_features if c not in train_data_pd.columns]
    print(f"[data_loader] tier features available: {len(available)}/{len(all_features)}")
    if missing:
        print(f"[data_loader] WARNING: missing tier features: {missing}")

    # Convert to polars
    train_data = pl.from_pandas(train_data_pd)
    del train_data_pd
    gc.collect()

    # Split: first train_months for fit, last val_months for val
    val_boundary = train_start + pd.DateOffset(months=cfg.train_months)
    val_boundary_str = val_boundary.strftime("%Y-%m")

    # auction_month column may be Timestamp or string; normalize for comparison
    if "auction_month" in train_data.columns:
        if train_data["auction_month"].dtype != pl.Utf8:
            train_data = train_data.with_columns(
                pl.col("auction_month").cast(pl.Utf8).str.slice(0, 7).alias("auction_month")
            )
        fit_df = train_data.filter(pl.col("auction_month") < val_boundary_str)
        val_df = train_data.filter(pl.col("auction_month") >= val_boundary_str)
    else:
        # Fallback: split by row proportion
        split = int(len(train_data) * cfg.train_months / (cfg.train_months + cfg.val_months))
        fit_df = train_data[:split]
        val_df = train_data[split:]

    del train_data
    gc.collect()
    print(f"[data_loader] fit: {fit_df.shape}, val: {val_df.shape}, mem: {mem_mb():.0f} MB")

    if we_inited_ray:
        ray.shutdown()
        print(f"[data_loader] Ray shutdown, mem: {mem_mb():.0f} MB")

    return fit_df, val_df


def load_test_data(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str,
    period_type: str,
) -> pl.DataFrame:
    """Load target-month test data for evaluation.

    This loads the data for the month actually being predicted (the forward
    month), which is distinct from the train/val data used for model fitting.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration.
    auction_month : str
        Auction month in YYYY-MM format.
    class_type : str
        "onpeak" or "offpeak".
    period_type : str
        Period type, e.g. "f0", "f1".

    Returns
    -------
    pl.DataFrame
        Test data for the target month.
    """
    if os.environ.get("SMOKE_TEST", "").lower() == "true":
        # Smoke: return a small synthetic test set
        _, val_df = _load_smoke(cfg)
        return val_df
    return _load_test_real(cfg, auction_month, class_type, period_type)


def _load_test_real(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str,
    period_type: str,
) -> pl.DataFrame:
    """Load real target-month test data via MisoDataLoader.

    Uses MisoDataLoader.load_test_data() to fetch the actual forward-month
    constraint data that the model is being evaluated against.
    """
    import gc
    import re
    import sys

    import pandas as pd

    print(f"[data_loader:test] mem before imports: {mem_mb():.0f} MB")

    src_path = "/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from shadow_price_prediction.data_loader import MisoDataLoader
    from shadow_price_prediction.config import PredictionConfig

    import ray
    we_inited_ray = not ray.is_initialized()
    if we_inited_ray:
        os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
        from pbase.config.ray import init_ray
        import pmodel
        import ml as shadow_ml
        init_ray(extra_modules=[pmodel, shadow_ml])

    pred_config = PredictionConfig()
    pred_config.class_type = class_type
    loader = MisoDataLoader(pred_config)

    # Compute market month from auction month + horizon
    auction_ts = pd.Timestamp(auction_month)
    m = re.match(r"f(\d+)", period_type)
    horizon = int(m.group(1)) if m else 3
    market_ts = auction_ts + pd.DateOffset(months=horizon)

    market_month_str = market_ts.strftime("%Y-%m")
    print(f"[data_loader:test] loading test data: auction={auction_month}, "
          f"market={market_month_str}, period_type={period_type}")

    # Use load_test_data_for_period directly (load_test_data has an index bug)
    test_data_pd = loader.load_test_data_for_period(
        auction_month=auction_ts,
        market_month=market_ts,
        period_type=period_type,
    )
    print(f"[data_loader:test] loaded {len(test_data_pd)} rows, mem: {mem_mb():.0f} MB")

    # Rename label → actual_shadow_price
    if "label" in test_data_pd.columns and "actual_shadow_price" not in test_data_pd.columns:
        test_data_pd = test_data_pd.rename(columns={"label": "actual_shadow_price"})

    test_df = pl.from_pandas(test_data_pd)
    del test_data_pd
    gc.collect()

    print(f"[data_loader:test] test_df: {test_df.shape}, mem: {mem_mb():.0f} MB")

    if we_inited_ray:
        ray.shutdown()
        print(f"[data_loader:test] Ray shutdown, mem: {mem_mb():.0f} MB")

    return test_df
