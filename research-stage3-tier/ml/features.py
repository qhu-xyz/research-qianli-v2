"""Feature preparation for tier classification pipeline.

Functions extract ordered feature columns from polars DataFrames,
fill nulls with 0.0, and return numpy arrays paired with monotone
constraint lists ready for XGBoost.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import TierConfig


def compute_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute derived interaction features from raw columns.

    These five features are products of raw MisoDataLoader columns.
    The function is idempotent — if the columns already exist they are
    overwritten.
    """
    return df.with_columns([
        (pl.col("hist_da") * pl.col("prob_exceed_100"))
            .alias("hist_physical_interaction"),
        (pl.col("expected_overload") * pl.col("prob_exceed_105"))
            .alias("overload_exceedance_product"),
        (pl.col("prob_band_95_100") * pl.col("expected_overload"))
            .alias("band_severity"),
        (pl.col("sf_max_abs") * pl.col("prob_exceed_100"))
            .alias("sf_exceed_interaction"),
        (pl.col("hist_da_max_season") * pl.col("prob_band_100_105"))
            .alias("hist_seasonal_band"),
    ])


def prepare_features(
    df: pl.DataFrame,
    cfg: TierConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from *df* using tier config.

    Returns
    -------
    X : np.ndarray
        Feature matrix with nulls filled to 0.0.
    monotone : list[int]
        Monotone constraint values aligned with columns of *X*.
    """
    cols = list(cfg.features)
    X = (
        df.select(cols)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64)
    )
    monotone = list(cfg.monotone_constraints)
    return X, monotone


def compute_tier_labels(
    actual_shadow_price: np.ndarray,
    config: TierConfig,
) -> np.ndarray:
    """Bin actual shadow prices into tier labels.

    Uses config.bins = [-inf, 0, 100, 1000, 3000, inf] with labels [4,3,2,1,0].
    Tier 0 = highest binding ([3000, inf)), Tier 4 = not binding ((-inf, 0)).

    Parameters
    ----------
    actual_shadow_price : np.ndarray
        Ground-truth shadow prices.
    config : TierConfig
        Tier configuration with bins.

    Returns
    -------
    np.ndarray
        Integer array of tier labels in {0, 1, 2, 3, 4}.
    """
    # bins[1:-1] = [0, 100, 1000, 3000] — the interior bin edges
    interior_edges = np.array(config.bins[1:-1])
    # np.digitize with right=False: value in [edge_i, edge_{i+1}) maps to bin i+1
    # bin indices: 0 = (-inf, 0), 1 = [0, 100), 2 = [100, 1000), 3 = [1000, 3000), 4 = [3000, inf)
    bin_indices = np.digitize(actual_shadow_price, interior_edges, right=False)
    # Map bin indices to tier labels: bin 0 -> tier 4, bin 1 -> tier 3, ..., bin 4 -> tier 0
    tier_map = np.array([4, 3, 2, 1, 0])
    return tier_map[bin_indices]


def compute_sample_weights(
    tier_labels: np.ndarray,
    config: TierConfig,
) -> np.ndarray:
    """Compute per-sample weights based on tier class weights.

    Parameters
    ----------
    tier_labels : np.ndarray
        Tier labels for each sample.
    config : TierConfig
        Tier configuration with class_weights.

    Returns
    -------
    np.ndarray
        Float64 array of sample weights.
    """
    weights = np.ones(len(tier_labels), dtype=np.float64)
    for tier, weight in config.class_weights.items():
        weights[tier_labels == tier] = weight
    return weights
