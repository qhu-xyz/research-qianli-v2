"""Feature preparation for shadow price classification pipeline."""

import resource

import numpy as np
import polars as pl

from ml.config import FeatureConfig


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prepare_features(
    df: pl.DataFrame, config: FeatureConfig
) -> tuple[np.ndarray, list[str]]:
    """Select feature columns, fill nulls, return numpy array + column names.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing at least the feature columns.
    config : FeatureConfig
        Feature configuration with column names.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    cols : list[str]
        List of feature column names.
    """
    cols = config.features
    print(f"[features] mem before prepare: {mem_mb():.0f} MB")

    # Compute interaction features only if requested by config
    interaction_cols = {"exceed_severity_ratio", "hist_physical_interaction", "overload_exceedance_product"}
    if interaction_cols & set(cols):
        df = df.with_columns([
            (pl.col("prob_exceed_110") / (pl.col("prob_exceed_90") + 1e-6))
                .alias("exceed_severity_ratio"),
            (pl.col("hist_da") * pl.col("prob_exceed_100"))
                .alias("hist_physical_interaction"),
            (pl.col("expected_overload") * pl.col("prob_exceed_105"))
                .alias("overload_exceedance_product"),
        ])

    # Verify all requested columns exist
    missing = set(cols) - set(df.columns) - interaction_cols
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing}")

    X = df.select(cols).fill_null(0).to_numpy()
    return X, cols


def compute_binary_labels(
    df: pl.DataFrame, threshold: float = 0.0
) -> np.ndarray:
    """Compute binary labels from actual_shadow_price.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with an actual_shadow_price column.
    threshold : float
        Shadow price threshold for binding classification.

    Returns
    -------
    labels : np.ndarray
        Binary array where 1 = binding (shadow_price > threshold).
    """
    return (df["actual_shadow_price"].to_numpy() > threshold).astype(int)


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced classification.

    Returns n_negative / n_positive, or 1.0 if no positives.
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
