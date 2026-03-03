"""Feature preparation for classifier and regressor pipelines.

Functions extract ordered feature columns from polars DataFrames,
fill nulls with 0.0, and return numpy arrays paired with monotone
constraint lists ready for XGBoost.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import ClassifierConfig, RegressorConfig


def prepare_clf_features(
    df: pl.DataFrame,
    cfg: ClassifierConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract classifier features from *df* using *cfg*.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(n_rows, 13)`` with nulls filled to 0.0.
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


def prepare_reg_features(
    df: pl.DataFrame,
    cfg: RegressorConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract regressor features from *df* using *cfg*.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(n_rows, 24)`` with nulls filled to 0.0.
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


def compute_binary_labels(
    df: pl.DataFrame,
    threshold: float = 0.0,
) -> np.ndarray:
    """Convert ``actual_shadow_price`` to binary labels.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain an ``actual_shadow_price`` column.
    threshold : float
        Values strictly greater than *threshold* are labelled 1, else 0.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_rows,)`` with values in {0, 1}.
    """
    return (
        df["actual_shadow_price"]
        .to_numpy()
        .astype(np.float64)
        .__gt__(threshold)
        .astype(int)
    )


def compute_regression_target(df: pl.DataFrame) -> np.ndarray:
    """Compute ``log1p(max(0, actual_shadow_price))`` regression target.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_rows,)``.
    """
    raw = df["actual_shadow_price"].to_numpy().astype(np.float64)
    return np.log1p(np.maximum(0.0, raw))


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    """Compute class-imbalance weight for XGBoost.

    Returns ``n_negative / n_positive``, or 1.0 when there are no positives.
    """
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return 1.0
    n_neg = int(np.sum(labels == 0))
    return n_neg / n_pos
