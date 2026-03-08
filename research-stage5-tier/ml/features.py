"""Feature preparation for LTR pipeline."""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import LTRConfig


def prepare_features(
    df: pl.DataFrame,
    cfg: LTRConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from df, fill nulls with 0."""
    cols = list(cfg.features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[features] WARNING: {len(missing)} features missing, filling with 0: {missing}")

    X_parts = []
    for c in cols:
        if c in df.columns:
            X_parts.append(df[c].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            X_parts.append(np.zeros(len(df), dtype=np.float64))

    X = np.column_stack(X_parts) if X_parts else np.zeros((len(df), 0))
    return X, list(cfg.monotone_constraints)


def compute_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Compute XGBoost query group sizes from query_month column.

    Data must be sorted by query_month before calling this.
    Returns array of group sizes (one per unique query_month).
    """
    months = df["query_month"].to_list()
    groups = []
    if not months:
        return np.array([], dtype=np.int32)

    current = months[0]
    count = 0
    for m in months:
        if m == current:
            count += 1
        else:
            groups.append(count)
            current = m
            count = 1
    groups.append(count)
    return np.array(groups, dtype=np.int32)
