# ml/features.py
"""Feature preparation for PJM LTR pipeline.

Identical to MISO features.py. Binding frequency enrichment happens in
the script layer (run_v2_ml.py), not here.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import LTRConfig


def _add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add v7_formula_score if not already present.

    IMPORTANT: This column MUST be named 'v7_formula_score' to match
    V10E_FEATURES in ml/config.py. The v2 script (run_v2_ml.py) computes
    it with per-slice blend weights; this fallback uses the V6.2B formula.
    """
    if "v7_formula_score" not in df.columns:
        has_cols = all(c in df.columns for c in
                       ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"])
        if has_cols:
            df = df.with_columns(
                (0.60 * pl.col("da_rank_value")
                 + 0.30 * pl.col("density_mix_rank_value")
                 + 0.10 * pl.col("density_ori_rank_value")
                ).alias("v7_formula_score")
            )
    return df


def prepare_features(
    df: pl.DataFrame,
    cfg: LTRConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from df, fill nulls with 0."""
    df = _add_derived_features(df)
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
    """Compute query group sizes from query_month column."""
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
