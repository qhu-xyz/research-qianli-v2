"""V6.2B rank reproduction utilities.

Reverse-engineered from the stored V6.2B parquet columns:

  score = 0.6 * da_rank_value + 0.3 * density_mix_rank_value + 0.1 * density_ori_rank_value
  rank  = dense_rank(score) / dense_rank(score).max()

This reproduces `rank` exactly (including tie behavior) for the months verified.
"""

from __future__ import annotations

import numpy as np


def v62b_score(
    da_rank_value: np.ndarray,
    density_mix_rank_value: np.ndarray,
    density_ori_rank_value: np.ndarray,
) -> np.ndarray:
    da = np.asarray(da_rank_value, dtype=float)
    mix = np.asarray(density_mix_rank_value, dtype=float)
    ori = np.asarray(density_ori_rank_value, dtype=float)
    return 0.6 * da + 0.3 * mix + 0.1 * ori


def dense_rank_normalized(values: np.ndarray) -> np.ndarray:
    """Dense rank 1..K, normalized by K -> in (0, 1], ties share rank.

    This matches the tie behavior we observe in V6.2B `rank` for months where the
    score has ties (K < n).
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    unique_sorted = np.sort(np.unique(x))
    dense_rank = np.searchsorted(unique_sorted, x) + 1  # 1..K
    k = int(dense_rank.max())
    return dense_rank.astype(float) / float(k)


def v62b_rank_from_columns(
    da_rank_value: np.ndarray,
    density_mix_rank_value: np.ndarray,
    density_ori_rank_value: np.ndarray,
) -> np.ndarray:
    return dense_rank_normalized(
        v62b_score(
            da_rank_value=da_rank_value,
            density_mix_rank_value=density_mix_rank_value,
            density_ori_rank_value=density_ori_rank_value,
        )
    )

