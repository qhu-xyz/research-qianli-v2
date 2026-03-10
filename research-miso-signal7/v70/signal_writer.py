"""Rank/tier computation and signal assembly helpers for V7.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_STAGE5 = Path(__file__).resolve().parent.parent.parent / "research-stage5-tier"
if str(_STAGE5) not in sys.path:
    sys.path.insert(0, str(_STAGE5))

from ml.config import MISO_AUCTION_SCHEDULE


def compute_rank_tier(
    scores: np.ndarray,
    v62b_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw ML scores to row-percentile rank (0,1] and tier (0-4).

    Sorting order (deterministic total order):
      1. ML score descending (higher = more binding = lower rank)
      2. V6.2B rank_ori ascending (lower = more binding, tie-break)
      3. Original index ascending (final tie-break)

    rank = row_position / n  (row-percentile, ~20% per tier)
    tier = ceil(rank * 5) - 1
    """
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([], dtype=int)

    # lexsort processes keys last-to-first:
    # tertiary=original_index, secondary=v62b_rank, primary=-scores
    order = np.lexsort((np.arange(n), v62b_rank, -scores))

    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(n) + 1) / n

    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier


def available_ptypes(auction_month: str) -> list[str]:
    """Return period types available for this auction month."""
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return MISO_AUCTION_SCHEDULE.get(month_num, ["f0"])
