# v70/signal_writer.py
"""Rank/tier computation and signal assembly for PJM V7.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PJM_AUCTION_SCHEDULE


def compute_rank_tier(
    scores: np.ndarray,
    v62b_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-percentile rank with V6.2B tie-breaking.

    1. ML score descending (higher = more binding = lower rank)
    2. V6.2B rank_ori ascending (lower = more binding, tie-break)
    3. Original index ascending (final tie-break)
    """
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([], dtype=int)
    order = np.lexsort((np.arange(n), v62b_rank, -scores))
    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(n) + 1) / n
    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier


def available_ptypes(auction_month: str) -> list[str]:
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])
