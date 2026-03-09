"""Rank/tier computation and signal assembly helpers for V7.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_STAGE5 = Path(__file__).resolve().parent.parent.parent / "research-stage5-tier"
if str(_STAGE5) not in sys.path:
    sys.path.insert(0, str(_STAGE5))

from ml.config import MISO_AUCTION_SCHEDULE
from ml.v62b_formula import dense_rank_normalized


def compute_rank_tier(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw ML scores to rank (0-1] and tier (0-4).

    Higher ML score = more binding = lower rank value.
    Rank: dense_rank(-scores) / K.  Tier: ceil(rank * 5) - 1.
    """
    rank = dense_rank_normalized(-scores)
    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier


def available_ptypes(auction_month: str) -> list[str]:
    """Return period types available for this auction month."""
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return MISO_AUCTION_SCHEDULE.get(month_num, ["f0"])
