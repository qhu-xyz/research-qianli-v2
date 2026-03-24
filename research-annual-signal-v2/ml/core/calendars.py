"""Market-agnostic round cutoff calendar.

Source: pbase MisoApTools / PjmApTools hardcoded dates.
These are the round CLOSE dates. History must use data strictly BEFORE this date.
"""
from __future__ import annotations

from datetime import date, timedelta

# ─── Round close dates (day-of-month templates) ──────────────────────

MISO_ANNUAL_ROUND_CLOSE: dict[int, date] = {
    1: date(1, 4, 8),    # April 8
    2: date(1, 4, 22),   # April 22
    3: date(1, 5, 5),    # May 5
}

PJM_ANNUAL_ROUND_CLOSE: dict[int, date] = {
    1: date(1, 4, 4),    # April 4
    2: date(1, 4, 11),   # April 11
    3: date(1, 4, 18),   # April 18
    4: date(1, 4, 25),   # April 25
}

VALID_MISO_ROUNDS: set[int] = {1, 2, 3}
VALID_PJM_ROUNDS: set[int] = {1, 2, 3, 4}


def get_round_close_date(planning_year: str, market_round: int, rto: str = "miso") -> date:
    """Return the round close date for a given PY and round.

    The close date is the day the auction bidding closes. No history on or
    after this date may be included in features (v1: day-level granularity).
    """
    py_year = int(planning_year[:4])
    if rto == "miso":
        if market_round not in VALID_MISO_ROUNDS:
            raise ValueError(f"Invalid MISO round: {market_round}. Valid: {sorted(VALID_MISO_ROUNDS)}")
        template = MISO_ANNUAL_ROUND_CLOSE[market_round]
    elif rto == "pjm":
        if market_round not in VALID_PJM_ROUNDS:
            raise ValueError(f"Invalid PJM round: {market_round}. Valid: {sorted(VALID_PJM_ROUNDS)}")
        template = PJM_ANNUAL_ROUND_CLOSE[market_round]
    else:
        raise ValueError(f"Unsupported RTO: {rto}")
    return date(py_year, template.month, template.day)


def get_history_cutoff_date(planning_year: str, market_round: int, rto: str = "miso") -> date:
    """Return the last date (inclusive) whose history may be used for features.

    v1 rule: exclude the entire close date. So cutoff = close_date - 1 day.
    """
    close = get_round_close_date(planning_year, market_round, rto)
    return close - timedelta(days=1)


def get_history_cutoff_month(planning_year: str, market_round: int, rto: str = "miso") -> str:
    """Return the last FULL month whose history is safe for all rounds.

    This is the month-level approximation used when daily cache is not available.
    For MISO: all rounds close in April or May, so the last full month is always March.
    This function exists for backward compatibility with the month-level pipeline.
    """
    cutoff = get_history_cutoff_date(planning_year, market_round, rto)
    if cutoff.day < 28:
        if cutoff.month == 1:
            return f"{cutoff.year - 1}-12"
        return f"{cutoff.year}-{cutoff.month - 1:02d}"
    if cutoff.month == 1:
        return f"{cutoff.year - 1}-12"
    return f"{cutoff.year}-{cutoff.month - 1:02d}"
