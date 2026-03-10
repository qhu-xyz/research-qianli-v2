"""V7.0 signal loader with V6.2B fallback.

Use this instead of ConstraintsSignal/ShiftFactorSignal directly.
If V7.0 exists for the requested month/ptype/ctype, load it.
Otherwise, transparently fall back to V6.2B.
"""
from __future__ import annotations

import logging

import pandas as pd
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

logger = logging.getLogger(__name__)

V62B_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"


def load_constraints(
    period_type: str,
    class_type: str,
    auction_month: pd.Timestamp,
    signal_name: str = V70_SIGNAL,
    fallback_signal: str = V62B_SIGNAL,
) -> tuple[pd.DataFrame, str]:
    """Load constraints signal, falling back to V6.2B if V7.0 is missing.

    Returns (dataframe, source_signal_name).

    Only falls back on FileNotFoundError or empty data. Other exceptions
    (schema errors, corruption, etc.) propagate so they are not silently
    masked as a normal fallback.
    """
    try:
        df = ConstraintsSignal("miso", signal_name, period_type, class_type).load_data(auction_month)
        if len(df) > 0:
            return df, signal_name
        logger.info(
            "V7.0 constraints empty for %s/%s/%s, falling back to V6.2B",
            period_type, class_type, auction_month.strftime("%Y-%m"),
        )
    except (FileNotFoundError, OSError):
        logger.info(
            "V7.0 constraints not found for %s/%s/%s, falling back to V6.2B",
            period_type, class_type, auction_month.strftime("%Y-%m"),
        )

    df = ConstraintsSignal("miso", fallback_signal, period_type, class_type).load_data(auction_month)
    return df, fallback_signal


def load_shift_factors(
    period_type: str,
    class_type: str,
    auction_month: pd.Timestamp,
    signal_name: str = V70_SIGNAL,
    fallback_signal: str = V62B_SIGNAL,
) -> tuple[pd.DataFrame, str]:
    """Load shift factors signal, falling back to V6.2B if V7.0 is missing.

    Returns (dataframe, source_signal_name).
    """
    try:
        df = ShiftFactorSignal("miso", signal_name, period_type, class_type).load_data(auction_month)
        if len(df) > 0:
            return df, signal_name
        logger.info(
            "V7.0 shift factors empty for %s/%s/%s, falling back to V6.2B",
            period_type, class_type, auction_month.strftime("%Y-%m"),
        )
    except (FileNotFoundError, OSError):
        logger.info(
            "V7.0 shift factors not found for %s/%s/%s, falling back to V6.2B",
            period_type, class_type, auction_month.strftime("%Y-%m"),
        )

    df = ShiftFactorSignal("miso", fallback_signal, period_type, class_type).load_data(auction_month)
    return df, fallback_signal
