"""V7.0 signal loader with V6.2B fallback.

Use this instead of ConstraintsSignal/ShiftFactorSignal directly.
If V7.0 exists for the requested month/ptype/ctype, load it.
Otherwise, transparently fall back to V6.2B.
"""
from __future__ import annotations

import pandas as pd
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

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
    """
    try:
        df = ConstraintsSignal("miso", signal_name, period_type, class_type).load_data(auction_month)
        if len(df) > 0:
            return df, signal_name
    except Exception:
        pass

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
    except Exception:
        pass

    df = ShiftFactorSignal("miso", fallback_signal, period_type, class_type).load_data(auction_month)
    return df, fallback_signal
