"""
MISO Sell April 2026 (auc2604) - Bid Pricing Engine

Migrated from March V5 (pmodel/notebook/ql/2026-mar/miso/sell_v5/miso_sell.py).

Architecture:
- Single module for f0/f1
- 6-bid structure with bp1=boosted formula for normal trades
- Counter param boost: x1.7 base, x2.5 mtm_coef/base_coef on bp2/bp3
- Period scaling: f0=1.0, f1=0.75
- 2-variable baseline (mtm1 + rev1 only)
- 30-day revenue lookback
- Picked trades: MTM-adjusted, mild counter params for all (prevail+counter)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 999)
pd.set_option("display.max_colwidth", 300)
pd.options.display.float_format = "{:,.2f}".format
np.set_printoptions(suppress=True)

# =============================================================================
# Configuration
# =============================================================================

REVENUE_LOOKBACK_DAYS = 30

PERIOD_CONFIG = {
    "f0": {
        "baseline_mtm_threshold": 1500,
        "baseline_coefficients": {
            "low": {"intercept": 0, "mtm1": 0.85, "rev1": 0.15},
            "high": {"intercept": 0, "mtm1": 0.90, "rev1": 0.10},
        },
        "rev1_cap": 4000,
        "scale": 1.0,
    },
    "f1": {
        "baseline_mtm_threshold": 1500,
        "baseline_coefficients": {
            "low": {"intercept": 0, "mtm1": 0.85, "rev1": 0.15},
            "high": {"intercept": 0, "mtm1": 0.90, "rev1": 0.10},
        },
        "rev1_cap": 4000,
        "scale": 0.75,
    },
}

# =============================================================================
# F0 Counter Params - 6 Bids (bp1=boosted formula, bp2/bp3 BOOSTED)
# =============================================================================

F0_COUNTER_PARAMS_BASE = {
    "aggressive": {
        "bp1": {
            "mtm_cap": 2500,
            "base_cap": 2500,
            "base": 0.69,
            "mtm_coef": 0.35,
            "base_coef": 0.35,
            "vol_rev": 0.001,
            "intercept": 300,
        },
        "bp2": {
            "mtm_cap": 2000,
            "base_cap": 2000,
            "base": 0.51,
            "mtm_coef": 0.25,
            "base_coef": 0.25,
            "vol_rev": 0.001,
            "intercept": 200,
        },
        "bp3": {
            "mtm_cap": 1000,
            "base_cap": 1000,
            "base": 0.425,
            "mtm_coef": 0.125,
            "base_coef": 0.125,
            "vol_rev": 0.001,
            "intercept": 200,
        },
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.15, "mtm_coef": -0.10, "base_coef": -0.10, "vol_rev": 0, "intercept": -50},
        "bp6": {"base": -0.30, "mtm_coef": -0.20, "base_coef": -0.20, "vol_rev": 0, "intercept": -100},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "mild": {
        "bp1": {
            "mtm_cap": 2200,
            "base_cap": 2200,
            "base": 0.597,
            "mtm_coef": 0.308,
            "base_coef": 0.308,
            "vol_rev": 0.00088,
            "intercept": 264,
        },
        "bp2": {
            "mtm_cap": 1760,
            "base_cap": 1760,
            "base": 0.442,
            "mtm_coef": 0.22,
            "base_coef": 0.22,
            "vol_rev": 0.00088,
            "intercept": 176,
        },
        "bp3": {
            "mtm_cap": 880,
            "base_cap": 880,
            "base": 0.374,
            "mtm_coef": 0.11,
            "base_coef": 0.11,
            "vol_rev": 0.00088,
            "intercept": 176,
        },
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.175, "mtm_coef": -0.117, "base_coef": -0.117, "vol_rev": 0, "intercept": -58},
        "bp6": {"base": -0.35, "mtm_coef": -0.233, "base_coef": -0.233, "vol_rev": 0, "intercept": -117},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "neutral": {
        "bp1": {
            "mtm_cap": 1700,
            "base_cap": 1700,
            "base": 0.459,
            "mtm_coef": 0.238,
            "base_coef": 0.238,
            "vol_rev": 0.00068,
            "intercept": 204,
        },
        "bp2": {
            "mtm_cap": 1360,
            "base_cap": 1360,
            "base": 0.34,
            "mtm_coef": 0.17,
            "base_coef": 0.17,
            "vol_rev": 0.00068,
            "intercept": 136,
        },
        "bp3": {
            "mtm_cap": 680,
            "base_cap": 680,
            "base": 0.289,
            "mtm_coef": 0.085,
            "base_coef": 0.085,
            "vol_rev": 0.00068,
            "intercept": 136,
        },
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.20, "mtm_coef": -0.133, "base_coef": -0.133, "vol_rev": 0, "intercept": -67},
        "bp6": {"base": -0.40, "mtm_coef": -0.267, "base_coef": -0.267, "vol_rev": 0, "intercept": -133},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "conservative": {
        "bp1": {
            "mtm_cap": 1250,
            "base_cap": 1250,
            "base": 0.344,
            "mtm_coef": 0.175,
            "base_coef": 0.175,
            "vol_rev": 0.0005,
            "intercept": 150,
        },
        "bp2": {
            "mtm_cap": 1000,
            "base_cap": 1000,
            "base": 0.255,
            "mtm_coef": 0.125,
            "base_coef": 0.125,
            "vol_rev": 0.0005,
            "intercept": 100,
        },
        "bp3": {
            "mtm_cap": 500,
            "base_cap": 500,
            "base": 0.2125,
            "mtm_coef": 0.0625,
            "base_coef": 0.0625,
            "vol_rev": 0.0005,
            "intercept": 100,
        },
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.225, "mtm_coef": -0.15, "base_coef": -0.15, "vol_rev": 0, "intercept": -75},
        "bp6": {"base": -0.45, "mtm_coef": -0.30, "base_coef": -0.30, "vol_rev": 0, "intercept": -150},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "plain": {
        "bp1": {
            "mtm_cap": 800,
            "base_cap": 800,
            "base": 0.23,
            "mtm_coef": 0.112,
            "base_coef": 0.112,
            "vol_rev": 0.00032,
            "intercept": 96,
        },
        "bp2": {
            "mtm_cap": 640,
            "base_cap": 640,
            "base": 0.17,
            "mtm_coef": 0.08,
            "base_coef": 0.08,
            "vol_rev": 0.00032,
            "intercept": 64,
        },
        "bp3": {
            "mtm_cap": 320,
            "base_cap": 320,
            "base": 0.136,
            "mtm_coef": 0.04,
            "base_coef": 0.04,
            "vol_rev": 0.00032,
            "intercept": 64,
        },
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.25, "mtm_coef": -0.167, "base_coef": -0.167, "vol_rev": 0, "intercept": -83},
        "bp6": {"base": -0.50, "mtm_coef": -0.333, "base_coef": -0.333, "vol_rev": 0, "intercept": -167},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
}

# =============================================================================
# F0 Prevail Params - 6 Bids
# =============================================================================

F0_PREVAIL_PARAMS_BASE = {
    "aggressive": {
        "bp1": {"base": 0.27, "mtm_coef": 0.07, "base_coef": 0.07, "vol_rev": 0.0005, "intercept": 75},
        "bp2": {"base": 0.20, "mtm_coef": 0.05, "base_coef": 0.05, "vol_rev": 0.0005, "intercept": 50},
        "bp3": {"base": 0.10, "mtm_coef": 0.025, "base_coef": 0.025, "vol_rev": 0.0003, "intercept": 25},
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.08, "mtm_coef": -0.05, "base_coef": -0.05, "vol_rev": 0, "intercept": -25},
        "bp6": {"base": -0.15, "mtm_coef": -0.10, "base_coef": -0.10, "vol_rev": 0, "intercept": -50},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "mild": {
        "bp1": {"base": 0.23, "mtm_coef": 0.056, "base_coef": 0.056, "vol_rev": 0.0004, "intercept": 60},
        "bp2": {"base": 0.17, "mtm_coef": 0.04, "base_coef": 0.04, "vol_rev": 0.0004, "intercept": 40},
        "bp3": {"base": 0.085, "mtm_coef": 0.02, "base_coef": 0.02, "vol_rev": 0.0002, "intercept": 20},
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.10, "mtm_coef": -0.06, "base_coef": -0.06, "vol_rev": 0, "intercept": -30},
        "bp6": {"base": -0.18, "mtm_coef": -0.12, "base_coef": -0.12, "vol_rev": 0, "intercept": -60},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "neutral": {
        "bp1": {"base": 0.162, "mtm_coef": 0.042, "base_coef": 0.042, "vol_rev": 0.0003, "intercept": 45},
        "bp2": {"base": 0.12, "mtm_coef": 0.03, "base_coef": 0.03, "vol_rev": 0.0003, "intercept": 30},
        "bp3": {"base": 0.06, "mtm_coef": 0.015, "base_coef": 0.015, "vol_rev": 0.00015, "intercept": 15},
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.12, "mtm_coef": -0.07, "base_coef": -0.07, "vol_rev": 0, "intercept": -35},
        "bp6": {"base": -0.21, "mtm_coef": -0.14, "base_coef": -0.14, "vol_rev": 0, "intercept": -70},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "conservative": {
        "bp1": {"base": 0.101, "mtm_coef": 0.028, "base_coef": 0.028, "vol_rev": 0.0002, "intercept": 30},
        "bp2": {"base": 0.075, "mtm_coef": 0.02, "base_coef": 0.02, "vol_rev": 0.0002, "intercept": 20},
        "bp3": {"base": 0.0375, "mtm_coef": 0.01, "base_coef": 0.01, "vol_rev": 0.0001, "intercept": 10},
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.14, "mtm_coef": -0.08, "base_coef": -0.08, "vol_rev": 0, "intercept": -40},
        "bp6": {"base": -0.24, "mtm_coef": -0.16, "base_coef": -0.16, "vol_rev": 0, "intercept": -80},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
    "plain": {
        "bp1": {"base": 0.041, "mtm_coef": 0.014, "base_coef": 0.014, "vol_rev": 0.0001, "intercept": 15},
        "bp2": {"base": 0.03, "mtm_coef": 0.01, "base_coef": 0.01, "vol_rev": 0.0001, "intercept": 10},
        "bp3": {"base": 0.015, "mtm_coef": 0.005, "base_coef": 0.005, "vol_rev": 0.00005, "intercept": 5},
        "bp4": {"base": 0, "mtm_coef": 0, "base_coef": 0, "vol_rev": 0, "intercept": 0},
        "bp5": {"base": -0.16, "mtm_coef": -0.09, "base_coef": -0.09, "vol_rev": 0, "intercept": -45},
        "bp6": {"base": -0.27, "mtm_coef": -0.18, "base_coef": -0.18, "vol_rev": 0, "intercept": -90},
        "volumes": [0.0, 0.1, 0.35, 0.60, 0.85, 1.0],
    },
}

# =============================================================================
# Parameter Scaling
# =============================================================================


def _scale_params(params: dict, scale: float) -> dict:
    """Scale parameters by a factor. Volumes and fixed_price are not scaled."""
    if scale == 1.0:
        return params

    scaled = {}
    for key, value in params.items():
        if key == "volumes":
            scaled[key] = value
        elif isinstance(value, dict):
            scaled[key] = {}
            for k, v in value.items():
                if k == "fixed_price":
                    scaled[key][k] = v
                elif k in ["mtm_cap", "base_cap", "intercept", "base", "mtm_coef", "base_coef", "vol_rev"]:
                    scaled[key][k] = v * scale
                else:
                    scaled[key][k] = v
        else:
            scaled[key] = value
    return scaled


def get_counter_params(period_type: str) -> dict:
    """Get counter params scaled for the given period type."""
    scale = PERIOD_CONFIG[period_type]["scale"]
    return {strategy: _scale_params(params, scale) for strategy, params in F0_COUNTER_PARAMS_BASE.items()}


def get_prevail_params(period_type: str) -> dict:
    """Get prevail params scaled for the given period type."""
    scale = PERIOD_CONFIG[period_type]["scale"]
    return {strategy: _scale_params(params, scale) for strategy, params in F0_PREVAIL_PARAMS_BASE.items()}


# =============================================================================
# Revenue Attachment
# =============================================================================


def _attach_historical_revenue(
    df: pd.DataFrame,
    class_type: str,
    lookback_days: int = REVENUE_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Attach historical revenue using MisoApTools with 30-day lookback."""
    from pbase.analysis.tools.all_positions import MisoApTools as ApTools

    aptools = ApTools()
    res = []

    for month in df["auction_date"].unique():
        one_month_trades = df[df["auction_date"] == month].copy()

        _, da_df, _ = aptools.tools.generate_monthly_stats_fast(
            path_df=one_month_trades.drop_duplicates(subset=["path"]),
            class_type=class_type,
            auction_month=month,
            month_1_last_n_days=lookback_days,
            dt_column_name=False,
            recent_n_months=14,
            get_da=True,
            get_mcp=False,
        )

        da_df = da_df.drop(columns=["1(rev_leak)", "0(rev)"], errors="ignore")
        one_month_trades = one_month_trades.merge(da_df, left_on="path", right_index=True, how="left")
        res.append(one_month_trades)

    df = pd.concat(res)
    return df


# =============================================================================
# Baseline Computation
# =============================================================================


def _compute_baseline(df: pd.DataFrame, period_type: str) -> pd.Series:
    """Compute 2-regime baseline using mtm1 + rev1 only (no mtm2/rev2)."""
    config = PERIOD_CONFIG[period_type]
    threshold = config["baseline_mtm_threshold"]
    coef = config["baseline_coefficients"]

    mtm1 = df["mtm_1st_mean"]
    rev1 = df["1(rev)"]

    rev1_cap = config.get("rev1_cap", None)
    if rev1_cap is not None:
        rev1 = rev1.clip(-rev1_cap, rev1_cap)

    low_mask = mtm1.abs() <= threshold
    baseline = pd.Series(index=df.index, dtype=float)

    c = coef["low"]
    baseline[low_mask] = c["intercept"] + c["mtm1"] * mtm1[low_mask] + c["rev1"] * rev1[low_mask]

    c = coef["high"]
    baseline[~low_mask] = c["intercept"] + c["mtm1"] * mtm1[~low_mask] + c["rev1"] * rev1[~low_mask]

    return baseline


# =============================================================================
# Bid Price Application
# =============================================================================


def _apply_counter_bid_prices(df: pd.DataFrame, mask: pd.Series, params: dict) -> pd.DataFrame:
    """Apply 6-bid counter flow prices. Caps on bp1/bp2/bp3."""
    for i, bp_key in enumerate(["bp1", "bp2", "bp3", "bp4", "bp5", "bp6"], start=1):
        bp = params[bp_key]

        if "fixed_price" in bp:
            df.loc[mask, f"bid_price_{i}"] = bp["fixed_price"]
        elif i <= 3:  # bp1, bp2, bp3 have caps
            mtm_cap = bp.get("mtm_cap", float("inf"))
            base_cap = bp.get("base_cap", float("inf"))
            mtm_term = bp["mtm_coef"] * np.minimum(np.abs(df.loc[mask, "mtm_1st_mean"]), mtm_cap)
            base_term = bp["base_coef"] * np.minimum(np.abs(df.loc[mask, "baseline"]), base_cap)
            df.loc[mask, f"bid_price_{i}"] = (
                df.loc[mask, "baseline"]
                - bp["base"] * np.abs(df.loc[mask, "baseline"])
                - mtm_term
                - base_term
                - bp["vol_rev"] * df.loc[mask, "vol_clipped"] * df.loc[mask, "abs_1rev_clipped"]
                - bp["intercept"]
            )
        else:  # bp4, bp5, bp6 no caps
            mtm_term = bp["mtm_coef"] * np.abs(df.loc[mask, "mtm_1st_mean"])
            base_term = bp["base_coef"] * np.abs(df.loc[mask, "baseline"])
            df.loc[mask, f"bid_price_{i}"] = (
                df.loc[mask, "baseline"]
                - bp["base"] * np.abs(df.loc[mask, "baseline"])
                - mtm_term
                - base_term
                - bp["vol_rev"] * df.loc[mask, "vol_clipped"] * df.loc[mask, "abs_1rev_clipped"]
                - bp["intercept"]
            )
    return df


def _apply_prevail_bid_prices(df: pd.DataFrame, mask: pd.Series, params: dict) -> pd.DataFrame:
    """Apply 6-bid prevail flow prices. No caps on any bid point."""
    for i, bp_key in enumerate(["bp1", "bp2", "bp3", "bp4", "bp5", "bp6"], start=1):
        bp = params[bp_key]

        if "fixed_price" in bp:
            df.loc[mask, f"bid_price_{i}"] = bp["fixed_price"]
        else:
            df.loc[mask, f"bid_price_{i}"] = (
                df.loc[mask, "baseline"]
                - bp["base"] * np.abs(df.loc[mask, "baseline"])
                - bp["mtm_coef"] * np.abs(df.loc[mask, "mtm_1st_mean"])
                - bp["base_coef"] * np.abs(df.loc[mask, "baseline"])
                - bp["vol_rev"] * df.loc[mask, "vol_clipped"] * df.loc[mask, "abs_1rev_clipped"]
                - bp["intercept"]
            )
    return df


def _assign_volumes(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Assign cumulative volumes based on strategy (6 volumes)."""
    vols = params["volumes"]
    for i, vol_pct in enumerate(vols, start=1):
        df[f"bid_volume_{i}"] = df["bid_volume"] * vol_pct
    return df


# =============================================================================
# Validation
# =============================================================================


def _validate_sell_monotonicity(
    df: pd.DataFrame, strategy_name: str, verbose: bool = True, num_bids: int = 6
) -> Tuple[bool, bool]:
    """Validate selling price and volume monotonicity (bp1 < bp2 < ... < bpN)."""
    price_mono_ok = True
    for i in range(1, num_bids):
        if not (df[f"bid_price_{i}"] <= df[f"bid_price_{i + 1}"]).all():
            price_mono_ok = False
            break

    volume_mono_ok = True
    for i in range(1, num_bids):
        if not (df[f"bid_volume_{i}"] <= df[f"bid_volume_{i + 1}"]).all():
            volume_mono_ok = False
            break

    if verbose:
        if not price_mono_ok:
            print(f"WARNING: Price monotonicity violated for {strategy_name}")
            violation_mask = pd.Series(False, index=df.index)
            for i in range(1, num_bids):
                violation_mask |= df[f"bid_price_{i}"] > df[f"bid_price_{i + 1}"]
            violations = df[violation_mask]
            if len(violations) > 0:
                print(f"  Violations: {len(violations)} rows")
        if not volume_mono_ok:
            print(f"WARNING: Volume monotonicity violated for {strategy_name}")

    return price_mono_ok, volume_mono_ok


# =============================================================================
# DataFrame Preparation
# =============================================================================


def _prepare_df_for_pricing(df: pd.DataFrame, class_type: str, period_type: str) -> pd.DataFrame:
    """Prepare dataframe with baseline and required columns for pricing."""
    df = df.copy()

    if "1(rev)" not in df.columns:
        df = _attach_historical_revenue(df, class_type)

    # Cap 1(rev) at -13000, then scale from (-13000, -700) to (-7000, -700)
    if "1(rev)" in df.columns:
        df.loc[df["1(rev)"] < -13000, "1(rev)"] = -13000
        mask = df["1(rev)"] < -700
        df.loc[mask, "1(rev)"] = -700 + (df.loc[mask, "1(rev)"] + 700) * (6300 / 12300)

    df["baseline"] = _compute_baseline(df, period_type)
    df = df[df.mtm_1st_mean.notna()].copy()

    df["trade_type"] = "sell"
    df["abs_1rev_clipped"] = np.abs(df["1(rev)"]).clip(upper=5000)

    if "bid_volume" not in df.columns:
        df["bid_volume"] = df.filter(like="bid_vol").max(axis=1)
    df["vol_clipped"] = df["bid_volume"].clip(upper=40)

    for i in range(1, 11):
        df[f"bid_price_{i}"] = np.nan
        df[f"bid_volume_{i}"] = np.nan

    return df


# =============================================================================
# Exposure Split (Picking)
# =============================================================================


def calculate_exposure_split(
    trades: pd.DataFrame,
    period_type: str,
    class_type: str,
    constraint_dict: Dict[str, float],
    sf_threshold: float = 0.07,
    sf_volume_threshold: float = 0.2,
    signal_name: str = "TEST.Signal.MISO.AUC_SPICE_BEFORE_CLEAR_V1.1.R1",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split trades into normal and picked groups based on shift factor exposure.

    Picking rules (positive SF only):
      - Rule 1: sf > sf_threshold for any constraint
      - Rule 2: sf * volume > sf_volume_threshold for any constraint

    For picked trades, MTM is adjusted:
      new_mtm = old_mtm - SF * (original_SP - (-target_SP))

    Returns:
        trades_normal: Normal trades (unmodified)
        trades_picked: Picked trades (MTM adjusted)
    """
    from pbase.analysis.tools.all_positions import MisoApTools
    from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

    assert constraint_dict, "constraint_dict must not be empty"

    aptools = MisoApTools()
    auction_date = trades.auction_date.iloc[0]
    ct = trades.class_type.iloc[0]

    signal = ConstraintsSignal(
        rto="MISO",
        signal_name=signal_name,
        period_type=period_type,
        class_type=ct,
    ).load_data(auction_month=auction_date)

    sf = ShiftFactorSignal(
        rto="MISO",
        signal_name=signal_name,
        period_type=period_type,
        class_type=ct,
    ).load_data(auction_month=auction_date)

    _, ppp, _, _ = aptools.get_signal_expo2(
        trades=trades,
        signal_sp=signal,
        signal_sf=sf,
        fill_value=0,
        zero_shadow_price="raise",
    )

    target_sp = pd.Series(constraint_dict)
    target_sp.index = target_sp.index.str.replace(" ", "")

    sf_pivot = ppp.pivot(index="path", columns="constraint", values="path_sf")
    sf_pivot.columns = sf_pivot.columns.str.replace(" ", "")

    available_constraints = [c for c in target_sp.index if c in sf_pivot.columns]
    assert len(available_constraints) > 0, (
        f"No matching constraints found. Available: {[c for c in sf_pivot.columns if 'ALDRICH' in c]}"
    )

    if len(available_constraints) != len(target_sp):
        missing = set(target_sp.index) - set(available_constraints)
        print(f"WARNING: Constraints not found in signal: {missing}")

    x = sf_pivot[available_constraints]
    path_volumes = trades.set_index("path")["bid_volume"].reindex(x.index).fillna(0)

    # Picking: positive SF only
    high_sf = x.gt(sf_threshold).any(axis=1)
    sf_vol = x.mul(path_volumes, axis=0)
    high_vol = sf_vol.gt(sf_volume_threshold).any(axis=1)
    picked_mask = high_sf | high_vol

    picked_paths = x[picked_mask].index.tolist()

    # Original shadow prices (negative, signal convention)
    original_sp = signal["shadow_price"].copy()
    original_sp.index = original_sp.index.str.replace(" ", "")
    original_sp_subset = original_sp.loc[available_constraints]

    # Build trades_picked with MTM adjustment
    trades_picked = trades[trades.path.isin(picked_paths)].copy()

    if len(trades_picked) > 0:
        new_sp = -target_sp.loc[available_constraints]
        sp_delta = original_sp_subset - new_sp
        x_picked = x.loc[picked_mask]
        mtm_adjust = (x_picked * sp_delta).sum(axis=1).to_frame(name="mtm_adjust")

        trades_picked = trades_picked.merge(mtm_adjust, left_on="path", right_index=True, how="left")
        trades_picked["mtm_1st_mean_original"] = trades_picked["mtm_1st_mean"].copy()
        trades_picked["mtm_1st_mean"] = trades_picked["mtm_1st_mean"] - trades_picked["mtm_adjust"]

        if verbose:
            print(f"Constraint: {available_constraints}")
            print(f"Picked: {picked_mask.sum()}, Normal: {(~picked_mask).sum()}")
            print(f"Original SP: {original_sp_subset.values}")
            print(f"MTM adjustment: min={mtm_adjust['mtm_adjust'].min():.2f}, "
                  f"max={mtm_adjust['mtm_adjust'].max():.2f}, mean={mtm_adjust['mtm_adjust'].mean():.2f}")
            print(f"Old MTM: mean={trades_picked['mtm_1st_mean_original'].mean():.2f}")
            print(f"New MTM: mean={trades_picked['mtm_1st_mean'].mean():.2f}")
            assert (trades_picked["mtm_1st_mean"] <= trades_picked["mtm_1st_mean_original"]).all(), (
                "MTM adjustment did not lower MTM for all picked trades!"
            )
            print("MTM goes DOWN for all picked trades ✓")

    trades_normal = trades[~trades.path.isin(picked_paths)].copy()

    return trades_normal, trades_picked


def apply_picked_pricing(
    df: pd.DataFrame,
    class_type: str,
    period_type: str,
    picked_strategy: str = "mild",
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply a single counter strategy to all picked trades (both prevail and counter)."""
    df = _prepare_df_for_pricing(df, class_type, period_type)
    params = get_counter_params(period_type)[picked_strategy]

    all_mask = pd.Series(True, index=df.index)
    df = _apply_counter_bid_prices(df, all_mask, params)
    df = _assign_volumes(df, params)

    df["flow_type"] = np.where(df["baseline"] > 100, "prevail", "counter")
    df["strategy"] = picked_strategy

    _validate_sell_monotonicity(df, f"picked_{picked_strategy}_{period_type}", verbose=verbose)

    return df


# =============================================================================
# Pricing Functions
# =============================================================================


def apply_pricing_all_strategies(
    df: pd.DataFrame,
    class_type: str,
    period_type: str,
    verbose: bool = True,
) -> List[Tuple[str, str, pd.DataFrame]]:
    """Apply all 5 pricing strategies to trades."""
    df_prepared = _prepare_df_for_pricing(df, class_type, period_type)
    counter_params = get_counter_params(period_type)
    prevail_params = get_prevail_params(period_type)

    results = []
    strategies = list(counter_params.keys())

    for strategy in strategies:
        df_copy = df_prepared.copy()

        prevail_mask = df_copy["baseline"] > 100
        counter_mask = ~prevail_mask

        df_copy = _apply_prevail_bid_prices(df_copy, prevail_mask, prevail_params[strategy])
        df_copy = _apply_counter_bid_prices(df_copy, counter_mask, counter_params[strategy])

        df_copy.loc[prevail_mask] = _assign_volumes(df_copy.loc[prevail_mask].copy(), prevail_params[strategy])
        df_copy.loc[counter_mask] = _assign_volumes(df_copy.loc[counter_mask].copy(), counter_params[strategy])

        _validate_sell_monotonicity(df_copy, f"{strategy}_{period_type}", verbose=verbose)

        prevail_df = df_copy[df_copy["baseline"] > 100].copy()
        counter_df = df_copy[df_copy["baseline"] <= 100].copy()

        prevail_df["flow_type"] = "prevail"
        prevail_df["strategy"] = strategy
        counter_df["flow_type"] = "counter"
        counter_df["strategy"] = strategy

        results.append((strategy, "prevail", prevail_df))
        results.append((strategy, "counter", counter_df))

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def generate_sell_files(
    data_path: str,
    class_type: str,
    period_type: str,
    output_dir: str,
    auction: str = "auc2604",
    version: str = "v2",
    auction_round: int = 1,
    constraint_dict: Optional[Dict[str, float]] = None,
    picked_strategy: str = "mild",
    verbose: bool = True,
) -> Dict:
    """Generate sell pricing files with optional constraint-based picking.

    If constraint_dict is provided, trades are split into normal and picked.
    Picked trades get MTM-adjusted counter pricing (1 file per slice).
    Normal trades get all 5 strategies × 2 flows (10 files per slice).
    """
    if period_type not in PERIOD_CONFIG:
        raise ValueError(f"Unsupported period type: {period_type}. Must be one of: {list(PERIOD_CONFIG.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    trades = pd.read_parquet(data_path)
    original_count = len(trades)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"MISO Sell Apr 2026 {version} - {period_type} {class_type}")
        print(f"Input: {data_path}")
        print(f"Total trades: {original_count}")
        print(f"Scale factor: {PERIOD_CONFIG[period_type]['scale']}")
        print(f"{'=' * 60}")

    files_generated = []

    # Split into normal and picked if constraints provided
    picked_count = 0
    if constraint_dict:
        trades_normal, trades_picked = calculate_exposure_split(
            trades=trades,
            period_type=period_type,
            class_type=class_type,
            constraint_dict=constraint_dict,
            verbose=verbose,
        )
        picked_count = len(trades_picked)

        # Process picked trades
        if picked_count > 0:
            if verbose:
                print(f"\nProcessing {picked_count} picked trades ({picked_strategy} counter, adjusted MTM)...")

            picked_priced = apply_picked_pricing(
                trades_picked, class_type, period_type, picked_strategy=picked_strategy, verbose=verbose,
            )

            ec_path = (f"{output_dir}/trades_to_sell_miso_{auction}{version}"
                       f"_exposure_controlled_{period_type}_{class_type}_{auction_round}.parquet")
            picked_priced.to_parquet(ec_path)
            files_generated.append(("mild", "exposure_controlled", ec_path))

            if verbose:
                print(f"  Saved: {Path(ec_path).name} ({picked_count} trades)")
    else:
        trades_normal = trades

    # Process normal trades
    if verbose:
        print(f"\nProcessing {len(trades_normal)} normal trades (all strategies)...")

    normal_results = apply_pricing_all_strategies(trades_normal, class_type, period_type, verbose=verbose)

    for strategy, flow_type, priced_df in normal_results:
        file_path = (f"{output_dir}/trades_to_sell_miso_{auction}{version}"
                     f"_priced_{flow_type}_{strategy}_{period_type}_{class_type}_{auction_round}.parquet")
        priced_df.to_parquet(file_path)
        files_generated.append((strategy, flow_type, file_path))

        if verbose:
            print(f"  Saved: {Path(file_path).name} ({len(priced_df)} trades)")

    return {
        "original_count": original_count,
        "normal_count": len(trades_normal),
        "picked_count": picked_count,
        "files_generated": files_generated,
    }
