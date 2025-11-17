"""Temporal feature engineering for time series data.

This module provides functions to create temporal features from timestamps,
including cyclical encodings and categorical time-based features.
"""

import numpy as np
import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    use_cyclical: bool = True,
) -> pd.DataFrame:
    """Add temporal features to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str, default 'timestamp'
        Name of timestamp column
    use_cyclical : bool, default True
        Use cyclical encoding (sin/cos) for periodic features

    Returns
    -------
    pd.DataFrame
        Dataframe with temporal features added

    Notes
    -----
    Added features:
    - year, month, day, day_of_week, day_of_year
    - hour (if timestamp includes time)
    - Cyclical encodings (if use_cyclical=True):
      - hour_sin, hour_cos
      - dow_sin, dow_cos (day of week)
      - month_sin, month_cos
      - doy_sin, doy_cos (day of year)
    - Categorical features:
      - is_weekend
      - is_peak (MISO on-peak hours)
      - is_super_peak (summer peak hours)
      - season (winter, spring, summer, fall)
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Extract basic components
    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    df["day_of_week"] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    df["day_of_year"] = df[timestamp_col].dt.dayofyear

    # Extract hour if timestamp includes time
    if df[timestamp_col].dt.hour.max() > 0:
        df["hour"] = df[timestamp_col].dt.hour
    else:
        # For date-only timestamps, use a default hour (12 = midday)
        df["hour"] = 12

    # Cyclical encoding (preserves periodicity)
    if use_cyclical:
        # Hour (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week (7-day cycle)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Month (12-month cycle)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Day of year (365-day cycle, captures seasonality)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Categorical features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # MISO on-peak hours: HE 7-22 (hours 6-21 in 0-indexed), weekdays only
    df["is_peak"] = (
        (df["hour"] >= 6) & (df["hour"] <= 21) & (df["day_of_week"] < 5)
    ).astype(int)

    # Super peak: Summer (June-August) peak hours HE 14-19 (hours 13-18)
    df["is_super_peak"] = (
        (df["month"].isin([6, 7, 8]))
        & (df["hour"] >= 13)
        & (df["hour"] <= 18)
        & (df["day_of_week"] < 5)
    ).astype(int)

    # Season
    df["season"] = df["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "fall",
            10: "fall",
            11: "fall",
        }
    )

    # One-hot encode season
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)

    # Winter peak hours (morning and evening heating load)
    df["is_winter_morning"] = (
        (df["month"].isin([12, 1, 2]))
        & (df["hour"].isin([6, 7, 8]))
        & (df["day_of_week"] < 5)
    ).astype(int)

    df["is_winter_evening"] = (
        (df["month"].isin([12, 1, 2]))
        & (df["hour"].isin([17, 18, 19]))
        & (df["day_of_week"] < 5)
    ).astype(int)

    # Spring wind peak (high wind generation, low load)
    df["is_spring_wind"] = (
        (df["month"].isin([3, 4, 5])) & (df["hour"] >= 10) & (df["hour"] <= 14)
    ).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "constraint_id",
    lags: list[int] = None,
) -> pd.DataFrame:
    """Add lag features for time series.

    WARNING: Only use for training on historical data where future values
    are not needed. For real-time prediction, lags must come from realized values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must be sorted by time within groups)
    value_col : str
        Column to create lags for
    group_col : str, default 'constraint_id'
        Column to group by (e.g., constraint_id)
    lags : list[int], optional
        Lag periods. If None, uses [1, 3, 7, 14, 30] for daily data

    Returns
    -------
    pd.DataFrame
        Dataframe with lag features added
    """
    if lags is None:
        # Default lags for 3-day periods (assuming data every 3 days)
        lags = [1, 2, 4, 8]  # 3, 6, 12, 24 days ago

    df = df.copy()

    for lag in lags:
        col_name = f"{value_col}_lag{lag}"
        df[col_name] = df.groupby(group_col)[value_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "constraint_id",
    windows: list[int] = None,
) -> pd.DataFrame:
    """Add rolling window features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must be sorted by time within groups)
    value_col : str
        Column to aggregate
    group_col : str, default 'constraint_id'
        Column to group by
    windows : list[int], optional
        Window sizes. If None, uses [4, 8, 16] for 3-day period data

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling features added
    """
    if windows is None:
        # Default windows for 3-day periods
        windows = [4, 8, 16]  # ~12, 24, 48 days

    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f"{value_col}_rolling_mean_{window}"] = (
            df.groupby(group_col)[value_col]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling max
        df[f"{value_col}_rolling_max_{window}"] = (
            df.groupby(group_col)[value_col]
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

        # Rolling std
        df[f"{value_col}_rolling_std_{window}"] = (
            df.groupby(group_col)[value_col]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df
