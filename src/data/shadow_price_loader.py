"""Load and aggregate shadow prices for MISO constraints.

This module provides functionality to load day-ahead shadow prices using the
get_da_shadow() API and aggregate them for 3-day periods matching the score data.
"""

from typing import Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


class ShadowPriceLoader:
    """Load and aggregate shadow prices for 3-day periods.

    Each score.parquet file corresponds to a 3-day period starting from
    the outage_date. Shadow prices are aggregated for these periods.
    """

    def __init__(
        self,
        get_da_shadow_func: Callable,
        aggregation_method: Literal["mean", "max", "median", "p95"] = "mean",
        binding_threshold: float = 0.5,
    ):
        """Initialize shadow price loader.

        Parameters
        ----------
        get_da_shadow_func : Callable
            Function to fetch shadow prices: get_da_shadow(st, et, class_type)
        aggregation_method : {'mean', 'max', 'median', 'p95'}, default 'mean'
            Method to aggregate shadow prices over 3-day period
        binding_threshold : float, default 0.5
            Threshold ($/MW) to classify binding events
        """
        self.get_da_shadow = get_da_shadow_func
        self.aggregation_method = aggregation_method
        self.binding_threshold = binding_threshold

    def load_shadow_prices(
        self,
        start_date: str,
        end_date: str,
        class_type: str = "constraint",
    ) -> pd.DataFrame:
        """Load shadow prices for a date range.

        Parameters
        ----------
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        class_type : str, default 'constraint'
            Constraint class type

        Returns
        -------
        pd.DataFrame
            Shadow price data with columns:
            - timestamp
            - constraint_id
            - shadow_price
            - binding_status
        """
        # Call the provided get_da_shadow function
        df = self.get_da_shadow(st=start_date, et=end_date, class_type=class_type)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Add binding status
        if "binding_status" not in df.columns:
            df["binding_status"] = df["shadow_price"] > self.binding_threshold

        return df

    def aggregate_for_period(
        self,
        shadow_df: pd.DataFrame,
        period_start: str,
        period_days: int = 3,
    ) -> pd.DataFrame:
        """Aggregate shadow prices for a specific period.

        Parameters
        ----------
        shadow_df : pd.DataFrame
            Shadow price data with 'timestamp', 'constraint_id', 'shadow_price'
        period_start : str
            Period start date 'YYYY-MM-DD'
        period_days : int, default 3
            Number of days in the period

        Returns
        -------
        pd.DataFrame
            Aggregated shadow prices with columns:
            - constraint_id
            - period_start
            - shadow_price_agg (aggregated value)
            - shadow_price_max (max in period)
            - shadow_price_mean (mean in period)
            - binding_frequency (% of hours binding)
            - n_hours (number of hours in period)
        """
        period_start_dt = pd.to_datetime(period_start)
        period_end_dt = period_start_dt + pd.Timedelta(days=period_days)

        # Filter to period
        period_mask = (shadow_df["timestamp"] >= period_start_dt) & (
            shadow_df["timestamp"] < period_end_dt
        )
        period_data = shadow_df[period_mask].copy()

        if len(period_data) == 0:
            # Return empty dataframe with expected columns
            return pd.DataFrame(
                columns=[
                    "constraint_id",
                    "period_start",
                    "shadow_price_agg",
                    "shadow_price_max",
                    "shadow_price_mean",
                    "shadow_price_median",
                    "shadow_price_p95",
                    "binding_frequency",
                    "n_hours",
                ]
            )

        # Aggregate by constraint
        agg_funcs = {
            "shadow_price": [
                "mean",
                "max",
                "median",
                lambda x: np.percentile(x, 95),
            ],
            "binding_status": "mean",
            "timestamp": "count",
        }

        agg_df = period_data.groupby("constraint_id").agg(agg_funcs).reset_index()

        # Flatten column names
        agg_df.columns = [
            "constraint_id",
            "shadow_price_mean",
            "shadow_price_max",
            "shadow_price_median",
            "shadow_price_p95",
            "binding_frequency",
            "n_hours",
        ]

        # Add primary aggregation based on method
        aggregation_map = {
            "mean": "shadow_price_mean",
            "max": "shadow_price_max",
            "median": "shadow_price_median",
            "p95": "shadow_price_p95",
        }

        agg_df["shadow_price_agg"] = agg_df[
            aggregation_map[self.aggregation_method]
        ].copy()

        # Add period start
        agg_df["period_start"] = period_start

        return agg_df

    def aggregate_for_score_data(
        self,
        score_df: pd.DataFrame,
        shadow_df: pd.DataFrame,
        period_days: int = 3,
    ) -> pd.DataFrame:
        """Aggregate shadow prices matching score data periods.

        Parameters
        ----------
        score_df : pd.DataFrame
            Score data with 'outage_date' and 'constraint_id'
        shadow_df : pd.DataFrame
            Shadow price data
        period_days : int, default 3
            Number of days per period

        Returns
        -------
        pd.DataFrame
            Score data joined with aggregated shadow prices
        """
        # Get unique (constraint_id, outage_date) combinations
        periods = (
            score_df[["constraint_id", "outage_date"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Aggregate shadow prices for each period
        shadow_agg_list = []

        for outage_date in tqdm(
            periods["outage_date"].unique(), desc="Aggregating shadow prices"
        ):
            period_shadow = self.aggregate_for_period(
                shadow_df, outage_date, period_days
            )

            if len(period_shadow) > 0:
                shadow_agg_list.append(period_shadow)

        if not shadow_agg_list:
            raise ValueError("No shadow price data found for any period")

        # Combine all periods
        shadow_agg = pd.concat(shadow_agg_list, ignore_index=True)

        # Join with score data
        result = score_df.merge(
            shadow_agg,
            left_on=["constraint_id", "outage_date"],
            right_on=["constraint_id", "period_start"],
            how="left",
        )

        return result

    def get_binding_statistics(
        self, df: pd.DataFrame, target_col: str = "shadow_price_agg"
    ) -> pd.DataFrame:
        """Get binding statistics by constraint.

        Parameters
        ----------
        df : pd.DataFrame
            Data with shadow prices
        target_col : str, default 'shadow_price_agg'
            Column name for shadow price

        Returns
        -------
        pd.DataFrame
            Statistics with columns:
            - constraint_id
            - n_periods
            - binding_rate
            - mean_shadow_price
            - max_shadow_price
            - mean_when_binding
        """
        stats = []

        for constraint_id, group in df.groupby("constraint_id"):
            binding_mask = group[target_col] > self.binding_threshold

            stat = {
                "constraint_id": constraint_id,
                "n_periods": len(group),
                "binding_rate": binding_mask.mean(),
                "mean_shadow_price": group[target_col].mean(),
                "max_shadow_price": group[target_col].max(),
                "mean_when_binding": (
                    group.loc[binding_mask, target_col].mean()
                    if binding_mask.sum() > 0
                    else 0.0
                ),
            }

            stats.append(stat)

        return pd.DataFrame(stats).sort_values("binding_rate", ascending=False)


def load_and_aggregate_shadow_prices(
    score_df: pd.DataFrame,
    get_da_shadow_func: Callable,
    start_date: str,
    end_date: str,
    aggregation_method: str = "mean",
    binding_threshold: float = 0.5,
    class_type: str = "constraint",
) -> pd.DataFrame:
    """Convenience function to load and aggregate shadow prices.

    Parameters
    ----------
    score_df : pd.DataFrame
        Score data with 'outage_date' and 'constraint_id'
    get_da_shadow_func : Callable
        Function to fetch shadow prices
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    aggregation_method : str, default 'mean'
        Aggregation method: 'mean', 'max', 'median', 'p95'
    binding_threshold : float, default 0.5
        Binding threshold ($/MW)
    class_type : str, default 'constraint'
        Constraint class type

    Returns
    -------
    pd.DataFrame
        Score data with aggregated shadow prices

    Examples
    --------
    >>> from src.data.score_loader import load_scores
    >>> from src.data.shadow_price_loader import load_and_aggregate_shadow_prices
    >>>
    >>> # Load score features
    >>> scores = load_scores('/path/to/density', '2024-01-01', '2024-03-31')
    >>>
    >>> # Load and aggregate shadow prices
    >>> data = load_and_aggregate_shadow_prices(
    ...     scores,
    ...     get_da_shadow,  # Your API function
    ...     '2024-01-01',
    ...     '2024-03-31'
    ... )
    """
    loader = ShadowPriceLoader(
        get_da_shadow_func=get_da_shadow_func,
        aggregation_method=aggregation_method,
        binding_threshold=binding_threshold,
    )

    # Load shadow prices for date range
    print(f"Loading shadow prices from {start_date} to {end_date}...")
    shadow_df = loader.load_shadow_prices(start_date, end_date, class_type)

    print(f"Loaded {len(shadow_df):,} shadow price records")

    # Aggregate for score data periods
    result = loader.aggregate_for_score_data(score_df, shadow_df)

    # Report statistics
    print("\nShadow Price Statistics:")
    print(f"Total records: {len(result):,}")
    print(f"Records with shadow price: {result['shadow_price_agg'].notna().sum():,}")
    print(
        f"Overall binding rate: {(result['shadow_price_agg'] > binding_threshold).mean():.2%}"
    )

    return result
