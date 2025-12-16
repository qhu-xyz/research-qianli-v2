"""
Data loading functions for shadow price prediction.
"""

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PredictionConfig

# Setup persistent cache for data loading
# memory = joblib.Memory(location='./.gemini_cache', verbose=0)


def fetch_constraints(cons_path):
    """Cached wrapper for loading constraints parquet."""
    return pd.read_parquet(cons_path / "constraints.parquet")


class BaseDataLoader(ABC):
    """Abstract base class for loading training and test data."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        # Dynamically load AP Tools
        module_name, class_name = self.config.iso.ap_tools_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ap_tools_class = getattr(module, class_name)
        self.aptools = ap_tools_class()

    @abstractmethod
    def fetch_da_shadow_wrapper(self, st, et, class_type):
        """Wrapper for fetching DA shadow prices with RTO-specific column renaming."""
        pass

    @abstractmethod
    def map_constraints_to_branches(
        self, data: pd.Series | pd.DataFrame, spice_map: pd.DataFrame, fill_map: pd.Series | None = None
    ) -> pd.Series | pd.DataFrame:
        """RTO-specific mapping of constraints to branches."""
        pass

    @abstractmethod
    def get_period_type(self, auction_month: pd.Timestamp, market_month: pd.Timestamp) -> str | None:
        """
        Determine the period type (e.g., 'f0', 'f1', 'q2') for a given auction and market month.
        """
        pass

    def fetch_da_shadow(self, st, et, class_type, remove_noise_floor: bool = True):
        da_shadow = self.fetch_da_shadow_wrapper(st, et, class_type)
        if remove_noise_floor:
            da_shadow = da_shadow[da_shadow["shadow_price"].abs() >= self.config.labeling.noise_floor]
        return da_shadow

    def get_required_period_types(self, test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None) -> set:
        """
        Determine which period types are needed based on test_periods.
        Returns a set of period types (e.g., {'f0', 'f1', 'q2'}) instead of group names.
        """
        # 1. Identify required Horizon Groups
        required_groups = []
        if not test_periods:
            # If no test periods specified, load all groups
            required_groups = self.config.horizon_groups
        else:
            for auction_month, market_month in test_periods:
                # Calculate forecast horizon
                horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

                for group in self.config.horizon_groups:
                    if group.min_horizon <= horizon <= group.max_horizon:
                        if group not in required_groups:
                            required_groups.append(group)

        # 2. Map Groups to Period Types
        required_period_types = set()
        for month, types in self.config.iso.auction_schedule.items():
            for p_type in types:
                horizons = []
                if p_type.startswith("f"):
                    # Fixed offset
                    offset = int(p_type[1:])
                    horizons.append(offset)
                elif p_type.startswith("q"):
                    # Quarter mapping
                    # q1: Jun-Aug, q2: Sep-Nov, q3: Dec-Feb, q4: Mar-May
                    start_months = {"q1": 6, "q2": 9, "q3": 12, "q4": 3}
                    for i in range(3):
                        horizons.append((start_months[p_type] + i - month) % 12)
                # Check if ANY of the resolved market months fall into a required group
                for horizon in horizons:
                    for group in required_groups:
                        if group.min_horizon <= horizon <= group.max_horizon:
                            required_period_types.add(p_type)
                            break
                    else:
                        continue  # Continue if inner loop didn't break
                    break  # Break outer loop if inner loop broke (found a match)

        return required_period_types

    def get_spice_map(self, cons_path: Path) -> pd.DataFrame:
        """
        Load constraints and create a mapping from constraint_id to branch_name.
        """
        cons = fetch_constraints(cons_path)
        gp = cons[cons["convention"] != 999].dropna(subset=["branch_name"]).groupby("constraint_id")
        spice_map = pd.concat([gp["branch_name"].apply(",".join), gp["type"].first()], axis=1)
        return spice_map

    def get_outage_dates(self, market_month: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Get outage dates for a given market month based on configuration frequency.

        Parameters:
        -----------
        market_month : pd.Timestamp
            The market month to generate dates for

        Returns:
        --------
        pd.DatetimeIndex
            Index of outage dates
        """
        return pd.date_range(
            market_month, market_month + pd.offsets.MonthBegin(1), freq=self.config.iso.outage_freq, inclusive="left"
        )

    # @lru_cache
    def get_historical_shadow(self, auction_month, market_month, constraint_period_type, class_type):
        run_at_day = self.config.iso.run_at_day
        cutoff_month = auction_month - pd.offsets.MonthBegin(1)
        run_at = cutoff_month.replace(day=run_at_day)
        season_start = market_month - pd.offsets.MonthBegin((market_month.month - 6) % 12 % 3)
        season_list, season_month_dict = [], {}
        for j in range(1, 4):
            season_month_dict[j] = 0
            for i in range(3):
                tmp_month = season_start + pd.offsets.MonthBegin(i) - pd.DateOffset(years=j)
                if tmp_month <= cutoff_month:
                    season_list.append(tmp_month)
                    season_month_dict[j] += 1
        recent_month_list = [cutoff_month - pd.offsets.MonthBegin(i) for i in range(3)]
        recent_month_list = [x for x in recent_month_list if x <= cutoff_month]

        def get_da(month_list):
            list_da = []
            for tmp_market_month in sorted(month_list, reverse=True):
                st = tmp_market_month
                et = tmp_market_month + pd.offsets.MonthBegin()
                tmp_sp = self.fetch_da_shadow(
                    st=st,
                    et=et,
                    class_type=class_type,
                )
                tmp_sp.index = tmp_sp.index.tz_localize(None)
                if tmp_market_month == cutoff_month:
                    tmp_sp = tmp_sp[tmp_sp.index <= run_at].copy()
                list_da.append(tmp_sp)
            da = pd.concat(list_da)
            return da

        recent_da_hist_discount = 1.3
        season_da_hist_discount = {1: 1.0, 2: 0.8, 3: 0.6}
        recent_da = get_da(recent_month_list)
        recent_da = (
            recent_da.groupby("constraint_id")["shadow_price"].sum().abs()
            / len(recent_month_list)
            * recent_da_hist_discount
        )
        season_da = get_da(season_list)
        season_da["planning_year"] = season_da.index.year + np.where(season_da.index.month < 6, -1, 0)
        season_da = season_da.groupby(["constraint_id", "planning_year"])["shadow_price"].sum().abs().reset_index()
        planning_year = auction_month.year
        if auction_month.month < 6:
            planning_year -= 1
        season_da["planning_year"] = planning_year - season_da["planning_year"]
        season_da["month_count"] = season_da["planning_year"].map(season_month_dict)
        season_da["shadow_discount"] = season_da["planning_year"].map(season_da_hist_discount)
        season_da["shadow_price"] *= season_da["shadow_discount"] / season_da["month_count"]
        season_da["planning_year"] = "season_hist_da_" + season_da["planning_year"].astype(str)
        season_da = season_da.pivot(index="constraint_id", columns="planning_year", values="shadow_price").fillna(0)
        # recent_da = np.log1p(recent_da)
        # season_da = np.log1p(season_da)

        # Use path template from config
        cons_path = Path(
            self.config.iso.data_paths.constraint_path_template.format(
                auction_month=auction_month.strftime("%Y-%m"),
                market_round=self.config.market_round,
                period_type=constraint_period_type if constraint_period_type else self.config.period_type,
                class_type=class_type,
            )
        )
        spice_map = self.get_spice_map(cons_path)

        recent_da = self.map_constraints_to_branches(recent_da, spice_map)
        season_da = self.map_constraints_to_branches(season_da, spice_map)

        return recent_da, season_da, spice_map

    def load_data_for_outage(
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        outage_date: pd.Timestamp,
        require_label: bool = True,
        # Optimization: Accept pre-calculated data
        da_label: pd.DataFrame | None = None,
        da_label_branch_map: pd.Series | None = None,
        recent_hist_da: pd.Series | None = None,
        season_hist_da: pd.Series | None = None,
        spice_map: pd.DataFrame | None = None,
        data_cache: dict | None = None,
    ) -> pd.DataFrame | None:
        """
        Load training data (score features + shadow price labels) for a single outage date.

        Parameters:
        -----------
        auction_month : pd.Timestamp
            The auction month
        market_month : pd.Timestamp
            The market month
        outage_date : pd.Timestamp
            The specific outage date to load data for
        require_label : bool, default=True
            If True, returns None if label data is missing.
            If False, returns features even if label is missing (for prediction).
        da_label : pd.DataFrame, optional
            Pre-loaded DA label data for the entire month
        da_label_branch_map : pd.Series, optional
            Pre-calculated branch map for the label data
        recent_hist_da : pd.Series, optional
            Pre-calculated recent historical shadow prices
        season_hist_da : pd.Series, optional
            Pre-calculated seasonal historical shadow prices
        spice_map : pd.DataFrame, optional
            Pre-loaded constraint mapping
        data_cache : dict, optional
            Shared cache for dataframes. Key: (auction_month, market_month, outage_date)

        Returns:
        --------
        pd.DataFrame or None
            DataFrame with columns: score features, label, branch_name, metadata
            Index: (constraint_id, flow_direction)
        """
        # Check cache first
        cache_key = (auction_month, market_month, outage_date)
        if data_cache is not None and cache_key in data_cache:
            # print(f"  ⚠️ Using cached data for {outage_date.strftime('%Y-%m-%d')}")
            return data_cache[cache_key].copy()

        # print(f"\n[Loading Data for Outage: {outage_date.strftime('%Y-%m-%d')}]")
        # Build paths using config templates
        # Use path template from config
        density_path = Path(
            self.config.iso.data_paths.density_path_template.format(
                auction_month=auction_month.strftime("%Y-%m"),
                market_month=market_month.strftime("%Y-%m"),
                market_round=self.config.market_round,
                outage_date=outage_date.strftime("%Y-%m-%d"),
            )
        )

        # Check if paths exist
        if not density_path.exists():
            raise FileNotFoundError(f"Density file not found: {density_path}")

            # Load density data
        density_df = pd.read_parquet(density_path / "density.parquet")
        try:
            density_multi_df = pd.read_parquet(density_path / "density_multi.parquet")
            density_df = pd.concat([density_df, density_multi_df], axis=0)
        except FileNotFoundError:
            pass
        # density_df = density_df.loc[['278858']]

        # Identify available flow points (columns)
        # Filter out non-numeric columns
        flow_points = []
        for col in density_df.columns:
            flow_points.append(int(col) / 100)

        if not flow_points:
            raise ValueError(f"No valid flow points found in {density_path}")

        # Define helper to calculate features for a given direction
        def calculate_direction_features(df, direction):
            # Create a copy to avoid modifying the original repeatedly
            # We only need the relevant columns for this direction
            # But we need to map them to the standard feature names (110, 105, etc.)

            res = pd.DataFrame(index=df.index)
            res["flow_direction"] = direction

            # Define the mapping from standard feature name (e.g., 110) to actual column (e.g., 110 or -110)
            # and the range for integration
            # --- Engineered Risk Features ---

            # if len(overload_x) > 1:
            #     cols = [str(x) for x in overload_x]
            #     res['prob_overload'] = np.trapezoid(y=df[cols].values, x=overload_x, axis=1)
            # else:
            #     res['prob_overload'] = 0.0

            # Prob Exceed X
            exceed_thresholds = [110, 105, 100, 95, 90, 85, 80]
            for t in exceed_thresholds:
                feat_name = f"prob_exceed_{t}"
                feat_name1 = f"prob_below_{t}"
                if direction == 1:
                    x_range = [x for x in flow_points if x >= t / 100]
                else:
                    x_range = [x for x in flow_points if x <= -t / 100]

                if len(x_range) > 1:
                    cols = [str(int(round(x * 100))) for x in x_range]
                    res[feat_name] = np.trapezoid(y=df[cols].values, x=x_range, axis=1)
                    res[feat_name1] = 1 - res[feat_name]
                else:
                    res[feat_name] = 0.0
                    res[feat_name1] = 0.0

            # --- Density Moments ---
            # Use all flow points to calculate full distribution moments
            # Assuming flow_points covers the range of interest for this direction
            # For moments, we use the density values over the whole range of x_range

            # Use full range of points available in df
            # Note: df cols are strings of flow_points.
            # We should use all flow_points available to capture full distribution
            moments_x = np.array(flow_points)
            moments_cols = [str(int(round(x * 100))) for x in moments_x]

            # Filter cols that exist in df
            existing_indices = [i for i, c in enumerate(moments_cols) if c in df.columns]
            if len(existing_indices) > 1:
                moments_x = moments_x[existing_indices]
                moments_cols = [moments_cols[i] for i in existing_indices]

                x_vals = moments_x
                y_vals = df[moments_cols].values

                # Broadcats x for integration
                # x_vals shape: (N_points,)
                # y_vals shape: (N_samples, N_points)

                # 1. Expectation (Mean)
                # Int x * f(x) dx
                # Multiply by direction so positive mean always implies "towards overload"
                density_mean = np.trapezoid(y=x_vals * y_vals, x=x_vals, axis=1)
                res["density_mean"] = density_mean * direction

                # 2. Variance
                # Int (x - mu)^2 * f(x) dx
                mu = density_mean.reshape(-1, 1)
                # Reshape x to broadcast properly against y if needed, but numpy broadcasts x (N,) against y (M,N) OK.
                # But (x - mu) is (N,) - (M,1) -> (M,N). Correct.
                variance = np.trapezoid(y=((x_vals - mu) ** 2) * y_vals, x=x_vals, axis=1)
                res["density_variance"] = variance

                # 3. Skewness
                # Int ((x - mu)/sigma)^3 * f(x) dx
                # Multiply by direction:
                # Direction 1: Positive skew -> Tail to right (overload). Keep as is.
                # Direction -1: Negative skew -> Tail to left (overload). Flip to positive.
                sigma = np.sqrt(variance).reshape(-1, 1) + 1e-9
                skewness = np.trapezoid(y=((x_vals - mu) ** 3 / sigma**3) * y_vals, x=x_vals, axis=1)
                res["density_skewness"] = skewness * direction

                # 4. Kurtosis
                # Int ((x - mu)/sigma)^4 * f(x) dx
                kurtosis = np.trapezoid(y=((x_vals - mu) ** 4 / sigma**4) * y_vals, x=x_vals, axis=1)
                res["density_kurtosis"] = kurtosis
            else:
                res["density_mean"] = 0.0
                res["density_variance"] = 0.0
                res["density_skewness"] = 0.0
                res["density_kurtosis"] = 0.0

            return res

        # Calculate for both directions
        pos_df = calculate_direction_features(density_df, 1)
        neg_df = calculate_direction_features(density_df, -1)

        # Combine
        full_df = pd.concat([pos_df, neg_df])

        # Load shadow prices for the outage period
        outage_st = outage_date
        outage_et = outage_date + pd.Timedelta(days=4)

        da_label_chunk = None
        if da_label is not None or require_label:
            # Use cached function for current month labels
            if da_label is None:
                if require_label:
                    raise ValueError("da_label is None")
            else:
                da_label = da_label.copy()  # Copy to avoid modifying cached object
                if da_label.index.tz is not None:
                    da_label.index = da_label.index.tz_localize(None)

                # --- Core + Decayed Context Labeling Strategy ---
                # Core Window: outage_st to outage_et (4 days) - Weight 1.0
                # Context Window: +/- config.labeling.context_window_days days around Core - Linear Decay

                context_window_days = self.config.labeling.context_window_days
                decay_end_weight = self.config.labeling.decay_end_weight

                # Ensure outage dates are naive for comparison
                outage_st_naive = outage_st.tz_localize(None) if outage_st.tzinfo else outage_st
                outage_et_naive = outage_et.tz_localize(None) if outage_et.tzinfo else outage_et

                # Define Windows

                context_st = outage_st_naive - pd.Timedelta(days=context_window_days)
                context_et = outage_et_naive + pd.Timedelta(days=context_window_days)

                # Get relevant data slice
                da_slice = da_label.loc[(da_label.index >= context_st) & (da_label.index < context_et)].copy()

                if not da_slice.empty:
                    # Calculate Weights
                    da_slice["weight"] = 0.0

                    # Vectorized weight calculation
                    # Core
                    is_core = (da_slice.index >= outage_st_naive) & (da_slice.index < outage_et_naive)
                    da_slice.loc[is_core, "weight"] = 1.0

                    # Context
                    unique_dates = da_slice.index.unique()
                    date_weights = {}
                    for d in unique_dates:
                        if outage_st_naive <= d < outage_et_naive:
                            date_weights[d] = 1.0
                        else:
                            # Calculate distance to nearest core boundary
                            # Dist is 1-based index (adjacent = 1)
                            if d < outage_st_naive:
                                dist = (outage_st_naive - d).days
                            else:
                                dist = (d - (outage_et_naive - pd.Timedelta(days=1))).days

                            # Linear decay map
                            # Dist 0 (Virtual Core Edge) -> 1.0
                            # Dist window -> decay_end_weight

                            if context_window_days > 0:
                                # y = 1.0 - slope * dist
                                # slope = (1.0 - end) / window
                                slope = (1.0 - decay_end_weight) / context_window_days
                                date_weights[d] = 1.0 - slope * dist
                            else:
                                date_weights[d] = decay_end_weight

                    # Map weights
                    da_slice["weight"] = da_slice.index.map(date_weights)

                    # Weighted Sum
                    da_slice["weighted_shadow"] = da_slice["shadow_price"].abs() * da_slice["weight"]
                    da_label_chunk = da_slice.groupby("constraint_id")["weighted_shadow"].sum()
                else:
                    da_label_chunk = None

            # Create mapping for label branch names
            # da_label_constraint_id_branch_map = self.get_branch_map(da_filtered)

        # Map to branch names
        if spice_map is None:
            raise ValueError("spice_map is required for mapping constraints to branches")
        full_df["branch_name"] = full_df.index.get_level_values("constraint_id").map(spice_map["branch_name"])

        if da_label_chunk is not None:
            da_label_chunk = self.map_constraints_to_branches(da_label_chunk, spice_map)

        # Assign labels
        if da_label_chunk is not None:
            full_df["label"] = full_df["branch_name"].map(da_label_chunk).fillna(0)
        else:
            full_df["label"] = np.nan
        full_df["recent_hist_da"] = full_df["branch_name"].map(recent_hist_da).fillna(0)
        full_df = full_df.merge(season_hist_da, left_on="branch_name", right_index=True, how="left")
        cols = full_df.filter(like="season_hist_da_").columns
        full_df[cols] = full_df[cols].fillna(0)
        full_df["hist_da"] = np.log1p(full_df["recent_hist_da"] + full_df[cols].sum(axis=1))

        # Sort by risk metrics
        full_df = full_df.sort_values(
            ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90"],
            ascending=False,
        )

        # Set Index
        full_df = full_df.reset_index().drop_duplicates(subset=["constraint_id"])

        # Add metadata
        full_df["auction_month"] = auction_month
        full_df["market_month"] = market_month
        full_df["outage_date"] = outage_date

        # Add engineered features for multi-period model
        # Forecast Horizon (months)
        horizon_months = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)
        full_df["forecast_horizon"] = horizon_months

        return full_df

    def load_training_data(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        required_period_types: set[str],
        branch_name: str | None = None,
        verbose: bool = True,
        data_cache: dict | None = None,
    ) -> pd.DataFrame:
        """
        Load and aggregate training data for a range of months.
        """
        all_data = []

        # Iterate through months in training range
        current_month = train_start
        while current_month < train_end:
            auction_month = current_month

            # Check available period types for this month
            available_types = self.config.iso.auction_schedule.get(auction_month.month, [])

            # Filter for required types
            types_to_load = [t for t in available_types if t in required_period_types]

            if not types_to_load:
                if verbose:
                    print(f"  Skipping {auction_month.strftime('%Y-%m')}: No required period types available.")
                current_month += pd.offsets.MonthBegin(1)
                continue

            if verbose:
                print(f"  Loading training data for {auction_month.strftime('%Y-%m')} (Types: {types_to_load})...")

            for p_type in types_to_load:
                market_st, market_et = self.aptools.tools.get_market_month_from_auction_month_and_period_trades(
                    auction_month=auction_month,
                    period_type=p_type,
                )

                for market_month in pd.date_range(market_st, market_et, freq="MS", inclusive="left"):
                    # 3. Load data for each outage date
                    # Note: We use p_type as the constraint_period_type

                    # Optimization: Pre-calculate data for the whole month
                    if market_month >= train_end:
                        print(f"    Skipping future month: {market_month.strftime('%Y-%m')}")
                        continue

                    # Load Historical Shadow Prices (Features)
                    recent_hist_da, season_hist_da, spice_map = self.get_historical_shadow(
                        auction_month, market_month, p_type, self.config.class_type
                    )

                    # Load Labels (DA Shadow Price)
                    # Extend fetch range by context window to ensure labels can be calculated for outages near month boundaries
                    pad_days = self.config.labeling.context_window_days
                    fetch_st = market_month - pd.Timedelta(days=pad_days)
                    fetch_et = market_month + pd.offsets.MonthBegin(1)  # + pd.Timedelta(days=pad_days)

                    da_label = self.fetch_da_shadow(
                        st=fetch_st,
                        et=fetch_et,
                        class_type=self.config.class_type,
                    )
                    if da_label.index.tz is not None:
                        da_label.index = da_label.index.tz_localize(None)

                    # Iterate through outage dates in this market month
                    for outage_date in self.get_outage_dates(market_month):
                        # Skip if outage date is in the future relative to train_end (shouldn't happen with lookback)
                        if outage_date >= train_end:
                            continue

                        daily_data = self.load_data_for_outage(
                            auction_month=auction_month,
                            market_month=market_month,
                            outage_date=outage_date,
                            require_label=True,
                            da_label=da_label,
                            recent_hist_da=recent_hist_da,
                            season_hist_da=season_hist_da,
                            spice_map=spice_map,
                            data_cache=data_cache,
                        )

                        if daily_data is not None:
                            # Add period type info
                            daily_data["period_type"] = p_type

                            # Filter by branch name if provided
                            if branch_name:
                                daily_data = daily_data[
                                    daily_data["branch_name"].str.contains(branch_name, case=False, na=False)
                                ]

                            all_data.append(daily_data)

            current_month += pd.offsets.MonthBegin(1)

        if not all_data:
            raise ValueError(f"No training data loaded for range {train_start} to {train_end}")

        return pd.concat(all_data, ignore_index=True)

    def load_test_data_for_period(
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        period_type: str | None,
        branch_name: str | None = None,
        verbose: bool = True,
        data_cache: dict | None = None,
    ) -> pd.DataFrame:
        """
        Load test data for a specific period (auction/market month + period type).
        """
        # Load Historical Features
        recent_hist_da, season_hist_da, spice_map = self.get_historical_shadow(
            auction_month, market_month, period_type, self.config.class_type
        )

        # Load Labels (Optional, for evaluation)
        da_label = None
        try:
            da_label = self.fetch_da_shadow(
                st=market_month,
                et=market_month + pd.offsets.MonthBegin(1),
                class_type=self.config.class_type,
                remove_noise_floor=False,
            )
            if da_label is not None:
                da_label.index = da_label.index.tz_localize(None)
                if verbose:
                    print(f"  Loaded labels for evaluation (Test Data): {len(da_label)} records")
        except Exception as e:
            if verbose:
                print(f"  Note: Could not load labels for {market_month.strftime('%Y-%m')}: {e}")
            da_label = None

        period_data = []
        for outage_date in self.get_outage_dates(market_month):
            daily_data = self.load_data_for_outage(
                auction_month=auction_month,
                market_month=market_month,
                outage_date=outage_date,
                require_label=False,  # Don't require labels for test data
                da_label=da_label,
                recent_hist_da=recent_hist_da,
                season_hist_da=season_hist_da,
                spice_map=spice_map,
                data_cache=data_cache,
            )

            if daily_data is not None:
                daily_data["period_type"] = period_type

                # Filter by branch name if provided
                if branch_name:
                    daily_data = daily_data[daily_data["branch_name"].str.contains(branch_name, case=False, na=False)]

                period_data.append(daily_data)

        if not period_data:
            raise ValueError(f"No test data loaded for {auction_month} {market_month}")

        return pd.concat(period_data, ignore_index=True)

    def load_test_data(
        self,
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        branch_name: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame | None:
        """
        Load test data for all configured test periods.

        Returns:
        --------
        Combined DataFrame with all test data across all periods
        """
        if test_periods is None:
            if verbose:
                print("⚠️ No test periods provided to load_test_data.")
            return None

        if verbose:
            print("\n[Loading Test Data]")
            print("=" * 80)
            print(f"Test periods: {len(test_periods)}")

        all_test_data = []

        for i, (auction_month, market_month) in enumerate(test_periods):
            if verbose and len(test_periods) > 1:
                print(f"\n--- Period {i + 1}/{len(test_periods)} ---")

            period_type = self.get_period_type(auction_month, market_month)
            period_data = self.load_test_data_for_period(
                auction_month,
                market_month,
                period_type=period_type,
                branch_name=branch_name,
                verbose=verbose,
            )
            all_test_data.append(period_data)

        # Combine all test data across periods
        if len(all_test_data) > 0:
            if len(test_periods) > 1 and verbose:
                print("\n" + "=" * 80)
                print(f"Combining data from {len(all_test_data)} test periods...")

            test_data = pd.concat(all_test_data, ignore_index=True)

            if verbose:
                print(f"  Total samples: {len(test_data):,}")
                print(f"  Unique constraints: {test_data.index.get_level_values('constraint_id').nunique():,}")

                # Check class distribution
                if test_data["label"].notna().any():
                    binding_count = (test_data["label"] > self.config.training.label_threshold).sum()
                    non_binding_count = len(test_data) - binding_count
                    print("\n  Class Distribution:")
                    print(f"    Binding: {binding_count:,} ({binding_count / len(test_data) * 100:.2f}%)")
                    print(f"    Non-binding: {non_binding_count:,} ({non_binding_count / len(test_data) * 100:.2f}%)")
                else:
                    print("\n  Labels: Not available (Test Data)")

            return test_data
        else:
            if verbose:
                print("\n⚠️  ERROR: No test data loaded!")
            return None


class MisoDataLoader(BaseDataLoader):
    """DataLoader implementation for MISO."""

    def fetch_da_shadow_wrapper(self, st, et, class_type):
        return self.aptools.tools.get_da_shadow_by_peaktype(
            st=st,
            et_ex=et,
            peak_type=class_type,
            offpeak_hrs=None,
        ).rename(columns={"monitored_facility": "constraint_id"})

    def map_constraints_to_branches(
        self, data: pd.Series | pd.DataFrame, spice_map: pd.DataFrame, fill_map: pd.Series | None = None
    ) -> pd.Series | pd.DataFrame:
        """
        Map the index (constraint_id) of the data to branch_name using spice_map.
        Aggregates by summing if multiple constraints map to the same branch.
        """
        mapped_index = pd.Series(data.index.map(spice_map["branch_name"]), index=data.index)

        if fill_map is not None:
            mapped_index = mapped_index.fillna(fill_map)

        data = data.copy()
        data.index = mapped_index.values

        # Group by branch name and sum
        return data.groupby(level=0).sum()

    def get_period_type(self, auction_month: pd.Timestamp, market_month: pd.Timestamp) -> str | None:
        q_period = f"q{(market_month.month - 6) % 12 // 3 + 1}"
        f_period = f"f{(market_month.year - auction_month.year) * 12 + market_month.month - auction_month.month}"
        period_type = None
        all_periods = self.aptools.tools._rto_period_type_class.get_valid_periods(auction_month)
        if q_period in all_periods:
            period_type = q_period
        elif f_period in all_periods:
            period_type = f_period
        return period_type


class PjmDataLoader(BaseDataLoader):
    """DataLoader implementation for PJM."""

    def fetch_da_shadow_wrapper(self, st, et, peak_type):
        return self.aptools.tools.get_da_shadow_by_peaktype(
            st=st,
            et_ex=et,
            peak_type=peak_type,
            offpeak_hrs=None,
        ).rename(columns={"monitored_facility": "constraint_id"})

    def map_constraints_to_branches(
        self, data: pd.Series | pd.DataFrame, spice_map: pd.DataFrame, fill_map: pd.Series | None = None
    ) -> pd.Series | pd.DataFrame:
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data_ori_cols = data.columns
        spice_map["monitored_facility"] = pd.Series(spice_map.index).str.split(":", n=1).str[0].values
        spice_map["match_str"] = spice_map["monitored_facility"].str.upper()
        map_nodup_consid = spice_map.reset_index()[["branch_name", "constraint_id"]].drop_duplicates()
        assert map_nodup_consid["constraint_id"].value_counts().max() == 1, (
            "one 'branch_name' has multiple 'constraint_id'"
        )
        map_nodup_moni = spice_map.drop_duplicates(subset=["branch_name", "match_str"]).drop_duplicates(
            subset=["match_str"]
        )

        data["match_str"] = data.index.str.upper()
        interface = map_nodup_moni.loc[map_nodup_moni["type"] == "interface"]

        interface_moni = data[
            data["match_str"].str.startswith(tuple(interface["match_str"] + " "))
            | data["match_str"].isin(interface["match_str"])
        ][["match_str"]].drop_duplicates()
        interface_moni["interface_monitored_facility"] = interface_moni["match_str"].str.split(" ", n=1).str[0]
        interface_moni["branch_name"] = interface_moni["interface_monitored_facility"].map(
            interface.set_index("match_str")["branch_name"]
        )
        data["branch_name"] = data["match_str"].map(map_nodup_moni.set_index("match_str")["branch_name"])
        data["interface_branch_name"] = data["match_str"].map(interface_moni.set_index("match_str")["branch_name"])
        data["branch_name"] = data["branch_name"].fillna(data["interface_branch_name"])
        data_by_branch = data.groupby("branch_name")[data_ori_cols].sum().squeeze()
        return data_by_branch

    def get_period_type(self, auction_month: pd.Timestamp, market_month: pd.Timestamp) -> str | None:
        f_period = f"f{(market_month.year - auction_month.year) * 12 + market_month.month - auction_month.month}"
        period_type = None
        all_periods = self.aptools.tools._rto_period_type_class.get_valid_periods(auction_month)
        if f_period in all_periods:
            period_type = f_period
        return period_type


# Alias for backward compatibility (optional, but pipeline will handle instantiation)
DataLoader = MisoDataLoader
