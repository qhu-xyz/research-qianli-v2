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
    def fetch_da_shadow_wrapper(self, st, et, peak_type):
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
                tmp_sp = self.fetch_da_shadow_wrapper(
                    st=st,
                    et=et,
                    peak_type=class_type,
                )
                tmp_sp.index = tmp_sp.index.tz_localize(None)
                if tmp_market_month == cutoff_month:
                    tmp_sp = tmp_sp[tmp_sp.index <= run_at].copy()
                list_da.append(tmp_sp)
            da = pd.concat(list_da)
            return da

        recent_da = get_da(recent_month_list)
        recent_da = recent_da.groupby("constraint_id")["shadow_price"].sum().abs() / len(recent_month_list)
        season_da = get_da(season_list)
        season_da["planning_year"] = season_da.index.year + np.where(season_da.index.month < 6, -1, 0)
        season_da = season_da.groupby(["constraint_id", "planning_year"])["shadow_price"].sum().abs().reset_index()
        planning_year = auction_month.year
        if auction_month.month < 6:
            planning_year -= 1
        season_da["planning_year"] = planning_year - season_da["planning_year"]
        season_da["month_count"] = season_da["planning_year"].map(season_month_dict)
        season_da["shadow_price"] /= season_da["month_count"]
        season_da["planning_year"] = "season_hist_da_" + season_da["planning_year"].astype(str)
        season_da = season_da.pivot(index="constraint_id", columns="planning_year", values="shadow_price").fillna(0)
        recent_da = np.log1p(recent_da)
        season_da = np.log1p(season_da)

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
        # density_df = density_df.loc[['278858']]

        # Identify available flow points (columns)
        # Filter out non-numeric columns
        flow_points = []
        for col in density_df.columns:
            # Convert to float first to handle "100.0", then int
            val = int(float(col))
            flow_points.append(val)

        if not flow_points:
            raise ValueError(f"No valid flow points found in {density_path}")

        flow_points = sorted(flow_points)

        # Define helper to calculate features for a given direction
        def calculate_direction_features(df, direction):
            # Create a copy to avoid modifying the original repeatedly
            # We only need the relevant columns for this direction
            # But we need to map them to the standard feature names (110, 105, etc.)

            res = pd.DataFrame(index=df.index)
            res["flow_direction"] = direction

            # Define the mapping from standard feature name (e.g., 110) to actual column (e.g., 110 or -110)
            # and the range for integration

            # We want features: 110, 105, 100, 95, 90, 85, 80, 70, 60
            # And diffs: 105_diff (105-110), 100_diff (100-105), etc.

            target_points = [110, 105, 100, 95, 90, 85, 80, 70, 60]

            for p in target_points:
                col_name = str(p * direction)  # e.g., "110" or "-110"
                if col_name in df.columns:
                    res[str(p)] = df[col_name]
                else:
                    res[str(p)] = 0.0

            # Calculate Diffs (Probability Mass in intervals)
            # 105_diff: Mass between 105 and 110 (or -110 and -105)
            # We always integrate from smaller absolute to larger absolute (or vice versa, area is positive)

            diff_intervals = [
                ("105_diff", 105, 110),
                ("100_diff", 100, 105),
                ("95_diff", 95, 100),
                ("90_diff", 90, 95),
                ("85_diff", 85, 90),
                ("80_diff", 80, 85),
                ("70_diff", 70, 80),
                ("60_diff", 60, 70),
            ]

            for name, start, end in diff_intervals:
                # For positive: integrate start to end
                # For negative: integrate -end to -start
                if direction == 1:
                    x_range = [x for x in flow_points if start <= x <= end]
                else:
                    x_range = [x for x in flow_points if -end <= x <= -start]

                if len(x_range) > 1:
                    cols = [str(x) for x in x_range]
                    res[name] = np.trapezoid(y=df[cols].values, x=x_range, axis=1)
                else:
                    res[name] = 0.0

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
                if direction == 1:
                    x_range = [x for x in flow_points if x >= t]
                else:
                    x_range = [x for x in flow_points if x <= -t]

                if len(x_range) > 1:
                    cols = [str(x) for x in x_range]
                    res[feat_name] = np.trapezoid(y=df[cols].values, x=x_range, axis=1)
                else:
                    res[feat_name] = 0.0

            # Safe Mass (80 to 90)
            if direction == 1:
                safe_x = [x for x in flow_points if 80 <= x <= 90]
            else:
                safe_x = [x for x in flow_points if -90 <= x <= -80]

            if len(safe_x) > 1:
                cols = [str(x) for x in safe_x]
                safe_mass = np.trapezoid(y=df[cols].values, x=safe_x, axis=1)
            else:
                safe_mass = 0.0

            res["risk_ratio"] = res["prob_exceed_100"] / (safe_mass + 1e-6)

            # Curvature at 100
            # (d110 - d100) - (d100 - d90)
            # For negative: (d-110 - d-100) - (d-100 - d-90) ??
            # Or just use the mapped columns: (res['110'] - res['100']) - (res['100'] - res['90'])
            res["curvature_100"] = (res["110"] - res["100"]) - (res["100"] - res["90"])

            # Log Density
            res["log_density_100"] = np.log1p(res["prob_exceed_100"])

            # --- Interaction Features (1.B) ---
            # risk_ratio * prob_exceed_100: Amplifies high-risk high-probability events
            res["interaction_risk_overload"] = res["risk_ratio"] * res["prob_exceed_100"]

            # curvature * prob_exceed_100: Distinguishes between "sharp" peaks and "flat" plateaus
            res["interaction_curvature_exceed"] = res["curvature_100"] * res["prob_exceed_100"]

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
        if require_label:
            # Use cached function for current month labels
            if da_label is None:
                raise ValueError("da_label is None")
            da_label = da_label.copy()  # Copy to avoid modifying cached object
            da_label.index = da_label.index.tz_localize(None)

            # Aggregate shadow prices for this outage period (SUM)
            da_filtered = da_label.loc[(da_label.index >= outage_st) & (da_label.index < outage_et)]
            da_label_chunk = da_filtered.groupby("constraint_id")["shadow_price"].sum().abs()

            # Create mapping for label branch names
            # da_label_constraint_id_branch_map = self.get_branch_map(da_filtered)

        # Map to branch names
        if spice_map is None:
            raise ValueError("spice_map is required for mapping constraints to branches")
        full_df["branch_name"] = full_df.index.get_level_values("constraint_id").map(spice_map["branch_name"])

        if da_label_chunk is not None:
            da_label_chunk = self.map_constraints_to_branches(da_label_chunk, spice_map)

        # recent_hist_da_index = pd.Series(recent_hist_da.index.map(spice_map), index=recent_hist_da.index)
        # recent_hist_da_index = recent_hist_da_index.fillna(recent_hist_da_branch)
        # recent_hist_da.index = recent_hist_da_index.values
        # recent_hist_da = recent_hist_da.groupby(level=0).sum()
        # season_hist_da_index = pd.Series(season_hist_da.index.map(spice_map), index=season_hist_da.index)
        # season_hist_da_index = season_hist_da_index.fillna(season_hist_da_branch)
        # season_hist_da.index = season_hist_da_index.values
        # season_hist_da = season_hist_da.groupby(level=0).sum()

        # Assign labels
        if da_label_chunk is not None:
            full_df["label"] = full_df["branch_name"].map(da_label_chunk).fillna(0)
        else:
            full_df["label"] = np.nan
        full_df["recent_hist_da"] = full_df["branch_name"].map(recent_hist_da).fillna(0)
        full_df = full_df.merge(season_hist_da, left_on="branch_name", right_index=True, how="left")
        cols = full_df.filter(like="season_hist_da_").columns
        full_df[cols] = full_df[cols].fillna(0)
        full_df["hist_da"] = full_df["recent_hist_da"] + full_df[cols].sum(axis=1)

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

                    # Load Labels (DA Shadow Price) for the whole month
                    da_label = self.fetch_da_shadow_wrapper(
                        st=market_month,
                        et=market_month + pd.offsets.MonthBegin(1),
                        peak_type=self.config.class_type,
                    )
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
        map_nodup_moni = spice_map.drop_duplicates(subset=["branch_name", "monitored_facility"]).drop_duplicates(
            subset=["monitored_facility"]
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
        data["branch_name"] = data["match_str"].map(map_nodup_moni.set_index("monitored_facility")["branch_name"])
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
