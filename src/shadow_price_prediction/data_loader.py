"""
Data loading functions for shadow price prediction.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from pbase.analysis.tools.all_positions import MisoApTools

from .config import AUCTION_SCHEDULE, PredictionConfig

# Setup persistent cache for data loading
# memory = joblib.Memory(location='./.gemini_cache', verbose=0)


@cache
def fetch_da_shadow(st, et, peak_type):
    """Cached wrapper for fetching DA shadow prices."""
    ap = MisoApTools()
    return ap.tools.get_da_shadow_by_peaktype(
        st=st,
        et_ex=et,
        peak_type=peak_type,
        offpeak_hrs=None,
    )


@cache
def fetch_constraints(cons_path):
    """Cached wrapper for loading constraints parquet."""
    return pd.read_parquet(cons_path / "constraints.parquet")


class DataLoader:
    """Handles loading of training and test data."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.aptools = MisoApTools()

    @staticmethod
    def get_required_period_types(test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None) -> set:
        """
        Determine which period types are needed based on test_periods.

        Parameters:
        -----------
        test_periods : Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]
            List of (auction_month, market_month) tuples

        Returns:
        --------
        Set of period type strings like: {'f0', 'f1', 'f2', 'f3', 'q2', 'q3', 'q4'}
        """
        if not test_periods:
            # If no test periods specified, load all period types
            return {"f0", "f1", "f2", "f3", "q2", "q3", "q4"}

        required_periods = set()

        for auction_month, market_month in test_periods:
            # Calculate forecast horizon
            horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

            if horizon == 0:
                required_periods.add("f0")
            elif horizon == 1:
                required_periods.add("f1")
            elif horizon == 2:
                required_periods.add("f2")
            elif horizon == 3:
                required_periods.add("f3")
            elif horizon > 3:
                # For quarters, conservatively add all quarter types
                # since we don't know exactly which quarter without detailed logic
                required_periods.update(["q2", "q3", "q4"])

        return required_periods

    def get_training_period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Calculate the training period based on the configured test auction month.

        Returns:
        --------
        train_start : pd.Timestamp
            Start of training period
        train_end : pd.Timestamp
            End of training period (exclusive)
        """
        # Training ends at the start of the test auction month
        train_end = self.config.test_auction_month
        # Training starts N months before
        train_start = train_end - pd.offsets.MonthBegin(self.config.training.train_months_lookback)

        return train_start, train_end

    @cache
    def get_historical_shadow(self, auction_month, market_month, class_type):
        run_at_day = 10
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
                tmp_sp = fetch_da_shadow(
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
        return recent_da, season_da

    def load_data_for_outage(
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        outage_date: pd.Timestamp,
        constraint_period_type: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Load training data (score features + shadow price labels) for a single outage date.

        Returns:
        --------
        DataFrame with columns: score features, label, branch_name, metadata
        Index: (constraint_id, flow_direction)
        """
        try:
            # Build paths using config templates
            density_path = Path(
                self.config.paths.density_path_template.format(
                    auction_month=auction_month.strftime("%Y-%m"),
                    market_month=market_month.strftime("%Y-%m"),
                    market_round=self.config.market_round,
                    outage_date=outage_date.strftime("%Y-%m-%d"),
                )
            )

            cons_path = Path(
                self.config.paths.constraint_path_template.format(
                    auction_month=auction_month.strftime("%Y-%m"),
                    market_round=self.config.market_round,
                    period_type=constraint_period_type if constraint_period_type else self.config.period_type,
                    class_type=self.config.class_type,
                )
            )

            # Check if paths exist
            if not density_path.exists():
                return None

            # Load density data
            density_df = pd.read_parquet(density_path / "density.parquet")

            # Identify available flow points (columns)
            # Filter out non-numeric columns
            flow_points = []
            for c in density_df.columns:
                try:
                    flow_points.append(int(c))
                except ValueError:
                    continue
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
                        res[name] = np.trapz(y=df[cols].values, x=x_range, axis=1)
                    else:
                        res[name] = 0.0

                # --- Engineered Risk Features ---

                # Prob Overload (Flow > 100 or Flow < -100) - REMOVED (Duplicate of prob_exceed_100)
                # if direction == 1:
                #     overload_x = [x for x in flow_points if x >= 100]
                # else:
                #     overload_x = [x for x in flow_points if x <= -100]

                # if len(overload_x) > 1:
                #     cols = [str(x) for x in overload_x]
                #     res['prob_overload'] = np.trapz(y=df[cols].values, x=overload_x, axis=1)
                # else:
                #     res['prob_overload'] = 0.0

                # Prob Exceed X
                exceed_thresholds = [110, 105, 100, 95, 90]
                for t in exceed_thresholds:
                    feat_name = f"prob_exceed_{t}"
                    if direction == 1:
                        x_range = [x for x in flow_points if x >= t]
                    else:
                        x_range = [x for x in flow_points if x <= -t]

                    if len(x_range) > 1:
                        cols = [str(x) for x in x_range]
                        res[feat_name] = np.trapz(y=df[cols].values, x=x_range, axis=1)
                    else:
                        res[feat_name] = 0.0

                # Safe Mass (80 to 90)
                if direction == 1:
                    safe_x = [x for x in flow_points if 80 <= x <= 90]
                else:
                    safe_x = [x for x in flow_points if -90 <= x <= -80]

                if len(safe_x) > 1:
                    cols = [str(x) for x in safe_x]
                    safe_mass = np.trapz(y=df[cols].values, x=safe_x, axis=1)
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

            # Use cached function for current month labels
            da_label = fetch_da_shadow(market_month, market_month + pd.offsets.MonthBegin(1), self.config.class_type)

            da_label = da_label.copy()  # Copy to avoid modifying cached object
            da_label.index = da_label.index.tz_localize(None)

            # Aggregate shadow prices for this outage period (SUM)
            da_label_chunk = (
                da_label.loc[(da_label.index >= outage_st) & (da_label.index < outage_et)]
                .groupby("constraint_id")["shadow_price"]
                .sum()
                .abs()
            )

            # Load constraint info using cache
            cons = fetch_constraints(cons_path)
            spice_map = (
                cons[cons["convention"] != 999]
                .dropna(subset=["branch_name"])
                .groupby("constraint_id")["branch_name"]
                .apply(",".join)
            )

            # Map to branch names
            # full_df index is constraint_id (repeated for pos and neg)
            if "constraint_id" in full_df.index.names:
                full_df["branch_name"] = full_df.index.get_level_values("constraint_id").map(spice_map)
            else:
                full_df["branch_name"] = full_df.index.map(spice_map)

            da_label_chunk.index = da_label_chunk.index.map(spice_map)
            da_label_chunk = da_label_chunk.groupby(level=0).sum()

            recent_hist_da, season_hist_da = self.get_historical_shadow(
                auction_month, market_month, self.config.class_type
            )
            recent_hist_da.index = recent_hist_da.index.map(spice_map)
            recent_hist_da = recent_hist_da.groupby(level=0).sum()
            season_hist_da.index = season_hist_da.index.map(spice_map)
            season_hist_da = season_hist_da.groupby(level=0).sum()

            # Assign labels
            full_df["label"] = full_df["branch_name"].map(da_label_chunk).fillna(0)
            full_df["recent_hist_da"] = full_df["branch_name"].map(recent_hist_da).fillna(0)
            full_df = full_df.merge(season_hist_da, left_on="branch_name", right_index=True, how="left")
            cols = full_df.filter(like="season_hist_da_").columns
            full_df[cols] = full_df[cols].fillna(0)

            # Sort by risk metrics
            full_df = full_df.sort_values(
                ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90"],
                ascending=False,
            )

            # Set Index
            full_df = (
                full_df.reset_index()
                .drop_duplicates(subset=["constraint_id"])
                .set_index(["constraint_id", "flow_direction"])
            )

            # Add metadata
            full_df["auction_month"] = auction_month
            full_df["market_month"] = market_month
            full_df["outage_date"] = outage_date

            # Add engineered features for multi-period model
            # Forecast Horizon (months)
            horizon_months = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)
            full_df["forecast_horizon"] = horizon_months

            return full_df

        except Exception as e:
            print(f"  ⚠️  Error loading data for {outage_date.strftime('%Y-%m-%d')}: {str(e)}")
            return None

    def load_training_data(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        required_period_types: set | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame | None:
        """
        Load all training data using multi-period strategy.

        Note: train_start is ignored. The method uses a fixed 12-month lookback
        from train_end (the target auction month).

        Parameters:
        -----------
        train_start : pd.Timestamp
            Ignored (kept for backward compatibility)
        train_end : pd.Timestamp
            Target auction month
        required_period_types : Optional[set]
            If provided, only load data for these period types (e.g., {'f0', 'f1'}).
            If None, load all period types from AUCTION_SCHEDULE.
        verbose : bool
            Print progress messages

        Strategy:
        1. Look back 12 months from train_end (exclusive).
        2. For each historical auction month, look up available period types from AUCTION_SCHEDULE.
        3. Filter by required_period_types if specified.
        4. Expand period types into (Market Month, Constraint Period Type) pairs.
        5. Load data for each pair.
        """
        if verbose:
            print("\n[Loading Training Data - Multi-Period Strategy]")
            print("-" * 80)
            if required_period_types:
                print(f"  Filtering to period types: {sorted(required_period_types)}")

        # Generate list of auction months in lookback period (12 months before train_end)
        # train_end is the target auction month
        lookback_start = train_end - pd.offsets.MonthBegin(12)
        training_months = pd.date_range(lookback_start, train_end, freq="MS", inclusive="left")

        if verbose:
            print(f"Training Window: {len(training_months)} months")
            print(f"  From: {training_months[0].strftime('%Y-%m')}")
            print(f"  To:   {training_months[-1].strftime('%Y-%m')}")

        # Collect all training data
        training_data_list = []

        # Helper to get market months for a period type
        def get_market_months(auction_month, period_type):
            st, et = self.aptools.tools.get_market_month_from_auction_month_and_period_trades(
                auction_month, period_type
            )
            return pd.date_range(st, et, freq="MS", inclusive="left")

        for auction_month in training_months:
            if verbose:
                print(f"\nProcessing Auction: {auction_month.strftime('%Y-%m')}...")

            # 1. Get available period types from schedule
            available_periods = AUCTION_SCHEDULE.get(auction_month.month, [])

            # 2. Filter by required_period_types if specified
            if required_period_types is not None:
                original_count = len(available_periods)
                available_periods = [p for p in available_periods if p in required_period_types]
                if verbose and len(available_periods) < original_count:
                    print(f"  Filtered: {original_count} → {len(available_periods)} period types")

            month_data_count = 0

            for p_type in available_periods:
                # 2. Expand to market months
                market_months = get_market_months(auction_month, p_type)

                for market_month in market_months:
                    # 3. Load data for each outage date
                    # Note: We use p_type as the constraint_period_type

                    # Optimization: Check if density file exists first?
                    # load_data_for_outage does that.

                    for outage_date in pd.date_range(
                        market_month,
                        market_month + pd.offsets.MonthBegin(1),
                        freq=self.config.outage_freq,
                        inclusive="left",
                    ):
                        # Skip if outage date is in the future relative to train_end (shouldn't happen with lookback)
                        if outage_date >= train_end:
                            continue

                        data = self.load_data_for_outage(
                            auction_month, market_month, outage_date, constraint_period_type=p_type
                        )

                        if data is not None and len(data) > 0:
                            training_data_list.append(data)
                            month_data_count += 1

            if verbose:
                print(f"  Loaded {month_data_count} samples across {len(available_periods)} period types")

        # Combine all training data
        if len(training_data_list) > 0:
            if verbose:
                print("\n" + "=" * 80)
                print(f"Combining {len(training_data_list)} datasets...")

            train_data_combined = pd.concat(training_data_list, axis=0)
            cols = train_data_combined.filter(like="season_hist_da").columns
            train_data_combined[cols] = train_data_combined[cols].fillna(0)

            if verbose:
                print("\n✓ Training Data Loaded Successfully")
                print(f"  Total samples: {len(train_data_combined):,}")
                print(
                    f"  Date range: {train_data_combined['outage_date'].min().strftime('%Y-%m-%d')} "
                    f"to {train_data_combined['outage_date'].max().strftime('%Y-%m-%d')}"
                )
                print(
                    f"  Unique constraints: {train_data_combined.index.get_level_values('constraint_id').nunique():,}"
                )

                # Check class distribution
                binding_count = (train_data_combined["label"] > self.config.training.label_threshold).sum()
                non_binding_count = len(train_data_combined) - binding_count
                print("\n  Class Distribution:")
                print(f"    Binding: {binding_count:,} ({binding_count / len(train_data_combined) * 100:.2f}%)")
                print(
                    f"    Non-binding: {non_binding_count:,} ({non_binding_count / len(train_data_combined) * 100:.2f}%)"
                )

            return train_data_combined
        else:
            if verbose:
                print("\n⚠️  ERROR: No training data loaded!")
            return None

    def load_test_data_for_period(
        self, auction_month: pd.Timestamp, market_month: pd.Timestamp, verbose: bool = True
    ) -> pd.DataFrame | None:
        """
        Load test data for all outage dates in a specific test period.

        Parameters:
        -----------
        auction_month : pd.Timestamp
            Auction month for this test period
        market_month : pd.Timestamp
            Market month for this test period
        verbose : bool
            Print progress messages

        Returns:
        --------
        Combined DataFrame with all test data for this period
        """
        if verbose:
            print(f"\n[Loading Test Data for Period: {auction_month.strftime('%Y-%m')}]")
            print("-" * 80)

        # Collect all test data
        test_data_list = []

        # Generate outage dates for test month
        test_outage_dates = pd.date_range(
            market_month, market_month + pd.offsets.MonthBegin(1), freq=self.config.outage_freq, inclusive="left"
        )

        if verbose:
            print(f"  Scanning for outage dates in {market_month.strftime('%Y-%m')}...")

        loaded_outage_count = 0
        for outage_date in test_outage_dates:
            data = self.load_data_for_outage(auction_month, market_month, outage_date)

            if data is not None and len(data) > 0:
                test_data_list.append(data)
                loaded_outage_count += 1
                if verbose:
                    print(f"  ✓ {outage_date.strftime('%Y-%m-%d')}: {len(data):,} constraints")

        # Combine all test data
        if len(test_data_list) > 0:
            if verbose:
                print("\n" + "=" * 80)
                print(f"Combining {len(test_data_list)} test datasets...")

            test_data = pd.concat(test_data_list, axis=0)

            if verbose:
                print("\n✓ Test Data Loaded Successfully")
                print(f"  Total samples: {len(test_data):,}")
                print(f"  Outage dates loaded: {loaded_outage_count}")
                print(
                    f"  Date range: {test_data['outage_date'].min().strftime('%Y-%m-%d')} "
                    f"to {test_data['outage_date'].max().strftime('%Y-%m-%d')}"
                )
                print(f"  Unique constraints: {test_data.index.get_level_values('constraint_id').nunique():,}")

                # Check class distribution
                binding_count = (test_data["label"] > self.config.training.label_threshold).sum()
                non_binding_count = len(test_data) - binding_count
                print("\n  Class Distribution:")
                print(f"    Binding: {binding_count:,} ({binding_count / len(test_data) * 100:.2f}%)")
                print(f"    Non-binding: {non_binding_count:,} ({non_binding_count / len(test_data) * 100:.2f}%)")

            return test_data
        else:
            if verbose:
                print("\n⚠️  ERROR: No test data loaded!")
            return None

    def load_test_data(
        self, test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None, verbose: bool = True
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

            period_data = self.load_test_data_for_period(auction_month, market_month, verbose)

            if period_data is not None:
                all_test_data.append(period_data)

        # Combine all test data across periods
        if len(all_test_data) > 0:
            if len(test_periods) > 1 and verbose:
                print("\n" + "=" * 80)
                print(f"Combining data from {len(all_test_data)} test periods...")

            test_data = pd.concat(all_test_data, axis=0)

            if len(test_periods) > 1 and verbose:
                print("\n✓ All Test Data Loaded Successfully")
                print(f"  Total samples across all periods: {len(test_data):,}")
                print(
                    f"  Date range: {test_data['outage_date'].min().strftime('%Y-%m-%d')} "
                    f"to {test_data['outage_date'].max().strftime('%Y-%m-%d')}"
                )

            return test_data
        else:
            if verbose:
                print("\n⚠️  ERROR: No test data loaded!")
            return None
