"""
Data loading functions for shadow price prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pbase.analysis.tools.all_positions import MisoApTools

from .config import AUCTION_SCHEDULE, PredictionConfig

# Setup persistent cache for data loading
# memory = joblib.Memory(location='./.gemini_cache', verbose=0)


# @lru_cache
def fetch_da_shadow(st, et, peak_type, aptools):
    """Cached wrapper for fetching DA shadow prices."""
    return aptools.tools.get_da_shadow_by_peaktype(
        st=st,
        et_ex=et,
        peak_type=peak_type,
        offpeak_hrs=None,
    )


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
            # elif horizon == 2:
            #     required_periods.add("f2")
            # elif horizon == 3:
            #     required_periods.add("f3")
            elif horizon >= 2:
                # For quarters, conservatively add all quarter types
                # since we don't know exactly which quarter without detailed logic
                required_periods.update(["f2", "f3", "q2", "q3", "q4"])

        return required_periods

    def get_branch_map(self, da):
        da_label_constraint_id_branch_map = da[["constraint_id", "branch_name"]].copy()
        da_label_constraint_id_branch_map = da_label_constraint_id_branch_map[
            (da_label_constraint_id_branch_map["branch_name"] != "None")
            & da_label_constraint_id_branch_map["branch_name"].notna()
        ].drop_duplicates(subset=["branch_name", "branch_name"])[["constraint_id", "branch_name"]]
        da_label_constraint_id_branch_map["branch_name"] = (
            da_label_constraint_id_branch_map["branch_name"].str.rsplit("(", n=1).str[0]
        )
        da_label_constraint_id_branch_map["branch_name"] = da_label_constraint_id_branch_map["branch_name"].str.strip()
        da_label_constraint_id_branch_map = da_label_constraint_id_branch_map.set_index("constraint_id")["branch_name"]
        da_label_constraint_id_branch_map = da_label_constraint_id_branch_map[
            ~da_label_constraint_id_branch_map.index.duplicated(keep="last")
        ]
        return da_label_constraint_id_branch_map

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
            market_month, market_month + pd.offsets.MonthBegin(1), freq=self.config.outage_freq, inclusive="left"
        )

    # @lru_cache
    def get_historical_shadow(self, auction_month, market_month, constraint_period_type, class_type):
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
                    aptools=self.aptools,
                )
                tmp_sp.index = tmp_sp.index.tz_localize(None)
                if tmp_market_month == cutoff_month:
                    tmp_sp = tmp_sp[tmp_sp.index <= run_at].copy()
                list_da.append(tmp_sp)
            da = pd.concat(list_da)
            return da

        recent_da = get_da(recent_month_list)

        # Get branch map (used for reference but not stored separately)
        _ = self.get_branch_map(recent_da)

        recent_da = recent_da.groupby("constraint_id")["shadow_price"].sum().abs() / len(recent_month_list)
        season_da = get_da(season_list)
        _ = self.get_branch_map(season_da)
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

        cons_path = Path(
            self.config.paths.constraint_path_template.format(
                auction_month=auction_month.strftime("%Y-%m"),
                market_round=self.config.market_round,
                period_type=constraint_period_type if constraint_period_type else self.config.period_type,
                class_type=self.config.class_type,
            )
        )
        cons = fetch_constraints(cons_path)
        spice_map = (
            cons[cons["convention"] != 999]
            .dropna(subset=["branch_name"])
            .groupby("constraint_id")["branch_name"]
            .apply(",".join)
        )

        recent_da_index = pd.Series(recent_da.index.map(spice_map), index=recent_da.index)
        # recent_da_index = recent_da_index.fillna(recent_da_branch)
        recent_da.index = recent_da_index.values
        recent_da = recent_da.groupby(level=0).sum()
        season_da_index = pd.Series(season_da.index.map(spice_map), index=season_da.index)
        # season_da_index = season_da_index.fillna(season_da_branch)
        season_da.index = season_da_index.values
        season_da = season_da.groupby(level=0).sum()
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
        spice_map: pd.Series | None = None,
        data_cache: dict | None = None,
    ) -> pd.DataFrame | None:
        """
        Load training data (score features + shadow price labels) for a single outage date.

        Parameters:
        -----------
        ...
        data_cache : dict, optional
            Shared cache for dataframes. Key: (auction_month, market_month, outage_date)
        """
        # Check cache first
        cache_key = (auction_month, market_month, outage_date)
        if data_cache is not None and cache_key in data_cache:
            # print(f"  ⚠️ Using cached data for {outage_date.strftime('%Y-%m-%d')}")
            return data_cache[cache_key].copy()
        """
        Load training data (score features + shadow price labels) for a single outage date.

        Returns:
        --------
        DataFrame with columns: score features, label, branch_name, metadata
        Index: (constraint_id, flow_direction)
        """
        # print(f"\n[Loading Data for Outage: {outage_date.strftime('%Y-%m-%d')}]")
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

            # Check if paths exist
            if not density_path.exists():
                return None

            # Load density data
            density_df = pd.read_parquet(density_path / "density.parquet")
            # density_df = density_df.loc[['278858']]

            # Identify available flow points (columns)
            # Filter out non-numeric columns
            flow_points = []
            for col in density_df.columns:
                try:
                    # Convert to float first to handle "100.0", then int
                    val = int(float(col))
                    flow_points.append(val)
                except ValueError:
                    continue

            if not flow_points:
                return None

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
                try:
                    # Use cached function for current month labels
                    if da_label is None:
                        raise ValueError("da_label is None")
                    da_label = da_label.copy()  # Copy to avoid modifying cached object
                    da_label.index = da_label.index.tz_localize(None)

                    # Aggregate shadow prices for this outage period (SUM)
                    da_filtered = da_label.loc[(da_label.index >= outage_st) & (da_label.index < outage_et)]
                    da_label_chunk = da_filtered.groupby("constraint_id")["shadow_price"].sum().abs()

                    # Create mapping for label branch names
                    da_label_constraint_id_branch_map = da_filtered[["constraint_id", "branch_name"]].copy()
                    da_label_constraint_id_branch_map = da_label_constraint_id_branch_map[
                        (da_label_constraint_id_branch_map["branch_name"] != "None")
                        & da_label_constraint_id_branch_map["branch_name"].notna()
                    ].drop_duplicates(subset=["branch_name", "branch_name"])[["constraint_id", "branch_name"]]
                    da_label_constraint_id_branch_map["branch_name"] = (
                        da_label_constraint_id_branch_map["branch_name"].str.rsplit("(", n=1).str[0]
                    )
                    da_label_constraint_id_branch_map["branch_name"] = da_label_constraint_id_branch_map[
                        "branch_name"
                    ].str.strip()
                    da_label_constraint_id_branch_map = da_label_constraint_id_branch_map.set_index("constraint_id")[
                        "branch_name"
                    ]
                    da_label_constraint_id_branch_map = da_label_constraint_id_branch_map[
                        da_label_constraint_id_branch_map.index.duplicated(keep="last")
                    ]
                except Exception as e:
                    print(f"    ⚠️ Failed to load labels for {market_month}: {e}")
                    da_label_chunk = None
                    raise ValueError(f"Failed to load labels for {market_month}: {e}") from e

            # Load constraint info using cache
            # cons = fetch_constraints(cons_path)
            # spice_map = (
            #     cons[cons["convention"] != 999]
            #     .dropna(subset=["branch_name"])
            #     .groupby("constraint_id")["branch_name"]
            #     .apply(",".join)
            # )

            # Map to branch names
            # full_df index is constraint_id (repeated for pos and neg)
            if "constraint_id" in full_df.index.names:
                full_df["branch_name"] = full_df.index.get_level_values("constraint_id").map(spice_map)
            else:
                full_df["branch_name"] = full_df.index.map(spice_map)

            if da_label_chunk is not None:
                da_label_chunk_index = pd.Series(da_label_chunk.index.map(spice_map), index=da_label_chunk.index)
                da_label_chunk_index = da_label_chunk_index.fillna(da_label_constraint_id_branch_map)
                da_label_chunk.index = da_label_chunk_index.values
                da_label_chunk = da_label_chunk.groupby(level=0).sum()

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
            import traceback

            traceback.print_exc()
            raise ValueError from None

    def load_training_data(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        required_period_types: set | None = None,
        branch_name: str | None = None,
        verbose: bool = True,
        data_cache: dict | None = None,
    ) -> pd.DataFrame | None:
        """
        Load all training data using multi-period strategy.

        Note: train_start is ignored. The method uses a fixed 12-month lookback
        from train_end (the target auction month exclusive).

        Parameters:
        -----------
        train_start : pd.Timestamp
            Ignored (kept for backward compatibility)
        train_end : pd.Timestamp
            Target auction month, exclusive.
        required_period_types : Optional[set]
            If provided, only load data for these period types (e.g., {'f0', 'f1'}).
        branch_name : Optional[str]
            If provided, only load data for branches containing this string.
        verbose : bool
            Print progress messages

        Returns:
        --------
        Combined DataFrame with all training data
        """
        auction_month = train_end
        if verbose:
            print(f"\n[Loading Training Data for Auction: {auction_month.strftime('%Y-%m')}]")
            print("-" * 80)

        training_data_list = []

        # Strategy:
        # 1. Look back 12 months from auction_month
        # 2. For each past auction month, find relevant market months
        # 3. Load data for those market months

        past_auction_months = pd.date_range(train_start, train_end, freq="MS", inclusive="left")

        # Helper to get market months for a given auction and period type
        def get_market_months(auc_month, p_type):
            if p_type.startswith("f"):
                # Forward month (f0 = current month, f1 = next month, etc.)
                offset = int(p_type[1:])
                return [auc_month + pd.DateOffset(months=offset)]
            elif p_type.startswith("q"):
                # Quarter (q1 = Jan-Mar, q2 = Apr-Jun, etc.)
                # This is more complex as it depends on the auction month relative to the quarter
                # Simplified: Return all months in the quarter that are valid for this auction
                # For now, let's assume standard quarterly products relative to auction
                # But actually, the period type defines the PRODUCT, not the time relative to auction?
                # No, period type like 'q2' usually means "Quarter 2" (Apr-Jun)
                # Let's use a simplified heuristic:
                # If auction is in Jan (1), q2 is Apr-Jun.
                # If auction is in Feb (2), q2 is Apr-Jun.
                # If auction is in Mar (3), q2 is Apr-Jun.
                # So we need to find the next occurrence of the quarter.

                # However, for historical training, we want to find where this product WAS traded.
                # This logic is tricky.
                # Let's stick to the existing logic if possible, or simplified one.
                # The existing logic (implied) was likely just f-products.
                # But we added q-products support.

                # Let's look at how we can map p_type to market months.
                # Actually, for training data, we want to find "similar" periods.
                # But here we are iterating over PAST auctions.
                # For a past auction, we want to load data that matches the requested period types.

                # Let's use a simple mapping for now:
                # f-types are relative.
                # q-types are absolute (calendar quarters).

                if p_type.startswith("f"):
                    offset = int(p_type[1:])
                    return [auc_month + pd.DateOffset(months=offset)]

                # For q-types, we need to know which months correspond to qN
                q_num = int(p_type[1:])
                # q1: 1,2,3; q2: 4,5,6; q3: 7,8,9; q4: 10,11,12
                start_month = (q_num - 1) * 3 + 1
                # We need to find the year.
                # If auction month is before the quarter start, it's same year.
                # If auction month is after, it's next year?
                # Usually auctions trade future quarters.

                current_year = auc_month.year
                # Try same year
                candidate = pd.Timestamp(year=current_year, month=start_month, day=1)
                if candidate < auc_month:
                    # If quarter start is in past, maybe it's next year?
                    # But we are looking at PAST auctions.
                    # If we are at past auction, we traded a future quarter.
                    candidate = pd.Timestamp(year=current_year + 1, month=start_month, day=1)

                return [candidate + pd.DateOffset(months=i) for i in range(3)]

            return []

        for i, auction_month in enumerate(past_auction_months):
            if verbose:
                print(
                    f"  Processing past auction: {auction_month.strftime('%Y-%m')} ({i + 1}/{len(past_auction_months)})"
                )

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
                _ = get_market_months(auction_month, p_type)  # For reference only
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

                    try:
                        # Fetch Full Month Labels
                        da_label = fetch_da_shadow(
                            market_month, market_month + pd.offsets.MonthBegin(1), self.config.class_type, self.aptools
                        )
                        da_label = da_label.copy()
                        da_label.index = da_label.index.tz_localize(None)

                        # Derive Branch Map from Labels (once per month)
                        da_label_branch_map = da_label[["constraint_id", "branch_name"]].copy()
                        da_label_branch_map = da_label_branch_map[
                            (da_label_branch_map["branch_name"] != "None") & da_label_branch_map["branch_name"].notna()
                        ].drop_duplicates(subset=["branch_name", "branch_name"])[["constraint_id", "branch_name"]]
                        da_label_branch_map["branch_name"] = (
                            da_label_branch_map["branch_name"].str.rsplit("(", n=1).str[0]
                        )
                        da_label_branch_map["branch_name"] = da_label_branch_map["branch_name"].str.strip()
                        da_label_branch_map = da_label_branch_map.set_index("constraint_id")["branch_name"]
                        da_label_branch_map = da_label_branch_map[da_label_branch_map.index.duplicated(keep="last")]
                    except Exception as e:
                        print(
                            f"    ⚠️ Failed to load labels for {auction_month} {market_month} with training period {train_start} - {train_end}: {e}"
                        )
                        da_label = None
                        da_label_branch_map = None
                        raise ValueError from e

                    # Fetch Historical Data

                    try:
                        recent_hist_da, season_hist_da, spice_map = self.get_historical_shadow(
                            auction_month, market_month, p_type, self.config.class_type
                        )
                    except Exception as e:
                        print(
                            f"    ⚠️ Failed to load historical data for {auction_month} {market_month} with training period {train_start} - {train_end}: {e}"
                        )
                        recent_hist_da = None
                        season_hist_da = None
                        spice_map = None
                        raise ValueError from e

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
                            auction_month,
                            market_month,
                            outage_date,
                            # Pass pre-calculated data
                            da_label=da_label,
                            recent_hist_da=recent_hist_da,
                            season_hist_da=season_hist_da,
                            spice_map=spice_map,
                            data_cache=data_cache,
                        )

                        if data is not None and len(data) > 0:
                            # Filter by branch_name if provided
                            if branch_name:
                                data = data[data["branch_name"].str.contains(branch_name, na=False)]

                            if len(data) > 0:
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
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        branch_name: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame | None:
        """
        Load test data for all outage dates in a specific test period.

        Parameters:
        -----------
        auction_month : pd.Timestamp
            Auction month for this test period
        market_month : pd.Timestamp
            Market month for this test period
        branch_name : Optional[str]
            If provided, only load data for branches containing this string.
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

        # Determine constraint_period_type based on horizon and AUCTION_SCHEDULE
        horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

        f_type = f"f{horizon}"
        q_num = ((market_month.month - 6) % 12) // 3 + 1
        q_type = f"q{q_num}"

        valid_periods = AUCTION_SCHEDULE.get(auction_month.month, [])

        # Priority: Quarter -> Month (matches notebook logic)
        if q_type in valid_periods:
            constraint_period_type = q_type
        elif f_type in valid_periods:
            constraint_period_type = f_type
        else:
            # Fallback to f-type if not found in schedule (or default)
            constraint_period_type = f_type

        if verbose:
            print(f"  Constraint Period Type: {constraint_period_type}")

        # Optimization: Pre-calculate historical data for the whole month
        # Labels are not required for test data, so we skip da_label pre-calc
        try:
            recent_hist_da, season_hist_da, spice_map = self.get_historical_shadow(
                auction_month, market_month, constraint_period_type, self.config.class_type
            )
        except Exception as e:
            if verbose:
                print(f"    ⚠️ Failed to load historical data for {market_month}: {e}")
            recent_hist_da = None
            season_hist_da = None
            spice_map = None

        loaded_outage_count = 0
        for outage_date in test_outage_dates:
            data = self.load_data_for_outage(
                auction_month,
                market_month,
                outage_date,
                require_label=False,
                # Pass pre-calculated data
                recent_hist_da=recent_hist_da,
                season_hist_da=season_hist_da,
                spice_map=spice_map,
            )

            if data is not None and len(data) > 0:
                # Filter by branch_name if provided
                if branch_name:
                    data = data[data["branch_name"].str.contains(branch_name, na=False)]

                if len(data) > 0:
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

            period_data = self.load_test_data_for_period(auction_month, market_month, verbose=verbose)

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
