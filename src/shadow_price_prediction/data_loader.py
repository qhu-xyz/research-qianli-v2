"""
Data loading functions for shadow price prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from pbase.analysis.tools.all_positions import MisoApTools

from .config import PredictionConfig


class DataLoader:
    """Handles loading of training and test data."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.aptools = MisoApTools()

    def get_training_period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
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

    def load_data_for_outage(
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        outage_date: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
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
                    auction_month=auction_month.strftime('%Y-%m'),
                    market_month=market_month.strftime('%Y-%m'),
                    market_round=self.config.market_round,
                    outage_date=outage_date.strftime('%Y-%m-%d')
                )
            )

            cons_path = Path(
                self.config.paths.constraint_path_template.format(
                    auction_month=auction_month.strftime('%Y-%m'),
                    market_round=self.config.market_round,
                    period_type=self.config.period_type,
                    class_type=self.config.class_type
                )
            )

            # Check if paths exist
            if not density_path.exists():
                return None

            # Load density data
            density_df = pd.read_parquet(density_path / 'density.parquet')
            
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
                res['flow_direction'] = direction
                
                # Define the mapping from standard feature name (e.g., 110) to actual column (e.g., 110 or -110)
                # and the range for integration
                
                # We want features: 110, 105, 100, 95, 90, 85, 80, 70, 60
                # And diffs: 105_diff (105-110), 100_diff (100-105), etc.
                
                target_points = [110, 105, 100, 95, 90, 85, 80, 70, 60]
                
                for p in target_points:
                    col_name = str(p * direction) # e.g., "110" or "-110"
                    if col_name in df.columns:
                        res[str(p)] = df[col_name]
                    else:
                        res[str(p)] = 0.0

                # Calculate Diffs (Probability Mass in intervals)
                # 105_diff: Mass between 105 and 110 (or -110 and -105)
                # We always integrate from smaller absolute to larger absolute (or vice versa, area is positive)
                
                diff_intervals = [
                    ('105_diff', 105, 110),
                    ('100_diff', 100, 105),
                    ('95_diff', 95, 100),
                    ('90_diff', 90, 95),
                    ('85_diff', 85, 90),
                    ('80_diff', 80, 85),
                    ('70_diff', 70, 80),
                    ('60_diff', 60, 70)
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
                
                # Prob Overload (Flow > 100 or Flow < -100)
                if direction == 1:
                    overload_x = [x for x in flow_points if x >= 100]
                else:
                    overload_x = [x for x in flow_points if x <= -100]
                
                if len(overload_x) > 1:
                    cols = [str(x) for x in overload_x]
                    res['prob_overload'] = np.trapz(y=df[cols].values, x=overload_x, axis=1)
                else:
                    res['prob_overload'] = 0.0
                    
                # Prob Exceed X
                exceed_thresholds = [110, 105, 100, 95, 90]
                for t in exceed_thresholds:
                    feat_name = f'prob_exceed_{t}'
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
                
                res['risk_ratio'] = res['prob_overload'] / (safe_mass + 1e-6)
                
                # Curvature at 100
                # (d110 - d100) - (d100 - d90)
                # For negative: (d-110 - d-100) - (d-100 - d-90) ?? 
                # Or just use the mapped columns: (res['110'] - res['100']) - (res['100'] - res['90'])
                res['curvature_100'] = (res['110'] - res['100']) - (res['100'] - res['90'])
                
                # Log Density
                res['log_density_100'] = np.log1p(res['100'])

                # --- Interaction Features (1.B) ---
                # risk_ratio * prob_overload: Amplifies high-risk high-probability events
                res['interaction_risk_overload'] = res['risk_ratio'] * res['prob_overload']
                
                # curvature * prob_exceed_100: Distinguishes between "sharp" peaks and "flat" plateaus
                # Note: prob_exceed_100 is same as prob_overload, but using the name for clarity
                res['interaction_curvature_exceed'] = res['curvature_100'] * res['prob_overload']
                
                return res

            # Calculate for both directions
            pos_df = calculate_direction_features(density_df, 1)
            neg_df = calculate_direction_features(density_df, -1)
            
            # Combine
            full_df = pd.concat([pos_df, neg_df])
            
            # Load shadow prices for the outage period
            outage_st = outage_date
            outage_et = outage_date + pd.Timedelta(days=4)

            da_label = self.aptools.tools.get_da_shadow_by_peaktype(
                market_month,
                market_month + pd.offsets.MonthBegin(1),
                self.config.class_type
            )
            da_label.index = da_label.index.tz_localize(None)

            # Aggregate shadow prices for this outage period (SUM)
            da_label_chunk = da_label.loc[
                (da_label.index >= outage_st) & (da_label.index < outage_et)
            ].groupby('constraint_id')['shadow_price'].sum()

            # Load constraint info
            cons = pd.read_parquet(cons_path / 'constraints.parquet')
            spice_map = (
                cons[cons['convention'] != 999]
                .dropna(subset=['branch_name'])
                .groupby('constraint_id')['branch_name']
                .apply(','.join)
            )

            # Map to branch names
            # full_df index is constraint_id (repeated for pos and neg)
            if 'constraint_id' in full_df.index.names:
                 full_df['branch_name'] = full_df.index.get_level_values('constraint_id').map(spice_map)
            else:
                 full_df['branch_name'] = full_df.index.map(spice_map)
                 
            da_label_chunk.index = da_label_chunk.index.map(spice_map)
            da_label_chunk = da_label_chunk.groupby(level=0).sum().abs()

            # Assign labels
            full_df['label'] = full_df['branch_name'].map(da_label_chunk).fillna(0)

            # Sort by risk metrics
            full_df = full_df.sort_values(
                ['prob_exceed_110', 'prob_exceed_105', 'prob_overload', 'prob_exceed_95', 'prob_exceed_90'], 
                ascending=False
            )
            
            # Set Index
            full_df = full_df.reset_index().drop_duplicates(subset=['constraint_id']).set_index(['constraint_id', 'flow_direction'])

            # Add metadata
            full_df['auction_month'] = auction_month
            full_df['market_month'] = market_month
            full_df['outage_date'] = outage_date

            return full_df

        except Exception as e:
            print(f"  ⚠️  Error loading data for {outage_date.strftime('%Y-%m-%d')}: {str(e)}")
            return None

    def load_training_data(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load all training data for the specified date range.

        Parameters:
        -----------
        train_start : pd.Timestamp
            Start of training period
        train_end : pd.Timestamp
            End of training period
        verbose : bool
            Print progress messages

        Returns:
        --------
        Combined DataFrame with all training data
        """
        if verbose:
            print("\n[Loading Training Data]")
            print("-" * 80)

        # Generate list of auction months in training period
        training_months = pd.date_range(train_start, train_end, freq='MS', inclusive='left')

        if verbose:
            print(f"Training months: {len(training_months)} months")
            print(f"  From: {training_months[0].strftime('%Y-%m')}")
            print(f"  To:   {training_months[-1].strftime('%Y-%m')}")

        # Collect all training data
        training_data_list = []

        for auction_month in training_months:
            market_month = auction_month

            if verbose:
                print(f"\nProcessing {auction_month.strftime('%Y-%m')}...")

            month_data_count = 0
            for outage_date in pd.date_range(
                market_month,
                market_month + pd.offsets.MonthBegin(1),
                freq=self.config.outage_freq,
                inclusive='left'
            ):
                if outage_date > train_end:
                    continue

                data = self.load_data_for_outage(auction_month, market_month, outage_date)

                if data is not None and len(data) > 0:
                    training_data_list.append(data)
                    month_data_count += 1
                    if verbose:
                        print(f"  ✓ {outage_date.strftime('%Y-%m-%d')}: {len(data):,} constraints")

            if month_data_count == 0 and verbose:
                print(f"  ⚠️  No data found for {auction_month.strftime('%Y-%m')}")

        # Combine all training data
        if len(training_data_list) > 0:
            if verbose:
                print(f"\n" + "=" * 80)
                print(f"Combining {len(training_data_list)} datasets...")

            train_data_combined = pd.concat(training_data_list, axis=0)

            if verbose:
                print(f"\n✓ Training Data Loaded Successfully")
                print(f"  Total samples: {len(train_data_combined):,}")
                print(f"  Date range: {train_data_combined['outage_date'].min().strftime('%Y-%m-%d')} "
                      f"to {train_data_combined['outage_date'].max().strftime('%Y-%m-%d')}")
                print(f"  Unique constraints: {train_data_combined.index.get_level_values('constraint_id').nunique():,}")

                # Check class distribution
                binding_count = (train_data_combined['label'] > 0).sum()
                non_binding_count = len(train_data_combined) - binding_count
                print(f"\n  Class Distribution:")
                print(f"    Binding: {binding_count:,} ({binding_count/len(train_data_combined)*100:.2f}%)")
                print(f"    Non-binding: {non_binding_count:,} ({non_binding_count/len(train_data_combined)*100:.2f}%)")

            return train_data_combined
        else:
            if verbose:
                print("\n⚠️  ERROR: No training data loaded!")
            return None

    def load_test_data_for_period(
        self,
        auction_month: pd.Timestamp,
        market_month: pd.Timestamp,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
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
            market_month,
            market_month + pd.offsets.MonthBegin(1),
            freq=self.config.outage_freq,
            inclusive='left'
        )

        if verbose:
            print(f"  Scanning for outage dates in {market_month.strftime('%Y-%m')}...")

        loaded_outage_count = 0
        for outage_date in test_outage_dates:
            data = self.load_data_for_outage(
                auction_month,
                market_month,
                outage_date
            )

            if data is not None and len(data) > 0:
                test_data_list.append(data)
                loaded_outage_count += 1
                if verbose:
                    print(f"  ✓ {outage_date.strftime('%Y-%m-%d')}: {len(data):,} constraints")

        # Combine all test data
        if len(test_data_list) > 0:
            if verbose:
                print(f"\n" + "=" * 80)
                print(f"Combining {len(test_data_list)} test datasets...")

            test_data = pd.concat(test_data_list, axis=0)

            if verbose:
                print(f"\n✓ Test Data Loaded Successfully")
                print(f"  Total samples: {len(test_data):,}")
                print(f"  Outage dates loaded: {loaded_outage_count}")
                print(f"  Date range: {test_data['outage_date'].min().strftime('%Y-%m-%d')} "
                      f"to {test_data['outage_date'].max().strftime('%Y-%m-%d')}")
                print(f"  Unique constraints: {test_data.index.get_level_values('constraint_id').nunique():,}")

                # Check class distribution
                binding_count = (test_data['label'] > 0).sum()
                non_binding_count = len(test_data) - binding_count
                print(f"\n  Class Distribution:")
                print(f"    Binding: {binding_count:,} ({binding_count/len(test_data)*100:.2f}%)")
                print(f"    Non-binding: {non_binding_count:,} ({non_binding_count/len(test_data)*100:.2f}%)")

            return test_data
        else:
            if verbose:
                print("\n⚠️  ERROR: No test data loaded!")
            return None

    def load_test_data(self, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Load test data for all configured test periods.

        Returns:
        --------
        Combined DataFrame with all test data across all periods
        """
        if verbose:
            print("\n[Loading Test Data]")
            print("=" * 80)
            print(f"Test periods: {len(self.config.test_periods)}")

        all_test_data = []

        for i, (auction_month, market_month) in enumerate(self.config.test_periods):
            if verbose and len(self.config.test_periods) > 1:
                print(f"\n--- Period {i + 1}/{len(self.config.test_periods)} ---")

            period_data = self.load_test_data_for_period(auction_month, market_month, verbose)

            if period_data is not None:
                all_test_data.append(period_data)

        # Combine all test data across periods
        if len(all_test_data) > 0:
            if len(self.config.test_periods) > 1 and verbose:
                print(f"\n" + "=" * 80)
                print(f"Combining data from {len(all_test_data)} test periods...")

            test_data = pd.concat(all_test_data, axis=0)

            if len(self.config.test_periods) > 1 and verbose:
                print(f"\n✓ All Test Data Loaded Successfully")
                print(f"  Total samples across all periods: {len(test_data):,}")
                print(f"  Date range: {test_data['outage_date'].min().strftime('%Y-%m-%d')} "
                      f"to {test_data['outage_date'].max().strftime('%Y-%m-%d')}")

            return test_data
        else:
            if verbose:
                print("\n⚠️  ERROR: No test data loaded!")
            return None
