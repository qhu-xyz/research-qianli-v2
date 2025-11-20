"""
Main pipeline for shadow price prediction.
"""
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from pbase.utils.ray import parallel_equal_pool

from .config import PredictionConfig
from .data_loader import DataLoader
from .models import ShadowPriceModels
from .anomaly_detection import AnomalyDetector
from .prediction import Predictor
from .evaluation import analyze_results


def _process_single_period(
    config: PredictionConfig,
    period_info: Tuple[int, pd.Timestamp, pd.Timestamp],
    train_only: bool = False,
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], int, pd.Timestamp, pd.Timestamp]:
    """
    Process a single test period: load data, train models, make predictions.

    This function is designed to be called in parallel via Ray.

    Parameters:
    -----------
    config : PredictionConfig
        Configuration for the pipeline
    period_info : Tuple[int, pd.Timestamp, pd.Timestamp]
        Tuple of (period_idx, test_auction_month, test_market_month)
    train_only : bool
        If True, only train models and return None
    verbose : bool
        Print progress messages

    Returns:
    --------
    results_per_outage : pd.DataFrame
        Per-outage-date predictions
    final_results : pd.DataFrame
        Monthly aggregated predictions
    period_idx : int
        Index of the period
    test_auction_month : pd.Timestamp
        Auction month for this period
    test_market_month : pd.Timestamp
        Market month for this period
    """
    # Import joblib here to avoid serialization issues
    import joblib
    import copy
    
    # Force joblib to use threading backend to avoid conflict with Ray's multiprocessing
    # This prevents the "resource_tracker" error: del registry[rtype][name] KeyError
    with joblib.parallel_backend('threading'):
        period_idx, test_auction_month, test_market_month = period_info
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[Processing Period {period_idx + 1}]")
            print(f"  Auction Month: {test_auction_month.strftime('%Y-%m')}")
            print(f"  Market Month:  {test_market_month.strftime('%Y-%m')}")
            print(f"{'='*80}")

        # Create a period-specific config (copy of main config)
        # We need to update the date parameters for this specific period
        period_config = copy.deepcopy(config)
        period_config.test_auction_month = test_auction_month
        period_config.test_market_month = test_market_month

        # Initialize components for this period
        period_data_loader = DataLoader(period_config)
        period_models = ShadowPriceModels(period_config)
        period_anomaly_detector = AnomalyDetector(period_config)
        period_predictor = Predictor(period_config, period_models, period_anomaly_detector)

        # Step 1: Calculate Training Period
        train_start, train_end = period_data_loader.get_training_period()
        
        if verbose:
            print(f"\n[STEP 1: Training Period Calculation]")
            print(f"  Training Range: {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}")

        # Step 2: Load Training Data
        if verbose:
            print(f"\n[STEP 2: Loading Training Data]")
        
        train_data = period_data_loader.load_training_data(train_start, train_end, verbose=verbose)
        
        if train_data is None or len(train_data) == 0:
            print(f"⚠️ No training data found for period {period_idx + 1}. Skipping.")
            return None, None, period_idx, test_auction_month, test_market_month

        # Step 2.5: Scale Features
        if verbose:
            print(f"  Scaling features...")
        
        scaler = StandardScaler()
        feature_cols = config.features.all_features
        
        # Fit and transform training data
        train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])

        # Step 3: Identify Test Branches
        # We need to know which branches are in the test set (outage period)
        # to know which specific models to train.
        # Since we are in a "simulated" environment where we might not have the future test data yet,
        # we can either:
        # A) Load the test data now (if available) to get the branches
        # B) Train on ALL branches found in training data (more expensive but safer)
        # C) Use a heuristic (e.g. top N branches)
        
        # Here we'll load the test data metadata/structure to get the branches
        # This is slightly inefficient as we might load it again for prediction, 
        # but ensures we train the right models.
        if verbose:
            print(f"\n[STEP 3: Identifying Test Branches]")

        # Load test data for the target month
        # We use the data loader's method but we need to iterate over outages in that month
        # For efficiency, let's just get the unique branches from the training data
        # and train models for all of them (or top N). 
        # Ideally, we should only train for branches that appear in the test set.
        
        # Let's load the test data to be precise
        test_data = period_data_loader.load_test_data_for_period(test_auction_month, test_market_month, verbose=False) # Don't be verbose here
        
        if test_data is None or len(test_data) == 0:
             print(f"⚠️ No test data found for period {period_idx + 1}. Skipping.")
             return None, None, period_idx, test_auction_month, test_market_month
             
        # Scale test data using the same scaler
        test_data[feature_cols] = scaler.transform(test_data[feature_cols])
             
        test_branches = set(test_data['branch_name'].unique())
        
        if verbose:
            print(f"  Found {len(test_branches)} unique branches in test set.")

        # Step 4: Train Models
        if verbose:
            print(f"\n[STEP 4: Training Models]")
        
        # Train classifiers
        period_models.train_classifiers(train_data, test_branches, verbose)
        
        # Step 5: Characterize Never-Binding Branches & Train Regressors
        if verbose:
            print(f"\n[STEP 5: Characterizing Branches & Training Regressors]")

        # Characterize never-binding branches
        period_anomaly_detector.characterize_never_binding_branches(train_data, verbose)

        # Train regressors
        period_models.train_regressors(train_data, test_branches, verbose)

        # Return early if train_only
        if train_only:
            if verbose:
                print(f"\n✅ Training Complete for Period {period_idx + 1}")
            return None, None, period_idx, test_auction_month, test_market_month

        # Step 6: Make predictions for this period
        if verbose:
            print(f"\n[STEP 6: Making Predictions for Period {period_idx + 1}]")

        # Create predictor and predict
        # We already loaded test_data in Step 3, reuse it
        results_per_outage, final_results = period_predictor.predict(test_data, verbose=verbose)
        
        if verbose:
            print(f"\n✅ Period {period_idx + 1} Complete")

        return results_per_outage, final_results, period_idx, test_auction_month, test_market_month


class ShadowPricePipeline:
    """Complete pipeline for shadow price prediction."""

    def __init__(self, config: PredictionConfig):
        """
        Initialize the pipeline.

        Parameters:
        -----------
        config : PredictionConfig
            Configuration for the pipeline
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.models = ShadowPriceModels(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.predictor = Predictor(config, self.models, self.anomaly_detector)

        # Store data for later use
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def run(
        self,
        train_only: bool = False,
        predict_only: bool = False,
        verbose: bool = True,
        use_parallel: bool = True,
        n_jobs: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run the complete prediction pipeline with parallel processing.

        For each test period (processed in parallel):
        1. Calculate training period based on that period's auction_month
        2. Train models on that period's training data
        3. Make predictions for that period
        4. Aggregate results across all periods

        Parameters:
        -----------
        train_only : bool
            If True, only train models without prediction
        predict_only : bool
            If True, skip training (assumes models are already trained)
        verbose : bool
            Print progress messages
        use_parallel : bool
            If True, use Ray parallel processing for multiple periods (default: True)
            If False, process periods sequentially
        n_jobs : int
            Number of parallel workers for Ray processing (default: 0 = auto-determine)
            Only used when use_parallel=True and multiple periods exist
            - 0: Auto-determine based on available CPUs
            - Positive integer: Use exactly this many workers
            - -1: Use all available CPUs

        Returns:
        --------
        results_per_outage : pd.DataFrame
            Per-outage-date predictions (combined across all periods)
        final_results : pd.DataFrame
            Monthly aggregated predictions (combined across all periods)
        metrics : dict
            Analysis metrics for both levels (combined across all periods)
        """
        if verbose:
            print("=" * 80)
            if use_parallel and len(self.config.test_periods) > 1:
                print("SHADOW PRICE PREDICTION PIPELINE - PARALLEL PER-PERIOD TRAINING MODE")
            else:
                print("SHADOW PRICE PREDICTION PIPELINE - PER-PERIOD TRAINING MODE")
            print("=" * 80)
            print(f"Test Periods: {len(self.config.test_periods)}")
            for i, (auction_month, market_month) in enumerate(self.config.test_periods):
                print(f"  {i + 1}. Auction: {auction_month.strftime('%Y-%m')}, Market: {market_month.strftime('%Y-%m')}")
            print(f"Class Type: {self.config.class_type}")
            print(f"Period Type: {self.config.period_type}")
            if use_parallel and len(self.config.test_periods) > 1:
                n_jobs_str = "auto-determined" if n_jobs == 0 else ("all CPUs" if n_jobs == -1 else str(n_jobs))
                print(f"\n🚀 Using Ray parallel processing for {len(self.config.test_periods)} periods")
                print(f"   Workers: {n_jobs_str}")

        if predict_only:
            raise NotImplementedError("predict_only mode not supported with per-period training")

        # Prepare parameter dictionaries for parallel processing
        # Each dict contains all kwargs needed for one _process_single_period call
        param_dict_list = [
            {
                'config': self.config,
                'period_info': (i, auction_month, market_month),
                'train_only': train_only,
                'verbose': verbose
            }
            for i, (auction_month, market_month) in enumerate(self.config.test_periods)
        ]

        # Process periods in parallel or sequentially
        if use_parallel and len(self.config.test_periods) > 1:
            if verbose:
                print("\n" + "=" * 80)
                print("[PARALLEL PROCESSING: Training and Predicting for All Periods]")
                print("=" * 80)

            # Use Ray parallel_equal_pool to process all periods in parallel
            period_results = parallel_equal_pool(
                func=_process_single_period,
                param_dict_list=param_dict_list,
                param_serialization="default",
                n_jobs=n_jobs,  # Configurable number of workers
                unordered=False,  # Maintain order of results
                raise_error=True,
                use_tqdm=True
            )
        else:
            # Sequential processing (for single period or if parallel disabled)
            if verbose and len(self.config.test_periods) > 1:
                print("\n[SEQUENTIAL PROCESSING: Processing Periods One by One]")

            period_results = []
            for param_dict in param_dict_list:
                result = _process_single_period(**param_dict)
                period_results.append(result)

        # Return early if train_only
        if train_only:
            if verbose:
                print("\n" + "=" * 80)
                print("✅ ALL PERIODS TRAINING COMPLETE!")
                print("=" * 80)
            return None, None, {}

        # Extract results from period_results
        all_results_per_outage = []
        all_final_results = []

        for results_per_outage, final_results, period_idx, auction_month, market_month in period_results:
            if results_per_outage is not None:
                all_results_per_outage.append(results_per_outage)
            if final_results is not None:
                all_final_results.append(final_results)

        # Combine results across all periods
        if verbose:
            print("\n" + "=" * 80)
            print("[COMBINING RESULTS ACROSS ALL PERIODS]")
            print("=" * 80)

        combined_results_per_outage = pd.concat(all_results_per_outage, axis=0)
        combined_final_results = pd.concat(all_final_results, axis=0)

        if verbose:
            print(f"\n✓ Results Combined")
            print(f"  Total samples (per-outage): {len(combined_results_per_outage):,}")
            print(f"  Total unique constraints (monthly): {len(combined_final_results):,}")
            print(f"  Date range: {combined_results_per_outage['outage_date'].min().strftime('%Y-%m-%d')} "
                  f"to {combined_results_per_outage['outage_date'].max().strftime('%Y-%m-%d')}")

        # Analyze combined results
        if verbose:
            print(f"\n[ANALYZING COMBINED RESULTS]")

        metrics = analyze_results(combined_results_per_outage, combined_final_results, verbose)

        if verbose:
            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETE!")
            print("=" * 80)

        return combined_results_per_outage, combined_final_results, metrics

    def predict_new_data(
        self,
        test_data: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions on new test data using already-trained models.

        Parameters:
        -----------
        test_data : pd.DataFrame
            New test data
        verbose : bool
            Print progress messages

        Returns:
        --------
        results_per_outage : pd.DataFrame
            Per-outage-date predictions
        final_results : pd.DataFrame
            Monthly aggregated predictions
        """
        if self.models.clf_default is None:
            raise ValueError("Models not trained. Run pipeline.run() first or set predict_only=False")

        return self.predictor.predict(test_data, verbose)
