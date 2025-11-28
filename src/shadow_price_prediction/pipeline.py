"""
Main pipeline for shadow price prediction.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from pbase.utils.ray import parallel_equal_pool

from .anomaly_detection import AnomalyDetector
from .config import PredictionConfig
from .data_loader import DataLoader
from .evaluation import analyze_results
from .models import ShadowPriceModels
from .prediction import Predictor


def _process_auction_month(
    config: PredictionConfig,
    auction_month: pd.Timestamp,
    market_months: list[pd.Timestamp],
    test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    train_only: bool = False,
    verbose: bool = True,
    output_dir: str | None = None,
    refresh: bool = False,
) -> tuple[list[tuple[pd.DataFrame | None, pd.DataFrame | None, pd.Timestamp, pd.Timestamp]], Any]:
    """
    Process a single auction month: train models once, predict for all market months.

    This function is designed to be called in parallel via Ray.

    Parameters:
    -----------
    config : PredictionConfig
        Configuration for the pipeline
    auction_month : pd.Timestamp
        Auction month to process
    market_months : List[pd.Timestamp]
        List of market months to predict for this auction month
    train_only : bool
        If True, only train models and return None
    verbose : bool
        Print progress messages

    Returns:
    --------
    List of tuples, each containing:
        results_per_outage : pd.DataFrame
            Per-outage-date predictions
        final_results : pd.DataFrame
            Monthly aggregated predictions
        auction_month : pd.Timestamp
            Auction month
        market_month : pd.Timestamp
            Market month
    """
    # Import joblib here to avoid serialization issues
    import copy

    import joblib

    # Force joblib to use threading backend to avoid conflict with Ray's multiprocessing
    with joblib.parallel_backend("threading"):
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"[Processing Auction Month: {auction_month.strftime('%Y-%m')}]")
            print(f"  Market Months: {len(market_months)}")
            for mm in sorted(market_months):
                print(f"    - {mm.strftime('%Y-%m')}")
            print(f"{'=' * 80}")

        # Create auction-specific config
        period_config = copy.deepcopy(config)

        # Initialize components
        period_data_loader = DataLoader(period_config)
        period_models = ShadowPriceModels(period_config)
        period_anomaly_detector = AnomalyDetector(period_config)

        # Step 1: Calculate Training Period
        # Training ends 2 months before the auction_month (to simulate data availability)
        # We use auction_month - 1 month as the exclusive upper bound, so data is loaded up to auction_month - 2 months
        train_end = auction_month - pd.DateOffset(months=1)
        train_start = train_end - pd.offsets.MonthBegin(period_config.training.train_months_lookback)

        if verbose:
            print("\n[STEP 1: Training Period Calculation]")
            print(f"  Training Range: {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}")

        # Step 2: Load Training Data
        if verbose:
            print("\n[STEP 2: Loading Training Data]")

        # Determine required period types based on test_periods (selective loading optimization)
        required_period_types = DataLoader.get_required_period_types(test_periods)
        if verbose:
            print(f"  Required period types: {sorted(required_period_types)}")

        train_data = period_data_loader.load_training_data(
            train_start, train_end, required_period_types=required_period_types, verbose=verbose
        )

        if train_data is None or len(train_data) == 0:
            print(f"⚠️ No training data found for auction month {auction_month.strftime('%Y-%m')}. Skipping.")
            return [(None, None, auction_month, mm) for mm in market_months], None

        # Step 2.5: Scale Features - MOVED TO PER-MODEL TRAINING
        # We no longer scale globally here. Scaling happens inside train_classifiers/train_regressors
        # for each horizon group independently.
        pass

        # Step 3: Identify Test Branches (from all market months)
        if verbose:
            print("\n[STEP 3: Identifying Test Branches Across All Market Months]")

        # Load test data for all market months to get unique branches
        # OPTIMIZATION: We load, extract branches, and discard data to save memory
        all_test_branches = set()
        test_data_cache = {}

        for market_month in market_months:
            if verbose:
                print(f"  Scanning {market_month.strftime('%Y-%m')}...")

            # Load only necessary columns if possible (currently loads all)
            test_data = period_data_loader.load_test_data_for_period(auction_month, market_month, verbose=False)

            if test_data is not None and len(test_data) > 0:
                all_test_branches.update(test_data["branch_name"].unique())

            # Cache data for later use (Step 6) since we have enough memory
            if test_data is not None:
                test_data_cache[market_month] = test_data

        if len(all_test_branches) == 0:
            print(f"⚠️ No test data found for any market month in auction {auction_month.strftime('%Y-%m')}. Skipping.")
            return [(None, None, auction_month, mm) for mm in market_months], None

        if verbose:
            print(f"  Found {len(all_test_branches)} unique branches across {len(market_months)} market month(s).")

        # Step 4: Train Models (ONCE for this auction month)
        if verbose:
            print(f"\n[STEP 4: Training Models for Auction Month {auction_month.strftime('%Y-%m')}]")

        # Train classifiers
        period_models.train_classifiers(train_data, all_test_branches, test_periods, verbose)

        # Step 5: Characterize Never-Binding Branches & Train Regressors
        if verbose:
            print("\n[STEP 5: Characterizing Branches & Training Regressors]")

        # Characterize never-binding branches per horizon group
        # Only process groups that are actually needed for test_periods
        required_groups = period_models._get_required_groups(test_periods)

        horizon_filters = {
            "f0": lambda df: df[df["forecast_horizon"] == 0],
            "f1": lambda df: df[df["forecast_horizon"] == 1],
            "medium": lambda df: df[df["forecast_horizon"].between(2, 3)],
            "long": lambda df: df[df["forecast_horizon"] > 3],
        }

        for horizon_group in required_groups:
            if horizon_group in horizon_filters:
                filter_func = horizon_filters[horizon_group]
                horizon_data = filter_func(train_data)
                if len(horizon_data) > 0:
                    period_anomaly_detector.characterize_never_binding_branches(horizon_data, horizon_group, verbose)

        # Train regressors
        period_models.train_regressors(train_data, all_test_branches, test_periods, verbose)

        # Return early if train_only
        if train_only:
            if verbose:
                print(f"\n✅ Training Complete for Auction Month {auction_month.strftime('%Y-%m')}")
            return [(None, None, auction_month, mm) for mm in market_months], None

        # Step 6: Make predictions for ALL market months (using the same trained models)
        if verbose:
            print("\n[STEP 6: Making Predictions for All Market Months]")

        # Create predictor
        period_predictor = Predictor(period_config, period_models, period_anomaly_detector)

        results = []
        for market_month in sorted(market_months):
            if verbose:
                print(f"  Predicting for market month: {market_month.strftime('%Y-%m')}")

            # Check if result already exists
            if output_dir is not None and not refresh:
                output_path = Path(output_dir)
                auc_month_str = auction_month.strftime("%Y-%m")
                market_month_str = market_month.strftime("%Y-%m")
                output_file = (
                    output_path
                    / f"auction_month={auc_month_str}/market_month={market_month_str}/class_type={config.class_type}/final_results.parquet"
                )

                if output_file.exists():
                    if verbose:
                        print(f"  ⚠️  Result already exists for {market_month.strftime('%Y-%m')}. Skipping.")
                    results.append((None, None, auction_month, market_month))
                    continue

            # Reload test data (trade-off: I/O vs Memory)
            # Use cached data if available
            if market_month in test_data_cache:
                test_data = test_data_cache[market_month]
            else:
                test_data = period_data_loader.load_test_data_for_period(auction_month, market_month, verbose=False)

            if test_data is None or len(test_data) == 0:
                if verbose:
                    print(f"  ⚠️  Skipping {market_month.strftime('%Y-%m')} (no test data)")
                results.append((None, None, auction_month, market_month))
                continue

            # Scale test data using horizon-specific scaler - REMOVED
            # Scaling is now handled inside Predictor.predict() per branch/model
            # feature_cols = config.features.all_features
            # horizon_months = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)
            # scaler = period_models.get_scaler(horizon_months)
            # if scaler:
            #     test_data[feature_cols] = scaler.transform(test_data[feature_cols])

            # test_data.to_parquet(f'/opt/temp/haoyan/test_data_{market_month.strftime("%Y-%m")}.parquet')
            results_per_outage, final_results, metrics = period_predictor.predict(test_data, verbose=False)

            # Save results immediately if output_dir is provided
            if output_dir is not None:
                output_path = Path(output_dir)
                auc_month_str = auction_month.strftime("%Y-%m")
                market_month_str = market_month.strftime("%Y-%m")
                output_file = (
                    output_path
                    / f"auction_month={auc_month_str}/market_month={market_month_str}/class_type={config.class_type}/"
                )
                output_file.mkdir(parents=True, exist_ok=True)
                final_results.to_parquet(output_file / "final_results.parquet")

                if verbose:
                    print(f"  Saved final results to: {output_file}")

            results.append((results_per_outage, final_results, auction_month, market_month))

            # Free memory
            del test_data
            import gc

            gc.collect()

        if verbose:
            print(f"\n✅ Auction Month {auction_month.strftime('%Y-%m')} Complete ({len(results)} predictions)")

        return results, period_models


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
        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None

        # Store trained models by auction month
        self.trained_models: dict[pd.Timestamp, ShadowPriceModels] = {}

    def run(
        self,
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | dict[pd.Timestamp, list[pd.Timestamp]],
        class_type: str | None = None,
        train_only: bool = False,
        predict_only: bool = False,
        verbose: bool = True,
        use_parallel: bool = True,
        n_jobs: int = 0,
        save_results: bool = False,  # Deprecated but kept for compatibility
        output_dir: str | None = None,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run the complete prediction pipeline with parallel processing.

        Optimized to group test periods by auction month to avoid redundant training.
        For each unique auction month:
        1. Train models once
        2. Predict for all associated market months

        Parameters:
        -----------
        test_periods : Union[List[Tuple], Dict[Timestamp, List[Timestamp]]] (REQUIRED)
            Either:
            - List of (auction_month, market_month) tuples
            - Dictionary mapping auction_month -> list of market_months
        class_type : str, optional
            Override the class_type in the configuration (e.g. 'onpeak', 'offpeak')
        train_only : bool
            If True, only train models without prediction
        predict_only : bool
            If True, skip training (assumes models are already trained)
        verbose : bool
            Print progress messages
        use_parallel : bool
            If True, use Ray parallel processing for multiple auction months (default: True)
            If False, process auction months sequentially
        n_jobs : int
            Number of parallel workers for Ray processing (default: 0 = auto-determine)
            Only used when use_parallel=True and multiple auction months exist
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
        # Validate test_periods
        if not test_periods:
            raise ValueError("test_periods is required and cannot be None or empty")

        # Use the provided test_periods
        periods_to_process = test_periods

        if periods_to_process is None or len(periods_to_process) == 0:
            raise ValueError(
                "No test periods provided. Pass test_periods to run() or configure them in PredictionConfig."
            )

        # Create run-specific config if class_type is provided
        if class_type is not None:
            run_config = replace(self.config, class_type=class_type)
        else:
            run_config = self.config

        # Group test periods by auction month
        from collections import defaultdict

        if isinstance(periods_to_process, dict):
            # Already in dict format: {auction_month: [market_months]}
            periods_by_auction = periods_to_process
        else:
            # Convert list of tuples to dict
            periods_by_auction = defaultdict(list)
            for auction_month, market_month in periods_to_process:
                periods_by_auction[auction_month].append(market_month)

        # Convert to sorted list of (auction_month, [market_months])
        auction_month_groups = sorted(periods_by_auction.items(), key=lambda x: x[0])

        if verbose:
            print("=" * 80)
            if use_parallel and len(auction_month_groups) > 1:
                print("SHADOW PRICE PREDICTION PIPELINE - PARALLEL PER-AUCTION-MONTH MODE")
            else:
                print("SHADOW PRICE PREDICTION PIPELINE - PER-AUCTION-MONTH MODE")
            print("=" * 80)
            print(f"Total Test Periods: {len(periods_to_process)}")
            print(f"Unique Auction Months: {len(auction_month_groups)}")
            print()
            for auction_month, market_months in auction_month_groups:
                print(f"  Auction: {auction_month.strftime('%Y-%m')} → {len(market_months)} market month(s)")
                for mm in sorted(market_months):
                    print(f"    - Market: {mm.strftime('%Y-%m')}")
            print(f"Class Type: {run_config.class_type}")
            print(f"Period Type: {run_config.period_type}")
            if use_parallel and len(auction_month_groups) > 1:
                n_jobs_str = "auto-determined" if n_jobs == 0 else ("all CPUs" if n_jobs == -1 else str(n_jobs))
                print(f"\n🚀 Using Ray parallel processing for {len(auction_month_groups)} auction months")
                print(f"   Workers: {n_jobs_str}")

        if predict_only:
            raise NotImplementedError("predict_only mode not supported with per-auction-month training")

        # Prepare parameter dictionaries for parallel processing
        # Each dict contains all kwargs needed for one _process_auction_month call
        param_dict_list = [
            {
                "config": run_config,
                "auction_month": auction_month,
                "market_months": market_months,
                "test_periods": test_periods if isinstance(test_periods, list) else None,
                "train_only": train_only,
                "verbose": verbose and (not use_parallel),  # Reduce verbosity in parallel
                "output_dir": output_dir,
                "refresh": refresh,
            }
            for auction_month, market_months in auction_month_groups
        ]

        # Process auction months in parallel or sequentially
        if use_parallel and len(auction_month_groups) > 1:
            if verbose:
                print("\n" + "=" * 80)
                print("[PARALLEL PROCESSING: Training and Predicting for All Auction Months]")
                print("=" * 80)

            # Use Ray parallel_equal_pool to process all auction months in parallel
            auction_results = parallel_equal_pool(
                func=_process_auction_month,
                param_dict_list=param_dict_list,
                param_serialization="default",
                n_jobs=n_jobs,  # Configurable number of workers
                unordered=False,  # Maintain order of results
                raise_error=True,
                use_tqdm=True,
            )
        else:
            # Sequential processing (for single auction month or if parallel disabled)
            if verbose and len(auction_month_groups) > 1:
                print("\n[SEQUENTIAL PROCESSING: Processing Auction Months One by One]")

            auction_results = []
            for param_dict in param_dict_list:
                result = _process_auction_month(**param_dict)
                auction_results.append(result)

        # Return early if train_only
        if train_only:
            if verbose:
                print("\n" + "=" * 80)
                print("✅ ALL AUCTION MONTHS TRAINING COMPLETE!")
                print("=" * 80)
            return None, None, {}

        # Extract results from auction_results
        # Each auction_results item is a tuple: (results_list, period_models)
        # where results_list is a list of (results_per_outage, final_results, auction_month, market_month)
        all_results_per_outage = []
        all_final_results = []

        for auction_result_item in auction_results:
            # Handle potential wrapping (e.g. if parallel pool returns [result])
            if isinstance(auction_result_item, list | tuple) and len(auction_result_item) == 1:
                auction_result_tuple = auction_result_item[0]
            else:
                auction_result_tuple = auction_result_item

            # Unpack tuple returned by _process_auction_month
            results_list, period_models = auction_result_tuple

            # Store trained models for this auction month
            # We can get the auction month from the first result in the list, or from the period_models config if needed
            # But simpler: we know results_list contains tuples with auction_month at index 2
            if results_list and len(results_list) > 0:
                auction_month = results_list[0][2]
                self.trained_models[auction_month] = period_models

            for results_per_outage, final_results, _, _ in results_list:
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
            print("\n✓ Results Combined")
            print(f"  Total samples (per-outage): {len(combined_results_per_outage):,}")
            print(f"  Total unique constraints (monthly): {len(combined_final_results):,}")
            print(
                f"  Date range: {combined_results_per_outage['outage_date'].min().strftime('%Y-%m-%d')} "
                f"to {combined_results_per_outage['outage_date'].max().strftime('%Y-%m-%d')}"
            )

        # Save results if requested - REMOVED (Moved to _process_auction_month for immediate saving)
        # if output_dir is not None:
        #     output_path = Path(output_dir)
        #
        #     for (auction_month, market_month), gp in combined_final_results.groupby(["auction_month", "market_month"]):
        #         auc_month_str = auction_month.strftime("%Y-%m")
        #         market_month_str = market_month.strftime("%Y-%m")
        #         output_file = output_path / f"auction_month={auc_month_str}/market_month={market_month_str}/"
        #         output_file.mkdir(parents=True, exist_ok=True)
        #         gp.to_parquet(output_file / "final_results.parquet")
        #
        #     if verbose:
        #         print(f"\n[SAVING RESULTS]")
        #         print(f"  Saved final results to: {output_path}")

        # Analyze combined results
        if verbose:
            print("\n[ANALYZING COMBINED RESULTS]")

        if combined_results_per_outage["actual_shadow_price"].notna().any():
            metrics = analyze_results(combined_results_per_outage, combined_final_results, verbose)
        else:
            if verbose:
                print("  Skipping metrics calculation (no labels available)")
            metrics = {}

        if verbose:
            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETE!")
            print("=" * 80)

        return combined_results_per_outage, combined_final_results, metrics

    def predict_new_data(
        self, test_data: pd.DataFrame, verbose: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
        metrics : dict
            Evaluation metrics
        """
        if not self.models.scalers_default:
            raise ValueError("Models not trained. Run pipeline.run() first or set predict_only=False")

        return self.predictor.predict(test_data, verbose)
