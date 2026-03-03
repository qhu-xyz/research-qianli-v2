"""
Main pipeline for shadow price prediction.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

# Set OMP_NUM_THREADS to 1 to avoid CPU thrashing when using Ray parallelism
# This must be done before importing numpy/pandas/xgboost in some cases
import pandas as pd

from pbase.utils.ray import parallel_equal_pool

from .anomaly_detection import AnomalyDetector
from .config import PredictionConfig
from .data_loader import DataLoader
from .evaluation import evaluate_split, aggregate_eval_metrics
from .models import ShadowPriceModels
from .prediction import Predictor


def _process_auction_month(
    config: PredictionConfig,
    auction_month: pd.Timestamp,
    market_months: list[pd.Timestamp],
    train_only: bool = False,
    verbose: bool = True,
    output_dir: str | None = None,
    refresh: bool = False,
    branch_name: str | None = None,
    # data_cache: dict | None = None, # Removed
) -> tuple[
    list[tuple[dict[Any, Any] | None, pd.DataFrame | None, Any, Any]],
    ShadowPriceModels | None,
    AnomalyDetector | None,
    pd.DataFrame | None,
    pd.Timestamp | None,
    dict,
]:
    # Import joblib here to avoid serialization issues
    import copy

    # If data_cache is a Ray ObjectRef, get it
    # if isinstance(data_cache, ray.ObjectRef):
    #     data_cache = ray.get(data_cache)

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
    output_dir : str, optional
        Directory to save intermediate results
    refresh : bool
        If True, force re-computation even if results exist
    branch_name : Optional[str]
        If provided, only process branches containing this string.
    n_jobs : int
        Number of parallel jobs for model training.

    Returns:
    --------
    Tuple containing:
        results : list of tuples (results_per_outage, final_results, auction_month, market_month)
        period_models : ShadowPriceModels
        period_anomaly_detector : AnomalyDetector
        train_data : pd.DataFrame
        auction_month : pd.Timestamp
    """
    # Import joblib here to avoid serialization issues
    import time

    start_time = time.time()

    if verbose:
        print(f"\n{'=' * 80}", flush=True)
        print(f"[Processing Auction Month: {auction_month.strftime('%Y-%m')}]", flush=True)
        print(f"  Market Months: {len(market_months)}")
        for mm in sorted(market_months):
            print(f"    - {mm.strftime('%Y-%m')}")
        if branch_name:
            print(f"  Filtering for branch: {branch_name}")
        print(f"{'=' * 80}")

    # Create auction-specific config
    period_config = copy.deepcopy(config)

    # Initialize components
    from .data_loader import BaseDataLoader

    period_data_loader: BaseDataLoader
    if config.iso.name == "PJM":
        from .data_loader import PjmDataLoader

        period_data_loader = PjmDataLoader(period_config)
    else:
        from .data_loader import MisoDataLoader

        period_data_loader = MisoDataLoader(period_config)

    period_models = ShadowPriceModels(period_config)
    period_anomaly_detector = AnomalyDetector(period_config)

    # Derive current test periods for this auction month
    current_test_periods = [(auction_month, mm) for mm in market_months]

    # Step 1: Calculate Training Period
    # Training ends 2 months before the auction_month (to simulate data availability)
    # We use auction_month - 1 month as the exclusive upper bound, so data is loaded up to auction_month - 2 months
    train_end = auction_month - pd.DateOffset(months=1)
    train_start = auction_month - pd.offsets.MonthBegin(period_config.training.train_months_lookback)

    if verbose:
        print("\n[STEP 1: Training Period Calculation]")
        print(f"  Training Range: {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}")

    # Step 2: Load Training Data
    if verbose:
        print("\n[STEP 2: Loading Training Data]")

    train_data = period_data_loader.load_training_data(
        train_start,
        train_end,
        # Determine required period types based on CURRENT auction/market months
        # This avoids loading unnecessary data types (e.g. long-term) if we only need short-term
        required_period_types=period_data_loader.get_required_period_types(
            [(auction_month, mm) for mm in market_months]
        ),
        branch_name=branch_name,
        verbose=verbose,
        # data_cache=data_cache # Removed
    )

    if train_data is None or len(train_data) == 0:
        print(f"⚠️ No training data found for auction month {auction_month.strftime('%Y-%m')}. Skipping.")
        return [(None, None, auction_month, mm) for mm in market_months], None, None, None, auction_month, {}

    # Apply label modification logic (before split so both fit and val get it)
    # If feature value is below threshold, set label to 0
    # Save original label in label_ori
    mod_rule = config.training.label_modification_rule
    if mod_rule is not None:
        feat_name, threshold = mod_rule
        if feat_name in train_data.columns:
            if verbose:
                print(f"  Applying label modification based on {feat_name} < {threshold}...")
            train_data["label_ori"] = train_data["label"]
            mask_zero_prob = train_data[feat_name] < threshold
            n_modified = mask_zero_prob.sum()
            if n_modified > 0:
                train_data.loc[mask_zero_prob, "label"] = 0
                if verbose:
                    print(f"    Modified {n_modified} labels to 0 where {feat_name} < {threshold}.")
        else:
            if verbose:
                print(f"  ⚠️ {feat_name} not found in training data. Skipping label modification.")

    # Split into fit / val / holdout based on time (6/3/3 split)
    val_boundary = train_start + pd.DateOffset(months=period_config.training.train_months)
    holdout_boundary = val_boundary + pd.DateOffset(months=period_config.training.val_months)

    fit_data = train_data[train_data["auction_month"] < val_boundary].copy()
    val_data = train_data[
        (train_data["auction_month"] >= val_boundary) & (train_data["auction_month"] < holdout_boundary)
    ].copy()
    # holdout_data (last 3 months) is not used during training — reserved for evaluation

    # Ensure all expected feature columns exist (e.g., season_hist_da_3 may be
    # absent if historical data only covers 2 planning years)
    for feat_name in period_config.features.all_features:
        for df in (fit_data, val_data, train_data):
            if feat_name not in df.columns:
                df[feat_name] = 0.0

    if verbose:
        print(f"\n[DATA SPLIT]")
        print(f"  Fit:     {len(fit_data):>8,} rows  ({train_start.strftime('%Y-%m')} to {(val_boundary - pd.DateOffset(months=1)).strftime('%Y-%m')})")
        print(f"  Val:     {len(val_data):>8,} rows  ({val_boundary.strftime('%Y-%m')} to {(holdout_boundary - pd.DateOffset(months=1)).strftime('%Y-%m')})")
        holdout_data_count = len(train_data) - len(fit_data) - len(val_data)
        print(f"  Holdout: {holdout_data_count:>8,} rows  ({holdout_boundary.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')})")

    # Step 3: Identify Test Branches (from all market months)
    if verbose:
        print("\n[STEP 3: Identifying Target Branches]")

    # Identify all unique (branch_name, flow_direction) pairs in the test set (future market months)
    # We need to know which branches to train models for
    if verbose:
        print("  Identifying target branches from test periods...")

    all_test_branches = set()
    test_data_by_month = {}

    # Load test data for all periods at once
    all_test_data = period_data_loader.load_test_data(
        test_periods=[(auction_month, mm) for mm in market_months],
        branch_name=branch_name,
        verbose=False,
    )

    if all_test_data is not None and not all_test_data.empty:
        # Populate test_data_by_month and all_test_branches
        for market_month, month_df in all_test_data.groupby("market_month"):
            test_data_by_month[market_month] = month_df

            # Filter branch generation based on test_unbind_rule
            # So that we don't train models for branches that are fully unbind
            df_for_branches = month_df
            mod_rule = config.training.test_unbind_rule
            if mod_rule is not None:
                feat_name, threshold = mod_rule
                if feat_name in df_for_branches.columns:
                    df_for_branches = df_for_branches[df_for_branches[feat_name] >= threshold]

            if not df_for_branches.empty:
                pairs = (
                    df_for_branches[["branch_name", "flow_direction"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                all_test_branches.update(pairs)

    if verbose:
        print(f"  Found {len(all_test_branches)} unique branch-flow pairs in test set.")

    # Step 4: Train Models (using fit_data for training, val_data for threshold optimization)
    if verbose:
        print(f"\n[STEP 4: Training Models for Auction Month {auction_month.strftime('%Y-%m')}]")
    # Train classifiers
    period_models.train_classifiers(fit_data, all_test_branches, current_test_periods, verbose, val_data=val_data)

    # Step 5: Characterize Never-Binding Branches & Train Regressors
    if verbose:
        print(f"\n[STEP 5: Training Regressors for Auction Month {auction_month.strftime('%Y-%m')}]")

    # Characterize never-binding branches per horizon group
    # Only process groups that are actually needed for test_periods
    required_groups = period_models._get_required_groups(current_test_periods)

    horizon_filters = {}
    for group in config.horizon_groups:
        # Capture group values in closure
        def make_filter(min_h, max_h):
            return lambda df: df[(df["forecast_horizon"] >= min_h) & (df["forecast_horizon"] <= max_h)]

        horizon_filters[group.name] = make_filter(group.min_horizon, group.max_horizon)

    for horizon_group in required_groups:
        if horizon_group in horizon_filters:
            filter_func = horizon_filters[horizon_group]
            horizon_data = filter_func(fit_data)
            if len(horizon_data) > 0:
                period_anomaly_detector.characterize_never_binding_branches(horizon_data, horizon_group, verbose)

    # Train regressors (on fit_data only — no val_data needed for regressors)
    period_models.train_regressors(fit_data, all_test_branches, current_test_periods, verbose)

    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"[Timing Training] _process_auction_month training for {auction_month.strftime('%Y-%m')} took {duration:.2f} seconds")

    # Step 5b: Evaluate on val and holdout splits
    eval_metrics: dict = {}
    eval_predictor = Predictor(period_config, period_models, period_anomaly_detector)

    if len(val_data) > 0:
        eval_metrics["val"] = evaluate_split(
            eval_predictor, val_data,
            f'VAL — {auction_month.strftime("%Y-%m")}',
            beta=2.0, verbose=verbose,
        )

    holdout_data = train_data[train_data["auction_month"] >= holdout_boundary].copy()
    if len(holdout_data) > 0:
        eval_metrics["holdout"] = evaluate_split(
            eval_predictor, holdout_data,
            f'HOLDOUT — {auction_month.strftime("%Y-%m")}',
            beta=2.0, verbose=verbose,
        )
    del holdout_data

    if train_only:
        if verbose:
            print(f"  Training complete for {auction_month.strftime('%Y-%m')}. Skipping prediction.")
        return [], period_models, period_anomaly_detector, train_data, auction_month, eval_metrics

    # Step 6: Make Predictions
    if verbose:
        print(f"\n[STEP 6: Making Predictions for Auction Month {auction_month.strftime('%Y-%m')}]")

    # Initialize predictor for this period
    period_predictor = Predictor(period_config, period_models, period_anomaly_detector)

    results: list[tuple[dict[Any, Any] | None, pd.DataFrame | None, Any, Any]] = []
    for market_month in sorted(market_months):
        if verbose:
            print(f"  Predicting for Market Month: {market_month.strftime('%Y-%m')}")

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

        test_data = test_data_by_month[market_month]
        results_per_outage, final_results, metrics = period_predictor.predict(test_data, verbose=False)

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

    end_time = time.time()
    duration = end_time - start_time
    print(f"[Timing] _process_auction_month for {auction_month.strftime('%Y-%m')} took {duration:.2f} seconds")

    return results, period_models, period_anomaly_detector, train_data, auction_month, eval_metrics


def extract_train_provenance(
    train_data: pd.DataFrame,
    auction_month: pd.Timestamp,
    class_type: str,
    feature_cols: list[str],
) -> dict:
    """Build training data provenance metadata for train_manifest.json.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data used for this run (before split).
    auction_month : pd.Timestamp
        The auction month this training data corresponds to.
    class_type : str
        e.g. 'onpeak' or 'offpeak'.
    feature_cols : list[str]
        Feature column names to compute summary stats for.

    Returns
    -------
    dict
        Provenance metadata including row counts, binding rate, date range,
        and per-feature summary statistics.
    """
    import numpy as np

    n_total = len(train_data)
    n_binding = int((train_data["label"] > 0).sum()) if "label" in train_data.columns else 0
    binding_rate = n_binding / n_total if n_total > 0 else 0.0

    # Date range
    if "auction_month" in train_data.columns:
        am_col = train_data["auction_month"]
        date_range = [str(am_col.min()), str(am_col.max())]
    else:
        date_range = ["unknown", "unknown"]

    # Feature stats (mean, std, min, max)
    feature_stats = {}
    for col in feature_cols:
        if col in train_data.columns:
            s = train_data[col]
            feature_stats[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
            }

    return {
        "auction_month": auction_month.strftime("%Y-%m"),
        "class_type": class_type,
        "training_date_range": date_range,
        "n_rows_total": n_total,
        "n_binding": n_binding,
        "binding_rate": round(binding_rate, 4),
        "feature_stats": feature_stats,
    }


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
        self.train_data: dict[pd.Timestamp, pd.DataFrame] = {}
        self.test_data: pd.DataFrame | None = None

        # Store trained models by auction month
        self.trained_models: dict[pd.Timestamp, ShadowPriceModels] = {}
        self.trained_predictors: dict[pd.Timestamp, Predictor] = {}

    def run(
        self,
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | dict[pd.Timestamp, list[pd.Timestamp]],
        class_type: str | None = None,
        train_only: bool = False,
        predict_only: bool = False,
        verbose: bool = True,
        use_parallel: bool = True,
        n_jobs: int = 0,
        output_dir: str | None = None,
        refresh: bool = False,
        branch_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, Predictor]]:
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
            If True, use Ray parallel processing for model training (branch-level).
            Auction months are always processed sequentially to avoid nested parallelism.
        n_jobs : int
            Number of parallel workers for Ray processing (default: 0 = auto-determine)
            Only used when use_parallel=True.
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

        # If parallel is enabled, we want to maximize parallelism by treating each (auction, market) pair as a separate task
        # This allows us to use n_jobs to parallelize across market months even within the same auction month
        if use_parallel:
            # Flatten to individual (auction_month, [market_month]) items
            if isinstance(periods_to_process, dict):
                auction_month_groups = []
                for auction_month, market_months in periods_to_process.items():
                    for mm in market_months:
                        auction_month_groups.append((auction_month, [mm]))
            else:
                auction_month_groups = [(am, [mm]) for am, mm in periods_to_process]

            # Sort for consistent order
            auction_month_groups.sort(key=lambda x: (x[0], x[1][0]))

        else:
            # Sequential mode: Group by auction month to optimize training (train once per auction month)
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
            # print(f"Period Type: {run_config.period_type}")  # Dynamic per market month
            if use_parallel and len(auction_month_groups) > 1:
                n_jobs_str = "auto-determined" if n_jobs == 0 else ("all CPUs" if n_jobs == -1 else str(n_jobs))
                print(f"\n🚀 Using Ray parallel processing for {len(auction_month_groups)} auction months")
                print(f"   Workers: {n_jobs_str}")

        if predict_only:
            raise NotImplementedError("predict_only mode not supported with per-auction-month training")

        # Prepare parameter dictionaries for parallel processing
        # Each dict contains all kwargs needed for one _process_auction_month call

        # Pre-load data into shared cache if parallel
        # Prepare parameter dictionaries for parallel processing
        # Each dict contains all kwargs needed for one _process_auction_month call

        param_dict_list = [
            {
                "config": run_config,
                "auction_month": auction_month,
                "market_months": market_months,
                "train_only": train_only,
                "verbose": verbose,  # Reduce verbosity in parallel
                "output_dir": output_dir,
                "refresh": refresh,
                "branch_name": branch_name,
                # "data_cache": None # Cache removed as per user request
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
                unordered=True,  # Maintain order of results
                raise_error=True,
                use_tqdm=True,
            )
        else:
            # Sequential processing (for single auction month or if parallel disabled)
            if verbose and len(auction_month_groups) > 1:
                print("\n[SEQUENTIAL PROCESSING: Processing Auction Months One by One]")

            auction_results = []
            for _i, param_dict in enumerate(param_dict_list):
                result = _process_auction_month(**param_dict)
                auction_results.append(result)

        if verbose:
            print("\n" + "=" * 80)
            print("✅ ALL AUCTION MONTHS TRAINING COMPLETE!")
            print("=" * 80)
        # Return early if train_only (but still collect eval metrics)
        if train_only:
            train_only_eval: list[tuple[pd.Timestamp, dict]] = []
            for auction_result_item in auction_results:
                if isinstance(auction_result_item, list | tuple) and len(auction_result_item) == 1:
                    auction_result_tuple = auction_result_item[0]
                else:
                    auction_result_tuple = auction_result_item
                _, period_models, period_anomaly_detector, _, auction_month, eval_metrics = auction_result_tuple
                if period_models is not None:
                    self.trained_models[auction_month] = period_models
                    self.trained_predictors[auction_month] = Predictor(
                        self.config, period_models, period_anomaly_detector
                    )
                if eval_metrics:
                    train_only_eval.append((auction_month, eval_metrics))
            metrics = aggregate_eval_metrics(train_only_eval, verbose=verbose)
            return None, None, metrics, None, self.trained_predictors

        # Extract results from auction_results
        # Each auction_results item is a tuple: (results_list, period_models, train_data)
        # where results_list is a list of (results_per_outage, final_results, auction_month, market_month)
        all_results_per_outage = []
        all_final_results = []
        all_train_data = []
        all_eval_metrics: list[tuple[pd.Timestamp, dict]] = []

        for auction_result_item in auction_results:
            # Handle potential wrapping (e.g. if parallel pool returns [result])
            if isinstance(auction_result_item, list | tuple) and len(auction_result_item) == 1:
                auction_result_tuple = auction_result_item[0]
            else:
                auction_result_tuple = auction_result_item

            # Unpack tuple returned by _process_auction_month
            results_list, period_models, period_anomaly_detector, train_data, auction_month, eval_metrics = auction_result_tuple

            if eval_metrics:
                all_eval_metrics.append((auction_month, eval_metrics))

            # Store trained models for this auction month
            if period_models is not None:
                self.trained_models[auction_month] = period_models

                # Create and store predictor
                predictor = Predictor(self.config, period_models, period_anomaly_detector)
                self.trained_predictors[auction_month] = predictor

            if train_data is not None:
                # self.train_data[auction_month] = train_data
                all_train_data.append(
                    train_data
                )  # Keep for logging stats if needed, or remove if memory concern. Used below for combined stats.

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

        combined_results_per_outage = (
            pd.concat(all_results_per_outage, axis=0) if all_results_per_outage else pd.DataFrame()
        )
        combined_final_results = pd.concat(all_final_results, axis=0) if all_final_results else pd.DataFrame()
        # combined_train_data is no longer returned, but we used it for stats.
        # Let's calculate total length from dict.
        total_train_samples = sum(len(df) for df in self.train_data.values())

        if verbose:
            print("\n✓ Results Combined")
            print(f"  Total samples (per-outage): {len(combined_results_per_outage):,}")
            print(f"  Total unique constraints (monthly): {len(combined_final_results):,}")
            if not combined_results_per_outage.empty:
                print(
                    f"  Date range: {combined_results_per_outage['outage_date'].min().strftime('%Y-%m-%d')} "
                    f"to {combined_results_per_outage['outage_date'].max().strftime('%Y-%m-%d')}"
                )
            if total_train_samples > 0:
                print(
                    f"  Total training samples: {total_train_samples:,} (across {len(self.train_data)} auction months)"
                )

        # Aggregate val/holdout evaluation metrics across auction months
        metrics = aggregate_eval_metrics(all_eval_metrics, verbose=verbose)

        if verbose:
            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETE!")
            print("=" * 80)

        return (
            combined_results_per_outage,
            combined_final_results,
            metrics,
            None,
            self.trained_predictors,
        )

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
        if not self.models.scalers_default_clf:
            raise ValueError("Models not trained. Run pipeline.run() first or set predict_only=False")

        return self.predictor.predict(test_data, verbose)
