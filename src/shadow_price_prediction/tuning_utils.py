"""
Utility functions for hyperparameter tuning with resumable search capability.
"""

import random
from pathlib import Path
from typing import Any

import pandas as pd


def load_previous_params(save_dir: Path) -> set[tuple]:
    """
    Scan the metrics folder and extract previously searched parameters.

    Args:
        save_dir: Base directory where results are saved

    Returns:
        Set of tuples representing previously searched parameter combinations
    """
    metrics_dir = save_dir / "metrics"
    seen_params: set[tuple] = set()

    if not metrics_dir.exists():
        print(f"Metrics directory {metrics_dir} does not exist. Starting fresh.")
        return seen_params

    # Find all metrics parquet files
    metric_files = list(metrics_dir.glob("iter_*.parquet"))

    if not metric_files:
        print("No previous metrics files found. Starting fresh.")
        return seen_params

    print(f"Found {len(metric_files)} previous metric files. Loading parameters...")

    for file in metric_files:
        df = pd.read_parquet(file)
        if len(df) > 0:
            # Extract parameter columns (exclude metrics)
            param_cols = [
                col
                for col in df.columns
                if col
                not in [
                    "F1",
                    "Precision",
                    "Recall",
                    "MAE",
                    "RMSE",
                    "R2",
                    "iteration",
                    "timestamp",
                    "run_id",
                    "metrics_file",
                    "per_outage_file",
                    "agg_file",
                ]
            ]

            # Get the parameter values
            params = df[param_cols].iloc[0].to_dict()

            # Convert to hashable tuple
            param_tuple = tuple(sorted(params.items()))
            seen_params.add(param_tuple)

    print(f"Loaded {len(seen_params)} previously searched parameter combinations.")
    return seen_params


def sample_params(
    param_grid: dict[str, list], seen_params: set[tuple] | None = None, max_retries: int = 1000
) -> dict[str, Any]:
    """
    Sample random parameters from the parameter grid, avoiding duplicates.

    Args:
        param_grid: Dictionary mapping parameter names to lists of possible values
        seen_params: Set of previously sampled parameter combinations (as tuples)
        max_retries: Maximum number of attempts to find a unique combination

    Returns:
        Dictionary of sampled parameters

    Raises:
        ValueError: If unable to find unique parameters after max_retries
    """
    if seen_params is None:
        seen_params = set()

    for _ in range(max_retries):
        # Sample parameters
        params = {}
        for param_name, param_values in param_grid.items():
            params[param_name] = random.choice(param_values)

        # Create a hashable representation
        param_tuple = tuple(sorted(params.items()))

        # Check if this combination has been seen before
        if param_tuple not in seen_params:
            seen_params.add(param_tuple)
            return params

    # If we get here, we couldn't find a unique combination
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    raise ValueError(
        f"Could not find unique parameters after {max_retries} attempts. "
        f"Total possible combinations: {total_combinations}, "
        f"Already sampled: {len(seen_params)}. "
        f"Consider increasing max_retries or expanding the parameter grid."
    )


def update_config_with_params(config, params):
    """
    Update the PredictionConfig with sampled hyperparameters.

    This function is designed to work with the parameter grid and should be
    called before running the pipeline.

    Args:
        config: PredictionConfig instance
        params: Dictionary of sampled parameters from sample_params

    Returns:
        Updated config
    """
    from xgboost import XGBClassifier, XGBRegressor

    from shadow_price_prediction.config import ModelConfig, ModelSpec

    # Get n_jobs from existing config to respect user setting
    # Default to 1 if not found, but try to use what's in the config
    base_n_jobs = 1
    if config.models.default_classifiers and "n_jobs" in config.models.default_classifiers[0].config.params:
        base_n_jobs = config.models.default_classifiers[0].config.params["n_jobs"]

    # --- Update XGBoost Classifier Parameters ---
    # Using unified XGBoost params
    xgb_clf_params = {
        "n_estimators": int(params["clf_xgb_n_estimators"]),
        "max_depth": int(params["clf_xgb_max_depth"]),
        "learning_rate": params["clf_xgb_learning_rate"],
        "gamma": params["clf_xgb_gamma"],
        "min_child_weight": params["clf_xgb_min_child_weight"],
        "reg_alpha": params["clf_xgb_reg_alpha"],
        "reg_lambda": params["clf_xgb_reg_lambda"],
        "random_state": 42,
        "n_jobs": base_n_jobs,
        "verbosity": 0,
        "eval_metric": "logloss",
    }

    # Update Ensemble Config (Classifiers)
    config.models.default_classifiers = [
        ModelSpec(XGBClassifier, ModelConfig(xgb_clf_params), params["clf_xgb_thres"]),
    ]
    config.models.branch_classifiers = [
        ModelSpec(XGBClassifier, ModelConfig(xgb_clf_params), params["clf_xgb_thres"]),
    ]

    # --- Update XGBoost Regressor Parameters ---
    # Using same unified XGBoost params
    xgb_reg_params = {
        "n_estimators": int(params["reg_xgb_n_estimators"]),
        "max_depth": int(params["reg_xgb_max_depth"]),
        "learning_rate": params["reg_xgb_learning_rate"],
        "gamma": params["reg_xgb_gamma"],
        "min_child_weight": params["reg_xgb_min_child_weight"],
        "reg_alpha": params["reg_xgb_reg_alpha"],
        "reg_lambda": params["reg_xgb_reg_lambda"],
        "random_state": 42,
        "n_jobs": base_n_jobs,
        "verbosity": 0,
        "objective": "reg:squarederror",
    }

    # Update Ensemble Config (Regressors)
    config.models.default_regressors = [
        ModelSpec(XGBRegressor, ModelConfig(xgb_reg_params), params["reg_xgb_thres"]),
    ]
    config.models.branch_regressors = [
        ModelSpec(XGBRegressor, ModelConfig(xgb_reg_params), params["reg_xgb_thres"]),
    ]

    # --- Update Horizon-Specific Weights ---
    # We iterate over configured horizon groups and look for corresponding weight params
    # Expected param format: "{group_name}_clf_xgb_w" and "{group_name}_reg_xgb_w"
    # Fallback to old "short_term"/"long_term" params if specific ones missing (for backward compat)

    n_classifiers = len(config.models.default_classifiers)
    n_regressors = len(config.models.default_regressors)

    for group in config.horizon_groups:
        # Classifier weights
        clf_w_param = f"{group.name}_clf_xgb_w"
        if clf_w_param in params:
            xgb_w = params[clf_w_param]
        elif group.weight_config_name == "short_term" and "short_term_clf_xgb_w" in params:
            xgb_w = params["short_term_clf_xgb_w"]
        elif group.weight_config_name == "long_term" and "long_term_clf_xgb_w" in params:
            xgb_w = params["long_term_clf_xgb_w"]
        else:
            xgb_w = 1.0 if n_classifiers == 1 else 0.5  # Default

        if n_classifiers == 1:
            config.models.clf_weights[group.name] = [1.0]
        else:
            # Distribute remaining weight equally among other models
            other_w = (1.0 - xgb_w) / (n_classifiers - 1)
            config.models.clf_weights[group.name] = [xgb_w] + [other_w] * (n_classifiers - 1)

        # Regressor weights
        reg_w_param = f"{group.name}_reg_xgb_w"
        if reg_w_param in params:
            xgb_w = params[reg_w_param]
        elif group.weight_config_name == "short_term" and "short_term_reg_xgb_w" in params:
            xgb_w = params["short_term_reg_xgb_w"]
        elif group.weight_config_name == "long_term" and "long_term_reg_xgb_w" in params:
            xgb_w = params["long_term_reg_xgb_w"]
        else:
            xgb_w = 1.0 if n_regressors == 1 else 0.5  # Default

        if n_regressors == 1:
            config.models.reg_weights[group.name] = [1.0]
        else:
            # Distribute remaining weight equally among other models
            other_w = (1.0 - xgb_w) / (n_regressors - 1)
            config.models.reg_weights[group.name] = [xgb_w] + [other_w] * (n_regressors - 1)

    # --- Update Threshold Config ---
    config.threshold.threshold_beta = params["threshold_beta"]

    if "label_threshold" in params:
        config.training.label_threshold = params["label_threshold"]

    return config


def run_single_experiment(iteration, params, test_periods, save_dir, iso_name="MISO"):
    """
    Run a single hyperparameter experiment. This function is designed to be
    called by parallel_equal_pool with PRE-SAMPLED parameters.

    Args:
        iteration: Iteration number
        params: Pre-sampled parameters (dict)
        test_periods: List of test periods for the pipeline
        save_dir: Directory to save results
        iso_name: Name of the ISO to run for (default: "MISO")

    Returns:
        Dictionary with iteration number, status, and file paths
    """
    import time

    from shadow_price_prediction.config import PredictionConfig
    from shadow_price_prediction.iso_configs import (
        MISO_HORIZON_GROUPS,
        MISO_ISO_CONFIG,
        PJM_HORIZON_GROUPS,
        PJM_ISO_CONFIG,
    )
    from shadow_price_prediction.pipeline import ShadowPricePipeline

    # Load and update config
    config = PredictionConfig()

    # Apply ISO configuration
    if iso_name == "PJM":
        config.iso = PJM_ISO_CONFIG
        config.horizon_groups = PJM_HORIZON_GROUPS
    else:
        config.iso = MISO_ISO_CONFIG
        config.horizon_groups = MISO_HORIZON_GROUPS

    config = update_config_with_params(config, params)
    config = update_config_with_params(config, params)

    # Initialize pipeline
    pipeline = ShadowPricePipeline(config)

    # Run pipeline (use_parallel=False to avoid nested Ray contexts)
    # Also set n_jobs=1 to prevent inner model training from trying to use Ray pool
    results_per_outage, final_results, metrics, _, _ = pipeline.run(
        test_periods=test_periods, verbose=False, use_parallel=False, n_jobs=1
    )

    # Generate unique ID for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = random.randint(1000, 9999)
    base_name = f"iter_{iteration}_{timestamp}_{run_id}"

    # Save per_outage results
    per_outage_dir = save_dir / "per_outage"
    per_outage_dir.mkdir(parents=True, exist_ok=True)
    per_outage_file = per_outage_dir / f"{base_name}.parquet"
    results_per_outage.to_parquet(per_outage_file)

    # Save aggregated results
    agg_dir = save_dir / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_file = agg_dir / f"{base_name}.parquet"
    final_results.to_parquet(agg_file)

    # Save metrics with parameters
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / f"{base_name}.parquet"

    # Create metrics DataFrame with parameters
    # Using ** unpacking to split params and metrics into separate columns
    metrics_data = {
        "iteration": iteration,
        "timestamp": timestamp,
        "run_id": run_id,
        **params,  # Unpack all sampled parameters as separate columns
        **(metrics["monthly"]),  # Unpack all metrics as separate columns
        "metrics_file": str(metrics_file),
        "per_outage_file": str(per_outage_file),
        "agg_file": str(agg_file),
    }
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_parquet(metrics_file)

    return {
        "iteration": iteration,
        "status": "success",
        "metrics": metrics,
        "metrics_file": str(metrics_file),
        "per_outage_file": str(per_outage_file),
        "agg_file": str(agg_file),
    }
