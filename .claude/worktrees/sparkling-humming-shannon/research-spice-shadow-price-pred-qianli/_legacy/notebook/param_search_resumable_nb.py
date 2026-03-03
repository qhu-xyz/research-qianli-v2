# %% [markdown]
# # Pipeline Hyperparameter Tuning (Resumable, Parallel)
# 
# This notebook performs resumable hyperparameter tuning on the shadow price prediction pipeline.
# 
# **Key Features:**
# - ✓ Auto-loads previously searched parameters
# - ✓ Ensures no duplicate parameter combinations
# - ✓ Can be interrupted and resumed without re-running experiments
# - ✓ Parallel execution using Ray
# 
# **Output Structure:**
# - `/opt/temp/haoyan/param_search/`
#     - `metrics/`: Parameters + key metrics
#     - `per_outage/`: Per-outage prediction results
#     - `agg/`: Monthly aggregated results

# %% [markdown]
## Setup & Imports

# %%
from pbase.config.ray import init_ray
import shadow_price_prediction

# Initialize Ray for parallel processing
extra_modules = [shadow_price_prediction]
init_ray(address='ray://10.8.0.36:10001', extra_modules=extra_modules)

import pandas as pd
import numpy as np
import random
from pathlib import Path
from pbase.utils.ray import parallel_equal_pool

# Import tuning utilities for resumable search
from shadow_price_prediction.tuning_utils import (
    load_previous_params,
    sample_params,
    run_single_experiment
)

# %% [markdown]
## 1. Define Parameter Search Space

# %%
param_grid = {
    # XGBoost Params (shared between classifier and regressor)
    'xgb_n_estimators': [100, 200, 300],
    'xgb_max_depth': [3, 4, 5, 6],
    'xgb_learning_rate': [0.05, 0.1, 0.2],
    'xgb_subsample': [0.8, 1.0],
    'xgb_colsample_bytree': [0.8, 1.0],
    'xgb_gamma': [0, 0.1, 0.2],
    'xgb_min_child_weight': [1, 5, 10],
    'xgb_reg_alpha': [0, 0.1, 1.0],
    'xgb_reg_lambda': [1, 10, 100],
    
    # Logistic Regression Params
    'lr_penalty': ['l2'],
    'lr_C': [0.1, 1.0, 10.0],
    
    # ElasticNet Params
    'enet_alpha': [0.1, 1.0, 10.0],
    'enet_l1_ratio': [0.1, 0.5, 0.9],
    
    # Threshold Config
    'threshold_beta': [0.1, 0.5, 1.0, 2.0, 3.0],
    
    # Ensemble weights
    'clf_xgb_thres': [0.3, 0.5, 0.7],
    'reg_xgb_thres': [0.3, 0.5, 0.7],
}

# Calculate total combinations
total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
    
print(f"Total possible parameter combinations: {total_combinations:,}")

# %% [markdown]
## 2. Configuration & Resume Previous Search

# %%
# Configuration
N_ITER = 50  # Total number of experiments to run (including previous ones)
BASE_OUTPUT_DIR = Path('/opt/temp/haoyan/param_search')

# Define Test Periods
TEST_PERIODS = [
    (pd.Timestamp('2025-09-01'), pd.Timestamp('2025-09-01')),
    (pd.Timestamp('2025-10-01'), pd.Timestamp('2025-10-01'))
]

# Load previously searched parameters
seen_params = load_previous_params(BASE_OUTPUT_DIR)

if seen_params:
    print(f"✓ Resuming: {len(seen_params)} combinations already tested")
    remaining_iterations = N_ITER - len(seen_params)
    
    if remaining_iterations <= 0:
        print(f"\n✓ All {N_ITER} experiments completed!")
        print("  To run more, increase N_ITER.")
    else:
        print(f"  Will run {remaining_iterations} more experiments")
else:
    print("✓ Starting fresh search")
    remaining_iterations = N_ITER

# %% [markdown]
## 3. Prepare Experiments

# %%
if remaining_iterations > 0:
    print(f"Preparing {remaining_iterations} new experiments...")
    
    experiment_params = []
    for i in range(remaining_iterations):
        iteration_idx = len(seen_params) + i
        
        experiment_params.append((
            iteration_idx,
            param_grid,
            TEST_PERIODS,
            BASE_OUTPUT_DIR,
            seen_params
        ))
    
    print(f"✓ Prepared {len(experiment_params)} experiments")
    print(f"  Output: {BASE_OUTPUT_DIR}")

# %% [markdown]
## 4. Run Parallel Execution

# %%
if remaining_iterations > 0:
    print(f"Starting parallel execution with {min(remaining_iterations, 10)} jobs...")
    
    results = parallel_equal_pool(
        func=run_single_experiment,
        params=experiment_params,
        n_jobs=min(remaining_iterations, 10),
        param_serialization='default',
        raise_error=False,
        use_tqdm=True
    )
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nExecution Complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total completed: {len(seen_params) + success_count} / {N_ITER}")
    
    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  Iteration {r['iteration']}: {r.get('error', 'Unknown')}")

# %% [markdown]
## 5. Analyze Best Results

# %%
# Load all metrics to find best results
metrics_dir = BASE_OUTPUT_DIR / 'metrics'

if metrics_dir.exists():
    all_metrics = []
    for file in metrics_dir.glob('iter_*.parquet'):
        try:
            df = pd.read_parquet(file)
            all_metrics.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")
    
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
        print(f"Total runs analyzed: {len(combined_metrics)}")
        
        # Best by F1 Score
        if 'F1' in combined_metrics.columns:
            best_idx = combined_metrics['F1'].idxmax()
            best_result = combined_metrics.iloc[best_idx]
            
            print("\n" + "=" * 60)
            print("Best Run (by F1 Score)")
            print("=" * 60)
            print(f"Iteration: {best_result.get('iteration', 'N/A')}")
            print(f"F1 Score: {best_result['F1']:.4f}")
            if 'MAE' in best_result:
                print(f"MAE: {best_result['MAE']:.4f}")
            if 'Precision' in best_result:
                print(f"Precision: {best_result['Precision']:.4f}")
            if 'Recall' in best_result:
                print(f"Recall: {best_result['Recall']:.4f}")
            
            # Print parameters
            print("\nParameters:")
            param_cols = [col for col in combined_metrics.columns 
                         if col not in ['F1', 'Precision', 'Recall', 'MAE', 'RMSE', 'R2',
                                       'iteration', 'timestamp', 'run_id', 'metrics_file',
                                       'per_outage_file', 'agg_file']]
            for col in param_cols:
                print(f"  {col}: {best_result[col]}")
        
        # Best by MAE (lower is better)
        if 'MAE' in combined_metrics.columns:
            best_idx = combined_metrics['MAE'].idxmin()
            best_result = combined_metrics.iloc[best_idx]
            
            print("\n" + "=" * 60)
            print("Best Run (by MAE)")
            print("=" * 60)
            print(f"Iteration: {best_result.get('iteration', 'N/A')}")
            print(f"MAE: {best_result['MAE']:.4f}")
            if 'F1' in best_result:
                print(f"F1 Score: {best_result['F1']:.4f}")
            if 'RMSE' in best_result:
                print(f"RMSE: {best_result['RMSE']:.4f}")
            
            # Print parameters
            print("\nParameters:")
            for col in param_cols:
                print(f"  {col}: {best_result[col]}")

# %% [markdown]
## 6. View Results Summary

# %%
if 'combined_metrics' in locals():
    # Display summary statistics
    print("Summary Statistics:")
    print("=" * 60)
    
    metric_cols = ['F1', 'Precision', 'Recall', 'MAE', 'RMSE', 'R2']
    available_metrics = [col for col in metric_cols if col in combined_metrics.columns]
    
    summary = combined_metrics[available_metrics].describe()
    print(summary)
    
    # Optionally display the top 5 runs by F1
    if 'F1' in combined_metrics.columns:
        print("\n" + "=" * 60)
        print("Top 5 Runs by F1 Score")
        print("=" * 60)
        
        display_cols = ['iteration'] + available_metrics
        top5 = combined_metrics.nlargest(5, 'F1')[display_cols]
        print(top5.to_string(index=False))

# %%
