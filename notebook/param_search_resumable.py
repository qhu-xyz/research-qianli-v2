"""
Resumable Hyperparameter Search Script

This script demonstrates how to use the tuning_utils module for resumable
hyperparameter search with automatic duplicate detection.

Key Features:
1. Auto-loads previously searched parameters from saved metrics files
2. Ensures no duplicate parameter combinations are sampled
3. Can be interrupted and resumed without re-running previous experiments
"""

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

# ============================================================================
# 1. Define Parameter Search Space
# ============================================================================
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
    'threshold_beta': [0.1, 0.5, 1.0, 2.0, 3.0],  # 2.0 favors recall
    
    # Ensemble weights
    'clf_xgb_thres': [0.3, 0.5, 0.7],
    'reg_xgb_thres': [0.3, 0.5, 0.7],
}

# ============================================================================
# 2. Configuration
# ============================================================================
N_ITER = 10  # Total number of experiments to run (including resumed ones)
BASE_OUTPUT_DIR = Path('/opt/temp/haoyan/param_search')

# Define Test Periods
TEST_PERIODS = [
    (pd.Timestamp('2025-09-01'), pd.Timestamp('2025-09-01')),
    (pd.Timestamp('2025-10-01'), pd.Timestamp('2025-10-01'))
]

# ============================================================================
# 3. Load Previously Searched Parameters (Resumable Search)
# ============================================================================
print("=" * 80)
print("RESUMABLE HYPERPARAMETER SEARCH")
print("=" * 80)

seen_params = load_previous_params(BASE_OUTPUT_DIR)

if seen_params:
    print(f"\n✓ Resuming from previous run: {len(seen_params)} combinations already tested")
    remaining_iterations = N_ITER - len(seen_params)
    
    if remaining_iterations <= 0:
        print(f"\n✓ All {N_ITER} experiments already completed!")
        print("  To run more experiments, increase N_ITER.")
        exit(0)
    else:
        print(f"  Will run {remaining_iterations} more experiments to reach {N_ITER} total")
else:
    print("\n✓ Starting fresh search (no previous results found)")
    remaining_iterations = N_ITER

# ============================================================================
# 4. Prepare Experiment Parameters
# ============================================================================
print("\n" + "=" * 80)
print(f"PREPARING {remaining_iterations} NEW EXPERIMENTS")
print("=" * 80)

# Calculate total possible combinations
total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)

print(f"\nParameter Grid Statistics:")
print(f"  Total possible combinations: {total_combinations:,}")
print(f"  Already tested: {len(seen_params)}")
print(f"  Remaining: {total_combinations - len(seen_params):,}")
print(f"  Will sample: {remaining_iterations}")

# Sample parameters for new experiments
experiment_params = []
for i in range(remaining_iterations):
    iteration_idx = len(seen_params) + i  # Continue numbering from where we left off
    
    # This will ensure we don't repeat any previously sampled combinations
    experiment_params.append((
        iteration_idx,
        param_grid,
        TEST_PERIODS,
        BASE_OUTPUT_DIR,
        seen_params  # Pass the shared set to track across iterations
    ))

print(f"\n✓ Prepared {len(experiment_params)} experiments")

# ============================================================================
# 5. Run Parallel Execution
# ============================================================================
print("\n" + "=" * 80)
print("STARTING PARALLEL EXECUTION")
print("=" * 80)
print(f"Output Directory: {BASE_OUTPUT_DIR}")
print(f"Parallel Jobs: {min(remaining_iterations, 10)}")  # Adjust based on cluster size

results = parallel_equal_pool(
    func=run_single_experiment,
    params=experiment_params,
    n_jobs=min(remaining_iterations, 10),  # Don't use more jobs than experiments
    param_serialization='default',
    raise_error=False,  # Don't stop all if one fails
    use_tqdm=True
)

# ============================================================================
# 6. Summary
# ============================================================================
print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)

success_count = sum(1 for r in results if r['status'] == 'success')
error_count = sum(1 for r in results if r['status'] == 'failed')

print(f"\nCurrent Run:")
print(f"  Successful: {success_count}")
print(f"  Failed: {error_count}")

print(f"\nOverall Progress:")
print(f"  Total completed: {len(seen_params) + success_count} / {N_ITER}")
print(f"  Remaining: {N_ITER - len(seen_params) - success_count}")

if error_count > 0:
    print("\n" + "=" * 80)
    print("ERRORS")
    print("=" * 80)
    for r in results:
        if r['status'] == 'failed':
            print(f"\nIteration {r['iteration']}:")
            print(f"  {r.get('error', 'Unknown error')}")

# ============================================================================
# 7. Find Best Results (from all runs, including previous)
# ============================================================================
print("\n" + "=" * 80)
print("BEST RESULTS")
print("=" * 80)

# Load all metrics files to find the best overall result
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
        
        # Find best by F1 Score
        if 'F1' in combined_metrics.columns:
            best_idx = combined_metrics['F1'].idxmax()
            best_result = combined_metrics.iloc[best_idx]
            
            print("\nBest Run (by F1 Score):")
            print(f"  Iteration: {best_result.get('iteration', 'N/A')}")
            print(f"  F1 Score: {best_result['F1']:.4f}")
            if 'MAE' in best_result:
                print(f"  MAE: {best_result['MAE']:.4f}")
            if 'metrics_file' in best_result:
                print(f"  Metrics File: {best_result['metrics_file']}")
        
        # Find best by MAE (lower is better)
        if 'MAE' in combined_metrics.columns:
            best_idx = combined_metrics['MAE'].idxmin()
            best_result = combined_metrics.iloc[best_idx]
            
            print("\nBest Run (by MAE):")
            print(f"  Iteration: {best_result.get('iteration', 'N/A')}")
            print(f"  MAE: {best_result['MAE']:.4f}")
            if 'F1' in best_result:
                print(f"  F1 Score: {best_result['F1']:.4f}")
            if 'metrics_file' in best_result:
                print(f"  Metrics File: {best_result['metrics_file']}")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
