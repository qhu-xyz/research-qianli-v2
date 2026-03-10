# Resumable Hyperparameter Search - User Guide

## Overview

Your hyperparameter search now supports:
- ✓ **Automatic duplicate detection**: Never tests the same parameter combination twice
- ✓ **Resumable execution**: Can interrupt and resume without losing progress
- ✓ **Auto-load previous results**: Automatically scans saved metrics to find already-tested parameters

## How It Works

### 1. Parameter Tracking
- Each experiment saves its parameters + metrics to `/opt/temp/haoyan/param_search/metrics/`
- When you start a new search, `load_previous_params()` scans this folder
- It extracts all previously tested parameter combinations and stores them in `seen_params`

### 2. Duplicate Prevention
- `sample_params(param_grid, seen_params)` ensures uniqueness
- It tries up to 1000 times to find a unique combination
- If a sampled combination already exists in `seen_params`, it resamples
- Once unique, it adds the combination to `seen_params` for future checks

### 3. Resumable Search
- You can interrupt the notebook at any time (Ctrl+C or kernel interrupt)
- When you restart, it automatically loads all previous results
- Only runs the remaining experiments to reach your target `N_ITER`

## Usage in Your Notebook

### Option 1: Update Your Existing Notebook

Replace the cell that defines `sample_params()` and `update_config_with_params()` with:

```python
# Import from tuning_utils instead of defining locally
from shadow_price_prediction.tuning_utils import (
    load_previous_params,
    sample_params,
    run_single_experiment
)
```

Then, before your main parallel execution loop, add:

```python
# Configuration
N_ITER = 50
BASE_OUTPUT_DIR = Path('/opt/temp/haoyan/param_search')

# Load previously searched parameters (RESUMABLE SEARCH)
seen_params = load_previous_params(BASE_OUTPUT_DIR)

if seen_params:
    print(f"✓ Resuming: {len(seen_params)} combinations already tested")
    remaining_iterations = N_ITER - len(seen_params)
else:
    print("✓ Starting fresh search")
    remaining_iterations = N_ITER

if remaining_iterations <= 0:
    print(f"All {N_ITER} experiments completed!")
```

Update your experiment preparation loop to pass `seen_params`:

```python
experiment_params = []
for i in range(remaining_iterations):
    iteration_idx = len(seen_params) + i
    
    experiment_params.append((
        iteration_idx,      # Iteration number
        param_grid,         # Parameter grid
        BASE_OUTPUT_DIR,    # Save directory
        seen_params         # Previously seen parameters
    ))
```

### Option 2: Use the New Notebook Template

I've created two files for you:

1. **`param_search_resumable.py`**: Standalone script version
2. **`param_search_resumable_nb.py`**: Jupyter-style version with cell markers

You can copy the cells from `param_search_resumable_nb.py` into a new notebook.

## Parameter Grid Compatibility

Your current parameter grid:
```python
param_grid = {
    'xgb_clf_n_estimators': [100, 200, 300],
    'xgb_clf_max_depth': [3, 4, 5, 6],
    'xgb_clf_learning_rate': [0.05, 0.1, 0.2],
    'xgb_clf_min_child_weight': [1, 5, 10],
    'lr_C': [0.1, 1.0, 10.0],
    'xgb_reg_n_estimators': [100, 200, 300],
    'xgb_reg_max_depth': [3, 4, 5],
    'xgb_reg_learning_rate': [0.05, 0.1, 0.2],
    'enet_alpha': [0.1, 1.0, 10.0],
    'enet_l1_ratio': [0.1, 0.5, 0.9],
    'threshold_beta': [0.1, 0.5, 1.0, 2.0, 3.0],
    'clf_xgb_thres': [0.3, 0.5, 0.7],
    'reg_xgb_thres': [0.3, 0.5, 0.7],
}
```

**Total combinations**: 3 × 4 × 3 × 3 × 3 × 3 × 3 × 3 × 3 × 3 × 5 × 3 × 3 = **1,417,500** possible combinations

The `tuning_utils.py` was updated to work with this exact parameter structure.

## Example Workflow

### First Run
```python
N_ITER = 10
# Runs 10 experiments
# Saves to /opt/temp/haoyan/param_search/metrics/
```

**Output**:
```
✓ Starting fresh search
Preparing 10 new experiments...
...
Successful: 10
Total completed: 10 / 10
```

### Interrupted Run
```python
N_ITER = 50
# Runs 20 experiments
# Gets interrupted after 15 complete
```

**Output**:
```
✓ Starting fresh search
Preparing 50 new experiments...
...
[INTERRUPTED]
Successful: 15
```

### Resume
```python
N_ITER = 50
# Resumes from where it left off
```

**Output**:
```
✓ Resuming: 25 combinations already tested  # 10 from first run + 15 from interrupted
Will run 25 more experiments
Preparing 25 new experiments...
...
Successful: 25
Total completed: 50 / 50
```

## File Structure

```
/opt/temp/haoyan/param_search/
├── metrics/
│   ├── iter_0_20251119_112345_1234.parquet   # Parameters + metrics
│   ├── iter_1_20251119_112350_5678.parquet
│   └── ...
├── per_outage/
│   ├── iter_0_20251119_112345_1234.parquet   # Per-outage predictions
│   └── ...
└── agg/
    ├── iter_0_20251119_112345_1234.parquet   # Monthly aggregated results
    └── ...
```

Each `iter_*` file has the same suffix across folders for easy correlation.

## Key Functions

### `load_previous_params(save_dir)`
- Scans the `metrics/` folder
- Extracts parameter combinations from all `iter_*.parquet` files
- Returns a `set()` of tuples representing tested combinations

### `sample_params(param_grid, seen_params)`
- Randomly samples from `param_grid`
- Checks if combination exists in `seen_params`
- If duplicate, resamples (up to 1000 attempts)
- Adds new combination to `seen_params` and returns it

### `run_single_experiment(args)`
- Designed for `parallel_equal_pool`
- Takes `(iteration, param_grid, save_dir, seen_params)` as input
- Samples parameters, runs pipeline, saves results
- Returns success/failure status

## Tips

1. **Set `N_ITER` to your total target**: Not the number of *new* experiments, but the total including previous ones
2. **Check completion**: The code will tell you if all experiments are done
3. **Analyze all results**: The analysis cell loads ALL metrics files, not just the current run
4. **Clean start**: Delete `/opt/temp/haoyan/param_search/` to start completely fresh

## Troubleshooting

**Q: "Could not find unique parameters after 1000 attempts"**
- Your search space is too small or you've exhausted it
- Check: Total combinations vs. already tested
- Solution: Expand parameter grid or increase `max_retries`

**Q: Duplicate parameters still appear**
- Ensure you're passing the same `seen_params` set to all iterations
- The set should be shared across the loop, not created fresh each time

**Q: Resume doesn't work**
- Check that `/opt/temp/haoyan/param_search/metrics/` exists
- Verify that previous parquet files are readable
- Look for error messages from `load_previous_params()`
