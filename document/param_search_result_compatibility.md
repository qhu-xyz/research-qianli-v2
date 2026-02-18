# Parameter Search Result Notebook - Multi-Period Compatibility

## Status
The `param_search_result.ipynb` notebook is **already compatible** with the multi-period implementation.

## Why No Changes Needed

1. **Results Analysis Only**: This notebook analyzes results from completed parameter search runs. It doesn't execute the training/prediction pipeline itself.

2. **Data Structure Unchanged**: The output structure (metrics, per_outage, agg files) remains the same:
   - Metrics files still contain F1, MAE, RMSE, etc.
   - Per-outage files still have the same columns (with 3 new feature columns added)
   - Aggregated files still have the same structure

3. **New Features Transparent**: The three new features (`forecast_horizon`, `season_sin`, `season_cos`) are automatically added during data loading and are included in the results, but don't break any existing analysis code.

## What Changed in the Results

If you run new parameter searches with the updated code, the results will include:

### Training Data
- More diverse training samples (pooled across period types)
- Samples from f0, f1, f2, f3, q2, q3, q4 (depending on availability)
- Each sample tagged with `forecast_horizon`, `season_sin`, `season_cos`

### Model Behavior
- Models now learn to adjust predictions based on forecast horizon
- Better generalization across different period types
- Improved handling of seasonal patterns

## Using the Notebook

The notebook works exactly as before:

```python
# Load and analyze results
metrics_dir = BASE_OUTPUT_DIR / "metrics"
all_metrics = []
for file in metrics_dir.glob("iter_*.parquet"):
    df = pd.read_parquet(file)
    all_metrics.append(df)

combined_metrics = pd.concat(all_metrics, ignore_index=True)

# Find best parameters
best_idx = combined_metrics["F1"].idxmax()
best_result = combined_metrics.iloc[best_idx]

# Create pipeline with best parameters
pipeline = create_pipeline_from_row(best_result)
```

## Note on Backward Compatibility

- Old results (from v1 code) can still be analyzed with this notebook
- New results (from multi-period code) can also be analyzed
- The only difference is that new results will have richer training data and potentially better performance
