# Multi-Period Implementation - Migration Guide

## Overview
The shadow price prediction framework has been extended to support multi-period training, enabling predictions for any `(auction_month, market_month)` pair across different period types (f0-f3, q2-q4).

## What Changed

### 1. Backup Created
All original code has been backed up to `src/shadow_price_prediction/v1/` for reference.

### 2. New Configuration (`config.py`)

#### Added Constants
- **`AUCTION_SCHEDULE`**: Maps auction months (1-12) to available period types
- **`PERIOD_MAPPING`**: Maps f-series period types to month offsets

#### Updated Features
Added three new features to `FeatureConfig`:
- **`forecast_horizon`** (int): Number of months between auction and market month
- **`season_sin`** (float): Sine component of market month cyclic encoding
- **`season_cos`** (float): Cosine component of market month cyclic encoding

### 3. Enhanced Data Loading (`data_loader.py`)

#### `load_data_for_outage`
- Added `constraint_period_type` parameter to support loading constraints from different period types
- Automatically calculates and adds `forecast_horizon`, `season_sin`, `season_cos` features

#### `load_training_data`
**Major Change**: Now implements multi-period pooling strategy:
1. Uses fixed 12-month lookback from `train_end` (ignores `train_start`)
2. For each historical auction month:
   - Looks up available period types from `AUCTION_SCHEDULE`
   - Expands period types into market months (e.g., q2 → Sep, Oct, Nov)
   - Loads data for each `(auction, market)` pair with correct constraint period type
3. Combines all samples into unified training set

#### `load_test_data_for_period`
- Already supported `(auction_month, market_month)` pairs
- Now benefits from new features added in `load_data_for_outage`

## How to Use

### Test Periods Format
Test periods are now explicitly `(auction_month, market_month)` tuples:

```python
test_periods = [
    (
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-09-01"),
    ),  # Jul auction, Sep market (q2)
    (
        pd.Timestamp("2025-08-01"),
        pd.Timestamp("2025-10-01"),
    ),  # Aug auction, Oct market (f2)
]

pipeline = ShadowPricePipeline(config)
results_per_outage, final_results, metrics = pipeline.run(test_periods=test_periods)
```

### Training Data Selection
The framework automatically:
- Looks back 12 months from each test auction month
- Loads all available period types for each historical month
- Pools data across different horizons (f0, f1, f2, f3, q2, q3, q4)
- Adds horizon and seasonality features for the model to learn from

### Model Training
Models are trained on the pooled dataset containing:
- Various forecast horizons (0-6+ months)
- Different seasons (via cyclic encoding)
- Multiple period types

The model learns to:
- Adjust predictions based on `forecast_horizon` (uncertainty increases with time)
- Capture seasonal patterns via `season_sin` and `season_cos`
- Generalize across period types

## Benefits

1. **Data Efficiency**: Maximizes training data by pooling across all period types
2. **Generalization**: Single model handles all horizons and seasons
3. **Scarcity Handling**: Can predict for rare period types (e.g., f3) by learning from abundant data (f0, f1)
4. **Flexibility**: Supports any valid `(auction, market)` pair without code changes

## Backward Compatibility

The pipeline interface remains the same:
- `test_periods` parameter still accepts list of tuples
- All existing code using `(auction_month, market_month)` pairs works without modification
- The only difference is richer training data and new features

## Next Steps

1. **Hyperparameter Tuning**: May need to retune models with new features
2. **Feature Importance**: Check if `forecast_horizon` is being used effectively
3. **Validation**: Compare performance on different horizons (f0 vs f1 vs f2)
