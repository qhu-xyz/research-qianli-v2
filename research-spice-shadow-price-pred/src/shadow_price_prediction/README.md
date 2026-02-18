# Shadow Price Prediction Package

A comprehensive package for predicting shadow prices in MISO electricity markets using ensemble-based two-stage models with branch-specific training and flow anomaly detection.

## Features

- **Parallel Per-Period Training**: Uses Ray to train models for multiple test periods in parallel
- **Per-Period Training**: For each test period, separate models are trained on that period's historical data
- **Two-Stage Prediction**: Classification (binding detection) followed by regression (shadow price magnitude)
- **Ensemble Models**: Combine multiple model types (XGBoost, LogisticRegression, RandomForest, etc.) with weighted averaging
- **Branch-Specific Models**: Separate model ensembles trained for each transmission branch
- **Dynamic Threshold Optimization**: F-beta score optimization for each branch
- **Flow Anomaly Detection**: Identifies unusual flow patterns in never-binding branches
- **Simplified Model Configuration**: Single `ModelConfig` class works with any sklearn/xgboost model
- **Configurable Parameters**: All hyperparameters and settings are easily configurable
- **Comprehensive Evaluation**: Metrics calculated at both outage-date and monthly aggregated levels

## Architecture

```
shadow_price_prediction/
├── config.py              # Configuration classes
├── data_loader.py         # Data loading from SPICE files
├── models.py              # XGBoost model training
├── anomaly_detection.py   # Flow anomaly detection
├── prediction.py          # Prediction pipeline
├── evaluation.py          # Metrics calculation
└── pipeline.py            # Main orchestration
```

## Quick Start

See `notebook/example.ipynb` for a complete example.

```python
from shadow_price_prediction import (
    ShadowPricePipeline,
    PredictionConfig,
    ModelConfig,
    ModelSpec,
    EnsembleConfig,
    XGBClassifier,
    LogisticRegression,
    XGBRegressor,
    ElasticNet
)
import pandas as pd

# Create configuration with custom ensemble
config = PredictionConfig(
    test_auction_month=pd.Timestamp('2025-10'),
    test_market_month=pd.Timestamp('2025-10'),
    period_type='f0',
    class_type='onpeak',
    market_round=1
)

# Or customize the ensemble (optional)
custom_ensemble = EnsembleConfig(
    default_classifiers=[
        ModelSpec(
            model_class=XGBClassifier,
            config=ModelConfig(params={'n_estimators': 200, 'max_depth': 4}),
            weight=0.5
        ),
        ModelSpec(
            model_class=LogisticRegression,
            config=ModelConfig(params={'C': 1.0, 'max_iter': 1000}),
            weight=0.5
        )
    ]
)

config.models = custom_ensemble  # Optional: override default ensemble

# Run pipeline
pipeline = ShadowPricePipeline(config)
results_per_outage, final_results, metrics = pipeline.run()
```

## Configuration

All parameters are configurable through the `PredictionConfig` class:

### Feature Configuration
- `all_features`: List of all flow features
- `step1_features`: Features for classification stage
- `step2_features`: Features for regression stage

### Simplified Model Configuration (v2.0+)
- **Single ModelConfig Class**: Works with any sklearn/xgboost model
- **Pass Actual Model Classes**: Use `XGBClassifier`, `LogisticRegression`, etc. directly
- **Dynamic Parameters**: All model parameters via `ModelConfig(params={...})`
- **Available Models**:
  - Classification: `XGBClassifier`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
  - Regression: `XGBRegressor`, `LinearRegression`, `ElasticNet`, `RandomForestRegressor`, `GradientBoostingRegressor`
- **Default Configuration**:
  - Classifiers: XGBoost (50%) + LogisticRegression (50%)
  - Regressors: XGBoost (50%) + ElasticNet (50%)
- Separate ensembles for default and branch-specific models
- Weights automatically normalized to sum to 1.0

**Example - Adding Any Model Parameter:**
```python
ModelSpec(
    model_class=XGBClassifier,
    config=ModelConfig(params={
        'n_estimators': 200,
        'max_depth': 4,
        'subsample': 0.8,  # Add any parameter!
        'colsample_bytree': 0.8,
        'random_state': 42
    }),
    weight=0.5
)
```

### Training Configuration
- `min_samples_for_branch_model`: Minimum samples to train branch-specific classifier
- `min_binding_samples_for_regression`: Minimum binding samples for regression
- `train_months_lookback`: Number of months of training data

### Threshold Optimization
- `threshold_range_start/end/steps`: Range for threshold search
- `fbeta`: Beta parameter for F-beta score

### Anomaly Detection
- `enabled`: Enable/disable anomaly detection
- `k_multiplier`: IQR multiplier for anomaly threshold
- `flow_feature`: Feature to use for anomaly detection

## Output

The pipeline returns:

1. **results_per_outage**: DataFrame with predictions for each outage date
   - Contains individual constraint-level predictions
   - Includes all flow features, actual/predicted shadow prices, probabilities

2. **final_results**: DataFrame with monthly aggregated predictions
   - Aggregates predictions across all outage dates
   - SUM of shadow prices, MAX of binding flags

3. **metrics**: Dictionary with comprehensive metrics
   - `monthly`: Metrics for aggregated results
   - `per_outage`: Metrics for each outage date

## Metrics

Calculated for both monthly and per-outage levels:

### Regression Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

### Classification Metrics
- Precision
- Recall
- F1-Score
- Accuracy
- Confusion Matrix (TP, FP, TN, FN)

## Example Use Cases

### 1. Standard Prediction with Default Ensemble
```python
from shadow_price_prediction import ShadowPricePipeline, PredictionConfig
import pandas as pd

config = PredictionConfig(
    test_auction_month=pd.Timestamp('2025-10'),
    test_market_month=pd.Timestamp('2025-10'),
    period_type='f0',
    class_type='onpeak',
    market_round=1
)

pipeline = ShadowPricePipeline(config)
results_per_outage, final_results, metrics = pipeline.run()
```

### 1b. Custom Model Ensemble (Simplified v2.0 API)
```python
from shadow_price_prediction import (
    ShadowPricePipeline, PredictionConfig,
    ModelConfig, ModelSpec, EnsembleConfig,
    XGBClassifier, RandomForestClassifier,
    XGBRegressor, ElasticNet
)
import pandas as pd

# Create custom ensemble with any models and parameters
custom_ensemble = EnsembleConfig(
    default_classifiers=[
        ModelSpec(
            model_class=XGBClassifier,
            config=ModelConfig(params={
                'n_estimators': 200,
                'max_depth': 4,
                'subsample': 0.8,
                'random_state': 42
            }),
            weight=0.6
        ),
        ModelSpec(
            model_class=RandomForestClassifier,
            config=ModelConfig(params={
                'n_estimators': 100,
                'max_depth': 10,
                'criterion': 'entropy',
                'random_state': 42
            }),
            weight=0.4
        )
    ],
    default_regressors=[
        ModelSpec(
            model_class=XGBRegressor,
            config=ModelConfig(params={'n_estimators': 200}),
            weight=0.7
        ),
        ModelSpec(
            model_class=ElasticNet,
            config=ModelConfig(params={'alpha': 1.0, 'l1_ratio': 0.5}),
            weight=0.3
        )
    ]
)

config = PredictionConfig(
    test_auction_month=pd.Timestamp('2025-10'),
    test_market_month=pd.Timestamp('2025-10'),
    period_type='f0',
    class_type='onpeak',
    market_round=1,
    models=custom_ensemble  # Use custom ensemble
)

pipeline = ShadowPricePipeline(config)
results_per_outage, final_results, metrics = pipeline.run()
```

### 2. Batch Processing (Multiple Periods with Parallel Training)
```python
# Process multiple test periods in parallel using Ray
# IMPORTANT: For each period (processed in PARALLEL), the pipeline will:
#   1. Calculate training period based on that period's auction_month
#   2. Train NEW models on that period's training data
#   3. Make predictions for that period
#   4. Combine results across all periods

test_periods = [
    (pd.Timestamp('2025-10'), pd.Timestamp('2025-10')),
    (pd.Timestamp('2025-11'), pd.Timestamp('2025-11')),
    (pd.Timestamp('2025-12'), pd.Timestamp('2025-12')),
]

config = PredictionConfig(
    test_periods=test_periods,  # Each period gets its own training data and models
    period_type='f0',
    class_type='onpeak',
    market_round=1
)

pipeline = ShadowPricePipeline(config)

# Run with parallel processing (default, auto-determine workers)
results_per_outage, final_results, metrics = pipeline.run(use_parallel=True)

# Control number of parallel workers with n_jobs parameter:
# - n_jobs=0 (default): Auto-determine based on available CPUs
# - n_jobs=3: Use exactly 3 workers (useful for 3 test periods)
# - n_jobs=-1: Use all available CPUs
results_per_outage, final_results, metrics = pipeline.run(use_parallel=True, n_jobs=3)

# Or run sequentially if needed
# results_per_outage, final_results, metrics = pipeline.run(use_parallel=False)

# Example training periods (all processed in PARALLEL):
# Period 1 (2025-10): trains on 2024-10 to 2025-09 (12 months) [Worker 1]
# Period 2 (2025-11): trains on 2024-11 to 2025-10 (12 months) [Worker 2]
# Period 3 (2025-12): trains on 2024-12 to 2025-11 (12 months) [Worker 3]

# Benefits of parallel processing:
# - 3 periods process simultaneously instead of sequentially
# - Significant speedup for multiple periods (3x faster for 3 periods on 3+ cores)
# - Each period uses independent Ray worker
# - Control resource usage with n_jobs parameter
```

### 3. Train Only (Save Models for Later)
```python
pipeline = ShadowPricePipeline(config)
pipeline.run(train_only=True)
```

### 4. Predict on New Data (Using Trained Models)
```python
pipeline = ShadowPricePipeline(config)
pipeline.run()  # Train models

# Later, with new test data
new_results_per_outage, new_final_results = pipeline.predict_new_data(new_test_data)
```

### 5. Custom Analysis
```python
from shadow_price_prediction import calculate_metrics, print_metrics_report

# Calculate custom metrics
custom_metrics = calculate_metrics(results_df)

# Print formatted report
print_metrics_report(custom_metrics, title="Custom Analysis")
```

## Dependencies

- pandas
- numpy
- xgboost
- scikit-learn
- pbase (MISO analysis tools)

## Version History

### 2.0.0 (Current)
**Major API Simplification, Per-Period Training & Parallel Processing**
- **Parallel Processing**: Uses Ray's `parallel_equal_pool` to train models for multiple test periods simultaneously
  - Significant speedup for batch processing (near-linear scaling with available cores)
  - Controlled via `use_parallel` parameter (default: True)
  - Configurable worker count via `n_jobs` parameter (0=auto, -1=all CPUs, or specific number)
  - Falls back to sequential processing for single period or when disabled
- **Per-Period Training**: For each test period in `test_periods`, the pipeline now:
  - Calculates training period based on that period's `auction_month`
  - Trains separate models on that period's historical data
  - Makes predictions for that period
  - Combines results across all periods
- Replaced multiple model config classes with single `ModelConfig` class
- Pass actual model classes (`XGBClassifier`, `LogisticRegression`, etc.) instead of string identifiers
- All model parameters are now dynamic via `ModelConfig(params={...})`
- Simplified `create_model()` function - no more string-based factory logic
- **Breaking Change**: Old config classes (`XGBoostClassifierConfig`, etc.) removed
- **Breaking Change**: Training period now calculated per test period (not once for all periods)
- **Breaking Change**: `predict_only` mode not supported with per-period training
- **Migration**: Replace `model_type='xgboost'` with `model_class=XGBClassifier`

### 1.0.0
Initial release with:
- Two-stage prediction (classification + regression)
- Ensemble models with weighted averaging
- Branch-specific models with dynamic thresholds
- Flow anomaly detection
- Separate config classes for each model type
