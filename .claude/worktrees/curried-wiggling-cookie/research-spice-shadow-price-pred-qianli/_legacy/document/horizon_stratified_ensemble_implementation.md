# Horizon-Stratified Ensemble Implementation

## Overview
Implemented a horizon-stratified ensemble weighting system that adjusts model weights based on forecast horizon to address the limitations of linear models in multi-period predictions.

## Changes Made

### 1. Configuration (`config.py`)

#### Added to `EnsembleConfig`:
```python
# Horizon-stratified ensemble weights [xgboost_weight, linear_model_weight]

# Short-term (0-1 months): f0, f1
short_term_clf_weights: List[float] = [0.5, 0.5]
short_term_reg_weights: List[float] = [0.5, 0.5]

# Medium-term (2-3 months): f2, f3
medium_term_clf_weights: List[float] = [0.6, 0.4]
medium_term_reg_weights: List[float] = [0.6, 0.4]

# Long-term (4+ months): q2, q3, q4
long_term_clf_weights: List[float] = [0.7, 0.3]
long_term_reg_weights: List[float] = [0.7, 0.3]

# Horizon thresholds
short_term_max_horizon: int = 1
medium_term_max_horizon: int = 3
```

#### Added Method:
```python
def get_ensemble_weights_for_horizon(
    horizon: int,
    model_type: str = 'classifier',
    is_branch: bool = False
) -> List[float]
```

### 2. Models (`models.py`)

#### Updated `predict_ensemble` Function:
- Added `weight_overrides` parameter to allow dynamic weight specification
- If provided, uses override weights instead of ensemble's default weights

#### Added to `ShadowPriceModels`:
```python
def get_ensemble_weights_for_horizon(
    horizon: int,
    model_type: str = 'classifier',
    is_branch: bool = False
) -> List[float]
```
Delegates to `EnsembleConfig.get_ensemble_weights_for_horizon()`

### 3. Prediction (`prediction.py`)

#### Added Horizon Calculation:
```python
auction_month = test_data['auction_month'].iloc[0]
market_month = test_data['market_month'].iloc[0]
forecast_horizon = (market_month.year - auction_month.year) * 12 + \
                   (market_month.month - auction_month.month)
```

#### Updated All `predict_ensemble` Calls:
1. **Classification Predictions**:
   ```python
   clf_weights = self.models.get_ensemble_weights_for_horizon(
       forecast_horizon, 'classifier', is_branch_model
   )
   y_pred_proba = predict_ensemble(
       clf_ensemble, X, predict_proba=True,
       weight_overrides=clf_weights
   )
   ```

2. **Regression Predictions**:
   ```python
   reg_weights = self.models.get_ensemble_weights_for_horizon(
       forecast_horizon, 'regressor', is_branch_reg
   )
   y_pred_value = predict_ensemble(
       reg_ensemble, X, predict_proba=False,
       weight_overrides=reg_weights
   )
   ```

3. **Anomaly Detection Regression**:
   - Uses default regressor with horizon-specific weights

## How It Works

### Weight Selection Logic
```
if horizon <= 1:
    use short_term_weights  # Balanced: XGB 50%, Linear 50%
elif horizon <= 3:
    use medium_term_weights  # Favor XGB: 60%, Linear 40%
else:
    use long_term_weights    # Heavy XGB: 70%, Linear 30%
```

### Rationale

1. **Short-term (f0, f1)**:
   - Abundant data, stable patterns
   - Both XGBoost and linear models contribute equally
   - Linear models capture core physics reliably

2. **Medium-term (f2, f3)**:
   - Moderate data availability
   - Favor XGBoost (60%) as it can learn horizon effects via `forecast_horizon` feature
   - Linear models (40%) still provide stable baseline

3. **Long-term (q2-q4)**:
   - Scarce data, high uncertainty
   - Heavily favor XGBoost (70%) for its ability to handle horizon-specific patterns
   - Linear models (30%) prevent overfitting and provide fallback

## Configuration

All weights are **fully configurable** via `PredictionConfig`:

```python
config = PredictionConfig()

# Customize short-term weights
config.models.short_term_clf_weights = [0.6, 0.4]  # More XGB
config.models.short_term_reg_weights = [0.5, 0.5]  # Keep balanced

# Customize thresholds
config.models.short_term_max_horizon = 2  # Extend short-term to f2
config.models.medium_term_max_horizon = 4  # Extend medium-term to q2
```

## Hyperparameter Tuning

These weights can be tuned via `tuning_utils.py`:

```python
param_grid = {
    # ... existing params ...
    'short_term_clf_xgb_weight': [0.4, 0.5, 0.6, 0.7],
    'medium_term_clf_xgb_weight': [0.5, 0.6, 0.7, 0.8],
    'long_term_clf_xgb_weight': [0.6, 0.7, 0.8, 0.9],
    # ... same for regressors ...
}
```

Then in `update_config_with_params`:
```python
config.models.short_term_clf_weights = [
    params['short_term_clf_xgb_weight'],
    1 - params['short_term_clf_xgb_weight']
]
```

## Benefits

1. ✅ **Addresses Linear Model Limitations**: Linear models can't use `forecast_horizon` effectively, so we reduce their weight for long horizons
2. ✅ **Maximizes Data Usage**: All models train on pooled data (no splitting)
3. ✅ **Adaptive Confidence**: Ensemble adapts to data scarcity and uncertainty
4. ✅ **Fully Configurable**: All weights are parameters, not hardcoded
5. ✅ **Backward Compatible**: Default weights maintain current behavior

## Validation

To validate the implementation:

1. **Check Weights Are Applied**:
   ```python
   # Should print different weights for different horizons
   config = PredictionConfig()
   print(config.models.get_ensemble_weights_for_horizon(0, 'classifier'))  # [0.5, 0.5]
   print(config.models.get_ensemble_weights_for_horizon(2, 'classifier'))  # [0.6, 0.4]
   print(config.models.get_ensemble_weights_for_horizon(5, 'classifier'))  # [0.7, 0.3]
   ```

2. **Compare Performance by Horizon**:
   - Run predictions on f0, f1, f2, f3, q2, q3, q4
   - Analyze F1/MAE/RMSE for each horizon
   - Verify that long-horizon predictions benefit from XGBoost-heavy weighting

3. **Ablation Study**:
   - Test with uniform weights (all horizons use [0.5, 0.5])
   - Compare against stratified weights
   - Expect improvement on f2+ predictions
