# Fixing Unrealistic Shadow Price Predictions

## Problem Description
The current shadow price prediction model exhibits two issues:
1.  **Unrealistically High Values**: Predictions can reach ~23 million, whereas typical high shadow prices are around 70k.
2.  **Negative Values**: The model predicts negative shadow prices, which is physically impossible as the training labels are absolute values (non-negative).

## Root Cause Analysis
- **Model Objective**: The XGBoost regressor uses `reg:squarederror` on the raw shadow price values. This objective function does not enforce non-negativity.
- **Target Distribution**: Shadow prices likely follow a heavy-tailed distribution (mostly small, occasional spikes). Regression on raw values is sensitive to these outliers, leading to over-extrapolation for high values.
- **Unconstrained Output**: Without a link function (like log) or specific objective (like Gamma/Poisson), the linear combination of trees can result in negative predictions for inputs that fall into "low value" leaf nodes with negative bias terms.

## Proposed Solutions

### 1. Log-Transformation of Target (Recommended)
Transform the target variable $y$ (shadow price) before training and inverse-transform the predictions.

**Transformation:**
$$ y' = \log(y + 1) $$

**Inverse Transformation:**
$$ \hat{y} = \exp(\hat{y}') - 1 $$

**Benefits:**
- **Enforces Non-Negativity**: Since $\exp(x)$ is always positive, the final prediction $\hat{y}$ will always be $> -1$. With a `max(0, ...)` clip, it guarantees non-negativity.
- **Compresses Range**: Reduces the impact of extreme outliers (e.g., 70k becomes ~11.2, while 10 becomes ~2.4). This stabilizes the gradient and prevents the model from chasing extreme values, reducing the likelihood of predicting millions.

**Implementation Steps:**
1.  In `train_regressors` (and `_train_single_branch_regressor`), apply `np.log1p(y_train)` before fitting.
2.  In `Predictor.predict`, apply `np.expm1(y_pred)` to the raw model output.

### 2. Clipping Predictions (Safety Net)
Apply a hard clip to the final predictions to ensure they stay within physical and practical limits.

**Implementation:**
```python
# In prediction.py
y_pred_shadow_price = np.clip(y_pred_shadow_price, 0, MAX_SHADOW_PRICE)
```
*   **Min**: 0 (Absolute lower bound)
*   **Max**: e.g., 50,000 or 100,000 (Based on historical max or market cap).

### 3. Outlier Removal in Training
Filter out extreme outliers from the training data if they represent data errors or rare events that the model shouldn't prioritize.

**Implementation:**
```python
# In train_regressors
mask = y_train <= REASONABLE_MAX_PRICE
X_train = X_train[mask]
y_train = y_train[mask]
```

## Implementation Plan

1.  **Modify `ShadowPriceModels.train_regressors`**:
    *   Apply `np.log1p` to `y_train` before passing it to `train_ensemble`.
    *   Apply `np.log1p` to `y_branch_reg` in `_train_single_branch_regressor`.

2.  **Modify `Predictor.predict`**:
    *   Apply `np.expm1` to the output of `reg.predict(X_branch_reg)`.
    *   Apply `np.maximum(0, ...)` to ensure no small negative values (floating point noise) remain.
    *   (Optional) Apply `np.minimum(..., 100000)` to cap extreme values.

3.  **Update `config.py`**:
    *   Add `max_shadow_price_clip` parameter if desired.
