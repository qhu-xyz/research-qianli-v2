# Suggestions for Improving Shadow Price Predictions for Anomalies and Never-Binding Constraints

## Problem Description
The current pipeline identifies "never-binding" branches and checks them for flow anomalies. When an anomaly is detected, the system correctly switches to using the **Default Regressor** (trained on all binding constraints). However, the predicted shadow price remains too small, even when the actual shadow price is significant.

## Root Cause Analysis
1.  **Model Training is Correct**: The default regressor is trained *only* on binding samples (`label > threshold`), so it is not biased by zero-value samples.
2.  **Feature Mismatch**: The anomaly detector triggers based on specific flow statistics (e.g., `prob_exceed_100`, `flow_mean`). However, the default regressor might rely on other features or interactions that are not present or are weak in these "never-binding" branches.
3.  **"Safe" Appearance**: To the default regressor, these branches likely still look "safe" (low congestion risk) based on the standard feature set, which is why they historically never bound. The anomaly detector sees a deviation, but the regressor doesn't see a high price signal.

## Recommendations

### 1. Apply Heuristic Multiplier or Floor (Recommended)
Since the model underestimates these "surprise" events, applying a heuristic adjustment is the most direct fix.
- **Multiplier**: Scale the prediction by a factor (e.g., 2.0x or 5.0x) when it comes from an anomaly detection.
- **Floor**: Enforce a minimum shadow price (e.g., $10 or $50) if an anomaly is detected.

**Implementation in `prediction.py`:**
```python
if is_anomaly:
    # ... existing code ...
    y_pred_shadow_price[idx] = predict_ensemble(...)

    # Apply heuristic boost
    y_pred_shadow_price[idx] = max(y_pred_shadow_price[idx] * 2.0, 20.0)
```

### 2. Feature Engineering: "Anomaly Score"
Feed the anomaly confidence/score directly into the regressor as a feature.
- **Action**: Add `anomaly_confidence` or `is_anomaly` as a feature column.
- **Benefit**: The model can learn that high anomaly scores correlate with higher prices (if such examples exist in training).
- **Challenge**: Requires retraining models and ensuring training data has this feature populated (which might be hard if anomalies are rare).

### 3. Quantile Regression
Train a separate version of the default regressor using **Quantile Loss** (e.g., 90th percentile) specifically for use in anomaly cases.
- **Benefit**: Predicts a "worst-case" price rather than an average price, which is more appropriate for risk management of anomalies.

### 4. Surrogate Model for Anomalies
If the default regressor is too conservative, train a simple surrogate model (e.g., a linear model on just `prob_exceed_100` and `flow_mean`) specifically for these cases.
- **Logic**: `Price = alpha * prob_exceed_100 + beta * flow_mean`
- **Benefit**: Direct link between the flow anomaly and price, bypassing complex interactions that might be suppressing the prediction in the main model.

## Immediate Next Step
Implement **Recommendation #1 (Heuristic Multiplier/Floor)** in `prediction.py` as a quick fix to verify if it improves the signal.
