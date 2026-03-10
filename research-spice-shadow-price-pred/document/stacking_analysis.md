# Analysis: Stacking Model vs. Simple Average Performance

## Executive Summary
The Stacking Model is currently underperforming compared to a simple weighted average. This behavior is commonly observed in time-series forecasting tasks when the stacking implementation does not strictly adhere to temporal causality. The primary reason is likely **temporal leakage during the generation of meta-features** (Out-of-Fold predictions), causing the meta-model to learn spurious correlations that do not hold in a true forward-looking test scenario.

## Root Cause Analysis

### 1. Temporal Leakage in Cross-Validation
The current `StackingModel` uses `GroupKFold` (grouped by `auction_month`) to generate Out-of-Fold (OOF) predictions for training the meta-model.

*   **How it works**: `GroupKFold` ensures that all samples from a specific month are either in the training set or the validation set for a given fold.
*   **The Problem**: It does **not** enforce temporal order.
    *   *Example*: In one fold, the model might train on data from **March 2025** and **May 2025** to predict **April 2025**.
    *   *Consequence*: The base models have access to "future" information (May) when generating predictions for the "past" (April). The meta-model then learns to trust these base models based on their performance in this artificially easier task.
    *   *Test Time*: When predicting for **June 2025**, the base models only have access to past data (up to May). Their performance will likely be worse than what the meta-model expects, leading to suboptimal weighting.

### 2. Meta-Model Overfitting
The meta-model (Logistic Regression) is trained on the OOF predictions. Because of the leakage described above, the OOF predictions are "too good" (optimistically biased). The meta-model learns weights that overfit to this optimistic scenario. When applied to new data where no future information is available, these weights are no longer optimal.

### 3. High Correlation of Base Models
Stacking works best when base models make **uncorrelated errors** (i.e., they are diverse).
*   If XGBoost and Logistic Regression make very similar mistakes on the difficult edge cases, the meta-model has little signal to leverage for correction.
*   In such cases, a simple average is often more robust because it reduces variance without adding the complexity (and potential overfitting) of learning a combiner model.

### 4. Data Scarcity for Meta-Learner
If the number of unique "events" (outage dates or months) is relatively small, the meta-model might not have enough diverse samples to learn a robust combination strategy, leading it to memorize noise in the training set.

## Recommendations

### 1. Switch to Time-Series Split for Stacking
To fix the leakage, the OOF predictions must be generated using a **Walk-Forward Validation** (or `TimeSeriesSplit`) approach.
*   *Train*: Jan-Feb -> *Predict*: March
*   *Train*: Jan-Mar -> *Predict*: April
*   *Train*: Jan-Apr -> *Predict*: May
This ensures that the meta-features for "May" were generated using a model that *only* saw data prior to May. This matches the inference-time reality.

### 2. Use Weighted Averaging (Current Best Approach)
Given the complexity of correctly implementing time-series stacking and the robustness of simple averaging:
*   **Stick to Weighted Averaging**: It is less prone to overfitting and easier to interpret.
*   **Optimize Weights**: Instead of learning weights via a meta-model for every prediction, you can tune the weights (e.g., `0.6 * XGB + 0.4 * LogReg`) as a hyperparameter in your `param_search` pipeline.

### 3. Regularize the Meta-Model
If you wish to persist with stacking:
*   Use **Non-Negative Least Squares (NNLS)** as the meta-model. It forces weights to be positive and sum to 1, acting as a "learned average" that is more constrained and stable than standard Logistic Regression.
