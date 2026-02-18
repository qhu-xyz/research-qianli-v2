# Shadow Price Prediction

A machine learning pipeline for predicting shadow prices in power markets (MISO). This project implements a two-stage prediction approach: first classifying whether a constraint will bind, and then regressing the shadow price if it is predicted to bind.

## Key Features

*   **Two-Stage Prediction**:
    1.  **Classification**: Predicts binding probability ($P(binding)$) using XGBoost/RandomForest classifiers.
    2.  **Regression**: Predicts shadow price magnitude for binding constraints using XGBoost/Linear regressors.
*   **Feature Scaling**: Automatically scales input features using `StandardScaler` before training and inference.
*   **Scaled Binding Probability**:
    *   Calculates a `binding_probability_scaled` metric that maps the decision threshold to 0.5.
    *   Allows for consistent interpretation of probabilities across different branches with varying optimal thresholds.
*   **Dynamic Thresholding**: Optimizes classification thresholds per branch to maximize F-beta score.
*   **Anomaly Detection**: Detects flow anomalies in typically non-binding branches to catch unexpected binding events.
*   **Resumable Hyperparameter Search**: Robust, parallelized parameter tuning with automatic state saving and resumption.

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

## Project Structure

```
.
├── src/
│   └── shadow_price_prediction/
│       ├── pipeline.py       # Main pipeline logic (data loading, scaling, training)
│       ├── prediction.py     # Prediction logic (inference, thresholding, aggregation)
│       ├── models.py         # Model definitions (Stacking, Ensemble)
│       ├── config.py         # Configuration dataclasses
│       ├── anomaly_detection.py # Flow anomaly detection
│       └── tuning_utils.py   # Hyperparameter search utilities
├── notebook/
│   └── param_search.ipynb    # Hyperparameter tuning notebook
├── document/
│   ├── resumable_search_guide.md # Guide for resumable parameter search
│   └── stacking_analysis.md      # Analysis of stacking model performance
└── examples/                 # Example scripts
```

## Usage

### Running the Pipeline

The core pipeline is encapsulated in the `ShadowPricePipeline` class in `src/shadow_price_prediction/pipeline.py`. It handles:
1.  Data loading for specified training and test periods.
2.  Feature scaling.
3.  Model training (classifiers and regressors).
4.  Prediction and result aggregation.

### Hyperparameter Search

Use `notebook/param_search.ipynb` to run parallel hyperparameter tuning. The system supports:
*   **Resumability**: Automatically loads previous results and skips completed parameter combinations.
*   **Parallel Execution**: Uses Ray for efficient parallel processing.
*   **Metric Tracking**: Logs detailed metrics (F1, Precision, Recall, MAE, MSE) for each run.

See `document/resumable_search_guide.md` for details.

## Methodology

### 1. Feature Scaling
Features are standardized ($z = \frac{x - \mu}{\sigma}$) using `StandardScaler`. The scaler is fit on the training data and applied to both training and test sets to ensure consistency.

### 2. Scaled Binding Probability
To normalize the interpretation of binding probabilities, we compute a scaled probability $f(p)$:

$$
f(p) = \begin{cases}
0.5 \times \frac{p}{t} & \text{if } p < t \\
0.5 + 0.5 \times \frac{p - t}{1 - t} & \text{if } p \ge t
\end{cases}
$$

Where $t$ is the optimal decision threshold. This ensures that $f(t) = 0.5$, making 0.5 the universal decision boundary for all branches.

### 3. Threshold Reporting
The pipeline reports the specific `threshold` used for each prediction, allowing for transparency in the decision-making process.
