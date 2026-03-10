# Strategies for Shadow Price Prediction under Data Scarcity

Beyond the implemented `seasonal_sin`/`cos` and `forecast_horizon` features, here are advanced strategies to improve prediction accuracy while specifically addressing the challenge of data scarcity (few binding samples per branch).

## 1. Global-Local Knowledge Transfer (Transfer Learning)

**Concept**: Instead of treating each branch as an isolated island, leverage patterns from the entire grid or similar branches.

*   **Cluster-Based Training**:
    *   **Idea**: Group branches into clusters based on their congestion patterns (e.g., "Summer Peaking", "Winter Peaking", "Base Load Congestion") or electrical characteristics (voltage level, zone).
    *   **Implementation**: Train a single "Cluster Model" for each group. Use this model's prediction as a strong feature (or "prior") for the branch-specific model.
    *   **Benefit**: Allows a branch with only 5 binding samples to learn from 500 samples of electrically similar branches.

*   **Global Feature Injection**:
    *   **Idea**: Train a massive "Global Model" on *all* data from *all* branches.
    *   **Implementation**: Feed the prediction of the Global Model as an input feature to the Branch-Specific Model.
    *   **Benefit**: The branch model only needs to learn the "residual" (difference) between the global average behavior and its specific behavior, which requires less data.

## 2. Advanced Feature Engineering

**Concept**: Create features that explicitly capture trends and physics, reducing the burden on the model to learn complex relationships from scratch.

*   **Rolling Statistics (Trend Features)**:
    *   **Idea**: Capture the recent "mood" of the branch.
    *   **Features**:
        *   `rolling_binding_prob_12m`: % of times this branch bound in the last 12 months.
        *   `rolling_max_price_12m`: Max shadow price observed in the last 12 months.
        *   `rolling_mean_price_binding`: Average price *when binding* in history.
    *   **Benefit**: Helps the model distinguish between a branch that *never* binds and one that binds rarely but severely.

*   **Interaction Features**:
    *   **Idea**: Explicitly model non-linear interactions that tree models might miss with small data.
    *   **Features**: `load * temperature`, `load * season_sin`, `horizon * season_cos`.
    *   **Benefit**: Linear models (ElasticNet) can use these to capture non-linearities without the high variance of trees.

## 3. Data Augmentation (Synthetic Data)

**Concept**: Artificially increase the number of "binding" samples to prevent the model from ignoring them as noise.

*   **SMOGN (Synthetic Minority Over-sampling for Regression)**:
    *   **Idea**: Generate synthetic samples for the minority class (binding events) and rare high-value shadow prices.
    *   **Implementation**: Use libraries like `smogn` to create plausible synthetic binding events by interpolating between existing binding samples.
    *   **Benefit**: Balances the dataset, forcing the model to pay attention to high shadow prices.

*   **Gaussian Noise Injection**:
    *   **Idea**: Make the model robust to small shifts in input.
    *   **Implementation**: During training, create copies of binding samples with slightly perturbed inputs (e.g., Load ± 1%, Temp ± 0.5°C).
    *   **Benefit**: Acts as a regularizer, preventing overfitting to the exact values of the few binding samples.

## 4. Model Constraints & Regularization

**Concept**: Restrict the model's freedom to prevent it from learning spurious correlations in sparse data.

*   **Monotonic Constraints (XGBoost)**:
    *   **Idea**: Enforce physical laws. Higher load should generally increase congestion risk.
    *   **Implementation**: Set monotonic constraints on features like `load` or `temperature`.
    *   **Benefit**: Prevents the model from learning "wiggly" decision boundaries that don't make physical sense just to fit a few outliers.

*   **Quantile Regression**:
    *   **Idea**: Instead of predicting the *mean* (which is dragged down by zeros), predict the *90th percentile*.
    *   **Implementation**: Change XGBoost objective to `reg:quantileerror` with `quantile_alpha=0.9`.
    *   **Benefit**: Focuses the model on "what's the worst that could happen?", which is often more valuable for FTR risk management than the average.

## 5. Hierarchical Bayesian Modeling

**Concept**: A statistically rigorous way to share information.

*   **Bayesian Priors**:
    *   **Idea**: Treat the "Global" parameters as the prior distribution, and the "Branch" data as the likelihood.
    *   **Implementation**: Use Bayesian Ridge Regression where the prior mean is 0 (or the global model's weight).
    *   **Benefit**: The model naturally shrinks to the "Global Average" when branch data is scarce, and shifts to "Branch Specific" as more data becomes available.

## Summary of Recommendations

| Strategy | Complexity | Impact on Scarcity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Global Feature Injection** | Low | High | **Start Here**. Easy to implement and very effective. |
| **Rolling Statistics** | Low | Medium | **Do this**. Adds valuable context. |
| **Monotonic Constraints** | Low | Medium | **Do this**. "Free" robustness. |
| **SMOGN / Augmentation** | High | High | Explore if regression performance is still poor. |
| **Bayesian Modeling** | High | Medium | Consider for a V2 if probabilistic bounds are needed. |
