# Strategies for FTR Period Types (f0-f3, q2-q4)

The FTR period types (`f0`, `f1`, `f2`, `f3`, `q2`, `q3`, `q4`) represent different **forecast horizons** and **durations**.
*   **Monthly Products (`f0`-`f3`)**: Short duration (1 month), varying horizon (0-3 months ahead). High volatility, sensitive to specific outages.
*   **Quarterly Products (`q2`-`q4`)**: Long duration (3 months), varying horizon. "Smoothed" behavior, sensitive to seasonal trends and long-term transmission status.

Here are strategies to leverage the relationships between these products to improve prediction accuracy.

## 1. Temporal Hierarchy & Reconciliation

**Concept**: The "Quarterly" price should roughly equal the average of the constituent "Monthly" prices (if they existed).

*   **Quarter-to-Month Decomposition**:
    *   **Idea**: If you have a strong signal for a Quarterly product (`q2`), use it to constrain the predictions for the corresponding months (`f1`, `f2`, `f3` if they overlap).
    *   **Implementation**:
        *   Train a model for the **Quarterly** average shadow price.
        *   Train models for **Monthly** deviations from that quarterly average.
        *   Prediction($Month_i$) = Prediction($Quarter$) + Prediction($Deviation_i$).
    *   **Benefit**: The Quarterly model learns the "base load" congestion risk from more stable data, while the Monthly models only need to learn the specific short-term variations.

*   **Month-to-Quarter Aggregation**:
    *   **Idea**: Construct a Quarterly prediction by aggregating Monthly predictions.
    *   **Implementation**:
        *   Predict shadow prices for all 3 months in the quarter using `f` models.
        *   Average them to get the `q` prediction.
    *   **Benefit**: Captures granular outage details that a coarse Quarterly model might miss.

## 2. Horizon-Based Transfer Learning

**Concept**: A branch that is congested in `f0` (prompt month) is likely to be congested in `f1` (next month), but with higher uncertainty.

*   **Chain of Predictions (Regressor Chain)**:
    *   **Idea**: Use the prediction for a shorter horizon as a feature for the longer horizon.
    *   **Implementation**:
        1. Predict `f0` (Prompt Month).
        2. Predict `f1` using `f0_prediction` as a feature.
        3. Predict `f2` using `f1_prediction` as a feature.
    *   **Why it works**: `f0` captures the *current* grid state best. Passing this information forward helps `f1` and `f2` models anchor their predictions to reality.

*   **Horizon Encoding**:
    *   **Idea**: Train a **Unified Model** for all monthly products (`f0`-`f3`) instead of separate models.
    *   **Implementation**:
        *   Add `forecast_horizon` (0, 1, 2, 3) as a numerical feature.
        *   Add `is_quarterly` (0/1) as a feature.
    *   **Benefit**: The model learns that "congestion risk generally decays with horizon" or "uncertainty increases with horizon" from the entire dataset, rather than relearning it for each period type.

## 3. Product-Specific Feature Engineering

**Concept**: Different products care about different drivers.

*   **For Monthly Products (`f0`-`f3`)**:
    *   **Focus**: **Specific Outages**.
    *   **Strategy**: Heavily weight features related to *specific* line outages scheduled for that exact month.
    *   **Feature**: `outage_overlap_ratio` (percentage of the month a critical line is out).

*   **For Quarterly Products (`q2`-`q4`)**:
    *   **Focus**: **Seasonal Trends & Long-Term Outages**.
    *   **Strategy**: Focus on "Seasonal" features and "Long-duration" outages.
    *   **Feature**: `season_sin`, `season_cos`, `long_term_outage_flag` (only count outages lasting > 2 weeks). Short 2-day outages usually wash out in a quarterly average.

## 4. Data Scarcity Strategy: "The Anchor Product"

**Concept**: `f0` (Prompt Month) usually has the most reliable "ground truth" training data because it's closest to the auction. `q4` has the least reliable signal.

*   **Anchor & Drift**:
    *   **Idea**: Treat the `f0` model as the "Anchor".
    *   **Implementation**:
        *   Train a robust `f0` model.
        *   For `f1`...`q4`, do not train from scratch. Instead, train a model to predict the **Drift** (change) from the `f0` baseline given the horizon.
        *   Prediction(`q2`) = Model(`f0`) + Drift(`horizon=4 months`).
    *   **Benefit**: If data for `q4` is scarce, the model defaults to the robust `f0` prediction, which is a better guess than a noisy model trained on few `q4` samples.

## Summary of Recommendations

| Strategy | Complexity | Impact | Recommendation |
| :--- | :--- | :--- | :--- |
| **Unified Horizon Model** | Low | High | **Strongly Recommended**. Train one model for all `f` products with `horizon` feature. |
| **Chain of Predictions** | Medium | High | **Do this**. Use `f0` pred to help `f1`, etc. |
| **Quarter-to-Month Decomp** | High | Medium | Good for ensuring consistency between `q` and `f` products. |
| **Anchor & Drift** | Medium | High | Excellent for handling the high uncertainty of `q4` products. |

### Immediate Action Plan

1.  **Unified Monthly Model**: Combine `f0`, `f1`, `f2`, `f3` data into a single training set. Add `forecast_horizon` as a feature.
2.  **Unified Quarterly Model**: Combine `q2`, `q3`, `q4` data. Add `forecast_horizon` (months to start of quarter) as a feature.
3.  **Feature Distinction**:
    *   For **Monthly** models, include all outage features.
    *   For **Quarterly** models, filter out short-duration outages (< 7 days) from the feature set to reduce noise.
