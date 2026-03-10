# Improvement Proposal: Multi-Feature Anomaly Detection

## Current Limitations
The current `AnomalyDetector` relies on a single feature (`prob_exceed_100`) and a static statistical threshold (`P99 + k*IQR`) derived from training data.
- **Single Point of Failure**: If `prob_exceed_100` is noisy or missing, detection fails.
- **Binary Threshold**: The decision is largely binary (threshold exceeded or not), with a simple linear confidence interpolation.
- **Hardcoded Downstream**: The prediction pipeline currently hardcodes the binary threshold to 0.5 when an anomaly is detected.

## Proposed Strategy: Weighted Multivariate Anomaly Score

We propose upgrading to a multivariate approach that combines signals from multiple relevant features (e.g., `prob_exceed_100`, `prob_exceed_98`, `prob_exceed_90`, `flow_max`).

### 1. Feature Selection & Statistics
Instead of one feature, we verify and track statistics for a configurable list of features.
For each branch $b$ and feature $f_i$ in training:
- Compute robust statistics: Median ($\mu_i$), IQR ($\sigma_i$), and Upper Bound ($U_i = P99 + k \cdot IQR$).

### 2. Normalized Anomaly Scores
For a new test sample $x$, compute a normalized anomaly score $s_i$ for each feature $f_i$:
$$ s_i = \max(0, \frac{x_i - U_i}{\sigma_i}) $$
This represents how many "deviations" (IQRs) the value is above the anomaly threshold. Values $\le 0$ are normal.

### 3. Aggregation (Ensemble Score)
Combine individual feature scores into a global anomaly score $S$.
$$ S = \sum w_i \cdot s_i $$
Where $w_i$ are weights for each feature (default equal or configurable).

### 4. Sigmoid Probability Mapping
Map the global score $S$ to a binding probability $P_{binding}$ using a sigmoid function to ensure smooth transition and bounded output $[0, 1]$.
$$ P_{binding} = \frac{1}{1 + e^{-\alpha(S - \beta)}} $$
- $\beta$: Shift parameter (score at which probability is 0.5).
- $\alpha$: Steepness parameter.

### 5. Dynamic Thresholding
Instead of a fixed `0.5`, the threshold $T$ can be dynamic or derived from the confidence of the anomaly. However, for compatibility, we can output the calculated probability $P_{binding}$ and let the standard threshold logic apply, or output a calibrated threshold.

## Implementation Steps

1.  **Config Update**:
    - Replace `flow_feature` string with `flow_features` list.
    - Add `anomaly_weights` (dict).
    - Add sigmoid parameters (`alpha`, `beta`).

2.  **Training Update**:
    - Calculate stats for *all* configured features.
    - Store covariance (optional, if using Mahalanobis) or just marginal stats (for simple weighted score).

3.  **Detection Update**:
    - Compute component scores $s_i$.
    - Compute aggregate $S$.
    - Return probability $P_{binding}$.

## Alternative: Isolation Forest
For more complex relationships, an `IsolationForest` can be trained per horizon-group (not per branch, to save compute) with identifying features (branch embeddings or IDs). However, this is heavier to implement. The statistical approach above preserves the per-branch specificity which is a key strength of the current system.
