# MISO Shadow Price Prediction - Practical Implementation Plan
## Handling Class Imbalance and Zero-Inflated Target

**Project**: MISO Constraint Shadow Price Prediction with Imbalanced Data
**Data Source**: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
**Target API**: `get_da_shadow(st, et, class_type)`
**Date**: 2025-11-16
**Critical Challenge**: **Severe class imbalance - most shadow prices are 0 (non-binding events)**

---

## Executive Summary

This implementation plan addresses the **critical challenge of class imbalance** in shadow price prediction where binding events (shadow price > 0) represent only 5-15% of historical data. A naive model would achieve 85-95% accuracy by always predicting 0, but would be useless for trading.

**Solution Strategy**: **Two-Stage Hybrid Modeling**
1. **Stage 1 (Classification)**: Predict if constraint will bind (binary: binding vs non-binding)
2. **Stage 2 (Regression)**: Predict shadow price magnitude given binding

**Key Innovation**: Combine classification (handles imbalance) with regression (predicts magnitude) to achieve both high binding detection accuracy and accurate shadow price estimates.

---

## 1. Problem Statement and Data Characteristics

### 1.1 MISO Market Context (FTR Trader Perspective)

**MISO Transmission Congestion Characteristics**:
- **Geography**: 15-state footprint (Great Lakes to Gulf Coast)
- **Key Constraints**:
  - North-to-South flowgates (wind-rich North Dakota to load centers)
  - Michigan import constraints
  - MISO-PJM seams
  - MISO-South integration interfaces
- **Binding Patterns**:
  - Seasonal: Summer (cooling load) and Winter (heating load, wind variability)
  - Diurnal: Peak hours (HE 14-19) more likely to bind
  - Weather-driven: Temperature extremes, wind lulls

**Constraint Binding Statistics** (Typical):
```
Binding Frequency by Constraint Type:
- Critical flowgates: 10-20% of hours
- Seasonal constraints: 5-15% of hours
- Interface limits: 2-10% of hours
- Local constraints: 1-5% of hours

Shadow Price Distribution when Binding:
- Median: $5-15/MW
- 75th percentile: $20-40/MW
- 95th percentile: $50-100/MW
- Extreme events: >$200/MW (rare but high impact)
```

### 1.2 Data Structure and Availability

#### Flow Percentage Density Data
```
Path: /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/

Structure:
density/
├── auction_month=YYYY-MM/
│   ├── market_month=YYYY-MM/
│   │   ├── market_round={1,2,3,...}/
│   │   │   ├── outage_date=YYYY-MM-DD/
│   │   │   │   ├── density.parquet     # Flow percentage KDE
│   │   │   │   ├── limit.parquet       # Constraint limits
│   │   │   │   └── score.parquet       # Additional scores

Expected Schema (density.parquet):
- constraint_id: str
- hour_ending: int (1-24)
- density_bins: array (e.g., [0, 1, 2, ..., 100] for flow %)
- density_values: array (KDE probability densities)
- topology_version: str
- generation_pattern_id: str
```

#### Shadow Price Data
```python
# Available via API call
shadow_price_data = get_da_shadow(
    st='2024-01-01',  # Start date
    et='2024-01-31',  # End date
    class_type='constraint'  # Type of constraint
)

Expected Return Schema:
- constraint_id: str
- timestamp: datetime (hourly)
- shadow_price: float ($/MW, 0 for non-binding)
- binding_status: bool (True if shadow_price > threshold)
- market_type: str ('DA' for day-ahead)
```

### 1.3 Class Imbalance Challenge

**Problem Visualization**:
```
Typical Shadow Price Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shadow Price = 0 (Non-binding): ████████████████████ 85-95%
Shadow Price > 0 (Binding):     ██ 5-15%
  ├─ $0-10:     █ 3-8%
  ├─ $10-50:    █ 2-5%
  └─ >$50:      ▏ <2%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implication:
- Naive model predicting always 0: 90% accuracy, 0% usefulness
- Need to maximize F1 score for binding classification
- Need accurate magnitude prediction for binding events
```

**Impact on Model Performance**:
| Approach | Accuracy | F1 (Binding) | Business Value |
|----------|----------|--------------|----------------|
| Always predict 0 | 90% | 0.00 | Useless |
| Balanced classification | 80% | 0.70 | Moderate |
| Two-stage hybrid | 85% | 0.80 | **High** |
| Weighted ensemble | 87% | 0.85 | **Very High** |

---

## 2. Recommended Modeling Approach: Two-Stage Hybrid

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                           │
│  - Flow percentage density (moments, quantiles, tail)       │
│  - Temporal features (hour, day, month, season)             │
│  - Contextual features (load, weather, outages)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        │                              │
┌───────▼────────┐            ┌────────▼────────┐
│   STAGE 1:     │            │  ALTERNATIVE:   │
│ CLASSIFICATION │            │  ZERO-INFLATED  │
│                │            │    REGRESSION   │
│ Will constrain │            │                 │
│ bind? (Y/N)    │            │ (Advanced)      │
└───────┬────────┘            └─────────────────┘
        │
        ├─ No (85-95%) ──► Shadow Price = 0
        │
        └─ Yes (5-15%)
               │
        ┌──────▼────────┐
        │   STAGE 2:    │
        │  REGRESSION   │
        │               │
        │ Shadow price  │
        │ magnitude     │
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ FINAL OUTPUT  │
        │  Shadow Price │
        │   Prediction  │
        └───────────────┘
```

### 2.2 Stage 1: Binding Classification Model

**Objective**: Predict P(binding) = P(shadow_price > threshold)

**Model Architecture**:
```python
# Recommended: LightGBM Classifier with class weights
from lightgbm import LGBMClassifier

stage1_classifier = LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=100,

    # CLASS IMBALANCE HANDLING (CRITICAL!)
    class_weight='balanced',  # Auto-compute weights
    # or manually: scale_pos_weight = (n_negative / n_positive)

    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)
```

**Target Variable**:
```python
# Binary classification target
y_binding = (shadow_price > binding_threshold).astype(int)

# Threshold selection (important!)
binding_threshold = 0.5  # $/MW (tune based on business needs)
# Higher threshold: Fewer false positives, miss marginal binding
# Lower threshold: More false positives, catch marginal binding
```

**Class Weight Calculation**:
```python
# Option 1: Automatic balanced weights
class_weight = 'balanced'

# Option 2: Manual calculation
n_non_binding = (y == 0).sum()
n_binding = (y == 1).sum()
scale_pos_weight = n_non_binding / n_binding  # Typically 5-20

# Option 3: Custom business-driven weights
# Weight binding events higher if missing them is more costly
class_weights = {
    0: 1.0,      # Non-binding
    1: 10.0,     # Binding (10x more important)
}
```

**Evaluation Metrics** (Critical for Imbalanced Data):
```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

def evaluate_binding_classifier(y_true, y_pred, y_pred_proba):
    """
    Comprehensive evaluation for imbalanced classification.
    """
    metrics = {}

    # Confusion matrix (most informative)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp

    # Key metrics for imbalanced data
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = f1_score(y_true, y_pred)

    # Business-critical: How often we correctly identify binding
    metrics['binding_detection_rate'] = tp / (tp + fn)  # Recall

    # Cost of false alarms: How often predicted binding is correct
    metrics['binding_precision'] = tp / (tp + fp)

    # ROC-AUC (threshold-independent)
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    # Precision-Recall AUC (better for imbalanced)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['pr_auc'] = auc(recall, precision)

    return metrics
```

**Target Performance**:
- **F1 Score**: > 0.75 (balance precision and recall)
- **Recall (Binding Detection)**: > 0.80 (catch 80% of binding events)
- **Precision**: > 0.60 (60% of predicted binding are correct)
- **PR-AUC**: > 0.70 (better metric than ROC-AUC for imbalanced data)

### 2.3 Stage 2: Shadow Price Magnitude Regression

**Objective**: Predict shadow_price | binding = True

**Model Architecture**:
```python
# Recommended: LightGBM Regressor
from lightgbm import LGBMRegressor

stage2_regressor = LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=20,  # Smaller dataset (only binding events)
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)
```

**Training Data**:
```python
# Train ONLY on binding events
binding_mask = shadow_price > binding_threshold
X_binding = X[binding_mask]
y_shadow_magnitude = shadow_price[binding_mask]

# Log-transform if shadow prices are right-skewed
y_log_shadow = np.log1p(y_shadow_magnitude)  # log(1 + x)
stage2_regressor.fit(X_binding, y_log_shadow)

# Inverse transform predictions
y_pred_magnitude = np.expm1(stage2_regressor.predict(X_test))
```

**Evaluation Metrics**:
```python
def evaluate_magnitude_regression(y_true, y_pred):
    """
    Evaluate regression on binding events only.
    """
    metrics = {}

    # Standard regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)

    # Percentage error (exclude very small values)
    mask = y_true > 5.0  # Only evaluate on shadow price > $5/MW
    if mask.sum() > 0:
        metrics['mape'] = mean_absolute_percentage_error(
            y_true[mask], y_pred[mask]
        )

    # Tail performance (high shadow prices most valuable)
    high_mask = y_true > np.percentile(y_true, 75)
    if high_mask.sum() > 0:
        metrics['mae_high_shadow'] = mean_absolute_error(
            y_true[high_mask], y_pred[high_mask]
        )

    return metrics
```

**Target Performance**:
- **MAE**: < 8 $/MW (on binding events)
- **RMSE**: < 15 $/MW
- **R²**: > 0.60 (challenging due to volatility)
- **MAPE**: < 30% (for shadow price > $5/MW)

### 2.4 Combining Stage 1 and Stage 2

```python
class TwoStageHybridModel:
    """
    Two-stage model for zero-inflated shadow price prediction.
    """

    def __init__(
        self,
        classifier,
        regressor,
        binding_threshold=0.5,
        probability_threshold=0.3,
    ):
        """
        Parameters
        ----------
        classifier : sklearn-compatible classifier
            Stage 1: Binding classification model
        regressor : sklearn-compatible regressor
            Stage 2: Shadow price magnitude model
        binding_threshold : float
            Shadow price threshold defining "binding" ($/MW)
        probability_threshold : float
            Probability threshold for classifying as binding
            Lower = more sensitive (catch more binding events)
            Higher = more specific (fewer false alarms)
        """
        self.classifier = classifier
        self.regressor = regressor
        self.binding_threshold = binding_threshold
        self.probability_threshold = probability_threshold

    def fit(self, X, y):
        """
        Train both stages.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Shadow prices (including zeros)
        """
        # Stage 1: Train binding classifier
        y_binding = (y > self.binding_threshold).astype(int)
        self.classifier.fit(X, y_binding)

        # Stage 2: Train magnitude regressor on binding events only
        binding_mask = y > self.binding_threshold
        X_binding = X[binding_mask]
        y_magnitude = y[binding_mask]

        # Log-transform for right-skewed distribution
        y_magnitude_log = np.log1p(y_magnitude)
        self.regressor.fit(X_binding, y_magnitude_log)

        return self

    def predict(self, X):
        """
        Predict shadow prices using two-stage approach.

        Returns
        -------
        predictions : array-like
            Predicted shadow prices
        """
        # Stage 1: Predict binding probability
        binding_proba = self.classifier.predict_proba(X)[:, 1]

        # Stage 2: Predict magnitude for likely binding events
        # Use probability threshold to classify
        predicted_binding = binding_proba > self.probability_threshold

        # Initialize predictions as zero
        predictions = np.zeros(len(X))

        # Only predict magnitude for events classified as binding
        if predicted_binding.sum() > 0:
            X_binding = X[predicted_binding]

            # Predict log-transformed magnitude
            magnitude_log = self.regressor.predict(X_binding)

            # Inverse transform
            magnitude = np.expm1(magnitude_log)

            # Assign predicted magnitudes
            predictions[predicted_binding] = magnitude

        return predictions

    def predict_with_confidence(self, X):
        """
        Predict with binding probability and prediction intervals.

        Returns
        -------
        dict with:
            - predictions: Shadow price predictions
            - binding_proba: Probability of binding
            - is_binding: Binary binding classification
            - magnitude: Predicted magnitude (if binding)
        """
        binding_proba = self.classifier.predict_proba(X)[:, 1]
        predicted_binding = binding_proba > self.probability_threshold

        predictions = np.zeros(len(X))
        magnitudes = np.zeros(len(X))

        if predicted_binding.sum() > 0:
            X_binding = X[predicted_binding]
            magnitude_log = self.regressor.predict(X_binding)
            magnitude = np.expm1(magnitude_log)

            predictions[predicted_binding] = magnitude
            magnitudes[predicted_binding] = magnitude

        return {
            'predictions': predictions,
            'binding_proba': binding_proba,
            'is_binding': predicted_binding,
            'magnitude': magnitudes,
        }
```

---

## 3. Feature Engineering from Flow Percentage Density

### 3.1 Data Loading Pipeline

```python
import pandas as pd
import numpy as np
from pathlib import Path
import glob

class MISOFlowDensityLoader:
    """
    Load and process MISO flow percentage density data.
    """

    def __init__(self, base_path='/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density'):
        self.base_path = Path(base_path)

    def load_auction_month(
        self,
        auction_month: str,
        market_months: list = None,
    ) -> pd.DataFrame:
        """
        Load all density data for a given auction month.

        Parameters
        ----------
        auction_month : str
            Format: 'YYYY-MM'
        market_months : list, optional
            Filter to specific market months

        Returns
        -------
        pd.DataFrame with columns:
            - constraint_id
            - hour_ending
            - auction_month
            - market_month
            - market_round
            - outage_date
            - density_bins (array)
            - density_values (array)
        """
        auction_path = self.base_path / f"auction_month={auction_month}"

        if not auction_path.exists():
            raise ValueError(f"Auction month {auction_month} not found")

        # Find all density parquet files
        pattern = str(auction_path / "**" / "density.parquet")
        density_files = glob.glob(pattern, recursive=True)

        if not density_files:
            raise ValueError(f"No density files found for {auction_month}")

        # Load and concatenate all files
        dfs = []
        for file_path in density_files:
            # Extract partition values from path
            parts = Path(file_path).parts
            partition_info = {}
            for part in parts:
                if '=' in part:
                    key, value = part.split('=')
                    partition_info[key] = value

            # Load parquet
            df = pd.read_parquet(file_path)

            # Add partition columns
            for key, value in partition_info.items():
                df[key] = value

            dfs.append(df)

        # Concatenate all dataframes
        result = pd.concat(dfs, ignore_index=True)

        # Filter by market_month if specified
        if market_months is not None:
            result = result[result['market_month'].isin(market_months)]

        return result

    def load_historical_range(
        self,
        start_auction: str,
        end_auction: str,
    ) -> pd.DataFrame:
        """
        Load density data for a range of auction months.

        Parameters
        ----------
        start_auction : str
            Start auction month 'YYYY-MM'
        end_auction : str
            End auction month 'YYYY-MM'

        Returns
        -------
        pd.DataFrame
            Concatenated density data
        """
        # Generate auction month list
        start_date = pd.to_datetime(start_auction + '-01')
        end_date = pd.to_datetime(end_auction + '-01')

        auction_months = pd.date_range(
            start_date, end_date, freq='MS'
        ).strftime('%Y-%m').tolist()

        # Load each auction month
        dfs = []
        for auction_month in auction_months:
            try:
                df = self.load_auction_month(auction_month)
                dfs.append(df)
                print(f"Loaded {auction_month}: {len(df)} records")
            except ValueError as e:
                print(f"Skipping {auction_month}: {e}")

        if not dfs:
            raise ValueError("No data loaded")

        return pd.concat(dfs, ignore_index=True)
```

### 3.2 Feature Extraction from Density Distributions

```python
class FlowDensityFeatureExtractor:
    """
    Extract features from flow percentage density distributions.
    """

    def __init__(self, density_bins=None):
        """
        Parameters
        ----------
        density_bins : array-like, optional
            Flow percentage bins (e.g., [0, 1, 2, ..., 100])
            If None, will be inferred from data
        """
        self.density_bins = density_bins

    def extract_features(self, density_values, density_bins=None):
        """
        Extract comprehensive features from a single density distribution.

        Parameters
        ----------
        density_values : array-like
            KDE probability densities
        density_bins : array-like, optional
            Flow percentage bins

        Returns
        -------
        dict
            Dictionary of features
        """
        if density_bins is None:
            density_bins = self.density_bins
        if density_bins is None:
            raise ValueError("density_bins must be provided")

        features = {}

        # Normalize density to ensure it's a valid PDF
        density_norm = density_values / (density_values.sum() + 1e-10)

        # --- Statistical Moments ---
        # Expected value (mean flow %)
        features['flow_mean'] = np.average(density_bins, weights=density_norm)

        # Variance and standard deviation
        variance = np.average(
            (density_bins - features['flow_mean'])**2,
            weights=density_norm
        )
        features['flow_std'] = np.sqrt(variance)
        features['flow_variance'] = variance

        # Coefficient of variation (normalized uncertainty)
        features['flow_cv'] = (
            features['flow_std'] / (features['flow_mean'] + 1e-6)
        )

        # Skewness (asymmetry)
        features['flow_skewness'] = np.average(
            ((density_bins - features['flow_mean']) / (features['flow_std'] + 1e-6))**3,
            weights=density_norm
        )

        # Kurtosis (tail heaviness)
        features['flow_kurtosis'] = np.average(
            ((density_bins - features['flow_mean']) / (features['flow_std'] + 1e-6))**4,
            weights=density_norm
        )

        # --- Quantile Features ---
        cumsum = np.cumsum(density_norm)

        def find_quantile(q):
            """Find flow% at quantile q."""
            idx = np.searchsorted(cumsum, q)
            return density_bins[min(idx, len(density_bins)-1)]

        features['flow_q05'] = find_quantile(0.05)
        features['flow_q10'] = find_quantile(0.10)
        features['flow_q25'] = find_quantile(0.25)
        features['flow_median'] = find_quantile(0.50)
        features['flow_q75'] = find_quantile(0.75)
        features['flow_q90'] = find_quantile(0.90)
        features['flow_q95'] = find_quantile(0.95)
        features['flow_q99'] = find_quantile(0.99)

        # Interquartile range
        features['flow_iqr'] = features['flow_q75'] - features['flow_q25']

        # --- Binding-Specific Features (CRITICAL for shadow price) ---
        # Probability of exceeding various thresholds
        features['prob_flow_gt_80'] = density_norm[density_bins >= 80].sum()
        features['prob_flow_gt_85'] = density_norm[density_bins >= 85].sum()
        features['prob_flow_gt_90'] = density_norm[density_bins >= 90].sum()
        features['prob_flow_gt_95'] = density_norm[density_bins >= 95].sum()
        features['prob_flow_gt_98'] = density_norm[density_bins >= 98].sum()
        features['prob_flow_gt_100'] = density_norm[density_bins >= 100].sum()

        # Expected flow conditional on high loading
        mask_high = density_bins >= 90
        if density_norm[mask_high].sum() > 0:
            features['expected_flow_when_high'] = np.average(
                density_bins[mask_high],
                weights=density_norm[mask_high]
            )
        else:
            features['expected_flow_when_high'] = 0.0

        # Tail mass (integral above threshold)
        features['tail_mass_90'] = density_norm[density_bins >= 90].sum()
        features['tail_mass_95'] = density_norm[density_bins >= 95].sum()
        features['tail_mass_100'] = density_norm[density_bins >= 100].sum()

        # Expected exceedance (for binding events)
        mask_binding = density_bins >= 95
        if density_norm[mask_binding].sum() > 0:
            features['expected_exceedance_95'] = np.average(
                density_bins[mask_binding] - 95,
                weights=density_norm[mask_binding]
            )
        else:
            features['expected_exceedance_95'] = 0.0

        # --- Distribution Shape Features ---
        # Entropy (measure of uncertainty)
        density_nonzero = density_norm[density_norm > 0]
        if len(density_nonzero) > 0:
            features['entropy'] = -np.sum(
                density_nonzero * np.log(density_nonzero + 1e-10)
            )
        else:
            features['entropy'] = 0.0

        # Mode (most likely flow %)
        features['flow_mode'] = density_bins[np.argmax(density_norm)]

        # Concentration ratio (peakedness)
        features['concentration_ratio'] = (
            density_norm.max() / (density_norm.mean() + 1e-10)
        )

        # Number of modes (multimodality indicator)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(density_norm, prominence=0.01)
        features['n_modes'] = len(peaks)

        # Distance from mode to 95% threshold
        features['mode_to_95_distance'] = 95 - features['flow_mode']

        # Probability mass below median (symmetry indicator)
        features['prob_below_median'] = cumsum[
            np.searchsorted(density_bins, features['flow_median'])
        ]

        return features

    def extract_batch_features(self, density_df):
        """
        Extract features for all rows in a density dataframe.

        Parameters
        ----------
        density_df : pd.DataFrame
            DataFrame with 'density_bins' and 'density_values' columns

        Returns
        -------
        pd.DataFrame
            Original dataframe with feature columns added
        """
        feature_rows = []

        for idx, row in density_df.iterrows():
            features = self.extract_features(
                density_values=row['density_values'],
                density_bins=row.get('density_bins', self.density_bins)
            )
            feature_rows.append(features)

        # Create feature dataframe
        feature_df = pd.DataFrame(feature_rows)

        # Combine with original dataframe
        result = pd.concat([
            density_df.reset_index(drop=True),
            feature_df
        ], axis=1)

        return result
```

### 3.3 Temporal and Contextual Features

```python
class TemporalFeatureEngineer:
    """
    Create temporal and contextual features.
    """

    @staticmethod
    def add_temporal_features(df, timestamp_col='timestamp'):
        """
        Add cyclical temporal features.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with timestamp column
        timestamp_col : str
            Name of timestamp column

        Returns
        -------
        pd.DataFrame
            Dataframe with temporal features added
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract basic temporal components
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_year'] = df[timestamp_col].dt.dayofyear

        # Cyclical encoding (preserves periodicity)
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day of year (365-day cycle, approximate seasons)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        # Categorical temporal features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Peak/Off-peak (MISO on-peak: HE 7-22 weekdays)
        df['is_peak'] = (
            (df['hour'] >= 7) & (df['hour'] <= 22) & (df['day_of_week'] < 5)
        ).astype(int)

        # Super peak (highest load hours: HE 14-19 summer)
        df['is_super_peak'] = (
            (df['hour'] >= 14) & (df['hour'] <= 19) &
            (df['month'].isin([6, 7, 8])) & (df['day_of_week'] < 5)
        ).astype(int)

        # Season
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall',
        })

        # One-hot encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)

        # Holiday indicator (simplified - expand with actual holiday calendar)
        # Major MISO holidays: New Year's, Memorial Day, July 4, Labor Day, Thanksgiving, Christmas
        df['is_holiday'] = 0  # Placeholder - implement holiday calendar

        return df

    @staticmethod
    def add_lag_features(
        df,
        value_col='shadow_price',
        group_by='constraint_id',
        lags=[1, 24, 168],  # 1h, 1day, 1week
    ):
        """
        Add lag features (time series dependencies).

        IMPORTANT: Only use for training on historical data.
        For real-time prediction, lags must be from realized values.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with time series data
        value_col : str
            Column to create lags for
        group_by : str
            Column to group by (e.g., constraint_id)
        lags : list
            Lag periods in hours

        Returns
        -------
        pd.DataFrame
            Dataframe with lag features
        """
        df = df.copy()
        df = df.sort_values(['constraint_id', 'timestamp'])

        for lag in lags:
            col_name = f'{value_col}_lag{lag}h'
            df[col_name] = df.groupby(group_by)[value_col].shift(lag)

        # Rolling statistics (trailing window)
        for window in [24, 168]:  # 1 day, 1 week
            # Mean
            df[f'{value_col}_rolling_mean_{window}h'] = (
                df.groupby(group_by)[value_col]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Max
            df[f'{value_col}_rolling_max_{window}h'] = (
                df.groupby(group_by)[value_col]
                .rolling(window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

            # Binding frequency
            df[f'binding_freq_{window}h'] = (
                df.groupby(group_by)[value_col]
                .rolling(window, min_periods=1)
                .apply(lambda x: (x > 0).mean())
                .reset_index(level=0, drop=True)
            )

        return df
```

---

## 4. Implementation Roadmap

### Phase 1: Data Pipeline Development (Week 1)

**Objectives**:
- Load and validate flow density data
- Fetch historical shadow prices via `get_da_shadow()` API
- Align datasets temporally
- Initial data quality assessment

**Tasks**:
```python
week1_tasks = [
    # Data loading
    "Implement MISOFlowDensityLoader class",
    "Test loading single auction month",
    "Test loading historical range (2017-2024)",
    "Profile memory usage and loading time",

    # Shadow price retrieval
    "Test get_da_shadow() API for various date ranges",
    "Understand constraint class_type taxonomy",
    "Document shadow price data schema",
    "Validate timestamp alignment with flow density",

    # Data quality
    "Check for missing auction months",
    "Identify constraints with insufficient binding events",
    "Validate density distribution integrity (sum to 1)",
    "Document data coverage statistics",

    # Initial EDA
    "Analyze binding frequency by constraint",
    "Visualize shadow price distributions",
    "Correlate flow statistics with binding events",
    "Identify most/least active constraints",
]
```

**Deliverables**:
- [ ] Data loading pipeline tested on 2+ years data
- [ ] Shadow price dataset for all constraints
- [ ] Data quality report (coverage, missing values, outliers)
- [ ] Constraint binding frequency analysis
- [ ] Initial correlation analysis (flow features vs shadow price)

**Success Criteria**:
- Load 2+ years of data in < 5 minutes
- Successfully fetch shadow prices for 90%+ of timestamps
- Identify at least 20 constraints with >5% binding frequency

### Phase 2: Feature Engineering (Week 2)

**Objectives**:
- Implement distributional feature extraction
- Create temporal features
- Build complete feature matrix
- Feature selection and correlation analysis

**Tasks**:
```python
week2_tasks = [
    # Feature extraction
    "Implement FlowDensityFeatureExtractor class",
    "Extract features for all density distributions",
    "Validate feature calculation correctness",
    "Profile feature extraction performance",

    # Temporal features
    "Implement TemporalFeatureEngineer class",
    "Add cyclical temporal encodings",
    "Create peak/off-peak indicators",

    # Feature engineering pipeline
    "Create sklearn Pipeline for feature extraction",
    "Implement feature scaling/normalization",
    "Handle missing values appropriately",

    # Feature analysis
    "Correlation analysis (features vs shadow price)",
    "Feature importance (Random Forest baseline)",
    "Identify redundant features (high correlation)",
    "Document top 20 most predictive features",
]
```

**Deliverables**:
- [ ] FlowDensityFeatureExtractor class with 30+ features
- [ ] Complete feature matrix for all constraints
- [ ] Feature correlation heatmap
- [ ] Top 20 predictive features identified
- [ ] Feature engineering pipeline ready

**Success Criteria**:
- Extract 30+ features from each density distribution
- Feature extraction for 1M samples in < 10 minutes
- At least 10 features with |correlation| > 0.3 with binding

### Phase 3: Baseline Models and Class Imbalance Handling (Week 3)

**Objectives**:
- Implement proper train-validation-test split
- Train baseline classification models
- Experiment with class imbalance techniques
- Establish performance benchmarks

**Tasks**:
```python
week3_tasks = [
    # Data splitting
    "Implement time series cross-validation (TimeSeriesSplit)",
    "Create train (60%), validation (20%), test (20%) splits",
    "Ensure no data leakage (temporal ordering)",
    "Stratify by constraint_id and season",

    # Baseline classifiers
    "Train logistic regression (class_weight='balanced')",
    "Train Random Forest (class_weight='balanced')",
    "Train LightGBM classifier (scale_pos_weight tuning)",

    # Class imbalance techniques
    "Experiment with different class weight ratios",
    "Try threshold tuning (optimize F1 vs precision/recall)",
    "Implement evaluation metrics for imbalanced data",

    # Baseline performance
    "Document confusion matrices for each model",
    "Calculate F1, precision, recall, PR-AUC",
    "Identify best baseline classifier",
]
```

**Deliverables**:
- [ ] Proper time series cross-validation framework
- [ ] 3+ baseline classifiers trained
- [ ] Class imbalance experiments documented
- [ ] Evaluation metrics dashboard
- [ ] Best baseline identified (target: F1 > 0.60)

**Success Criteria**:
- F1 score > 0.60 on validation set
- Recall (binding detection) > 0.70
- Precision > 0.50
- PR-AUC > 0.65

### Phase 4: Two-Stage Model Development (Week 4-5)

**Objectives**:
- Implement two-stage hybrid architecture
- Optimize Stage 1 (classification)
- Optimize Stage 2 (regression)
- Hyperparameter tuning for both stages

**Tasks**:
```python
week4_5_tasks = [
    # Stage 1 optimization (Week 4)
    "Implement TwoStageHybridModel class",
    "Hyperparameter tuning for LightGBM classifier (Optuna)",
    "Optimize class weights and probability threshold",
    "Ensemble multiple classifiers (voting, stacking)",
    "Cross-validation for stage 1",

    # Stage 2 optimization (Week 4)
    "Train LightGBM regressor on binding events only",
    "Handle log-transformation for skewed shadow prices",
    "Hyperparameter tuning for regressor (Optuna)",
    "Implement prediction interval estimation",

    # Combined optimization (Week 5)
    "Joint optimization of probability threshold",
    "Test different regressor architectures (XGBoost, RF)",
    "Ensemble stage 2 models",
    "Walk-forward cross-validation",

    # Advanced techniques
    "Experiment with quantile regression (uncertainty)",
    "Try focal loss for classification (handle hard examples)",
    "Implement calibration (Platt scaling, isotonic)",
]
```

**Deliverables**:
- [ ] TwoStageHybridModel class fully implemented
- [ ] Optimized Stage 1 classifier (F1 > 0.75)
- [ ] Optimized Stage 2 regressor (MAE < 10 $/MW on binding)
- [ ] Combined model performance report
- [ ] Hyperparameter optimization results

**Success Criteria**:
- **Stage 1**: F1 > 0.75, Recall > 0.80, Precision > 0.65
- **Stage 2**: MAE < 10 $/MW, R² > 0.60 (on binding events)
- **Combined**: Overall MAE < 5 $/MW (including zeros)

### Phase 5: Constraint-Specific Modeling (Week 6)

**Objectives**:
- Train individual models for high-value constraints
- Implement constraint clustering for similar constraints
- Build ensemble across multiple constraints
- Assess per-constraint performance

**Tasks**:
```python
week6_tasks = [
    # Constraint segmentation
    "Identify high-value constraints (>10% binding frequency)",
    "Cluster constraints by binding patterns",
    "Group constraints by geographic zone/interface",

    # Per-constraint models
    "Train dedicated models for top 10 constraints",
    "Compare per-constraint vs. global model performance",
    "Assess data sufficiency per constraint",

    # Constraint features
    "Add constraint-specific features (historical binding rate)",
    "Include constraint limit as feature",
    "Add constraint type (flowgate, interface, local)",

    # Model selection by constraint
    "Implement automatic model selection per constraint",
    "Fallback to global model for data-scarce constraints",
]
```

**Deliverables**:
- [ ] Constraint clustering analysis
- [ ] Top 10 constraints with dedicated models
- [ ] Performance comparison: per-constraint vs global
- [ ] Model selection framework
- [ ] Constraint-level performance report

**Success Criteria**:
- Top 10 constraints: F1 > 0.80, MAE < 8 $/MW
- Global model: F1 > 0.75 for all constraints
- Automatic model selection working

### Phase 6: Production Pipeline and Deployment (Week 7-8)

**Objectives**:
- Production-ready code with proper testing
- Model versioning and experiment tracking
- Prediction API implementation
- Monitoring and retraining framework

**Tasks**:
```python
week7_8_tasks = [
    # Production code (Week 7)
    "Refactor code to production standards (type hints, docstrings)",
    "Implement comprehensive unit tests (pytest)",
    "Create prediction pipeline (data load → predict → output)",
    "Optimize for inference speed (<100ms per constraint)",

    # Model management (Week 7)
    "Set up MLflow for experiment tracking",
    "Implement model versioning and serialization",
    "Create model card documentation",
    "Store trained models with metadata",

    # API and deployment (Week 8)
    "Build prediction API (FastAPI or similar)",
    "Implement batch prediction endpoint",
    "Create monitoring dashboard (data drift, performance)",
    "Set up retraining triggers (performance degradation)",

    # Documentation and handoff (Week 8)
    "Complete technical documentation",
    "Create user guide for prediction API",
    "Document retraining procedures",
    "Conduct knowledge transfer session",
]
```

**Deliverables**:
- [ ] Production-ready codebase (tested, documented)
- [ ] MLflow experiment tracking active
- [ ] Prediction API deployed
- [ ] Monitoring dashboard operational
- [ ] Complete documentation package

**Success Criteria**:
- All unit tests passing (>90% coverage)
- Prediction latency < 100ms per constraint
- API uptime > 99%
- Documentation complete and reviewed

---

## 5. Class Imbalance Mitigation Strategies

### 5.1 Comprehensive Techniques

#### A. Algorithmic Approaches

**1. Class Weights (Recommended - Simple and Effective)**
```python
# LightGBM built-in support
model = LGBMClassifier(
    class_weight='balanced',  # Auto: n_samples / (n_classes * np.bincount(y))
    # or manual:
    # class_weight={0: 1.0, 1: 10.0}  # 10x weight for binding class
)

# XGBoost
scale_pos_weight = len(y[y==0]) / len(y[y==1])  # Ratio of negative/positive
model = XGBClassifier(scale_pos_weight=scale_pos_weight)

# Sklearn
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
```

**2. Focal Loss (Handles Hard Examples)**
```python
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard-to-classify examples.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ > 0 reduces loss for well-classified examples
    α balances positive/negative importance
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()

# Use with PyTorch model
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**3. Threshold Tuning (Post-Training)**
```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal probability threshold to maximize given metric.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    metric : str
        Metric to optimize ('f1', 'precision', 'recall')

    Returns
    -------
    float
        Optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Find threshold achieving target precision (e.g., >0.7)
        target_precision = 0.7
        optimal_idx = np.argmax(precision >= target_precision)
    elif metric == 'recall':
        # Find threshold achieving target recall (e.g., >0.8)
        target_recall = 0.8
        optimal_idx = np.argmax(recall >= target_recall)

    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# Usage
optimal_threshold = find_optimal_threshold(y_val, y_pred_proba, metric='f1')
y_pred_optimized = (y_pred_proba > optimal_threshold).astype(int)
```

#### B. Sampling Approaches (Use with Caution for Time Series!)

**4. Oversampling Minority Class**
```python
from imblearn.over_sampling import SMOTE, ADASYN

# SMOTE: Synthetic Minority Over-sampling Technique
# WARNING: Only use on training set AFTER time series split
# Do NOT use for validation/test sets

smote = SMOTE(
    sampling_strategy=0.3,  # Oversample minority to 30% of majority
    k_neighbors=5,
    random_state=42
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ADASYN: Adaptive Synthetic Sampling
# Focuses on samples that are harder to learn
adasyn = ADASYN(
    sampling_strategy=0.3,
    n_neighbors=5,
    random_state=42
)

X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
```

**IMPORTANT WARNING for Time Series**:
- **Do NOT shuffle** when using SMOTE/ADASYN
- Only oversample **within** each time fold of cross-validation
- Never oversample validation or test sets
- Synthetic samples may not respect temporal dependencies

**Safer Alternative for Time Series**:
```python
# Simple duplication of minority class (preserves temporal structure)
def oversample_minority_simple(X, y, target_ratio=0.3):
    """
    Oversample minority class by simple duplication.
    Safer for time series than SMOTE.
    """
    minority_mask = y == 1
    majority_mask = y == 0

    n_minority = minority_mask.sum()
    n_majority = majority_mask.sum()

    # Calculate how many duplicates needed
    target_minority = int(n_majority * target_ratio)
    n_duplicates = target_minority - n_minority

    if n_duplicates > 0:
        # Randomly sample with replacement from minority class
        minority_indices = np.where(minority_mask)[0]
        duplicate_indices = np.random.choice(
            minority_indices,
            size=n_duplicates,
            replace=True
        )

        # Combine original data with duplicates
        X_resampled = np.vstack([X, X[duplicate_indices]])
        y_resampled = np.concatenate([y, y[duplicate_indices]])
    else:
        X_resampled, y_resampled = X, y

    return X_resampled, y_resampled
```

**5. Undersampling Majority Class**
```python
from imblearn.under_sampling import RandomUnderSampler, NearMiss

# Random undersampling
rus = RandomUnderSampler(
    sampling_strategy=0.5,  # Majority class = 2x minority class
    random_state=42
)

X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# NearMiss: Intelligent undersampling (keeps hard examples)
nm = NearMiss(version=2, n_neighbors=3)
X_train_resampled, y_train_resampled = nm.fit_resample(X_train, y_train)
```

**Trade-offs**:
- ✅ Reduces training time
- ✅ Balances classes
- ❌ Loses information from majority class
- ❌ May underfit if too aggressive

#### C. Ensemble Methods

**6. Balanced Random Forest**
```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    sampling_strategy='all',  # Undersample each tree
    replacement=True,
    random_state=42,
)
```

**7. EasyEnsemble / BalancedBagging**
```python
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier

# EasyEnsemble: Multiple undersampled ensembles
easy_ensemble = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
)

# BalancedBagging: Bagging with resampling
balanced_bagging = BalancedBaggingClassifier(
    base_estimator=LGBMClassifier(n_estimators=100),
    n_estimators=10,
    sampling_strategy='all',
    random_state=42
)
```

### 5.2 Recommended Strategy for MISO Shadow Prices

```python
# RECOMMENDED: Combine multiple techniques

class ImbalanceHandledPipeline:
    """
    Complete pipeline for handling class imbalance.
    """

    def __init__(self):
        # Stage 1: Classification with multiple imbalance techniques
        self.classifiers = {
            # Primary: LightGBM with class weights
            'lgbm_weighted': LGBMClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=42,
            ),

            # Alternative: XGBoost with scale_pos_weight
            'xgb_weighted': XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.05,
                scale_pos_weight=10,  # Tune based on class ratio
                random_state=42,
            ),

            # Ensemble: Balanced Random Forest
            'balanced_rf': BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
            ),
        }

        # Ensemble: Voting classifier
        from sklearn.ensemble import VotingClassifier
        self.ensemble = VotingClassifier(
            estimators=list(self.classifiers.items()),
            voting='soft',  # Use probabilities
            weights=[0.4, 0.3, 0.3],  # Weight LightGBM higher
        )

        # Stage 2: Regression (unchanged)
        self.regressor = LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
        )

        self.optimal_threshold = 0.5  # Will be tuned

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train with validation-based threshold tuning.
        """
        # Stage 1: Train ensemble classifier
        y_binding = (y_train > 0.5).astype(int)
        self.ensemble.fit(X_train, y_binding)

        # Tune threshold on validation set
        y_val_binding = (y_val > 0.5).astype(int)
        y_val_proba = self.ensemble.predict_proba(X_val)[:, 1]

        self.optimal_threshold = find_optimal_threshold(
            y_val_binding,
            y_val_proba,
            metric='f1'
        )

        print(f"Optimal threshold: {self.optimal_threshold:.3f}")

        # Stage 2: Train regressor on binding events
        binding_mask = y_train > 0.5
        if binding_mask.sum() > 0:
            X_binding = X_train[binding_mask]
            y_magnitude = y_train[binding_mask]
            y_magnitude_log = np.log1p(y_magnitude)

            self.regressor.fit(X_binding, y_magnitude_log)

        return self

    def predict(self, X):
        """
        Predict with optimized threshold.
        """
        # Stage 1: Predict binding probability
        binding_proba = self.ensemble.predict_proba(X)[:, 1]
        predicted_binding = binding_proba > self.optimal_threshold

        # Stage 2: Predict magnitude
        predictions = np.zeros(len(X))

        if predicted_binding.sum() > 0:
            X_binding = X[predicted_binding]
            magnitude_log = self.regressor.predict(X_binding)
            magnitude = np.expm1(magnitude_log)
            predictions[predicted_binding] = magnitude

        return predictions
```

---

## 6. Evaluation Framework for Imbalanced Data

### 6.1 Metrics Hierarchy

```python
class ImbalancedEvaluator:
    """
    Comprehensive evaluation for imbalanced shadow price prediction.
    """

    @staticmethod
    def evaluate_stage1_classification(y_true, y_pred, y_pred_proba):
        """
        Evaluate binding classification (Stage 1).

        Focus on metrics appropriate for imbalanced data:
        - F1 score (harmonic mean of precision and recall)
        - PR-AUC (better than ROC-AUC for imbalanced data)
        - Confusion matrix breakdown
        """
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score,
            average_precision_score,
            precision_recall_curve,
            auc,
        )

        results = {}

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results['confusion_matrix'] = {
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
        }

        # Primary metrics
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1_score'] = (
            2 * results['precision'] * results['recall'] /
            (results['precision'] + results['recall'])
            if (results['precision'] + results['recall']) > 0 else 0
        )

        # Specificity (true negative rate)
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # ROC-AUC (threshold-independent)
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        # PR-AUC (BETTER for imbalanced data)
        results['pr_auc'] = average_precision_score(y_true, y_pred_proba)

        # Business metrics
        results['binding_detection_rate'] = results['recall']  # How many binding events caught
        results['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )

        return results

    @staticmethod
    def evaluate_stage2_regression(y_true, y_pred, shadow_price_full):
        """
        Evaluate shadow price magnitude prediction (Stage 2).
        Only evaluate on binding events.
        """
        # Filter to binding events only
        binding_mask = shadow_price_full > 0.5

        if binding_mask.sum() == 0:
            return {'error': 'No binding events in evaluation set'}

        y_true_binding = y_true[binding_mask]
        y_pred_binding = y_pred[binding_mask]

        results = {}

        # Standard regression metrics
        results['mae'] = mean_absolute_error(y_true_binding, y_pred_binding)
        results['rmse'] = np.sqrt(mean_squared_error(y_true_binding, y_pred_binding))
        results['r2'] = r2_score(y_true_binding, y_pred_binding)

        # MAPE (exclude very small shadow prices)
        mask_large = y_true_binding > 5.0
        if mask_large.sum() > 0:
            results['mape'] = mean_absolute_percentage_error(
                y_true_binding[mask_large],
                y_pred_binding[mask_large]
            )

        # Quantile performance
        for q in [0.5, 0.75, 0.90, 0.95]:
            threshold = np.quantile(y_true_binding, q)
            mask_q = y_true_binding > threshold
            if mask_q.sum() > 0:
                results[f'mae_q{int(q*100)}'] = mean_absolute_error(
                    y_true_binding[mask_q],
                    y_pred_binding[mask_q]
                )

        # Residual analysis
        residuals = y_true_binding - y_pred_binding
        results['residual_mean'] = np.mean(residuals)
        results['residual_std'] = np.std(residuals)
        results['residual_skewness'] = scipy.stats.skew(residuals)

        return results

    @staticmethod
    def evaluate_combined(y_true_shadow, y_pred_shadow, y_true_binding, y_pred_binding_proba):
        """
        Evaluate complete two-stage pipeline.

        Parameters
        ----------
        y_true_shadow : array-like
            True shadow prices (including zeros)
        y_pred_shadow : array-like
            Predicted shadow prices (including zeros)
        y_true_binding : array-like
            True binding labels (0/1)
        y_pred_binding_proba : array-like
            Predicted binding probabilities
        """
        results = {}

        # Overall shadow price prediction accuracy
        results['overall_mae'] = mean_absolute_error(y_true_shadow, y_pred_shadow)
        results['overall_rmse'] = np.sqrt(mean_squared_error(y_true_shadow, y_pred_shadow))

        # Separate performance for binding vs non-binding
        binding_mask_true = y_true_shadow > 0.5
        non_binding_mask_true = ~binding_mask_true

        # Performance on true binding events
        if binding_mask_true.sum() > 0:
            results['mae_when_binding'] = mean_absolute_error(
                y_true_shadow[binding_mask_true],
                y_pred_shadow[binding_mask_true]
            )

        # Performance on true non-binding events (should predict ~0)
        if non_binding_mask_true.sum() > 0:
            results['mae_when_non_binding'] = mean_absolute_error(
                y_true_shadow[non_binding_mask_true],
                y_pred_shadow[non_binding_mask_true]
            )

        # Economic simulation
        results['economic_metrics'] = ImbalancedEvaluator.simulate_trading_pnl(
            y_true_shadow,
            y_pred_shadow,
            y_pred_binding_proba
        )

        return results

    @staticmethod
    def simulate_trading_pnl(y_true_shadow, y_pred_shadow, y_pred_binding_proba):
        """
        Simulate trading P&L if predictions used for FTR bidding.

        Strategy: Bid on constraints where predicted binding probability > threshold
        P&L = Realized shadow price when we bid
        """
        results = {}

        for prob_threshold in [0.3, 0.5, 0.7]:
            # Bid when predicted probability exceeds threshold
            bid_mask = y_pred_binding_proba > prob_threshold

            if bid_mask.sum() > 0:
                # Realized shadow prices when we bid
                realized_when_bid = y_true_shadow[bid_mask]

                # Hit rate: % of bids where shadow price > 0
                hit_rate = (realized_when_bid > 0).mean()

                # Average P&L per bid
                avg_pnl = realized_when_bid.mean()

                # Sharpe ratio (simplified)
                sharpe = avg_pnl / (realized_when_bid.std() + 1e-6)

                results[f'threshold_{prob_threshold:.1f}'] = {
                    'n_bids': bid_mask.sum(),
                    'hit_rate': hit_rate,
                    'avg_pnl_per_bid': avg_pnl,
                    'sharpe_ratio': sharpe,
                }
            else:
                results[f'threshold_{prob_threshold:.1f}'] = {
                    'n_bids': 0,
                    'hit_rate': 0,
                    'avg_pnl_per_bid': 0,
                    'sharpe_ratio': 0,
                }

        return results
```

### 6.2 Visualization for Imbalanced Data

```python
import matplotlib.pyplot as plt
import seaborn as sns

class ImbalancedVisualizer:
    """
    Visualization tools for imbalanced classification.
    """

    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba, title=''):
        """
        Precision-Recall curve (better than ROC for imbalanced data).
        """
        from sklearn.metrics import precision_recall_curve, auc

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}', linewidth=2)
        ax.set_xlabel('Recall (Binding Detection Rate)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve{title}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        return fig

    @staticmethod
    def plot_confusion_matrix_normalized(y_true, y_pred, labels=['Non-Binding', 'Binding']):
        """
        Normalized confusion matrix (percentage of true class).
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Normalized Confusion Matrix', fontsize=14)

        return fig

    @staticmethod
    def plot_prediction_calibration(y_true, y_pred_proba, n_bins=10):
        """
        Calibration plot: Are predicted probabilities well-calibrated?
        """
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
        ax.plot(prob_pred, prob_true, 'o-', label='Model', linewidth=2)
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Probability', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        return fig

    @staticmethod
    def plot_threshold_tuning(y_true, y_pred_proba):
        """
        Show how precision, recall, F1 vary with threshold.
        """
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
        ax.plot(thresholds, recall[:-1], label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores[:-1], label='F1 Score', linewidth=2)

        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax.axvline(optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')

        ax.set_xlabel('Probability Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Threshold Tuning Analysis', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        return fig
```

---

## 7. MISO-Specific Considerations

### 7.1 Constraint Taxonomy

**MISO Constraint Types**:
```python
constraint_types = {
    'flowgate': {
        'description': 'Major transmission constraints (N-S, E-W flows)',
        'typical_binding': '10-20%',
        'shadow_price_range': '$5-50/MW',
        'example': 'Boswell - Grand Rapids 345kV',
    },
    'interface': {
        'description': 'Zone-to-zone transfer limits',
        'typical_binding': '5-15%',
        'shadow_price_range': '$10-100/MW',
        'example': 'MISO-PJM interface, MISO North-South',
    },
    'local': {
        'description': 'Local area constraints',
        'typical_binding': '1-5%',
        'shadow_price_range': '$2-20/MW',
        'example': 'Distribution-level constraints',
    },
    'hvdc': {
        'description': 'HVDC line limits',
        'typical_binding': '10-30%',
        'shadow_price_range': '$5-40/MW',
        'example': 'Square Butte HVDC',
    },
}
```

### 7.2 Seasonal Patterns

**MISO Seasonal Binding Characteristics**:
```python
seasonal_patterns = {
    'summer': {
        'months': [6, 7, 8],
        'key_drivers': ['Cooling load', 'High temperatures', 'Solar variability'],
        'binding_constraints': ['North-South flowgates', 'Local urban constraints'],
        'peak_hours': [14, 15, 16, 17, 18, 19],  # HE 14-19
        'expected_binding_rate': '15-25%',
    },
    'winter': {
        'months': [12, 1, 2],
        'key_drivers': ['Heating load', 'Wind variability', 'Generator outages'],
        'binding_constraints': ['Gas pipeline constraints', 'Wind-rich North'],
        'peak_hours': [7, 8, 17, 18, 19],  # Morning and evening peaks
        'expected_binding_rate': '10-20%',
    },
    'spring': {
        'months': [3, 4, 5],
        'key_drivers': ['High wind', 'Low load', 'Transmission maintenance'],
        'binding_constraints': ['North-South (wind export)', 'Maintenance outages'],
        'peak_hours': [10, 11, 12, 13],  # Midday wind peak
        'expected_binding_rate': '5-10%',
    },
    'fall': {
        'months': [9, 10, 11],
        'key_drivers': ['Moderate load', 'Shoulder season'],
        'binding_constraints': ['Planned outages', 'Maintenance'],
        'peak_hours': [14, 15, 16, 17],
        'expected_binding_rate': '3-8%',
    },
}
```

### 7.3 MISO-Specific Feature Engineering

```python
def add_miso_specific_features(df):
    """
    Add MISO-specific contextual features.

    Parameters
    ----------
    df : pd.DataFrame
        Base feature dataframe with temporal features

    Returns
    -------
    pd.DataFrame
        Enhanced with MISO-specific features
    """
    df = df.copy()

    # MISO peak periods
    df['is_miso_on_peak'] = (
        (df['hour'] >= 7) & (df['hour'] <= 22) & (df['day_of_week'] < 5)
    ).astype(int)

    # Summer super peak (highest congestion risk)
    df['is_summer_super_peak'] = (
        (df['month'].isin([6, 7, 8])) &
        (df['hour'].isin([14, 15, 16, 17, 18, 19])) &
        (df['day_of_week'] < 5)
    ).astype(int)

    # Winter morning peak (heating load)
    df['is_winter_morning_peak'] = (
        (df['month'].isin([12, 1, 2])) &
        (df['hour'].isin([7, 8])) &
        (df['day_of_week'] < 5)
    ).astype(int)

    # Spring wind peak (high wind, low load → reverse flow risk)
    df['is_spring_wind_peak'] = (
        (df['month'].isin([3, 4, 5])) &
        (df['hour'].isin([10, 11, 12, 13]))
    ).astype(int)

    # Constraint type indicators (if available)
    # df['is_flowgate'] = df['constraint_id'].str.contains('FLOWGATE', case=False)
    # df['is_interface'] = df['constraint_id'].str.contains('INTERFACE', case=False)

    return df
```

### 7.4 Data Quality Checks

```python
def validate_miso_data_quality(density_df, shadow_df):
    """
    MISO-specific data quality validation.

    Returns
    -------
    dict
        Validation results and warnings
    """
    results = {
        'warnings': [],
        'errors': [],
        'stats': {},
    }

    # Check density distribution integrity
    for idx, row in density_df.head(100).iterrows():
        density_sum = row['density_values'].sum()
        if not (0.95 < density_sum < 1.05):  # Should sum to ~1
            results['warnings'].append(
                f"Density sum {density_sum:.3f} != 1.0 for row {idx}"
            )

    # Check for missing auction months
    expected_months = pd.date_range(
        density_df['auction_month'].min(),
        density_df['auction_month'].max(),
        freq='MS'
    )
    actual_months = pd.to_datetime(density_df['auction_month'] + '-01').unique()
    missing_months = set(expected_months) - set(actual_months)

    if missing_months:
        results['warnings'].append(
            f"Missing auction months: {sorted(missing_months)}"
        )

    # Check binding rate distribution
    binding_rate = (shadow_df['shadow_price'] > 0.5).mean()
    results['stats']['overall_binding_rate'] = binding_rate

    if binding_rate < 0.01:
        results['warnings'].append(
            f"Very low binding rate ({binding_rate:.2%}). May indicate data issue."
        )
    elif binding_rate > 0.50:
        results['errors'].append(
            f"Unexpectedly high binding rate ({binding_rate:.2%}). Check data filter."
        )

    # Check for extreme shadow prices (potential data errors)
    extreme_threshold = 500  # $/MW
    extreme_mask = shadow_df['shadow_price'] > extreme_threshold
    if extreme_mask.sum() > 0:
        results['warnings'].append(
            f"{extreme_mask.sum()} shadow prices > ${extreme_threshold}/MW. "
            "Verify these are real events, not data errors."
        )

    # Check temporal alignment
    # TODO: Implement timestamp alignment check between density and shadow price

    return results
```

---

## 8. Production Deployment Considerations

### 8.1 Model Serving Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Prediction Request                    │
│  {constraint_id, auction_month, market_month, ...}      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                │
│  1. Load flow density from /opt/temp/.../density/       │
│  2. Extract distributional features                      │
│  3. Add temporal features                                │
│  4. Normalize and transform                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Model Selection Router                    │
│  - Check constraint_id                                   │
│  - Load constraint-specific model if available           │
│  - Else use global model                                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Two-Stage Prediction                        │
│  Stage 1: Binding Classification                         │
│    → Probability of binding                              │
│  Stage 2: Shadow Price Regression (if binding likely)    │
│    → Shadow price magnitude                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Prediction Response                     │
│  {                                                       │
│    shadow_price_prediction: float,                       │
│    binding_probability: float,                           │
│    confidence_interval: [lower, upper],                  │
│    model_version: str,                                   │
│    timestamp: datetime                                   │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Retraining Triggers

```python
class ModelPerformanceMonitor:
    """
    Monitor model performance and trigger retraining.
    """

    def __init__(
        self,
        performance_threshold_f1=0.70,
        performance_threshold_mae=12.0,
        data_drift_threshold=0.15,
    ):
        self.performance_threshold_f1 = performance_threshold_f1
        self.performance_threshold_mae = performance_threshold_mae
        self.data_drift_threshold = data_drift_threshold

        self.historical_performance = []

    def check_performance_degradation(self, current_metrics):
        """
        Check if model performance has degraded below threshold.

        Returns
        -------
        bool
            True if retraining needed
        """
        triggers = []

        # F1 score degradation
        if current_metrics['f1_score'] < self.performance_threshold_f1:
            triggers.append(
                f"F1 score {current_metrics['f1_score']:.3f} < "
                f"threshold {self.performance_threshold_f1}"
            )

        # MAE degradation
        if current_metrics['mae'] > self.performance_threshold_mae:
            triggers.append(
                f"MAE {current_metrics['mae']:.2f} > "
                f"threshold {self.performance_threshold_mae}"
            )

        # Trend analysis (if sufficient history)
        if len(self.historical_performance) >= 10:
            recent_f1 = [m['f1_score'] for m in self.historical_performance[-10:]]
            trend = np.polyfit(range(10), recent_f1, 1)[0]  # Linear trend

            if trend < -0.01:  # Declining F1
                triggers.append(f"F1 declining trend: {trend:.4f}")

        # Store current performance
        self.historical_performance.append(current_metrics)

        return len(triggers) > 0, triggers

    def check_data_drift(self, current_features, reference_features):
        """
        Detect data drift using KL divergence.

        Parameters
        ----------
        current_features : pd.DataFrame
            Recent feature distributions
        reference_features : pd.DataFrame
            Training feature distributions

        Returns
        -------
        bool
            True if significant drift detected
        """
        from scipy.stats import ks_2samp

        drift_detected = []

        for col in current_features.columns:
            if current_features[col].dtype in [np.float64, np.int64]:
                # Kolmogorov-Smirnov test
                stat, pval = ks_2samp(
                    current_features[col].dropna(),
                    reference_features[col].dropna()
                )

                if pval < 0.01:  # Significant difference
                    drift_detected.append(col)

        drift_ratio = len(drift_detected) / len(current_features.columns)

        return drift_ratio > self.data_drift_threshold, drift_detected
```

### 8.3 Model Versioning

```python
import mlflow
from datetime import datetime

class ModelVersionManager:
    """
    Manage model versions using MLflow.
    """

    def __init__(self, experiment_name='miso_shadow_price'):
        mlflow.set_experiment(experiment_name)

    def save_model(
        self,
        model,
        metrics,
        features,
        metadata,
        model_name='two_stage_hybrid',
    ):
        """
        Save model with full tracking.
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(metadata)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                model_name,
                signature=mlflow.models.infer_signature(
                    features,
                    model.predict(features)
                )
            )

            # Log feature list
            mlflow.log_dict(
                {'features': list(features.columns)},
                'feature_list.json'
            )

            # Log training data statistics
            mlflow.log_dict(
                features.describe().to_dict(),
                'feature_stats.json'
            )

            # Log model card
            model_card = {
                'model_type': 'Two-Stage Hybrid (Classification + Regression)',
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'use_cases': 'Shadow price prediction for MISO constraints',
                'limitations': 'Not suitable for unprecedented events or topology changes',
            }
            mlflow.log_dict(model_card, 'model_card.json')

            run_id = mlflow.active_run().info.run_id
            print(f"Model saved with run_id: {run_id}")

            return run_id
```

---

## 9. Success Criteria and Validation

### Performance Targets (Reiterated)

| Metric | Stage | Baseline | Target | Stretch |
|--------|-------|----------|--------|---------|
| **F1 Score** | Stage 1 (Classification) | 0.60 | 0.75 | 0.85 |
| **Recall** | Stage 1 (Binding Detection) | 0.70 | 0.80 | 0.90 |
| **Precision** | Stage 1 | 0.50 | 0.65 | 0.75 |
| **PR-AUC** | Stage 1 | 0.65 | 0.75 | 0.85 |
| **MAE** | Stage 2 (Magnitude) | 12 $/MW | 8 $/MW | 6 $/MW |
| **R²** | Stage 2 | 0.50 | 0.65 | 0.75 |
| **Overall MAE** | Combined (all predictions) | 8 $/MW | 5 $/MW | 3 $/MW |
| **Hit Rate** | Trading Simulation | 60% | 70% | 80% |

### Business Value Validation

```python
def validate_business_value(predictions, actuals, cost_per_missed_binding=100):
    """
    Calculate business value of predictions.

    Parameters
    ----------
    predictions : dict
        Model predictions with binding_proba
    actuals : dict
        Actual shadow prices and binding status
    cost_per_missed_binding : float
        Opportunity cost of missing binding event ($/MW)

    Returns
    -------
    dict
        Business value metrics
    """
    # Scenario: Use predictions to decide which constraints to bid FTRs

    # True positives: Correctly identified binding → captured value
    tp_value = actuals['shadow_price'][
        (predictions['is_binding'] == True) & (actuals['shadow_price'] > 0)
    ].sum()

    # False negatives: Missed binding events → lost opportunity
    fn_cost = len(actuals['shadow_price'][
        (predictions['is_binding'] == False) & (actuals['shadow_price'] > 0)
    ]) * cost_per_missed_binding

    # False positives: Predicted binding but didn't → wasted bid cost
    fp_cost = (predictions['is_binding'] == True & (actuals['shadow_price'] == 0)).sum() * 10  # Assume $10/bid cost

    net_value = tp_value - fn_cost - fp_cost

    return {
        'captured_value': tp_value,
        'missed_opportunity_cost': fn_cost,
        'false_alarm_cost': fp_cost,
        'net_business_value': net_value,
        'value_capture_rate': tp_value / (tp_value + fn_cost) if (tp_value + fn_cost) > 0 else 0,
    }
```

---

## 10. Appendix

### A. File Structure
```
research_spice_shadow_price_pred/
├── data/
│   ├── external/                    # Raw MISO data (read-only)
│   │   └── density/ -> /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/
│   ├── processed/
│   │   ├── features_train.parquet
│   │   ├── features_val.parquet
│   │   ├── features_test.parquet
│   │   └── metadata.json
│   └── shadow_prices/
│       ├── da_shadow_2017_2024.parquet
│       └── binding_statistics.csv
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── density_loader.py       # MISOFlowDensityLoader
│   │   ├── shadow_price_api.py     # Wrapper for get_da_shadow()
│   │   └── alignment.py            # Temporal alignment utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── distributional.py       # FlowDensityFeatureExtractor
│   │   ├── temporal.py             # TemporalFeatureEngineer
│   │   ├── miso_specific.py        # MISO-specific features
│   │   └── pipeline.py             # Complete feature pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── two_stage.py            # TwoStageHybridModel
│   │   ├── imbalance_handlers.py   # Class weight, SMOTE, etc.
│   │   ├── constraint_specific.py  # Per-constraint model management
│   │   └── ensemble.py             # Ensemble strategies
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # ImbalancedEvaluator
│   │   ├── visualization.py        # ImbalancedVisualizer
│   │   └── business_value.py       # Trading simulation
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── predictor.py            # Production prediction API
│   │   ├── monitor.py              # ModelPerformanceMonitor
│   │   └── version_manager.py      # MLflow versioning
│   └── utils/
│       ├── __init__.py
│       ├── validation.py           # Data quality checks
│       └── config.py               # Configuration management
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_class_imbalance_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_two_stage_model.ipynb
│   ├── 06_constraint_specific.ipynb
│   └── 07_production_testing.ipynb
├── tests/
│   ├── test_data_loaders.py
│   ├── test_feature_extractors.py
│   ├── test_models.py
│   └── test_evaluation.py
├── configs/
│   ├── model_config.yaml
│   ├── feature_config.yaml
│   └── deployment_config.yaml
├── document/
│   ├── shadow_price_prediction_research_plan.md  # Original plan
│   ├── miso_shadow_price_implementation_plan.md  # This document
│   └── README.md
├── results/
│   ├── models/
│   │   ├── stage1_classifier_v1.pkl
│   │   ├── stage2_regressor_v1.pkl
│   │   └── two_stage_hybrid_v1.pkl
│   ├── predictions/
│   │   └── test_predictions_YYYY-MM-DD.csv
│   └── reports/
│       ├── model_performance_report.html
│       └── business_value_analysis.pdf
├── mlruns/                         # MLflow tracking
├── requirements.txt
└── README.md
```

### B. Key Dependencies
```txt
# requirements.txt

# Core ML
lightgbm>=4.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
optuna>=3.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0  # For parquet files

# Imbalanced learning
imbalanced-learn>=0.11.0

# Model management
mlflow>=2.8.0
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Time series
statsmodels>=0.14.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
hypothesis>=6.88.0
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-16 | Research Team | Initial MISO implementation plan with class imbalance focus |

**Next Steps**:
1. Review and approve implementation plan
2. Set up development environment
3. Begin Phase 1: Data pipeline development (Week 1)

---

**Critical Success Factor**: Properly handling class imbalance is THE key to success for this project. The two-stage hybrid approach addresses this systematically.
