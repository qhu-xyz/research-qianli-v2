# Shadow Price Prediction Using Flow Percentage Density
## Research Plan and Methodology

**Project**: ML-based Shadow Price Prediction for Transmission Constraints
**Author**: Power Trading Quantitative Research Team
**Date**: 2025-11-16
**Status**: Research Planning Phase

---

## Executive Summary

This research plan outlines a machine learning approach to predict transmission constraint shadow prices using flow percentage density distributions derived from kernel density estimation (KDE) on historical generation and load patterns. The approach bypasses the need for generator cost curves by learning the relationship between flow distributions and realized shadow prices from historical data.

**Key Innovation**: Utilizing distributional features (flow percentage density) rather than point estimates to capture the full uncertainty in constraint loading, enabling better shadow price prediction without requiring detailed generator cost curves.

---

## 1. Problem Formulation

### 1.1 Market Context (FTR Trader Perspective)

**Shadow Price Definition**:
```
Shadow Price (λ) = Marginal value of relaxing a constraint by 1 MW
                 = ∂(Total Generation Cost) / ∂(Constraint Limit)
```

**Physical Interpretation**:
- When a constraint binds, shadow price represents the LMP differential it creates
- Shadow price = 0 when constraint is not binding
- Shadow price > 0 indicates congestion, creating price separation
- Magnitude reflects severity of constraint binding

**Business Value**:
- Shadow price forecasts enable FTR path valuation
- Identify high-value congestion opportunities
- Risk management for transmission-constrained positions
- Inform bidding strategies in FTR auctions

### 1.2 Technical Problem Statement

**Objective**: Predict constraint shadow prices (λ_t) given flow percentage density distribution

**Inputs**:
- Flow percentage density: P(flow% | topology, gen/load pattern)
  - Generated via KDE on 2 years historical same-season data
  - Represents probability distribution of constraint utilization
  - Updated topology ensures structural accuracy

**Output**:
- Shadow price prediction: λ̂_t ($/MWh or $/MW)
- Confidence intervals: [λ̂_lower, λ̂_upper]
- Probability of binding: P(λ > 0)

**Challenge**:
- Cannot run OPF due to missing generator cost curves
- Must learn shadow price from distributional features + historical realizations
- Handle non-linear, potentially discontinuous relationship

### 1.3 Modeling Approach

**Core Hypothesis**:
*Shadow prices are predictable from flow percentage density distributions because:*
1. Flow density captures likelihood and severity of constraint binding
2. Historical shadow prices embed market cost structure
3. Seasonal patterns in gen/load capture cost curve dynamics implicitly

**Model Type**: Supervised learning regression with distributional inputs

---

## 2. Data Requirements and Sources

### 2.1 Primary Data Assets

#### A. Flow Percentage Density (Predictor)
```python
# Structure
flow_density = {
    'constraint_id': str,
    'timestamp': pd.Timestamp,
    'season': str,  # Winter, Spring, Summer, Fall
    'density_bins': np.ndarray,  # [0%, 5%, 10%, ..., 100%]
    'density_values': np.ndarray,  # KDE probabilities
    'topology_version': str,
    'generation_date': pd.Timestamp
}
```

**Critical Attributes**:
- Bin resolution (recommend: 1% or 2% bins for granularity)
- Seasonal alignment (same-season historical patterns)
- Topology version tracking (structural changes matter)
- KDE bandwidth selection (affects smoothness vs. overfitting)

#### B. Historical Shadow Prices (Target)
```python
# Structure
shadow_price_history = {
    'constraint_id': str,
    'timestamp': pd.Timestamp,  # Hourly or 5-minute
    'shadow_price': float,  # $/MW or $/MWh
    'binding_status': bool,  # True if λ > threshold
    'lmp_differential': float,  # Price separation caused
    'market_type': str,  # Day-ahead vs Real-time
}
```

**Data Requirements**:
- Minimum 2 years historical shadow prices (same period as flow data)
- Hourly granularity minimum (5-minute if available)
- Aligned with flow density timestamps
- Include both binding and non-binding periods

#### C. Contextual Features (Enhancement)
```python
# Additional predictive features
contextual_data = {
    'load_level': float,  # Total system load
    'renewable_penetration': float,  # Wind/solar as % of load
    'temperature': float,  # Weather proxy
    'hour_of_day': int,
    'day_of_week': int,
    'month': int,
    'is_peak': bool,
    'season': str,
    'holiday': bool,
    'outages': list,  # Concurrent outages
}
```

### 2.2 Data Quality Requirements

**Flow Density Data**:
- [ ] Consistent KDE methodology across all constraints
- [ ] Topology version control (detect network changes)
- [ ] Sufficient historical samples (min 50 per season-hour combination)
- [ ] Outlier detection (extreme flow scenarios)
- [ ] Missing data handling strategy

**Shadow Price Data**:
- [ ] Clean separation of binding vs. non-binding periods
- [ ] Outlier investigation (extreme prices may be valid)
- [ ] Alignment with flow density timestamps
- [ ] Data source validation (market operator official data)
- [ ] Handle market design changes (rule changes affect prices)

---

## 3. Feature Engineering Strategy

### 3.1 Distributional Feature Extraction

**From Flow Percentage Density, extract:**

#### A. Statistical Moments
```python
features_statistical = {
    'flow_mean': E[flow%],  # Expected flow percentage
    'flow_median': Median[flow%],
    'flow_std': Std[flow%],  # Dispersion/uncertainty
    'flow_skewness': Skew[flow%],  # Tail asymmetry
    'flow_kurtosis': Kurt[flow%],  # Tail heaviness
    'flow_cv': Std/Mean,  # Coefficient of variation
}
```

#### B. Quantile Features
```python
features_quantiles = {
    'flow_q05': 5th percentile,
    'flow_q25': 25th percentile,
    'flow_q50': 50th percentile (median),
    'flow_q75': 75th percentile,
    'flow_q95': 95th percentile,
    'flow_q99': 99th percentile,  # Extreme scenarios
    'iqr': Q75 - Q25,  # Interquartile range
}
```

#### C. Binding-Specific Features
```python
# Probability and severity of constraint binding
features_binding = {
    'prob_binding': P(flow% > 95%),  # Likely to bind
    'prob_high_loading': P(flow% > 90%),
    'prob_critical': P(flow% > 98%),
    'expected_exceedance': E[flow% | flow% > 95%],  # Conditional expectation
    'tail_mass': Integral of density above 90%,
}
```

#### D. Distribution Shape Features
```python
features_shape = {
    'entropy': -Σ p(x) log p(x),  # Uncertainty measure
    'mode_location': argmax(density),  # Most likely flow
    'n_modes': Count of local maxima,  # Multimodality
    'concentration_ratio': max(density) / mean(density),
}
```

### 3.2 Temporal Features

```python
features_temporal = {
    # Cyclical encoding (preserves periodicity)
    'hour_sin': sin(2π * hour / 24),
    'hour_cos': cos(2π * hour / 24),
    'day_sin': sin(2π * day_of_week / 7),
    'day_cos': cos(2π * day_of_week / 7),
    'month_sin': sin(2π * month / 12),
    'month_cos': cos(2π * month / 12),

    # Categorical
    'is_peak': bool,  # On-peak hours
    'is_super_peak': bool,  # Highest load hours
    'is_weekend': bool,
    'is_holiday': bool,
    'season': OneHotEncoded(['Winter', 'Spring', 'Summer', 'Fall']),
}
```

### 3.3 Lag Features (Time Series Dependencies)

```python
features_lagged = {
    # Recent shadow price history (if available)
    'shadow_price_lag1h': λ_{t-1},
    'shadow_price_lag24h': λ_{t-24},  # Same hour yesterday
    'shadow_price_lag168h': λ_{t-168},  # Same hour last week

    # Rolling statistics
    'shadow_price_rolling_mean_24h': Mean(λ_{t-24:t}),
    'shadow_price_rolling_max_24h': Max(λ_{t-24:t}),
    'binding_frequency_7d': % hours binding in last 7 days,

    # Flow density changes
    'flow_mean_change_24h': flow_mean_t - flow_mean_{t-24},
    'flow_std_change_24h': flow_std_t - flow_std_{t-24},
}
```

### 3.4 Cross-Sectional Features

```python
features_network = {
    # If multiple constraints available
    'nearby_constraint_shadow_prices': [λ_1, λ_2, ..., λ_k],
    'correlated_constraint_flows': Correlations with related constraints,
    'zone_congestion_index': Aggregate congestion metric,
    'interface_total_flow': Sum of related interface flows,
}
```

---

## 4. Model Architecture Selection

### 4.1 Baseline Models (ML Engineer Perspective)

#### Model 1: Linear Regression with Distributional Features
```python
from sklearn.linear_model import Ridge, Lasso

model_linear = Ridge(alpha=1.0)
# Features: Statistical moments + quantiles + temporal
# Pros: Interpretable, fast, good baseline
# Cons: Cannot capture non-linearity in binding threshold
```

**Expected Performance**:
- R² ~ 0.4-0.6 (if relationship moderately linear)
- Best for well-behaved constraints with gradual binding

#### Model 2: Quantile Regression
```python
from sklearn.linear_model import QuantileRegressor

model_quantile = QuantileRegressor(quantile=0.5, alpha=0.1)
# Also train for quantile=0.1, 0.9 for confidence intervals
# Pros: Provides prediction intervals, robust to outliers
# Cons: Requires training multiple models for different quantiles
```

**Expected Performance**:
- Better for asymmetric shadow price distributions
- Provides uncertainty quantification naturally

### 4.2 Tree-Based Models (Primary Recommendation)

#### Model 3: XGBoost/LightGBM (Best for Tabular Data)
```python
import lightgbm as lgb

model_gbm = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
)
```

**Pros**:
- Handles non-linearity in binding threshold naturally
- Feature importance for interpretability
- Robust to outliers and missing values
- Fast training and prediction

**Cons**:
- Requires careful hyperparameter tuning
- Can overfit if not properly regularized

**Expected Performance**:
- R² ~ 0.7-0.85 (strong non-linear relationships)
- MAE: 20-40% of shadow price range

#### Model 4: Random Forest (Ensemble Baseline)
```python
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
```

**Pros**:
- Stable, less prone to overfitting than individual trees
- Out-of-bag error estimation
- Parallelizable

**Cons**:
- Slower than LightGBM
- Typically lower performance than gradient boosting

### 4.3 Advanced Models (If Baseline Insufficient)

#### Model 5: Deep Learning - Tabular Neural Network
```python
import torch
import torch.nn as nn

class ShadowPriceNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

**When to Use**:
- Very large dataset (>100k samples)
- Complex feature interactions
- When tree-based models plateau

**Pros**:
- Can learn complex non-linear patterns
- Flexible architecture

**Cons**:
- Requires more data
- Harder to interpret
- Longer training time

#### Model 6: Probabilistic Models - Gaussian Process
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
model_gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-5,
    n_restarts_optimizer=10,
)
```

**When to Use**:
- Small to medium datasets (<10k samples)
- Need uncertainty quantification
- Interpretable predictions with confidence

**Pros**:
- Natural uncertainty quantification
- Works well with limited data
- Principled Bayesian approach

**Cons**:
- O(n³) computational complexity (slow for large data)
- Difficult to scale

### 4.4 Ensemble Strategy (Recommended Final Approach)

```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('lgbm', model_lgbm),
    ('xgb', model_xgb),
    ('rf', model_rf),
], weights=[0.5, 0.3, 0.2])  # Weight based on validation performance
```

**Rationale**:
- Combines strengths of multiple models
- Reduces overfitting risk
- More robust predictions

---

## 5. Model Training and Validation Strategy

### 5.1 Data Splitting (Time Series Aware)

```python
# Critical: Respect temporal ordering, avoid look-ahead bias

# Approach 1: Simple Train-Validation-Test Split
train_data = data['2023-01-01':'2024-06-30']  # 18 months
val_data = data['2024-07-01':'2024-09-30']    # 3 months
test_data = data['2024-10-01':'2024-12-31']   # 3 months

# Approach 2: Walk-Forward Cross-Validation (Preferred)
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=24*7)  # 7-day gap to avoid leakage
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Train and evaluate
```

**Key Principles**:
- ✅ Train only on past data, validate/test on future
- ✅ Include gap between train/validation to simulate real forecasting
- ✅ Ensure all seasons represented in training
- ❌ Never shuffle time series data
- ❌ Don't use future information in features (strict causality)

### 5.2 Cross-Validation Strategy

```python
def time_series_cv_score(model, X, y, n_splits=5):
    """
    Time series cross-validation with walk-forward approach.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)
    scores = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'binding_accuracy': [],  # Classification of binding vs non-binding
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Train
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Predict
        y_pred = model.predict(X.iloc[val_idx])
        y_true = y.iloc[val_idx]

        # Evaluate
        scores['mae'].append(mean_absolute_error(y_true, y_pred))
        scores['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
        scores['r2'].append(r2_score(y_true, y_pred))
        scores['binding_accuracy'].append(
            accuracy_score(y_true > 0, y_pred > 0)
        )

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
```

### 5.3 Evaluation Metrics

#### Primary Metrics
```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

def evaluate_shadow_price_model(y_true, y_pred):
    """Comprehensive evaluation for shadow price predictions."""

    metrics = {}

    # Regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)

    # Percentage errors (excluding near-zero shadow prices)
    non_zero_mask = np.abs(y_true) > 1.0
    if non_zero_mask.sum() > 0:
        metrics['mape'] = mean_absolute_percentage_error(
            y_true[non_zero_mask],
            y_pred[non_zero_mask]
        )

    # Binding classification (critical for trading)
    binding_threshold = 0.5  # $/MW
    y_true_binding = y_true > binding_threshold
    y_pred_binding = y_pred > binding_threshold

    metrics['binding_accuracy'] = accuracy_score(y_true_binding, y_pred_binding)
    metrics['binding_precision'] = precision_score(y_true_binding, y_pred_binding)
    metrics['binding_recall'] = recall_score(y_true_binding, y_pred_binding)
    metrics['binding_f1'] = f1_score(y_true_binding, y_pred_binding)

    # Economic metrics
    # Simulate trading P&L if used for FTR bidding
    # Assume bid when predicted shadow price > threshold
    bid_threshold = 5.0  # $/MW
    bid_signal = y_pred > bid_threshold

    if bid_signal.sum() > 0:
        metrics['avg_profit_when_bid'] = y_true[bid_signal].mean()
        metrics['hit_rate'] = (y_true[bid_signal] > 0).mean()

    # Tail performance (extreme events matter for risk)
    high_shadow_mask = y_true > np.percentile(y_true, 90)
    if high_shadow_mask.sum() > 0:
        metrics['mae_high_congestion'] = mean_absolute_error(
            y_true[high_shadow_mask],
            y_pred[high_shadow_mask]
        )

    return metrics
```

#### Diagnostic Metrics
```python
def diagnostic_analysis(y_true, y_pred, feature_names, model):
    """
    Deep diagnostic analysis of model behavior.
    """
    diagnostics = {}

    # Residual analysis
    residuals = y_true - y_pred
    diagnostics['residual_mean'] = np.mean(residuals)  # Should be ~0
    diagnostics['residual_std'] = np.std(residuals)
    diagnostics['residual_skewness'] = scipy.stats.skew(residuals)
    diagnostics['residual_kurtosis'] = scipy.stats.kurtosis(residuals)

    # Heteroscedasticity test (residual variance vs. prediction)
    from scipy.stats import spearmanr
    diagnostics['heteroscedasticity_corr'] = spearmanr(
        np.abs(residuals),
        y_pred
    )[0]

    # Feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        diagnostics['top_10_features'] = importance_df.head(10)

    # Prediction calibration (for probabilistic models)
    # Check if predicted quantiles match empirical quantiles

    return diagnostics
```

### 5.4 Hyperparameter Optimization

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit

def objective(trial):
    """
    Optuna objective for LightGBM hyperparameter tuning.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X_train):
        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
            early_stopping_rounds=50,
            verbose=0,
        )
        y_pred = model.predict(X_train.iloc[val_idx])
        mae = mean_absolute_error(y_train.iloc[val_idx], y_pred)
        scores.append(mae)

    return np.mean(scores)

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best MAE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 6. Implementation Roadmap

### Phase 1: Data Preparation and EDA (Weeks 1-2)

**Objectives**:
- Consolidate flow density and shadow price datasets
- Perform comprehensive exploratory data analysis
- Validate data quality and alignment

**Tasks**:
```python
# Week 1: Data Collection and Cleaning
tasks_week1 = [
    "Load and validate flow percentage density data",
    "Load and validate historical shadow price data",
    "Check temporal alignment (timestamps match)",
    "Handle missing values and outliers",
    "Document data sources and versions",
]

# Week 2: Exploratory Data Analysis
tasks_week2 = [
    "Analyze shadow price distributions by season/hour",
    "Correlate flow statistics with shadow prices",
    "Identify binding patterns (frequency, severity)",
    "Visualize flow density distributions",
    "Statistical tests for stationarity, autocorrelation",
]
```

**Deliverables**:
- [ ] Data quality report
- [ ] EDA notebook with key insights
- [ ] Feature correlation analysis
- [ ] Initial hypotheses about predictive relationships

### Phase 2: Feature Engineering (Week 3)

**Objectives**:
- Extract comprehensive features from flow density distributions
- Create temporal and lag features
- Feature selection and dimensionality reduction

**Tasks**:
```python
tasks_week3 = [
    "Implement distributional feature extractors (moments, quantiles)",
    "Create binding-specific features (prob of binding, tail mass)",
    "Engineer temporal features (cyclical encoding)",
    "Generate lag features (if applicable)",
    "Feature correlation analysis and selection",
    "Create feature engineering pipeline (sklearn Pipeline)",
]
```

**Deliverables**:
- [ ] Feature extraction library
- [ ] Feature engineering pipeline
- [ ] Feature importance preliminary analysis
- [ ] Final feature set documentation

### Phase 3: Baseline Model Development (Week 4)

**Objectives**:
- Establish baseline performance with simple models
- Validate data pipeline and evaluation framework

**Tasks**:
```python
tasks_week4 = [
    "Implement train-validation-test split (time series aware)",
    "Train linear regression baseline",
    "Train random forest baseline",
    "Implement evaluation metrics framework",
    "Document baseline performance",
    "Residual analysis and diagnostics",
]
```

**Deliverables**:
- [ ] Baseline model performance report
- [ ] Evaluation metrics dashboard
- [ ] Residual diagnostic plots
- [ ] Initial performance benchmarks

### Phase 4: Advanced Model Development (Weeks 5-6)

**Objectives**:
- Train gradient boosting models (LightGBM, XGBoost)
- Hyperparameter optimization
- Model comparison and selection

**Tasks**:
```python
# Week 5: Model Training
tasks_week5 = [
    "Train LightGBM with default parameters",
    "Train XGBoost with default parameters",
    "Implement time series cross-validation",
    "Initial hyperparameter tuning (grid search)",
    "Feature importance analysis",
]

# Week 6: Optimization and Ensemble
tasks_week6 = [
    "Optuna hyperparameter optimization (100+ trials)",
    "Train ensemble models (stacking, voting)",
    "Model selection based on validation performance",
    "Final model training on full training set",
    "Comprehensive evaluation on test set",
]
```

**Deliverables**:
- [ ] Trained models with optimized hyperparameters
- [ ] Model comparison report
- [ ] Feature importance analysis
- [ ] Selected production model

### Phase 5: Model Validation and Diagnostics (Week 7)

**Objectives**:
- Rigorous out-of-sample testing
- Diagnostic analysis (residuals, calibration)
- Scenario testing and stress testing

**Tasks**:
```python
tasks_week7 = [
    "Comprehensive test set evaluation",
    "Residual analysis (normality, heteroscedasticity)",
    "Prediction interval calibration",
    "Scenario testing (extreme weather, high load)",
    "Model failure mode analysis",
    "Documentation of limitations and assumptions",
]
```

**Deliverables**:
- [ ] Model validation report
- [ ] Diagnostic analysis document
- [ ] Scenario test results
- [ ] Known limitations documentation

### Phase 6: Production Implementation (Week 8)

**Objectives**:
- Production-ready code
- Model deployment pipeline
- Monitoring and retraining framework

**Tasks**:
```python
tasks_week8 = [
    "Refactor code to production standards",
    "Create prediction API/pipeline",
    "Implement model versioning (MLflow)",
    "Set up monitoring (data drift, performance drift)",
    "Documentation (model card, API docs)",
    "Handoff to operations team",
]
```

**Deliverables**:
- [ ] Production prediction pipeline
- [ ] Model card documentation
- [ ] Deployment guide
- [ ] Monitoring dashboard

---

## 7. Risk Factors and Mitigation

### 7.1 Data Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Flow density not predictive** | High | Medium | Validate correlation in EDA; add contextual features |
| **Insufficient historical data** | High | Low | Minimum 2 years required; supplement with similar constraints |
| **Topology changes invalidate model** | Medium | Medium | Track topology versions; retrain when network changes |
| **Missing shadow price data** | Medium | Low | Validate data completeness; impute carefully if necessary |
| **KDE bandwidth selection** | Low | Medium | Sensitivity analysis on bandwidth; cross-validate |

### 7.2 Modeling Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Overfitting to historical patterns** | High | High | Rigorous cross-validation; regularization; ensemble |
| **Binding threshold non-linearity** | Medium | Medium | Tree-based models handle naturally; threshold features |
| **Extreme event underrepresentation** | High | Medium | Oversample extreme events; scenario analysis |
| **Temporal dependencies ignored** | Medium | Low | Include lag features; time series cross-validation |
| **Distribution shift (regime change)** | High | Low | Monitor model performance; retrain on recent data |

### 7.3 Business/Trading Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Model predictions used inappropriately** | High | Medium | Clear documentation of use cases and limitations |
| **Overconfidence in predictions** | Medium | High | Provide prediction intervals; uncertainty quantification |
| **Market microstructure changes** | Medium | Low | Monitor market rule changes; model retraining triggers |
| **Constraint characteristics change** | Medium | Medium | Regular model performance monitoring |

---

## 8. Model Limitations and Assumptions

### 8.1 Key Assumptions

1. **Historical Relationship Stability**:
   - Shadow price relationship to flow density is stable over time
   - Market cost structure doesn't change dramatically

2. **Flow Density Sufficiency**:
   - Flow percentage distribution contains sufficient information
   - KDE accurately represents future flow scenarios

3. **Seasonal Consistency**:
   - Same-season historical patterns are representative
   - 2 years of data captures typical variability

4. **Topology Accuracy**:
   - Latest topology correctly represents network
   - Topology changes are properly tracked

### 8.2 Known Limitations

**What the Model CAN Predict**:
- ✅ Shadow prices under normal operating conditions
- ✅ Binding probability based on flow patterns
- ✅ Relative congestion severity across constraints
- ✅ Seasonal and hourly patterns

**What the Model CANNOT Predict Well**:
- ❌ Shadow prices during unprecedented events (outside training distribution)
- ❌ Impact of major topology changes (new lines, retirements)
- ❌ Effects of market rule changes
- ❌ Exact shadow prices during extreme volatility
- ❌ Shadow prices without corresponding historical flow patterns

### 8.3 Model Boundary Conditions

**Use With Confidence**:
- Flow patterns similar to historical (within training distribution)
- Stable network topology
- Normal market operations
- Seasonal patterns aligned with training data

**Use With Caution**:
- Flow patterns at edge of historical range
- Recently modified network topology
- Transitional seasons
- Low-frequency binding constraints (sparse data)

**Do NOT Use**:
- Major topology changes without retraining
- Market design changes
- Unprecedented weather events
- Generator mix significantly different from training period

---

## 9. Success Criteria

### 9.1 Technical Performance Targets

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| **R² Score** | 0.50 | 0.70 | 0.80 |
| **MAE ($/MW)** | 15 | 10 | 7 |
| **RMSE ($/MW)** | 25 | 18 | 12 |
| **Binding Accuracy** | 75% | 85% | 90% |
| **Binding F1 Score** | 0.60 | 0.75 | 0.85 |

### 9.2 Business Value Targets

| Objective | Measurement | Target |
|-----------|-------------|--------|
| **FTR Bid Accuracy** | Hit rate on predicted profitable paths | >70% |
| **Risk Reduction** | VaR improvement vs. no model | 20% reduction |
| **Computational Efficiency** | Prediction latency | <100ms per constraint |
| **Operational Stability** | Model uptime | >99% |

### 9.3 Validation Checkpoints

- [ ] **Week 2**: Data quality validated, EDA complete
- [ ] **Week 4**: Baseline models trained, metrics framework working
- [ ] **Week 6**: Advanced models trained, performance targets met
- [ ] **Week 7**: Validation complete, diagnostics satisfactory
- [ ] **Week 8**: Production deployment ready

---

## 10. Future Enhancements

### 10.1 Short-term Improvements (3-6 months)

1. **Incorporate Weather Forecasts**:
   - Integrate temperature, wind, solar forecasts
   - Improve seasonal predictions

2. **Multi-Constraint Modeling**:
   - Joint prediction of correlated constraints
   - Network-wide congestion modeling

3. **Uncertainty Quantification**:
   - Conformal prediction for prediction intervals
   - Quantile regression forests

4. **Real-time Updates**:
   - Online learning for model adaptation
   - Incremental retraining

### 10.2 Long-term Research Directions (6-12 months)

1. **Deep Learning for Distribution Inputs**:
   - CNN/RNN to directly process flow density curves
   - Attention mechanisms for temporal patterns

2. **Causal Inference**:
   - Identify causal drivers of shadow prices
   - Counterfactual analysis (what-if scenarios)

3. **Reinforcement Learning**:
   - Optimize bidding strategy directly
   - Learn from realized P&L

4. **Physics-Informed ML**:
   - Embed power flow equations as constraints
   - Hybrid OPF-ML approach

---

## 11. Appendices

### A. Code Structure

```
research_spice_shadow_price_pred/
├── data/
│   ├── raw/                    # Raw flow density and shadow price data
│   ├── processed/              # Cleaned and aligned datasets
│   └── features/               # Engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loaders.py         # Data loading utilities
│   │   └── preprocessors.py   # Data cleaning
│   ├── features/
│   │   ├── distributional.py  # Flow density feature extraction
│   │   ├── temporal.py        # Temporal features
│   │   └── pipeline.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── baseline.py        # Linear, RF models
│   │   ├── gradient_boosting.py  # LightGBM, XGBoost
│   │   ├── ensemble.py        # Ensemble strategies
│   │   └── utils.py           # Model utilities
│   ├── evaluation/
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── diagnostics.py     # Model diagnostics
│   │   └── visualization.py   # Plotting utilities
│   └── deployment/
│       ├── predictor.py       # Production prediction API
│       └── monitor.py         # Model monitoring
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluation.py
├── configs/
│   ├── model_config.yaml      # Model hyperparameters
│   └── data_config.yaml       # Data paths and settings
├── document/
│   └── shadow_price_prediction_research_plan.md  # This document
├── results/
│   ├── models/                # Saved models
│   ├── predictions/           # Model predictions
│   └── reports/               # Performance reports
└── requirements.txt
```

### B. Technology Stack

**Core ML Libraries**:
- `scikit-learn`: Baseline models, preprocessing, metrics
- `lightgbm`: Primary gradient boosting model
- `xgboost`: Alternative gradient boosting
- `catboost`: Categorical feature handling (if needed)
- `optuna`: Hyperparameter optimization

**Data Processing**:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical functions, KDE validation

**Visualization**:
- `matplotlib`: Static plots
- `seaborn`: Statistical visualizations
- `plotly`: Interactive dashboards

**Model Management**:
- `mlflow`: Experiment tracking, model versioning
- `joblib`: Model serialization

**Time Series**:
- `statsmodels`: Time series diagnostics
- `tsfresh`: Automated time series feature extraction (optional)

**Testing**:
- `pytest`: Unit testing
- `hypothesis`: Property-based testing

### C. References and Resources

**Power Systems and Shadow Prices**:
- Schweppe et al. (1988): "Spot Pricing of Electricity"
- Wood & Wollenberg: "Power Generation, Operation, and Control"
- ISO/RTO Market Manuals (PJM, ERCOT, CAISO)

**Machine Learning for Power Systems**:
- Artificial Intelligence for the Grid: "Machine Learning in Power System Operations" (IEEE)
- Time Series Forecasting: "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
- Distribution Regression: "Distributional Random Forests" (NeurIPS papers)

**Statistical Methods**:
- Hastie et al.: "The Elements of Statistical Learning"
- Bishop: "Pattern Recognition and Machine Learning"
- Quantile Regression: Koenker & Bassett (1978)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-16 | Research Team | Initial research plan |

**Approval**:
- [ ] Technical Lead
- [ ] Trading Desk Manager
- [ ] Risk Management

**Next Review**: End of Phase 1 (Week 2)
