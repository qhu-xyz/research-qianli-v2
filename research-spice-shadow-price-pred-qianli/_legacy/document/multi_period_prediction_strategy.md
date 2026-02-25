# Multi-Period Prediction Strategy: Handling Data Scarcity with Linear Models

## Problem Statement

We need to predict shadow prices for different forecast horizons (f0, f1, f2, f3, q2, q3, q4) while addressing two challenges:
1. **Data Scarcity**: Some period types (e.g., f3) have limited historical data
2. **Model Limitations**: Linear models (LogisticRegression, ElasticNet) cannot effectively use `forecast_horizon` as a feature due to their inability to learn non-linear interactions

## Recommended Strategy: Hybrid Ensemble Approach

### Option A: Horizon-Stratified Ensembles (Recommended)

Train **separate ensemble weights** for different horizon groups, but share the same base model training data.

#### Implementation

```python
# Group horizons by similarity
HORIZON_GROUPS = {
    "short_term": [0, 1],  # f0, f1 - Most data, highest confidence
    "medium_term": [2, 3],  # f2, f3 - Moderate data
    "long_term": [4, 5, 6],  # q2, q3, q4 - Least data, lowest confidence
}

# For each horizon group, learn different ensemble weights
# Example for short_term:
#   XGBoost: 60%, LogReg: 40%
# Example for long_term:
#   XGBoost: 80%, LogReg: 20%  (trust tree model more when uncertain)
```

**Rationale**:
- **XGBoost** learns horizon effects automatically via `forecast_horizon` feature
- **LogReg/ElasticNet** focus on the core physics (density, flow) which is horizon-agnostic
- **Ensemble weights** adjust based on horizon uncertainty
- **Shared training data** maximizes data usage (no splitting by horizon)

#### Advantages
✅ Addresses data scarcity (all data used for all models)
✅ XGBoost handles horizon-specific patterns
✅ Linear models contribute stable baseline predictions
✅ Ensemble weights adapt to horizon uncertainty

#### Disadvantages
⚠️ Requires tuning 3 sets of ensemble weights instead of 1
⚠️ More complex configuration

---

### Option B: XGBoost-Only for Long Horizons

Use the ensemble (XGBoost + Linear) for short horizons, but **XGBoost-only** for long horizons.

```python
def get_models_for_horizon(horizon):
    if horizon <= 1:  # f0, f1
        return [XGBoost, LogReg, ElasticNet]  # Full ensemble
    elif horizon <= 3:  # f2, f3
        return [XGBoost, LogReg]  # Drop ElasticNet
    else:  # q2, q3, q4
        return [XGBoost]  # XGBoost only
```

**Rationale**:
- Linear models add value when data is abundant and patterns are stable
- For rare horizons, trust the model that can learn from `forecast_horizon`

#### Advantages
✅ Simpler logic
✅ Avoids ensemble complexity for uncertain predictions
✅ XGBoost can still learn from pooled data

#### Disadvantages
⚠️ Loses diversity for long-horizon predictions
⚠️ No fallback if XGBoost overfits

---

### Option C: Feature Engineering for Linear Models

Create **interaction features** specifically for linear models to capture horizon effects.

```python
# Add these features ONLY for LogReg/ElasticNet
interaction_features = [
    "horizon_x_prob_overload",  # horizon * prob_overload
    "horizon_x_density_100",  # horizon * density_100
    "horizon_x_curvature_100",  # horizon * curvature_100
    "horizon_squared",  # horizon²
    "is_short_term",  # horizon <= 1 (binary)
    "is_long_term",  # horizon >= 4 (binary)
]
```

**Rationale**:
- Manually encode the non-linear relationships that linear models can't learn
- `horizon_x_prob_overload` allows LogReg to learn: "High prob_overload matters more at short horizons"

#### Advantages
✅ Linear models can now use horizon information
✅ Maintains ensemble diversity across all horizons
✅ Interpretable (can see which interactions matter)

#### Disadvantages
⚠️ Requires domain knowledge to choose interactions
⚠️ Increases feature dimensionality (risk of overfitting for ElasticNet)
⚠️ Still limited compared to XGBoost's automatic interaction discovery

---

### Option D: Transfer Learning / Domain Adaptation

Train on abundant data (f0, f1), then **fine-tune** on scarce data (f3, q4).

```python
# Step 1: Pre-train on f0, f1 data
model_pretrained = train_on_abundant_data(f0_f1_data)

# Step 2: Fine-tune on f3 data (even if small)
model_f3 = fine_tune(model_pretrained, f3_data, learning_rate=0.01)
```

**Rationale**:
- The core physics (which constraints bind) is similar across horizons
- Only the uncertainty/variance changes with horizon
- Fine-tuning adapts the model to horizon-specific patterns

#### Advantages
✅ Leverages abundant data to bootstrap rare-horizon models
✅ Works well when f0 and f3 are similar (just noisier)

#### Disadvantages
⚠️ Requires separate model storage for each horizon
⚠️ XGBoost doesn't support fine-tuning natively (would need custom implementation)
⚠️ Complexity in deployment

---

## Final Recommendation: **Option A (Horizon-Stratified Ensembles)**

### Why This is Best

1. **Maximizes Data Usage**: All models train on all available data (pooled across horizons)
2. **Leverages Model Strengths**:
   - XGBoost learns horizon effects via `forecast_horizon` feature
   - Linear models provide stable baseline (horizon-agnostic physics)
3. **Adaptive Confidence**: Ensemble weights reflect uncertainty
   - Short-term: Balanced ensemble (both models confident)
   - Long-term: XGBoost-heavy (linear models less reliable)
4. **Minimal Code Changes**: Just modify ensemble weight selection logic

### Implementation Plan

#### 1. Update `config.py`
```python
@dataclass
class EnsembleConfig:
    # Define horizon-specific weights
    short_term_clf_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    medium_term_clf_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    long_term_clf_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])

    short_term_reg_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    medium_term_reg_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    long_term_reg_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])
```

#### 2. Update `models.py`
```python
def get_ensemble_weights(self, horizon, model_type="classifier"):
    """Select ensemble weights based on forecast horizon."""
    if horizon <= 1:
        weights = (
            self.config.ensemble.short_term_clf_weights
            if model_type == "classifier"
            else self.config.ensemble.short_term_reg_weights
        )
    elif horizon <= 3:
        weights = (
            self.config.ensemble.medium_term_clf_weights
            if model_type == "classifier"
            else self.config.ensemble.medium_term_reg_weights
        )
    else:
        weights = (
            self.config.ensemble.long_term_clf_weights
            if model_type == "classifier"
            else self.config.ensemble.long_term_reg_weights
        )
    return weights
```

#### 3. Update `prediction.py`
```python
# When making predictions, calculate horizon and use appropriate weights
horizon = (market_month.year - auction_month.year) * 12 + (
    market_month.month - auction_month.month
)
clf_weights = self.models.get_ensemble_weights(horizon, "classifier")
reg_weights = self.models.get_ensemble_weights(horizon, "regressor")

# Use these weights for ensemble averaging
y_pred_proba = weighted_average(clf_predictions, clf_weights)
y_pred_value = weighted_average(reg_predictions, reg_weights)
```

### Validation Strategy

1. **Holdout Test**: Reserve f2, f3 data for testing
2. **Metrics by Horizon**:
   - F1 Score for each horizon group
   - MAE/RMSE for each horizon group
3. **Ensemble Weight Tuning**: Use grid search or Bayesian optimization to find optimal weights for each horizon group

### Expected Outcomes

- **f0, f1**: High accuracy (abundant data, both models contribute)
- **f2, f3**: Good accuracy (moderate data, XGBoost-weighted)
- **q2, q3, q4**: Acceptable accuracy (scarce data, XGBoost-dominant, but linear models prevent overfitting)

---

## Alternative: If Simplicity is Preferred

If the horizon-stratified approach is too complex, use **Option B (XGBoost-Only for Long Horizons)** as a simpler fallback. This requires minimal code changes and still addresses data scarcity by pooling training data.
