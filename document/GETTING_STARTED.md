# Getting Started with MISO Shadow Price Prediction

## 📁 Documentation Overview

You now have two comprehensive research plans:

1. **`shadow_price_prediction_research_plan.md`** - **Theoretical Foundation**
   - General methodology and best practices
   - Model architecture recommendations
   - Feature engineering strategies
   - Comprehensive ML/statistical approaches

2. **`miso_shadow_price_implementation_plan.md`** - **Practical Implementation** ⭐
   - MISO-specific data paths and structure
   - **Critical focus on class imbalance** (most shadow prices = 0)
   - Two-stage hybrid modeling approach
   - Production-ready code templates
   - 8-week implementation roadmap

## 🚀 Quick Start (Start Here!)

### What You Need to Know First

**The Critical Challenge**:
```
Shadow Price Distribution in MISO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shadow Price = 0 (Non-binding): ████████████████████ 85-95%
Shadow Price > 0 (Binding):     ██ 5-15%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: A naive model predicting always 0 gets 90% accuracy but is USELESS!
Solution: Two-stage approach (classify binding → predict magnitude)
```

### Recommended Reading Order

#### For Implementation (Start Here):
1. **Read**: `miso_shadow_price_implementation_plan.md` Section 1-2
   - Understand the class imbalance problem
   - Learn the two-stage hybrid solution
   - ~20 minutes

2. **Read**: Section 3.1-3.2 (Data Loading & Feature Engineering)
   - How to load flow density from `/opt/temp/tmp/pw_data/...`
   - How to extract 30+ features from distributions
   - ~30 minutes

3. **Skim**: Section 4 (Two-Stage Model Architecture)
   - Stage 1: Binding classification (with class weights)
   - Stage 2: Shadow price regression (on binding events only)
   - ~15 minutes

4. **Review**: Section 6 (Implementation Roadmap)
   - 8-week plan with clear deliverables
   - Week-by-week tasks
   - ~10 minutes

#### For Deep Understanding:
5. **Read**: `shadow_price_prediction_research_plan.md` Section 3-4
   - Comprehensive feature engineering details
   - Alternative model architectures
   - ~45 minutes

## 📊 Key Technical Decisions (Already Made)

### ✅ Model Architecture: Two-Stage Hybrid

**Why?** Handles severe class imbalance effectively

```python
# Stage 1: Classification (Binding vs Non-Binding)
classifier = LGBMClassifier(
    class_weight='balanced',  # ← KEY: Handles imbalance
    n_estimators=500,
    max_depth=10,
)

# Stage 2: Regression (Shadow Price IF Binding)
regressor = LGBMRegressor(
    n_estimators=500,
    max_depth=8,
)
# Only trained on binding events (shadow_price > 0)
```

**Alternative Considered**: Single regression model
- ❌ Problem: Predicts mostly 0, misses binding events
- ❌ Poor F1 score for binding detection

### ✅ Class Imbalance Handling: Multiple Techniques

1. **Class Weights** (Primary) - Built into LightGBM
2. **Threshold Tuning** - Optimize F1 score on validation set
3. **Ensemble** - Combine multiple classifiers
4. **SMOTE** (Optional) - Use cautiously with time series

### ✅ Features: Distributional Statistics (30+ features)

**From Flow Percentage Density**:
- Statistical moments: mean, std, skewness, kurtosis
- Quantiles: Q05, Q25, Q50, Q75, Q90, Q95, Q99
- **Binding-specific** (most important!):
  - `prob_flow_gt_95`: P(flow > 95%)
  - `tail_mass_95`: Probability mass above 95%
  - `expected_exceedance_95`: Expected flow when > 95%

### ✅ Evaluation Metrics: Imbalance-Aware

**Primary Metrics**:
- **F1 Score** (not accuracy!) - Target: > 0.75
- **PR-AUC** (not ROC-AUC) - Target: > 0.75
- **Recall** (binding detection) - Target: > 0.80
- **MAE** (on binding events) - Target: < 8 $/MW

**Why not accuracy?** Always predicting 0 → 90% accuracy but useless!

## 🎯 Performance Targets

| Metric | What It Measures | Target | Why It Matters |
|--------|------------------|--------|----------------|
| **F1 Score** | Balance of precision/recall | > 0.75 | Catch binding events without too many false alarms |
| **Recall** | % of binding events detected | > 0.80 | Don't miss profitable FTR opportunities |
| **Precision** | % of predicted binding are correct | > 0.65 | Avoid wasting bid costs on false alarms |
| **MAE (Binding)** | Average error on shadow price magnitude | < 8 $/MW | Accurate valuation for profitable paths |
| **Hit Rate** | % profitable when we bid | > 70% | Business profitability |

## 🔧 Setup and First Steps

### Step 1: Verify Data Access

```bash
# Check flow density data
ls /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/auction_month=2024-01/

# Should see: market_month=YYYY-MM/ directories
```

### Step 2: Test Shadow Price API

```python
# Test get_da_shadow() function
shadow_data = get_da_shadow(
    st='2024-01-01',
    et='2024-01-31',
    class_type='constraint'
)

print(f"Retrieved {len(shadow_data)} shadow price records")
print(f"Binding rate: {(shadow_data['shadow_price'] > 0).mean():.2%}")
```

### Step 3: Explore Class Imbalance

```python
import pandas as pd

# Load one month of shadow prices
shadow_df = get_da_shadow('2024-01-01', '2024-01-31', 'constraint')

# Analyze binding rate
binding_rate = (shadow_df['shadow_price'] > 0.5).mean()
print(f"Binding rate: {binding_rate:.2%}")

# Distribution of shadow prices
print("\nShadow Price Distribution:")
print(shadow_df['shadow_price'].describe())

# This will show the severe imbalance!
```

### Step 4: Start with Week 1 Tasks

From `miso_shadow_price_implementation_plan.md` Section 6, Phase 1:

**Week 1 Tasks**:
```python
week1_checklist = [
    "✓ Implement MISOFlowDensityLoader class",
    "✓ Test loading single auction month",
    "✓ Test get_da_shadow() API",
    "✓ Analyze binding frequency by constraint",
    "✓ Visualize shadow price distributions",
    "✓ Document data coverage",
]
```

## 📚 Code Templates Available

The implementation plan includes complete, production-ready code:

1. **Data Loading**: `MISOFlowDensityLoader` class (Section 3.1)
2. **Feature Extraction**: `FlowDensityFeatureExtractor` class (Section 3.2)
3. **Temporal Features**: `TemporalFeatureEngineer` class (Section 3.3)
4. **Two-Stage Model**: `TwoStageHybridModel` class (Section 2.4)
5. **Imbalance Handlers**: Multiple techniques (Section 5)
6. **Evaluation**: `ImbalancedEvaluator` class (Section 6.1)
7. **Monitoring**: `ModelPerformanceMonitor` class (Section 8.2)

## ⚠️ Critical Warnings

### ❌ DON'T:
- **Never use accuracy** as primary metric (misleading with imbalance!)
- **Never shuffle** time series data in train/test split
- **Never use SMOTE** without understanding time series implications
- **Never skip** class imbalance handling (model will be useless!)

### ✅ DO:
- **Use F1 score and PR-AUC** as primary metrics
- **Use TimeSeriesSplit** for cross-validation
- **Apply class weights** in classification models
- **Validate binding detection** separately from magnitude prediction
- **Provide prediction intervals**, not just point estimates

## 🎓 Learning Resources

### Understanding Class Imbalance
- Section 5 of `miso_shadow_price_implementation_plan.md`
- Techniques: Class weights, threshold tuning, SMOTE, focal loss

### Two-Stage Modeling
- Section 2 of `miso_shadow_price_implementation_plan.md`
- Why it works: Separates binding classification from magnitude prediction

### MISO Market Context
- Section 7 of `miso_shadow_price_implementation_plan.md`
- Seasonal patterns, constraint types, peak periods

## 📞 Quick Reference

### Data Paths
```
Flow Density: /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/
Shadow Prices: get_da_shadow(st, et, class_type='constraint')
```

### Key Classes to Implement
```
MISOFlowDensityLoader → Load density parquet files
FlowDensityFeatureExtractor → Extract 30+ features
TwoStageHybridModel → Main prediction model
ImbalancedEvaluator → Evaluate with proper metrics
```

### Success Criteria
```
Stage 1 (Classification): F1 > 0.75, Recall > 0.80
Stage 2 (Regression): MAE < 8 $/MW on binding events
Combined: Overall MAE < 5 $/MW, Hit Rate > 70%
```

## 🚦 Next Actions

### Immediate (Today):
1. [ ] Read Sections 1-2 of `miso_shadow_price_implementation_plan.md`
2. [ ] Verify access to flow density data
3. [ ] Test `get_da_shadow()` API
4. [ ] Analyze class imbalance in your data

### This Week:
1. [ ] Implement `MISOFlowDensityLoader`
2. [ ] Load 1 month of flow density data
3. [ ] Extract shadow prices for same month
4. [ ] Calculate binding rates per constraint
5. [ ] Create initial visualizations

### Week 2:
1. [ ] Implement `FlowDensityFeatureExtractor`
2. [ ] Extract features for all constraints
3. [ ] Feature correlation analysis
4. [ ] Identify top 20 predictive features

## 💡 Pro Tips

1. **Start with one constraint**: Pick a high-binding constraint (>10% binding rate) for initial development

2. **Visualize class imbalance**: Create histograms showing shadow price distribution BEFORE modeling

3. **Baseline first**: Train a simple logistic regression with class weights to establish baseline

4. **Monitor both stages**: Track F1 (stage 1) and MAE (stage 2) separately

5. **Use MLflow**: Track all experiments from day 1 (prevents "which model was that?" problems)

## 📖 Full Documentation Index

1. **`README.md`** - This file (start here!)
2. **`miso_shadow_price_implementation_plan.md`** - Practical implementation ⭐
   - Section 1: Problem statement and data
   - Section 2: Two-stage model architecture
   - Section 3: Feature engineering
   - Section 4: Implementation roadmap (8 weeks)
   - Section 5: Class imbalance techniques
   - Section 6: Evaluation framework
   - Section 7: MISO-specific considerations
   - Section 8: Production deployment
3. **`shadow_price_prediction_research_plan.md`** - Theoretical foundation
   - Deep dive into ML methodology
   - Alternative architectures
   - Comprehensive feature engineering

---

**Ready to Start?** → Begin with Week 1, Task 1: Implement `MISOFlowDensityLoader` 🚀

**Questions?** → Review Section 7.4 (Data Quality Checks) and Section 5 (Class Imbalance) in the implementation plan
