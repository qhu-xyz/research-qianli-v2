# Shadow Price Prediction Research - Quick Start Guide

## 📁 Document Overview

This directory contains the research planning documentation for ML-based shadow price prediction using flow percentage density distributions.

## 🎯 **START HERE**: `GETTING_STARTED.md`

Quick start guide with setup instructions, reading order, and critical technical decisions.

## 📄 Main Documents

### ⭐ **miso_shadow_price_implementation_plan.md** (PRACTICAL IMPLEMENTATION)
**Use this for actual implementation with your MISO data!**

Comprehensive implementation plan with:
- **Critical Focus**: Handling severe class imbalance (85-95% shadow prices = 0)
- **Two-Stage Hybrid Model**: Classification (binding?) → Regression (magnitude)
- Real data paths: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
- API integration: `get_da_shadow(st, et, class_type)`
- Production-ready code templates (MISOFlowDensityLoader, TwoStageHybridModel, etc.)
- Class imbalance techniques: Class weights, SMOTE, threshold tuning, focal loss
- Evaluation framework for imbalanced data (F1, PR-AUC, not accuracy!)
- 8-week implementation roadmap

**Key Sections**:
- §1: Problem Statement - MISO data structure and class imbalance challenge
- §2: Two-Stage Architecture - Why and how it handles imbalance
- §3: Feature Engineering - Complete code for distributional feature extraction
- §5: Class Imbalance Strategies - Multiple techniques with code
- §6: Evaluation Framework - Metrics that actually matter for imbalanced data

### 📚 **shadow_price_prediction_research_plan.md** (THEORETICAL FOUNDATION)
General methodology and machine learning best practices:
- Problem formulation and business context
- Comprehensive feature engineering strategies
- Model architecture recommendations (beyond two-stage)
- Risk analysis and success criteria

**Key Sections**:
- §1: Problem Formulation - Market context and technical problem statement
- §3: Feature Engineering - Extract predictive features from flow density distributions
- §4: Model Selection - Recommended architectures (LightGBM, XGBoost, Ensemble)
- §6: Implementation Roadmap - Week-by-week execution plan
- §7: Risk Factors - Mitigation strategies for known challenges

## 🎯 Quick Start: What to Read First

### For Technical Implementation:
1. **Section 3**: Feature Engineering Strategy
   - How to extract features from flow percentage density
   - Distributional features (moments, quantiles, tail mass)
   - Temporal and lag features

2. **Section 4**: Model Architecture Selection
   - LightGBM/XGBoost as primary recommendation
   - Ensemble strategy for robustness
   - Hyperparameter optimization approach

3. **Section 5**: Validation Strategy
   - Time series cross-validation (critical!)
   - Evaluation metrics specific to shadow prices
   - Diagnostic analysis

### For Project Planning:
1. **Section 6**: Implementation Roadmap
   - 8-week timeline with clear milestones
   - Week-by-week deliverables
   - Resource requirements

2. **Section 9**: Success Criteria
   - Technical performance targets (R² > 0.70, MAE < 10 $/MW)
   - Business value targets (>70% hit rate)
   - Validation checkpoints

### For Risk Assessment:
1. **Section 7**: Risk Factors and Mitigation
   - Data risks (flow density not predictive, topology changes)
   - Modeling risks (overfitting, regime changes)
   - Business risks (inappropriate use, overconfidence)

2. **Section 8**: Model Limitations
   - What the model CAN predict
   - What the model CANNOT predict
   - Boundary conditions for safe usage

## 🔑 Key Insights

### Core Innovation
**Using flow percentage density distributions instead of point estimates** captures:
- Full uncertainty in constraint loading
- Tail risk (extreme flow scenarios)
- Probabilistic binding information

### Recommended Approach
1. **Feature Engineering**: Extract 30+ features from flow density (moments, quantiles, binding probabilities)
2. **Model**: LightGBM + XGBoost ensemble
3. **Validation**: Time series walk-forward cross-validation
4. **Target**: R² > 0.70, Binding Accuracy > 85%

### Critical Success Factors
- ✅ Time series cross-validation (avoid look-ahead bias)
- ✅ Proper feature engineering from distributions
- ✅ Ensemble models for robustness
- ✅ Regular retraining (topology/market changes)

## 📊 Expected Performance

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| R² | 0.50 | 0.70 | 0.80 |
| MAE | 15 $/MW | 10 $/MW | 7 $/MW |
| Binding Accuracy | 75% | 85% | 90% |

## 🚀 Next Steps

### Immediate (Week 1-2):
1. [ ] Review full research plan
2. [ ] Validate data availability (flow density + shadow prices)
3. [ ] Set up development environment
4. [ ] Begin exploratory data analysis

### Phase 1 Deliverables:
- Data quality report
- EDA notebook with correlation analysis
- Feature correlation study
- Initial hypotheses validation

## 📚 Technical Stack

**Core ML**: LightGBM, XGBoost, scikit-learn, Optuna
**Data**: NumPy, Pandas, SciPy
**Tracking**: MLflow
**Visualization**: Matplotlib, Seaborn, Plotly

## ⚠️ Important Warnings

### Data Handling
- **Never** shuffle time series data
- **Always** respect temporal ordering in train/validation splits
- **Include** gap between train/validation to avoid look-ahead bias

### Model Usage
- **Do NOT use** for major topology changes without retraining
- **Provide** prediction intervals, not just point estimates
- **Monitor** model performance continuously (data drift)

### Feature Engineering
- **Critical**: Extract distributional features correctly (moments, quantiles)
- **Test**: KDE bandwidth sensitivity
- **Validate**: Flow density alignment with shadow price timestamps

## 🔗 Related Resources

**Appendix A**: Recommended code structure
**Appendix B**: Technology stack details
**Appendix C**: Academic references and ISO/RTO market manuals

## 📞 Support

For questions or issues:
1. Review Section 8 (Model Limitations) for boundary conditions
2. Check Section 7 (Risk Factors) for known challenges
3. Consult Appendix C for additional references

---

**Last Updated**: 2025-11-16
**Version**: 1.0
**Status**: Research Planning Complete, Ready for Implementation
