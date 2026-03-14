# v0005 Changes Summary — Iteration 2 (feat-eng-20260303-060938)

## Hypothesis

H7: Does further training window expansion (14→18 months) continue the positive AUC trend seen in iter 1?

## Changes Made

### 1. Training window expansion: 14 → 18 months
- **File**: `ml/config.py` line 95
- **Change**: `train_months: int = 14` → `train_months: int = 18`
- This gives the model 80% more training data than v0's original 10-month window and 29% more than v0004's 14-month window

### 2. Feature importance diagnostic (new)
- **File**: `ml/benchmark.py`
- Added gain-based feature importance extraction during benchmark evaluation
- Captures per-month importance from each trained XGBoost model
- Saves to `registry/v0005/feature_importance.json` (separate from metrics.json)
- Implementation: importance dict injected into per-month metrics with `_feature_importance` key, popped before aggregation

### 3. Test update
- **File**: `ml/tests/test_config.py`
- Updated `test_pipeline_config_defaults` assertion: `train_months == 14` → `train_months == 18`

### No other changes
- No feature changes (17 features preserved)
- No hyperparameter changes (v0 defaults)
- No threshold changes (beta=0.7)
- No changes to evaluate.py or gates.json

## Results

### v0005 Aggregate Metrics (12 months, f0, onpeak, train_months=18)

| Metric | v0 | v0004 (iter1) | v0005 | Δ vs v0 | Δ vs v0004 |
|--------|-----|--------------|-------|---------|------------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | +0.0013 | -0.0002 |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | -0.0007 | -0.0023 |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | +0.0044 | -0.0012 |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | +0.0032 | -0.0007 |
| S1-BRIER | 0.1503 | 0.1516 | 0.1525 | +0.0022 | +0.0009 |
| S1-VCAP@500 | 0.0908 | 0.0843 | 0.0845 | -0.0063 | +0.0002 |

### Win/Loss vs v0 Baseline

| Metric | W/L | Mean Δ |
|--------|-----|--------|
| S1-AUC | 7W/5L | +0.0013 |
| S1-AP | 6W/6L | -0.0007 |
| S1-VCAP@100 | 10W/2L | +0.0044 |
| S1-NDCG | 8W/4L | +0.0032 |
| S1-BRIER | 3W/9L | +0.0022 |
| S1-VCAP@500 | 5W/7L | -0.0063 |

### Win/Loss vs v0004 (previous iteration, train_months=14)

| Metric | W/L | Mean Δ |
|--------|-----|--------|
| S1-AUC | 7W/5L | -0.0002 |
| S1-AP | 6W/6L | -0.0023 |
| S1-VCAP@100 | 6W/6L | -0.0012 |
| S1-NDCG | 7W/5L | -0.0007 |

### Gate Status
- **Group A**: All PASS
- **Group B**: All PASS
- **Overall**: PASS

### Outcome Assessment

**Diminishing returns confirmed.** v0005 (18-month window) shows virtually identical aggregate AUC to v0004 (14-month window), with AUC 0.8361 vs 0.8363 (-0.0002). The 14→18 expansion provides no marginal improvement — older 2019 data dilutes more than it diversifies.

vs v0 baseline, v0005 is essentially indistinguishable from v0004: same AUC (+0.0013 vs +0.0015), similar VCAP@100 (+0.0044 vs +0.0056), and AP is now slightly negative (-0.0007 vs +0.0015). The extra 4 months of training added noise to AP.

**Window expansion is exhausted as a lever.** The productive range was 10→14 months. Going further provides no benefit.

### Feature Importance (First Empirical Data)

Top features by mean gain across 12 months:

| Rank | Feature | Mean Gain | Std |
|------|---------|-----------|-----|
| 1 | hist_da_trend | 0.5392 | 0.0187 |
| 2 | hist_physical_interaction | 0.1425 | 0.0227 |
| 3 | hist_da | 0.1128 | 0.0056 |
| 4 | prob_below_90 | 0.0512 | 0.0108 |
| 5 | prob_exceed_90 | 0.0310 | 0.0087 |
| ... | ... | ... | ... |
| 16 | exceed_severity_ratio | 0.0038 | 0.0010 |
| 17 | density_skewness | 0.0031 | 0.0003 |

**Key findings:**
1. **hist_da_trend dominates** with 54% of total gain — by far the most informative feature
2. **hist_physical_interaction** (interaction feature) is #2 at 14% — validates iter 1's interaction features
3. **hist_da** is #3 at 11% — historical DA shadow price is key
4. **Physical features** (prob_exceed/below) account for ~18% collectively
5. **Distribution shape features** (skewness, kurtosis, cv) are collectively <1% of gain — candidate for pruning
6. **exceed_severity_ratio** and **density_skewness** are the weakest features (<0.4% gain each)

The model is overwhelmingly driven by historical shadow price signals (features #1, #2, #3 = 79% of gain). The physical flow features provide useful but secondary signal. Distribution shape features are near-zero contributors.
