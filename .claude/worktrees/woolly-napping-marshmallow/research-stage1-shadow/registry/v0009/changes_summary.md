# v0009 Changes Summary — Iter 2

## Hypothesis H11: Derived Interaction Features + colsample_bytree Tuning

### What Changed

1. **Added 3 derived interaction features** (26 → 29 features):
   - `band_severity` = `prob_band_95_100 * expected_overload` (monotone: +1) — combines near-binding mass with overload severity
   - `sf_exceed_interaction` = `sf_max_abs * prob_exceed_100` (monotone: +1) — combines network topology with physical exceedance
   - `hist_seasonal_band` = `hist_da_max_season * prob_band_100_105` (monotone: +1) — combines seasonal historical extremes with mild overload band

2. **Increased colsample_bytree** from 0.8 to 0.9 — ensures ~26/29 features per tree, reducing critical feature dropout for top-100 ranking

### Files Modified

| File | Change |
|------|--------|
| `ml/config.py` | Added 3 features to `FeatureConfig.step1_features`; changed `colsample_bytree` from 0.8 to 0.9 |
| `ml/features.py` | Added 3 new features to `interaction_cols` set and `with_columns` computation block |
| `ml/data_loader.py` | Added synthetic data generation for 3 new features in `_load_smoke()` |
| `ml/tests/test_config.py` | Updated feature count (26→29), monotone constraints string, feature names list, colsample_bytree assertion |

### Results (12-month benchmark, f0 onpeak)

| Metric | v0009 mean | v0008 (champion) | Delta | v0009 bot2 | L3 Floor | Status |
|--------|-----------|-------------------|-------|-----------|----------|--------|
| S1-AUC | 0.8495 | 0.8498 | -0.0003 | 0.8189 | 0.7999 | PASS |
| S1-AP | 0.4445 | 0.4418 | +0.0027 | 0.3712 | 0.3526 | PASS |
| S1-NDCG | 0.7359 | 0.7346 | +0.0013 | 0.6648 | 0.6463 | PASS |
| S1-VCAP@100 | 0.0266 | 0.0240 | +0.0026 | 0.0089 | -0.0139 | PASS |
| S1-BRIER | 0.1376 | 0.1383 | -0.0007 | 0.1452 | — | improved |

### Feature Importance (new features)

| Feature | Mean Gain | Rank (of 29) |
|---------|-----------|-------------|
| `hist_seasonal_band` | 11.75% | #2 |
| `sf_exceed_interaction` | 4.00% | #7 |
| `band_severity` | 1.38% | #11 |

Combined new feature importance: 17.13% — substantial signal contribution.

### Key Observations

- **VCAP@100 improved**: bot2 0.0089 vs champion 0.0061 (+0.0028). Primary target of this iteration.
- **NDCG maintained**: bot2 0.6648 vs champion 0.6663 (-0.0015), still well above floor.
- **hist_seasonal_band dominated**: At 11.75% mean gain (rank #2), this interaction captures seasonal binding patterns that individual features couldn't express.
- **sf_exceed_interaction solid**: At 4.00% gain, the network-topology × exceedance interaction contributes meaningfully.
- **band_severity modest**: At 1.38%, less impactful than expected — near-boundary mass × overload may be partially redundant with existing features.
- **2021-04 (worst NDCG month)**: NDCG 0.6529 vs champion 0.6510 (+0.0019). Slight improvement but still structurally weak.
- **All L3 floors pass** with comfortable headroom.
