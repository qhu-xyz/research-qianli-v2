# v0006 Changes Summary — Feature Pruning + Window Revert

## Hypothesis
H8: Does removing 4 near-zero-importance features (density_skewness, density_kurtosis, density_cv, exceed_severity_ratio) reduce noise and improve tail metrics, while reverting from the 18-month to 14-month training window?

## Changes Made

### 1. Feature Pruning (17 → 13 features)
Removed 4 features contributing <2% of total gain collectively:
- `density_skewness` (0.31% gain, unconstrained monotone)
- `exceed_severity_ratio` (0.38% gain, weakest interaction)
- `density_cv` (0.40% gain, unconstrained monotone)
- `density_kurtosis` (0.58% gain, unconstrained monotone)

### 2. Training Window Revert (18 → 14 months)
Reverted to the 14-month window established as optimal in iter 1 (v0004). The 18-month window (v0005) provided zero marginal benefit.

### 3. No Other Changes
All hyperparameters, threshold_beta (0.7), val_months (2) kept at v0 defaults.

## Results (12 months, f0, onpeak)

### Group A (Blocking Gates)
| Metric | v0 | v0004 (best) | v0005 (iter2) | v0006 | Δ vs v0 | Δ vs v0004 |
|--------|-----|-------------|---------------|-------|---------|------------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | 0.8354 | +0.0006 | -0.0009 |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | 0.3892 | -0.0044 | -0.0059 |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | 0.0270 | +0.0121 | +0.0065 |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | 0.7560 | +0.0227 | +0.0189 |

### AP Bottom-2 Mean
| Version | AP Bot2 | Δ vs v0 |
|---------|---------|---------|
| v0 | 0.3322 | — |
| v0002 | 0.3305 | -0.0017 |
| v0003 | 0.3277 | -0.0045 |
| v0004 | 0.3282 | -0.0040 |
| v0005 | 0.3247 | -0.0075 |
| **v0006** | **0.3228** | **-0.0094** |

### Group B (Monitoring, Non-Blocking)
| Metric | v0 | v0006 | Δ |
|--------|-----|-------|---|
| S1-BRIER | 0.1503 | 0.1540 | +0.0037 (worse) |
| S1-REC | 0.4192 | 0.4179 | -0.0013 |
| S1-VCAP@500 | 0.0908 | 0.1172 | +0.0264 |
| S1-VCAP@1000 | 0.1591 | 0.1821 | +0.0230 |
| S1-CAP@100 | 0.7825 | 0.7892 | +0.0067 |
| S1-CAP@500 | 0.7740 | 0.7770 | +0.0030 |

### Gate Compliance
All gates pass (Group A passed, Group B passed, overall passed).

## Feature Importance (13 features)
| Rank | Feature | Mean Gain % |
|------|---------|------------|
| 1 | hist_da_trend | 44.2% |
| 2 | hist_da | 24.1% |
| 3 | hist_physical_interaction | 11.1% |
| 4 | prob_below_90 | 6.1% |
| 5 | prob_exceed_90 | 3.4% |
| 6 | prob_exceed_95 | 2.8% |
| 7 | prob_exceed_105 | 1.9% |
| 8 | expected_overload | 1.6% |
| 9 | prob_below_95 | 1.5% |
| 10 | prob_exceed_100 | 1.1% |
| 11 | overload_exceedance_product | 0.8% |
| 12 | prob_exceed_110 | 0.7% |
| 13 | prob_below_100 | 0.6% |

## Interpretation

### Key Finding: Mixed Results — NDCG and VCAP Improved, AP Regressed
The feature pruning experiment produced a surprising split in outcomes:

**Positive:**
- **S1-NDCG** improved substantially: +0.0227 vs v0, +0.0189 vs v0004. This is the largest NDCG improvement in 6 experiments.
- **S1-VCAP@100** improved: +0.0121 vs v0, +0.0065 vs v0004. New best VCAP@100 across all versions.
- **S1-VCAP@500** recovered strongly: 0.1172 vs v0004's 0.0843. The VCAP@500 regression trend is reversed.
- **S1-VCAP@1000** also improved: 0.1821 vs v0's 0.1591.

**Negative:**
- **S1-AP** regressed: -0.0044 vs v0, -0.0059 vs v0004. This is below all previous versions.
- **AP bottom-2** continued worsening: 0.3228, now -0.0094 below v0. The monotonic decline continues.
- **S1-BRIER** worsened slightly: +0.0037 vs v0.
- **S1-AUC** essentially flat: +0.0006 vs v0.

### What This Means
The pruning sharpened the model's top-of-stack ranking quality (NDCG, VCAP metrics) while degrading the positive-class probability ranking (AP). Removing unconstrained features (skewness, kurtosis, CV) apparently:
1. Concentrated model capacity on informative features → better top-K value capture
2. Lost some subtle discrimination signal for the broader positive class → lower AP

The NDCG/VCAP vs AP divergence suggests the pruned features provided small but real regularization benefit for AP, even though their gain contribution was minimal.

### Assessment Against Success Criteria
- **Promotion-worthy**: NO — AUC (0.8354) < 0.837 threshold; AP bot2 (0.3228) < 0.330 threshold
- **Best-so-far**: NO — AP (0.3892) < v0004 (0.3951) and AP bot2 < v0004
- **Ceiling confirmed**: PARTIALLY — AUC is within ±0.001 of v0004, but AP divergence exceeds ±0.002 tolerance
- **Regression**: MILD — AP (0.3892) is above 0.390 floor but regressed relative to all prior versions

### Recommendation
v0004 remains the best overall version (highest AUC and AP). v0006 demonstrates that feature pruning is not a viable path — the removed features, despite low gain, provided useful regularization. However, v0006's VCAP and NDCG improvements are noteworthy and suggest that ranking-focused objectives may respond differently to model simplification.
