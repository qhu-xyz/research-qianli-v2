# Feature Overlap Analysis: Stage 1 v0009 vs Stage 2 Regressor

## Current State

**Stage 1 classifier (v0009)**: 29 features
**Stage 2 frozen classifier (v0006)**: 13 features
**Stage 2 regressor**: 24 features = 13 (from v0006) + 11 additional

## Feature Inventory

### Core Features (in all configs)

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor |
|---|---------|----------|----------|-----------------|--------------|
| 1 | prob_exceed_110 | Y | Y | Y | Y |
| 2 | prob_exceed_105 | Y | Y | Y | Y |
| 3 | prob_exceed_100 | Y | Y | Y | Y |
| 4 | prob_exceed_95 | Y | Y | Y | Y |
| 5 | prob_exceed_90 | Y | Y | Y | Y |
| 6 | prob_below_100 | Y | Y | Y | Y |
| 7 | prob_below_95 | Y | Y | Y | Y |
| 8 | prob_below_90 | Y | Y | Y | Y |
| 9 | expected_overload | Y | Y | Y | Y |
| 10 | hist_da | Y | Y | Y | Y |
| 11 | hist_da_trend | Y | Y | Y | Y |

### Distribution Shape (in some configs)

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor | Notes |
|---|---------|----------|----------|-----------------|--------------|-------|
| 12 | density_skewness | Y | Y | Y | Y* | In v0006 clf, kept |
| 13 | density_kurtosis | Y | Y | Y | Y* | In v0006 clf, kept |
| 14 | density_cv | Y | N | Y** | N | Dropped in v0009 |
| 15 | density_mean | N | **Y** | N | Y | **Overlap: now in both** |
| 16 | density_variance | N | **Y** | N | Y | **Overlap: now in both** |
| 17 | density_entropy | N | **Y** | N | Y | **Overlap: now in both** |

*In S2, these are inherited as classifier features
**density_cv was in v0006 but not in v0009; current S2 frozen clf has it but v0009 doesn't

### Near-Boundary Bands

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor | Notes |
|---|---------|----------|----------|-----------------|--------------|-------|
| 18 | tail_concentration | N | **Y** | N | Y | **Overlap** |
| 19 | prob_band_95_100 | N | **Y** | N | Y | **Overlap** |
| 20 | prob_band_100_105 | N | **Y** | N | Y | **Overlap** |

### Network Topology (v0009 only)

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor | Notes |
|---|---------|----------|----------|-----------------|--------------|-------|
| 21 | sf_max_abs | N | Y | N | N | New in v0009, not in S2 |
| 22 | sf_mean_abs | N | Y | N | N | New in v0009, not in S2 |
| 23 | sf_std | N | Y | N | N | New in v0009, not in S2 |
| 24 | sf_nonzero_frac | N | Y | N | N | New in v0009, not in S2 |
| 25 | is_interface | N | Y | N | N | New in v0009, not in S2 |
| 26 | constraint_limit | N | Y | N | N | New in v0009, not in S2 |

### Derived Interactions (v0009 only)

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor | Notes |
|---|---------|----------|----------|-----------------|--------------|-------|
| 27 | hist_physical_interaction | N | Y | N | N | hist_da × prob_exceed_100 |
| 28 | overload_exceedance_product | N | Y | N | N | expected_overload × prob_exceed_105 |
| 29 | hist_da_max_season | N | Y | N | N | Seasonal max hist DA |
| 30 | band_severity | N | Y | N | N | prob_band_95_100 × expected_overload |
| 31 | sf_exceed_interaction | N | Y | N | N | sf_max_abs × prob_exceed_100 |
| 32 | hist_seasonal_band | N | Y | N | N | hist_da_max_season × prob_band_100_105 |

### Stage 2 Only (regressor-specific)

| # | Feature | S1 v0006 | S1 v0009 | S2 clf (frozen) | S2 regressor | Notes |
|---|---------|----------|----------|-----------------|--------------|-------|
| 33 | prob_exceed_85 | N | N | N | Y | Truly S2-specific |
| 34 | prob_exceed_80 | N | N | N | Y | Truly S2-specific |
| 35 | recent_hist_da | N | N | N | Y | Truly S2-specific |
| 36 | season_hist_da_1 | N | N | N | Y | Truly S2-specific |
| 37 | season_hist_da_2 | N | N | N | Y | Truly S2-specific |

## Summary

If we update the frozen classifier to v0009:

- **Classifier features**: 13 → 29 (add topology, interactions, bands, density features)
- **"Additional" regressor features after removing overlap**: 11 → 5
  - Keep: prob_exceed_85, prob_exceed_80, recent_hist_da, season_hist_da_1, season_hist_da_2
  - Remove (now in classifier): tail_concentration, prob_band_95_100, prob_band_100_105, density_mean, density_variance, density_entropy
- **Total regressor features**: 29 + 5 = 34

The 5 remaining unique regressor features are all about:
- Extended exceedance tail (prob_exceed_85, 80) — do very overloaded lines have bigger shadow prices?
- Temporal patterns in historical DA (recent_hist_da, seasonal components) — does recent price history predict magnitude?

## Implication

Stage 2's original value proposition was "the regressor sees features the classifier doesn't." With v0009, that unique feature space shrinks from 11 to 5. The remaining 5 features are reasonable but limited. The regressor's main value now comes from:

1. **The regression target itself** (log1p shadow price vs binary bind/not-bind) — same features can learn different things
2. **The gating mechanism** (training only on binding samples) — focuses the regressor on magnitude estimation
3. **Different HP** (400 trees, depth 5, LR 0.05 vs 200 trees, depth 4, LR 0.1) — deeper model for regression

This suggests that **the architectural value of stage 2 is NOT primarily in additional features**, but in the **task decomposition** (classification vs regression) and **training strategy** (all samples vs binding-only). This insight matters for the design direction.
