# Review — feat-eng-3-20260304-102111, Iteration 1

**Reviewer**: Claude
**Version**: v0009 (39 regressor features)
**Champion**: v0007 (34 regressor features)
**Date**: 2026-03-04

## Summary

v0009 adds 5 features to the regressor (`density_skewness`, `density_kurtosis`, `density_cv`, `season_hist_da_3`, `prob_below_85`), expanding from 34 to 39 features. The hypothesis was sound: these distributional shape features were already proven in the classifier but missing from the regressor. The screening correctly selected Hypothesis B (39 features) over Hypothesis A (37 features).

**However, v0009 is an exact duplicate of v0008.** The configs are byte-identical, the metrics are byte-identical. This iteration re-ran an experiment that was already completed in a prior batch. The version counter was incremented but no new information was generated. This is a process issue — the worker should have detected the duplicate and either (a) referenced v0008 directly or (b) skipped the iteration. The results themselves are valid, but the version is redundant.

## Gate-by-Gate Analysis (v0009 vs v0007 Champion)

### Group A (Hard Gates) — Three-Layer Detail

| Gate | v0007 Mean | v0009 Mean | Delta | L1 Mean | L2 Tail (fails) | v0007 Bot2 | v0009 Bot2 | L3 Regr | Overall |
|------|-----------|-----------|-------|---------|-----------------|-----------|-----------|---------|---------|
| EV-VC@100 | 0.0699 | 0.0762 | **+9.0%** | P (floor=0.0664) | P (0 fails) | 0.0071 | 0.0065 | P (-0.0006) | **P** |
| EV-VC@500 | 0.2294 | 0.2329 | **+1.5%** | P (floor=0.2179) | P (0 fails) | 0.0662 | 0.0718 | P (+0.0056) | **P** |
| EV-NDCG | 0.7513 | 0.7548 | **+0.5%** | P (floor=0.7137) | P (0 fails) | 0.6502 | 0.6446 | P (-0.0056) | **P** |
| Spearman | 0.3932 | 0.3910 | **-0.6%** | P (floor=0.3736) | P (0 fails) | 0.2669 | 0.2705 | P (+0.0036) | **P** |

All Group A gates pass all three layers. The Spearman mean degradation (-0.0022) is small and the tail actually improved (+0.0036 on bottom_2_mean).

### Group B (Monitor Gates)

| Gate | v0007 Mean | v0009 Mean | Delta | Overall |
|------|-----------|-----------|-------|---------|
| C-RMSE | 2916.4 | 2827.4 | **-3.1%** (improved) | P |
| C-MAE | 1151.5 | 1136.7 | **-1.3%** (improved) | P |
| EV-VC@1000 | 0.3173 | 0.3152 | -0.7% | P |
| R-REC@500 | 0.0355 | 0.0356 | +0.3% | P |

All Group B gates pass. C-RMSE and C-MAE improvements are notable — the new distributional features genuinely help regression calibration.

## Per-Month Analysis

### EV-VC@100 Month-by-Month Deltas (v0009 − v0007)

| Month | v0007 | v0009 | Delta | Direction |
|-------|-------|-------|-------|-----------|
| 2020-09 | 0.0270 | 0.0697 | +0.0427 | improved |
| 2020-11 | 0.1036 | 0.1491 | +0.0455 | improved |
| 2021-01 | 0.0544 | 0.0401 | -0.0143 | degraded |
| 2021-03 | 0.0223 | 0.0254 | +0.0031 | improved |
| 2021-05 | 0.0002 | 0.0012 | +0.0010 | improved |
| 2021-07 | 0.1119 | 0.1028 | -0.0091 | degraded |
| 2021-09 | 0.1223 | 0.2000 | **+0.0777** | large improvement |
| 2021-11 | 0.0392 | 0.0118 | -0.0274 | degraded |
| 2022-03 | 0.1239 | 0.0936 | -0.0303 | degraded |
| 2022-06 | 0.0141 | 0.0169 | +0.0028 | improved |
| 2022-09 | 0.0277 | 0.0298 | +0.0021 | improved |
| 2022-12 | 0.1922 | 0.1737 | -0.0185 | degraded |

**Pattern**: 7 months improved, 5 degraded. The +9% mean improvement is driven by 3 large wins (2020-09: +0.043, 2020-11: +0.046, 2021-09: +0.078). The degraded months (2021-11: -0.027, 2022-03: -0.030, 2022-12: -0.019) show the model is redistributing value capture rather than uniformly improving.

### Statistical Consistency

The improvement is **not uniformly distributed** across months. The coefficient of variation for EV-VC@100 deltas is very high. The +9% mean is credible but fragile — removing the single best month (2021-09) drops the mean improvement to approximately +4%.

### Seasonal Pattern in Weak Months

Persistent weak months across all versions:
- **2021-05**: EV-VC@100 near zero (0.0002→0.0012). This is structurally intractable at the regressor level — likely a classifier issue (very low precision in this month).
- **2022-06**: EV-VC@100 remains very low (0.014→0.017). EV-NDCG at 0.607, Spearman at 0.274. Summer month with unusual congestion patterns.
- **2021-11**: EV-VC@100 actually degraded (0.039→0.012), Spearman=0.267 (worst). Late fall — possible seasonal regime shift.
- **2022-09**: EV-VC@500=0.062 (worst), EV-NDCG=0.682.

The weak months cluster in late spring/summer (May, June) and fall (September, November). No version has meaningfully improved 2021-05 or 2022-06 — these appear to be classifier-limited months where P(binding) predictions are poor, and even perfect regression would not rescue EV rankings.

## Code Review Findings

1. **No ML code changes** — confirmed via `git diff`. The feature additions use raw data loader columns already defined in `_ALL_REGRESSOR_FEATURES`. This is correct and low-risk.

2. **Config correctness**: v0009/config.json has 39 features and 39 monotone constraints — lengths match. Feature order and constraint values match the direction plan.

3. **Duplicate version (PROCESS ISSUE)**: v0009 is byte-identical to v0008 (same config, same metrics, same per-month results). The worker re-ran an already-completed experiment. This wasted compute and created version counter pollution. The orchestrator should detect when a proposed config matches an existing version and short-circuit.

4. **Zero-filled features**: 5 of 39 features (`hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`) are consistently zero-filled because the data loader doesn't provide them. This means the effective feature count is 34, not 39. The 5 new features bring the effective count from 29 to 34. This pre-existing condition inherited from v0007 should be addressed — zero columns waste tree splits and model capacity.

## Regression Quality (C-RMSE, C-MAE, Spearman on Binding-Only)

- **C-RMSE**: 2827 (vs 2916, -3.1%) — solid improvement in regression accuracy
- **C-MAE**: 1137 (vs 1152, -1.3%) — modest improvement
- **Spearman**: 0.391 (vs 0.393, -0.6%) — slight degradation in rank correlation

The regression calibration improvements (C-RMSE, C-MAE) are meaningful and directionally correct — distributional shape features help estimate magnitudes. The slight Spearman degradation suggests the model is becoming better calibrated in absolute terms but marginally less monotonic. This is acceptable for the business objective (EV ranking quality).

### Worst-month C-RMSE: 2022-06 at 5897 (vs mean 2827)
This month has C-MAE of 2280, nearly 2x the mean. The regressor struggles badly in this month, consistent with it being classifier-limited.

## EV Ranking Quality

The EV ranking metrics tell a positive story:
- **EV-VC@100 +9%**: Material improvement in top-100 value capture — the most economically important metric
- **EV-VC@500 +1.5%**: Modest broader improvement
- **EV-NDCG +0.5%**: Small ranking quality improvement

The pipeline is producing better EV rankings, particularly at the top. The improvement is concentrated in the most important metric (VC@100).

## Regressor Feature Importance

### Recommendations on Feature Set

1. **Remove zero-filled features**: `hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band` are always zero. They cannot contribute to model quality and waste tree splits. Pruning them would reduce noise and potentially improve generalization.

2. **Feature importance analysis needed**: With 39 nominal features (34 effective), it would be valuable to run a feature importance analysis to identify which of the 5 new features are actually contributing. If `season_hist_da_3` or `prob_below_85` show near-zero importance, they could be pruned back to a 37-feature model.

## Unified vs Gated Mode

Not explored in this iteration (correctly — the direction constrained to feature engineering only). The gated mode remains appropriate for now. Given that the weakest months (2021-05, 2022-06) appear classifier-limited, unified mode would not help — the regressor already can't fix classifier errors.

## Gate Calibration Assessment

### Current Floors (gates.json v4, 0.95x champion mean)

| Gate | Floor | v0009 Mean | Margin | Assessment |
|------|-------|-----------|--------|------------|
| EV-VC@100 | 0.0664 | 0.0762 | +14.8% | Reasonable |
| EV-VC@500 | 0.2179 | 0.2329 | +6.9% | Tight but OK |
| EV-NDCG | 0.7137 | 0.7548 | +5.8% | Tight |
| Spearman | 0.3736 | 0.3910 | +4.7% | Tight — could block borderline improvements |

The recalibrated gates (v4) are a significant improvement over the old v0-exact floors. However, Spearman at 4.7% margin is still tight. Given the -0.6% Spearman degradation in this iteration, a future iteration with slightly worse Spearman could fail L1 despite strong EV improvements. The floor of 0.3736 is appropriate if we want to enforce rank correlation quality, but the team should be aware that Spearman is the binding constraint.

### Tail Floor Assessment

| Gate | Tail Floor | Worst v0009 Month | Margin |
|------|-----------|-------------------|--------|
| EV-VC@100 | 0.000135 | 0.0012 (2021-05) | 8.9x headroom |
| EV-VC@500 | 0.0536 | 0.0617 (2022-09) | +15% headroom |
| EV-NDCG | 0.5434 | 0.6069 (2022-06) | +11.7% headroom |
| Spearman | 0.2363 | 0.2674 (2021-11) | +13.2% headroom |

Tail floors are appropriately calibrated. EV-VC@100 tail_floor is essentially zero (0.000135) — it provides no practical safety. This is fine given that 2021-05's near-zero performance is structurally unavoidable.

### L3 Noise Tolerance

The gate_calibration notes correctly identify that `noise_tolerance=0.02` is not scale-aware. For EV-VC@100 with bottom_2_mean ~0.007, the tolerance makes L3 trivially easy (threshold goes to -0.013). For this iteration it doesn't matter (all pass), but for future rigor, percentage-based tolerance would be better.

## Recommendations for Next Iteration

### Priority 1: Remove Zero-Filled Features (34 → 34 effective)
Remove the 5 features that are always zero: `hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`. This is pure cleanup — should not change results if XGBoost already ignores constant columns, but will reduce config pollution and make feature importance analysis cleaner.

### Priority 2: Hyperparameter Tuning (now that feature set is stabilized)
The feature set has been expanded from 29→34 effective features. The hyperparameters (mcw=25, depth=5, n_est=400) were tuned for 29 features. With 34 features:
- **colsample_bytree=0.8**: With more features, the model sees ~27 per tree. Could experiment with 0.7 to increase diversity.
- **min_child_weight=25**: Could be reduced to 15-20 to allow finer splits on the new distributional features.
- **n_estimators=400**: May benefit from increase to 500-600 with the lower learning rate, especially with more features.

### Priority 3: Value-Weighted Training
Explore `value_weighted=True` to emphasize high-value constraints during training. Since EV-VC@100 is the primary business metric, weighting training by shadow price magnitude could improve top-of-ranking predictions.

### Not Recommended
- **Unified regressor**: Weak months are classifier-limited, not regressor-limited. Unified mode would not help.
- **Additional raw features**: All available raw columns are now included. New features would require feature engineering (interactions, transformations).
- **Classifier changes**: Frozen — out of scope.

## Process Issues

1. **Duplicate version**: v0009 = v0008. The orchestrator should hash the config and check for duplicates before running a full benchmark.
2. **Version counter inflation**: v0008 was already registered with identical results. v0009 should have been a no-op or a reference to v0008.

## Verdict

**PROMOTABLE** — v0009/v0008 passes all gates on all three layers and shows material improvement on the primary business metric (EV-VC@100 +9%). The Spearman trade-off (-0.6%) is acceptable. However, since v0009 is identical to v0008, promoting either is equivalent. The team should acknowledge this is not a new result but a re-confirmation of v0008.
