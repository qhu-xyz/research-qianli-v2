# Claude Review — v0010 (feat-eng-3-20260303-104101, iter3)

## Summary

v0010 tests hypothesis H12: increasing n_estimators from 200 to 300 and decreasing learning_rate from 0.1 to 0.07, targeting finer-grained splits with a 29-feature model that was potentially under-treed. The result is a **null outcome** — all metrics are within noise of the champion v0009. The largest Group A mean delta is AP at -0.0021, well within the 0.02 noise tolerance. W/L ratios are at or near 50/50 across all Group A metrics (AUC 6W/5L/1T, AP 6W/6L, NDCG 5W/6L/1T, VCAP@100 5W/7L). No metric shows a consistent directional signal.

This was the predicted outcome from the direction document, which explicitly stated: "if results are flat, that confirms the model has reached its capacity ceiling with 29 features." The null result is informative — it closes the hyperparameter tuning search space for this feature set. The model's performance is bounded by the feature information, not by insufficient tree capacity. **v0009 should remain champion.** The pipeline is ready for HUMAN_SYNC.

## Gate-by-Gate Analysis (Group A)

### Three-Layer Gate Summary (v0010 vs v0009 champion)

| Gate | v0010 Mean | Floor | L1 | v0010 Bot2 | L3 Floor | L3 Margin | L3 | Overall |
|------|-----------|-------|-----|-----------|----------|-----------|-----|---------|
| S1-AUC | 0.8496 | 0.7848 | P (+0.065) | 0.8172 | 0.7989 | +0.0183 | P | **P** |
| S1-AP | 0.4424 | 0.3436 | P (+0.099) | 0.3748 | 0.3512 | +0.0236 | P | **P** |
| S1-VCAP@100 | 0.0254 | -0.0351 | P (+0.060) | 0.0070 | -0.0111 | +0.0181 | P | **P** |
| S1-NDCG | 0.7359 | 0.6833 | P (+0.053) | 0.6685 | 0.6448 | +0.0237 | P | **P** |

All Group A gates pass all 3 layers. No months below any tail_floor (L2 all pass with 0 failures).

### Mean Comparison (v0010 vs v0009)

| Metric | v0009 Mean | v0010 Mean | Delta | W/L | Interpretation |
|--------|-----------|-----------|-------|-----|----------------|
| S1-AUC | 0.8495 | 0.8496 | +0.0001 | 6W/5L/1T | Noise |
| S1-AP | 0.4445 | 0.4424 | **-0.0021** | 6W/6L | Noise (largest delta, still within tolerance) |
| S1-VCAP@100 | 0.0266 | 0.0254 | -0.0012 | 5W/7L | Noise |
| S1-NDCG | 0.7359 | 0.7359 | +0.0000 | 5W/6L/1T | Noise |
| S1-BRIER | 0.1376 | 0.1374 | -0.0002 | 6W/5L/1T | Noise (marginal improvement) |

### Bottom-2 Mean Comparison

| Metric | v0009 Bot2 | v0010 Bot2 | Delta | Assessment |
|--------|-----------|-----------|-------|------------|
| S1-AUC | 0.8189 | 0.8172 | -0.0017 | Slight tail degradation (within noise) |
| S1-AP | 0.3712 | 0.3748 | **+0.0036** | Tail improvement (positive signal) |
| S1-NDCG | 0.6648 | 0.6685 | **+0.0037** | Tail improvement (relieves tightest L3 constraint) |
| S1-VCAP@100 | 0.0089 | 0.0070 | -0.0019 | Tail degradation (within noise) |
| S1-BRIER | 0.1452 | 0.1448 | -0.0004 | Marginal tail improvement |

**Bot2 summary**: Mixed. NDCG and AP bot2 improved while AUC and VCAP@100 bot2 degraded. The NDCG bot2 improvement (+0.0037) is the single most positive signal from this experiment — it relieved what was the tightest L3 constraint (0.6648 → 0.6685, expanding margin from +0.020 to +0.024).

### Per-Month Detail (v0010 - v0009 deltas)

| Month | AUC Δ | AP Δ | VCAP@100 Δ | NDCG Δ | Net |
|-------|-------|------|------------|--------|-----|
| 2020-09 | +0.0010 | +0.0015 | +0.0012 | +0.0011 | 4W/0L |
| 2020-11 | +0.0008 | +0.0012 | +0.0045 | +0.0064 | 4W/0L |
| 2021-01 | -0.0003 | -0.0038 | -0.0024 | -0.0079 | 0W/4L |
| 2021-04 | +0.0011 | +0.0018 | -0.0021 | -0.0010 | 2W/2L |
| 2021-06 | 0.0000 | +0.0057 | +0.0003 | 0.0000 | 2W/0L |
| 2021-08 | -0.0005 | -0.0077 | -0.0059 | +0.0023 | 1W/3L |
| 2021-10 | -0.0005 | -0.0098 | -0.0089 | -0.0017 | 0W/4L |
| 2021-12 | -0.0014 | -0.0124 | -0.0002 | -0.0020 | 0W/4L |
| 2022-03 | +0.0011 | +0.0041 | +0.0002 | **+0.0083** | 4W/0L |
| 2022-06 | +0.0011 | -0.0053 | -0.0002 | -0.0034 | 1W/3L |
| 2022-09 | +0.0008 | +0.0029 | +0.0006 | +0.0021 | 4W/0L |
| 2022-12 | **-0.0020** | **-0.0031** | -0.0017 | **-0.0039** | 0W/4L |

### Seasonal Pattern Analysis

- **2022-12 regressed across all 4 Group A metrics**: This is the persistent weakest month (AUC 0.8161, lowest). The additional trees slightly hurt — likely the model is overfitting to patterns that don't generalize to this late-2022 distribution shift month. Deltas are small (all < 0.004) but uniformly negative.
- **2022-03 showed the largest single improvement** (NDCG +0.0083): This was the 2nd worst NDCG month for v0009 (0.6768) and improved to 0.6851. This is the clearest positive signal from H12, suggesting the finer-grained splits helped ranking quality in this specific spring transition month.
- **High-performing months degraded on AP**: 2021-01 (-0.0038), 2021-10 (-0.0098), 2021-12 (-0.0124) — all months where v0009 was already strong. The model appears to redistribute rather than uniformly improve.
- **Early months (2020-09, 2020-11) improved uniformly**: Small positive deltas across all metrics. Possibly more trees help when training data is sparser (earlier in the rolling window).

### Group B Monitoring

| Gate | v0010 Mean | v0009 Mean | Floor | Delta | Status |
|------|-----------|-----------|-------|-------|--------|
| S1-BRIER | 0.1374 | 0.1376 | 0.1703 | -0.0002 | PASS (+0.033 headroom) |
| S1-REC | 0.4289 | 0.4280 | 0.1000 | +0.0009 | PASS |
| S1-VCAP@500 | 0.0952 | 0.0955 | 0.0408 | -0.0003 | PASS |
| S1-VCAP@1000 | 0.1472 | 0.1483 | 0.1091 | -0.0011 | PASS |
| S1-CAP@100 | 0.7083 | 0.7158 | 0.7325 | -0.0075 | **FAIL** (L1 + L3; 4th consecutive) |
| S1-CAP@500 | 0.7153 | 0.7188 | 0.7240 | -0.0035 | **FAIL** (L1 + L3; 4th consecutive) |

**CAP@100 declined further** to 0.7083 from 0.7158. This is the lowest CAP@100 since v0007. Additional trees appear to slightly degrade threshold-dependent capture metrics, likely because the higher model capacity spreads probability mass more finely, making the hard threshold less effective at capturing the exact top-K predictions that happen to be positive. This reinforces the need for floor relaxation at HUMAN_SYNC.

## Code Review Findings

### Changes Reviewed

1. **`ml/config.py`** (lines 79, 81):
   - `n_estimators: int = 300` (was 200) ✓
   - `learning_rate: float = 0.07` (was 0.1) ✓
   - All other hyperparameters unchanged ✓
   - Feature list unchanged (29 features) ✓
   - Pipeline config unchanged (train_months=14, beta=0.7) ✓

2. **`ml/tests/test_config.py`** (lines 61, 63):
   - `assert hc.n_estimators == 300` (updated from 200) ✓
   - `assert hc.learning_rate == 0.07` (updated from 0.1) ✓
   - Feature count still asserts 29 ✓
   - All other assertions unchanged ✓

### Code Quality Assessment

- **Minimal, correct changes** — exactly 2 values changed in config + 2 test assertions updated
- **No scope creep** — no features, loader, evaluation, or gate changes
- **HUMAN-WRITE-ONLY files untouched** (gates.json, evaluate.py) ✓
- **No bugs possible** — this is a pure hyperparameter change with no logic modifications
- **Gradient magnitude preserved**: 300 × 0.07 = 21 vs 200 × 0.1 = 20 — consistent with direction document's reasoning

### Feature Importance Redistribution

The changes_summary notes interaction features dropped from 17.13% to 15.8% combined importance. This is expected behavior: with 300 trees (vs 200), individual feature importances get diluted as more trees split on a wider feature set. This is NOT a signal of information loss — it's a mechanical artifact of having more splitting opportunities. hist_da_trend remains the top feature at 39.7%, and the importance ranking is stable.

## Statistical Rigor

With 12 eval months, the evidence is unambiguous:

- **No W/L ratio exceeds 6/6**: The best is AUC at 6W/5L/1T and BRIER at 6W/5L/1T — both indistinguishable from a coin flip. Under the null hypothesis (no effect), P(6+ wins in 12 trials) = 0.387.
- **No mean delta exceeds 0.0021**: AP's -0.0021 is 0.05σ (AP std = 0.045) — far below any reasonable detection threshold.
- **Bot2 movements are mixed**: 2 improved (AP +0.0036, NDCG +0.0037), 2 degraded (AUC -0.0017, VCAP@100 -0.0019). The bot2 changes are the largest absolute numbers but still within the 0.02 noise tolerance.
- **Per-month analysis shows redistribution, not improvement**: 4 months improved on all 4 metrics, 4 months degraded on all 4 metrics, 4 months were mixed. This is the signature of noise.

**Conclusion**: H12 is a confirmed null. The effect of 300 trees with lr=0.07 vs 200 trees with lr=0.1 is indistinguishable from random variation on this dataset.

## Promotion Recommendation

**DO NOT PROMOTE v0010. Keep v0009 as champion.**

Rationale:
1. All metrics are within noise — no evidence of improvement
2. AP mean regressed -0.0021 (v0009 had the best AP in pipeline history at 0.4445)
3. VCAP@100 bot2 degraded from 0.0089 to 0.0070 — losing ground on a metric that was hard-won in iter 2
4. AUC bot2 degraded -0.0017
5. CAP@100/500 degraded further
6. The only positive signals (NDCG/AP bot2 improvement) are within noise range

v0009 represents the optimal configuration: 29 features, n_estimators=200, learning_rate=0.1, colsample_bytree=0.9. Promoting v0010 would add 50% training time for zero measurable benefit.

## Recommendations for HUMAN_SYNC

### 1. Model Capacity Ceiling Confirmed

10 experiments (v0-v0010) with the same underlying XGBoost architecture have converged. The optimization frontier is:
- **AUC**: ~0.849-0.850 (stable since v0007)
- **AP**: ~0.440-0.445 (peaked at v0009)
- **NDCG**: ~0.735-0.736 (stable since v0007)
- **BRIER**: ~0.137-0.138 (improving trend may have more room)

Further gains within this architecture require fundamentally new signal sources (new data, not new features derived from existing data) or architectural changes (ensemble methods, multi-stage pipeline, temporal modeling).

### 2. Gate Calibration Recommendations (Final, Cumulative)

1. **VCAP@100 floor**: Tighten from -0.035 to **0.0**. Recommended for 4 consecutive iterations. Current values (0.025 mean, 0.007 bot2) are far above zero.

2. **CAP@100/500 floors**: Relax by **0.03** to 0.7025 and 0.6940. Failing for 4 consecutive champion versions (v0007-v0010). The model profile is ranking-first — this is a deliberate architectural choice, not a deficiency.

3. **Noise tolerance**: Keep at **0.02**. Observed per-metric bot2 deltas between versions are typically 0.001-0.004. The 0.02 tolerance is adequate. Could tighten to 0.015 in a future batch if desired — all L3 margins would still pass.

4. **Group A mean floors**: No changes needed. All pass with 0.05+ headroom.

### 3. What Next Batch Should Explore

The feature engineering search space within the current architecture is exhausted. Potential directions for the next batch:
- **Temporal features**: The persistent weakness in late-2022 months suggests distribution shift. Time-aware features (e.g., months since training window start, volatility regime indicators) might help.
- **Multi-stage pipeline**: Use Stage 1 probabilities as input to a Stage 2 model that incorporates portfolio-level information.
- **Alternative loss functions**: XGBoost supports custom objectives. A ranking-specific loss (e.g., pairwise, lambda) may improve NDCG without sacrificing AUC.

These are structural changes requiring human design decisions, reinforcing that this batch should conclude at HUMAN_SYNC.
