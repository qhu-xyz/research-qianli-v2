# Claude Review — v0009 (feat-eng-3-20260303-104101, iter2)

## Summary

v0009 adds 3 derived interaction features (band_severity, sf_exceed_interaction, hist_seasonal_band) targeting VCAP@100 recovery, and increases colsample_bytree from 0.8 to 0.9. The primary objective — recovering VCAP@100 from v0008's 4W/8L regression — was achieved: VCAP@100 W/L improved to 6W/5L/1T with bot2 improving +0.0028 (0.0089 vs 0.0061). AP continued its upward trajectory (+0.0027 mean, 9W/3L), NDCG improved modestly (+0.0013, 7W/5L), and BRIER improved -0.0007. AUC is essentially flat (-0.0003, 4W/8L) — within noise given the 0.017 standard deviation across months.

The 3 new features contribute 17.13% combined importance, with hist_seasonal_band dominating at 11.75% (rank #2). This is the highest single-feature importance gain from any iteration, validating that seasonal historical extremes interacted with physical overload bands capture a signal the tree model couldn't efficiently approximate from the raw components. All Group A gates pass all 3 layers with comfortable margins. **Recommend promotion.**

## Gate-by-Gate Analysis (Group A)

### Aggregate Comparison (v0009 vs v0008 champion)

| Gate | v0009 Mean | v0008 Mean | Delta | W/L | v0009 Bot2 | v0008 Bot2 | Bot2 Delta | L3 Floor | L3 Margin |
|------|-----------|-----------|-------|-----|-----------|-----------|-----------|----------|-----------|
| S1-AUC | 0.8495 | 0.8498 | -0.0003 | 4W/8L | 0.8189 | 0.8199 | -0.0010 | 0.7999 | +0.019 |
| S1-AP | 0.4445 | 0.4418 | **+0.0027** | **9W/3L** | 0.3712 | 0.3726 | -0.0014 | 0.3526 | +0.019 |
| S1-VCAP@100 | 0.0266 | 0.0240 | **+0.0026** | **6W/5L/1T** | **0.0089** | 0.0061 | **+0.0028** | -0.0139 | +0.023 |
| S1-NDCG | 0.7359 | 0.7346 | +0.0013 | 7W/5L | 0.6648 | 0.6663 | -0.0015 | 0.6463 | +0.019 |

### Per-Month Detail (v0009 vs v0008)

| Month | AUC Δ | AP Δ | VCAP@100 Δ | NDCG Δ |
|-------|-------|------|------------|--------|
| 2020-09 | -0.0010 | +0.0005 | -0.0038 | -0.0055 |
| 2020-11 | +0.0001 | +0.0014 | -0.0014 | -0.0080 |
| 2021-01 | +0.0016 | **+0.0132** | -0.0016 | -0.0064 |
| 2021-04 | +0.0011 | +0.0024 | +0.0029 | +0.0021 |
| 2021-06 | -0.0004 | -0.0049 | +0.0049 | +0.0064 |
| 2021-08 | -0.0004 | +0.0045 | **+0.0239** | **+0.0125** |
| 2021-10 | -0.0002 | +0.0022 | +0.0007 | +0.0113 |
| 2021-12 | -0.0030 | **+0.0120** | -0.0029 | -0.0004 |
| 2022-03 | -0.0003 | -0.0025 | 0.0000 | -0.0049 |
| 2022-06 | -0.0009 | +0.0029 | -0.0008 | +0.0050 |
| 2022-09 | -0.0012 | -0.0002 | +0.0060 | +0.0032 |
| 2022-12 | +0.0008 | +0.0016 | +0.0026 | +0.0003 |

### Seasonal Pattern Analysis

**VCAP@100 gains concentrated in mid-year months**: The largest VCAP@100 improvements are 2021-08 (+0.0239), 2022-09 (+0.0060), 2021-06 (+0.0049), and 2021-04 (+0.0029). These are spring/summer months where binding patterns shift — exactly where hist_seasonal_band (seasonal extremes × overload band) would add the most signal. The early-period months (2020-09, 2020-11, 2021-01) show small VCAP@100 regressions, suggesting the interaction features are less informative when the model has shorter history to draw from.

**NDCG improvements in structurally weak months**: 2021-08 (+0.0125), 2021-10 (+0.0113), 2021-06 (+0.0064) all improved. 2021-04 (the persistent worst NDCG month at 0.6529) improved +0.0021 vs champion — marginal but directionally correct.

**AUC flat across the board**: No month shows AUC change > ±0.003, confirming the interaction features don't add new discriminative signal — they refine ranking quality within the already-separated classes.

### Group B Monitoring

| Gate | v0009 Mean | v0008 Mean | Floor | Status |
|------|-----------|-----------|-------|--------|
| S1-BRIER | 0.1376 | 0.1383 | 0.1703 | PASS (improved, +0.033 headroom) |
| S1-REC | 0.4280 | 0.4228 | 0.1000 | PASS (+0.0052 improvement) |
| S1-VCAP@500 | 0.0955 | 0.0955 | 0.0408 | PASS (unchanged) |
| S1-VCAP@1000 | 0.1483 | 0.1479 | 0.1091 | PASS (unchanged) |
| S1-CAP@100 | 0.7158 | 0.7142 | 0.7325 | **FAIL** (L1: mean below floor) |
| S1-CAP@500 | 0.7188 | 0.7175 | 0.7240 | **FAIL** (L1: mean below floor) |

CAP@100/500 remain below floors, consistent with v0008. Both improved very slightly (+0.0016 and +0.0013). The floor relaxation recommendation from iter 1 still applies.

## Code Review Findings

### Changes Reviewed

1. **`ml/config.py`** — 3 new features correctly appended with monotone=+1. colsample_bytree changed from 0.8 to 0.9. All other HPs unchanged. **No issues.**

2. **`ml/features.py`** — 3 new features added to `interaction_cols` set and `with_columns` block. Formulas match direction spec exactly:
   - `band_severity = prob_band_95_100 * expected_overload` ✓
   - `sf_exceed_interaction = sf_max_abs * prob_exceed_100` ✓
   - `hist_seasonal_band = hist_da_max_season * prob_band_100_105` ✓

   Source feature verification block correctly unchanged (new features are computed, not loaded). **No issues.**

3. **`ml/data_loader.py`** — Smoke data generators for 3 new features correctly use existing synthetic columns. Note: the smoke-generated values are overwritten by `features.py`'s `prepare_features()` anyway, but having them in smoke data maintains consistency with the existing pattern (same as `hist_physical_interaction` and `overload_exceedance_product`). **No issues.**

4. **`ml/tests/test_config.py`** — Feature count updated to 29, monotone constraint string updated, feature names list updated, colsample_bytree assertion updated to 0.9. All assertions verified against actual config.py values. **No issues.**

### Code Quality Assessment

- Changes are minimal and focused — exactly 4 files modified, all following established patterns
- No unnecessary refactoring or scope creep
- Monotone constraints correctly set to +1 for all 3 multiplicative interactions (product of two +1 monotone features is monotone +1)
- No HUMAN-WRITE-ONLY files modified
- No changes to gates.json, evaluate.py, threshold, train_months, or other fixed parameters

## Statistical Rigor

With 12 eval months:

- **AP at 9W/3L**: Probability of 9+ wins under null (p=0.5) is ~0.073 — approaching significance. The improvement is broad-based, not driven by outlier months.
- **VCAP@100 at 6W/5L/1T**: Not statistically significant on its own, but the recovery from 4W/8L (v0008 vs v0007) to 6W/5L is directionally strong. The bot2 improvement (+0.0028) is the more meaningful signal — it shows the worst months improved.
- **NDCG at 7W/5L**: Modest. Gains concentrated in structurally weak months (2021-04, 2021-08, 2021-10) — exactly where improvement has the most value.
- **AUC at 4W/8L**: All month deltas within ±0.003 — noise, not signal. Mean delta of -0.0003 is negligible relative to std of 0.017.

**Conclusion**: The improvements are consistent and targeted rather than statistically overwhelming. The feature importance data (17.13% combined) confirms these features capture real signal, not random variation.

## Feature Importance Analysis

| Feature | Mean Gain | Rank | Assessment |
|---------|-----------|------|------------|
| hist_seasonal_band | 11.75% | #2 | **Exceptional** — strongest single feature added in any iteration. Seasonal history × overload band captures something the tree model couldn't approximate efficiently. |
| sf_exceed_interaction | 4.00% | #7 | **Strong** — network topology × exceedance is a natural interaction. Complements sf_max_abs (#11 at 1.20% previously). |
| band_severity | 1.38% | #11 | **Modest** — near-boundary mass × overload severity. Lower than expected, likely because prob_band_95_100 (#5) and expected_overload (#9) already interact well through tree splits. |

Combined 17.13% is the highest feature-block importance in pipeline history (vs 10.3% for v0008's 7 features, 4.66% for v0007's 6 features). This suggests these interactions capture genuinely new signal rather than redundant information.

## Recommendations for Next Iteration

1. **Consider n_estimators increase to 300**: With 29 features and confirmed strong signal from interaction terms, the model may benefit from additional trees to capture finer-grained interactions. Risk is low — monitor for overfitting via BRIER degradation.

2. **Investigate 2021-04 (persistent worst NDCG)**: Despite slight improvement (+0.0021), NDCG 0.6529 remains an outlier. This month may have structural data issues (spring transition, different binding patterns) that no feature engineering can resolve. Consider whether this month should be flagged as a known outlier or if there's a feature specifically targeting spring transition patterns.

3. **Do NOT add more interaction features**: 29 features with 17.13% from interactions is well-balanced. Adding more interactions risks diluting signal and increasing collinearity. The diminishing returns from band_severity (1.38%) suggest the interaction feature space is approaching saturation.

4. **Consider learning_rate reduction**: If n_estimators increases to 300, reducing learning_rate from 0.1 to 0.07 maintains the effective model capacity while potentially improving generalization. This is a low-risk change.

5. **colsample_bytree at 0.9 appears optimal**: No evidence of overfitting (BRIER improved), and the VCAP@100 recovery confirms the feature coverage improvement was beneficial. Do NOT increase to 1.0 — maintain regularization.

## Gate Calibration Suggestions

1. **VCAP@100 floor (-0.035)**: Still excessively loose. v0009 mean is 0.0266, bot2 is 0.0089. Recommend tightening floor to 0.0 at HUMAN_SYNC (reaffirming iter 1 recommendation).

2. **CAP@100/500 floors (0.7325 / 0.7240)**: Still failing for v0009 (0.7158 / 0.7188). Model profile has definitively shifted toward ranking quality over threshold-dependent capture since v0007. Recommend relaxing both by 0.03 to 0.7025 / 0.6940 (reaffirming iter 1 recommendation). These have been failing for 3 consecutive versions — the floors are misaligned, not the model.

3. **Group A floors**: All pass with 0.05+ headroom on L1. No changes needed.

4. **L3 non-regression tolerance (0.02)**: Adequate. Tightest margins are AUC and AP bot2 at +0.019 from floor — comfortably within tolerance. If v0009 is promoted, the new L3 floors would be:
   - AUC: 0.8189 - 0.02 = 0.7989
   - AP: 0.3712 - 0.02 = 0.3512
   - VCAP@100: 0.0089 - 0.02 = -0.0111
   - NDCG: 0.6648 - 0.02 = 0.6448

   These are all reasonable. NDCG floor 0.6448 is slightly tighter than the current 0.6463 — manageable but worth noting.

## Promotion Recommendation

**PROMOTE v0009 to champion.** Rationale:
- Primary target (VCAP@100 recovery) achieved: W/L improved from 4W/8L to 6W/5L/1T, bot2 +0.0028
- AP improved +0.0027 mean (9W/3L) — best AP in pipeline history (0.4445)
- NDCG improved +0.0013 mean (7W/5L), with gains in structurally weak months
- AUC flat within noise (-0.0003)
- BRIER improved (better calibration)
- All Group A gates pass all 3 layers with comfortable margins
- Feature importance confirms real signal addition (17.13% combined)
- No code quality issues
