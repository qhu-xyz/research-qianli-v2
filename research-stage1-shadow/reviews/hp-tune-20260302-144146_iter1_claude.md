# Claude Review — Iteration 1 (hp-tune-20260302-144146)

**Version**: v0002
**Hypothesis**: H4 — Interaction features provide new discriminative signal for ranking quality
**Date**: 2026-03-02

## Summary

v0002 adds 3 physically-motivated interaction features (`exceed_severity_ratio`, `hist_physical_interaction`, `overload_exceedance_product`) while reverting hyperparameters to v0 defaults. The hypothesis was that pre-computed interactions would provide discriminative signal that depth-4 trees cannot efficiently discover.

**Verdict: Hypothesis weakly supported for ranking (NDCG, AP) but refuted for discrimination (AUC).** AUC is unchanged at 0.8348 (5W/6L/1T — indistinguishable from noise). AP improves by +0.0010 mean (7W/5L), and NDCG by +0.0016 mean (8W/4L). These are positive but extremely small signals. The interaction features slightly improve how the model orders positive predictions but do not improve its ability to separate positives from negatives. All gates pass all 3 layers. This is a safe but marginal iteration.

The critical finding is that **AUC is firmly stuck at ~0.835** — neither HP tuning (v0003) nor interaction features (v0002) can move it. Combined with the v0003 result, this confirms the model is at the information ceiling of the 14 base features derived from probabilistic power flow. Further feature engineering within this feature set will yield diminishing returns.

## Gate-by-Gate Analysis

### Group A (Blocking)

| Gate | v0 Mean | v0002 Mean | Delta | v0 Bot2 | v0002 Bot2 | Delta Bot2 | Mean vs Floor | Pass |
|------|---------|------------|-------|---------|------------|------------|---------------|------|
| S1-AUC | 0.8348 | 0.8348 | +0.0000 | 0.8105 | 0.8105 | +0.0000 | +0.0500 | YES |
| S1-AP | 0.3936 | 0.3946 | +0.0010 | 0.3322 | 0.3305 | -0.0017 | +0.0510 | YES |
| S1-VCAP@100 | 0.0149 | 0.0158 | +0.0009 | 0.0014 | 0.0006 | -0.0008 | +0.0509 | YES |
| S1-NDCG | 0.7333 | 0.7349 | +0.0016 | 0.6716 | 0.6703 | -0.0013 | +0.0516 | YES |

### Group B (Monitor)

| Gate | v0 Mean | v0002 Mean | Delta | Pass |
|------|---------|------------|-------|------|
| S1-BRIER | 0.1503 | 0.1505 | +0.0002 | YES |
| S1-VCAP@500 | 0.0908 | 0.0865 | **-0.0043** | YES |
| S1-VCAP@1000 | 0.1591 | 0.1560 | **-0.0031** | YES |
| S1-REC | 0.4192 | 0.4233 | +0.0041 | YES |
| S1-CAP@100 | 0.7825 | 0.7925 | +0.0100 | YES |
| S1-CAP@500 | 0.7740 | 0.7758 | +0.0018 | YES |

### Layer-by-Layer Detail

**Layer 1 (Mean Quality)**: All gates pass with ~0.05 headroom. No change from v0.

**Layer 2 (Tail Safety)**: Zero months below tail_floor for any gate (identical to v0).

**Layer 3 (Tail Non-Regression)**: All pass. However, a noteworthy pattern: **all 4 Group A bottom_2_mean values slightly degraded** (AP -0.0017, VCAP@100 -0.0008, NDCG -0.0013, AUC +0.0000). The gains are concentrated in the middle of the distribution while the worst months got slightly worse. All within the 0.02 non-regression tolerance, but the direction is concerning.

### Seasonal/Temporal Pattern Analysis

**Strongest months** (consistent improvement across AP, NDCG, VCAP@100):
- 2021-01: NDCG +0.042 (biggest single improvement), VCAP@100 +0.027, AP +0.006
- 2020-09: AP +0.006, NDCG +0.007, VCAP@100 +0.004

**Weakest months** (consistent degradation):
- 2022-03: AP -0.005, NDCG -0.013, VCAP@100 -0.013
- 2021-10: NDCG -0.024, VCAP@100 -0.019 (despite AP being flat)
- 2022-12: NDCG -0.007, VCAP@100 -0.0005 (already weakest AUC month)

**Pattern**: Early months (2020–2021-H1) benefit more than late months (2022). The late-2022 weakness persists unchanged — this is consistent with distribution shift that no feature engineering within the current feature set can fix.

### VCAP@500 and VCAP@1000 Regression

A notable finding: while VCAP@100 improved (+0.0009 mean), **VCAP@500 and VCAP@1000 both degraded** (-0.0043 and -0.0031 respectively). This suggests the interaction features improve ranking at the very top (top-100) but slightly harm ordering at broader K values. This is consistent with the interaction features acting as "confidence boosters" for the most extreme cases while adding noise to borderline cases.

## Code Review

### Changes Correctness

1. **Feature computation in `ml/features.py`**: Correct. The 3 interaction features are computed from base columns via `df.with_columns()` before `df.select(cols)`. The `1e-6` epsilon in `exceed_severity_ratio` prevents division by zero. Polars expressions are idiomatic and efficient.

2. **HP revert in `ml/config.py`**: Correctly reverted all 4 parameters to v0 defaults (n_estimators=200, max_depth=4, lr=0.1, min_child_weight=10). Verified in diff.

3. **FeatureConfig update**: 3 new features appended with correct monotone constraint (+1). Total 17 features, monotone string updated correctly.

4. **Test updates**: All assertions updated from 14→17. Test for `prepare_features` correctly creates DataFrames from only the 14 base features, verifying that interaction computation works.

### Issues Found

**MINOR — Stale docstrings**:
- `features.py:24` docstring says "containing at least the 14 feature columns" — should say 14 base feature columns
- `features.py:31` says "Feature matrix of shape (n_samples, 14)" — should say 17
- `config.py:46` property docstring says "Return list of feature names (14 items)" — should say 17

These are cosmetic but could confuse future workers.

**CARRIED — No new issues introduced**:
- Threshold leakage (HIGH) — unchanged, affects threshold-dependent metrics only
- Threshold `>` vs `>=` mismatch (MEDIUM) — unchanged
- `scale_pos_weight_auto` dead config (LOW) — unchanged

## Statistical Assessment

With 12 evaluation months:

| Metric | Win/Loss | Mean Delta | Magnitude |
|--------|----------|------------|-----------|
| AUC | 5W/6L/1T | +0.0000 | Noise (zero effect) |
| AP | 7W/5L | +0.0010 | Marginal (within 1σ of noise at 0.043 std) |
| NDCG | 8W/4L | +0.0016 | Marginal (within 1σ of noise at 0.047 std) |
| VCAP@100 | 8W/4L | +0.0009 | Marginal (VCAP@100 has very high variance, std=0.013) |

The NDCG 8W/4L is the most encouraging signal, but the per-month standard deviation (0.047) dwarfs the +0.0016 mean improvement. A binomial test for 8/12 wins gives p≈0.19 (not significant). The massive 2021-01 NDCG outlier (+0.042) is driving much of the mean improvement — without it, NDCG mean delta drops to ~-0.002.

**Conclusion**: No statistically significant improvement on any metric. The direction specified 8/12 AUC wins as the consistency target — actual result is 5/12, well below threshold.

## Recommendations for Next Iteration

### 1. Do NOT promote v0002
The improvements are statistically indistinguishable from noise. Bottom-2 values regressed on 3/4 Group A metrics. No basis for promotion.

### 2. Acknowledge the feature ceiling
Two iterations have now demonstrated the information ceiling:
- v0003 (HP tuning): AUC 0W/11L — deeper trees don't help
- v0002 (interaction features): AUC 5W/6L — pre-computed interactions don't help

The 14 probabilistic power flow features plus their interactions capture approximately all available signal for binary bind/not-bind classification. Further engineering within this feature space (more interactions, polynomial features, etc.) is very unlikely to break through.

### 3. Next iteration should pursue genuinely new information sources

The most promising directions that introduce **new information** not captured by the current features:

**A. Temporal/seasonal features** (RECOMMENDED): The persistent late-2022 weakness (AUC 0.809, AP 0.315) suggests distribution shift. Features like `month_of_year` (cyclic-encoded), `months_since_covid`, or `season` could help the model adapt to temporal patterns. These would be unconstrained (monotone=0) since seasonal effects are non-monotone.

**B. Constraint-level features**: If available in the data, features like `historical_bind_count_last_6m`, `constraint_age`, or `line_rating_change` would provide constraint-specific context beyond aggregate flow probabilities.

**C. Cross-constraint context**: Features capturing whether nearby or related constraints are also showing high exceedance could improve discrimination for constraint clusters.

### 4. Consider longer training windows for stability
The 10-month training window may be too short for late-2022 distribution shift. Experiment with train_months=12 or 14 as a simpler lever before adding new features.

### 5. Avoid further HP tuning
v0003 already demonstrated that HP changes within reasonable ranges cannot improve ranking metrics with the current feature set. This avenue is exhausted.

## Gate Calibration Assessment

Current gate calibration remains appropriate:
- All Group A floors have +0.05 headroom — neither too tight (blocking good candidates) nor too loose (allowing regressions)
- Layer 3 is effectively disabled (champion=null) — acceptable; will become meaningful once a champion is promoted
- BRIER headroom is +0.020 (tightest Group B gate) — adequate since interaction features have negligible impact on calibration
- No calibration changes recommended at this point (only 2 real-data iterations beyond v0)

**One observation**: VCAP@100 has such high variance (min=0.0000 in v0002, max=0.0461) and negative floor (-0.0351) that it is effectively non-binding. If the goal is to prioritize top-100 value capture, the floor should eventually be tightened. However, premature to act now — the metric's high variance means we need more iterations to understand its natural range.
