# Claude Review — Iteration 1 (Batch hp-tune-20260302-134412)

**Version**: v0003
**Hypothesis**: H3 — Hyperparameter tuning (deeper trees + slower learning) improves ranking quality
**Date**: 2026-03-02

## Summary

The hypothesis that standard XGBoost hyperparameter tuning (max_depth 4→6, n_estimators 200→400, learning_rate 0.1→0.05, min_child_weight 10→5) would improve ranking quality over the v0 baseline **is not supported by the data**. All Group A ranking metrics (AUC, AP, NDCG) degraded slightly, with AUC showing a consistent loss across 11 of 12 months. The lone positive signal — BRIER improving by 0.004 (better calibration, 12/12 months) — is a Group B monitor metric and does not constitute a ranking quality improvement. VCAP@100 mean improved by +0.0015 but this is driven by a single month (2021-01: +0.014) and the bottom-2 VCAP@100 actually worsened.

The v0 defaults were already near-optimal for this dataset and feature set. The model's discrimination ceiling appears to be constrained by feature informativeness rather than tree complexity. All gates pass because floors have 0.05 headroom, but no metric improved meaningfully. **v0003 should not be promoted.**

## Gate-by-Gate Analysis

### Group A (Blocking)

| Gate | v0 Mean | v0003 Mean | Δ Mean | v0 Bot2 | v0003 Bot2 | Δ Bot2 | Win/Loss | Verdict |
|------|---------|------------|--------|---------|------------|--------|----------|---------|
| S1-AUC | 0.8348 | 0.8323 | **-0.0025** | 0.8105 | 0.8089 | -0.0016 | 0W/11L/1T | Degraded |
| S1-AP | 0.3936 | 0.3921 | **-0.0015** | 0.3322 | 0.3299 | -0.0023 | 4W/8L | Degraded |
| S1-VCAP@100 | 0.0149 | 0.0164 | +0.0015 | 0.0014 | 0.0007 | -0.0007 | 7W/5L | Mixed (mean ↑, tail ↓) |
| S1-NDCG | 0.7323 | 0.7323 | **-0.0010** | 0.6716 | 0.6675 | -0.0041 | 4W/8L | Degraded |

**Layer-by-layer:**
- **L1 (Mean Quality)**: All pass with large headroom. AUC: 0.8323 vs floor 0.7848 (+0.048), AP: 0.3921 vs floor 0.3436 (+0.049).
- **L2 (Tail Safety)**: All pass — 0 months below any tail_floor.
- **L3 (Tail Non-Regression)**: All pass. Largest bottom-2 degradation is NDCG (-0.0041), within the 0.02 tolerance. But the direction is uniformly negative: AUC -0.0016, AP -0.0023, VCAP@100 -0.0007, NDCG -0.0041.

**Critical finding**: AUC degrades in **every single month** except 2022-12 (tied at 0.8088). This is not noise — it's a systematic signal that the deeper/slower configuration produces marginally worse discrimination. The consistency (0W/11L) is more informative than the magnitude (-0.0025 mean).

### Group B (Monitor)

| Gate | v0 Mean | v0003 Mean | Δ Mean | Win/Loss | Notes |
|------|---------|------------|--------|----------|-------|
| S1-BRIER | 0.1503 | 0.1462 | **-0.0041** (improved) | 12W/0L | Only universally positive signal |
| S1-REC | 0.4192 | 0.4220 | +0.0028 | 7W/5L | Noisy |
| S1-CAP@100 | 0.7825 | 0.7833 | +0.0008 | 2W/2L/8T | Noise |
| S1-CAP@500 | 0.7740 | 0.7712 | -0.0028 | 3W/8L | Slight degradation |
| S1-VCAP@500 | 0.0908 | 0.0867 | -0.0041 | 5W/7L | Degraded |
| S1-VCAP@1000 | 0.1591 | 0.1567 | -0.0024 | 3W/9L | Degraded |

**BRIER is the standout**: 12/12 months improved. The deeper model with slower learning produces better-calibrated probability estimates even though the ranking order (AUC/AP) is marginally worse. This is a meaningful calibration improvement, but BRIER is not a blocking gate.

### Seasonal/Temporal Patterns

Per-month AUC deltas show no seasonal pattern — v0003 underperforms v0 uniformly across all seasons. The weakest months remain unchanged:
- Late-2022 (2022-09, 2022-12) remains the weak tail for AUC and AP in both versions
- 2021-04 and 2021-08 remain weak for NDCG in both versions
- The HP tuning did not address the late-2022 distribution shift, as predicted in the direction document as a risk

### Statistical Rigor

With 12 evaluation months:
- AUC: 0W/11L is extremely unlikely under H0 (binomial p ≈ 0.003 for ≤0 wins out of 11 non-ties). The AUC degradation is statistically significant despite its small magnitude.
- AP: 4W/8L (p ≈ 0.19) — not significant, consistent with noise.
- NDCG: 4W/8L — same as AP, not significant.
- BRIER: 12W/0L (p ≈ 0.0002) — the calibration improvement is statistically significant.

**Conclusion**: The AUC degradation and BRIER improvement are both real effects. The deeper trees with slower learning improve probability calibration but slightly hurt discrimination, producing a marginally less separable ranking.

## Code Review Findings

The diff is clean and minimal:

1. **ml/config.py**: Exactly the 4 parameter changes specified in the direction document. No unintended changes. Unchanged parameters (subsample, colsample_bytree, reg_alpha, reg_lambda, random_state) are confirmed intact.

2. **ml/tests/test_config.py**: Test assertions updated to match new defaults. `test_hyperparam_defaults` and `test_hyperparam_to_dict` both updated correctly. The `len(d) == 9` assertion is still correct.

3. **No changes to**: evaluate.py, pipeline.py, features, threshold config, or any HUMAN-WRITE-ONLY files. Correct.

4. **No bugs or edge cases identified.** The changes are safe and fully reversible.

### Previously reported issues (from smoke test iterations, status check):
- **Threshold `>` vs `>=` mismatch** (Codex HIGH from smoke-v7): Not addressed in this iteration (not in scope per direction). Still open.
- **Threshold leakage** (Codex HIGH from smoke-v6): Not addressed. Still deferred per D5/D10.
- These are pre-existing issues from the v0 baseline; they affect v0003 and v0 equally and do not invalidate the comparison.

## Recommendations for Next Iteration

### Primary recommendation: Feature engineering, not more HP tuning

The v0 hyperparameters were already well-tuned for this data. The model is **feature-limited, not complexity-limited**. Evidence:
- Deeper trees (6 vs 4) did not improve discrimination despite 2x more trees
- AUC degraded systematically — the model can't extract more signal from these 14 features by fitting them harder
- BRIER improved, suggesting the model can calibrate better, but the underlying ranking signal is exhausted

**Concrete directions:**
1. **Interaction features**: Create explicit interactions between the most informative features (e.g., `prob_exceed_110 * hist_da`, `expected_overload * prob_below_100`). The model has monotone constraints that may prevent it from learning these interactions naturally.
2. **Temporal features**: Month-of-year, season indicators, or lagged binding rates to help address the late-2022 distribution shift.
3. **Feature selection**: Investigate if any of the 14 features are adding noise rather than signal (density_skewness, density_kurtosis, density_cv have unconstrained monotonicity — they may inject noise).

### Secondary recommendation: Revert HP changes

Since v0003 shows no improvement and a statistically significant AUC degradation, the next iteration should revert to v0 hyperparameters (max_depth=4, n_estimators=200, lr=0.1, min_child_weight=10) and explore a different lever (features or training window). Carrying the HP changes forward would mean starting from a marginally worse baseline.

### Lower priority:
- Address the threshold `>` vs `>=` mismatch (carried from smoke-v7, Codex HIGH) — won't affect ranking metrics but is a correctness issue.
- Consider training window adjustment (12 months instead of 10) to give the model more late-period data for handling distribution shift.

## Gate Calibration

Current Group A floors are set at v0_mean - 0.05 with tail_floor at v0_min - 0.10. For this iteration:

- **Floors are appropriately calibrated.** All gates pass comfortably (headroom ~0.05 on mean, ~0.10 on tail). The 0.05/0.10 offsets are reasonable for an early-stage pipeline that needs room to explore.
- **No calibration changes recommended.** We have only one real-data iteration (v0003) beyond the baseline. Tightening floors now would be premature. After 3-4 iterations with real data, reassess whether floors should tighten (especially if improvements plateau).
- **S1-VCAP@100 floor is effectively non-binding** (-0.0351 floor, all months positive). This is acceptable — VCAP@100 is the highest-variance Group A metric (std=0.013, coefficient of variation ~0.83) and a tighter floor would cause spurious gate failures.
- **S1-BRIER headroom is tight (0.02)** but the metric improved in v0003. Keep as-is; if future iterations push BRIER up toward the floor, revisit at HUMAN_SYNC.

## Verdict

**Do not promote v0003.** While all gates technically pass, the iteration produced no meaningful improvement on any Group A metric and showed a statistically significant (though small) degradation in AUC. The hypothesis is refuted. Redirect effort toward feature engineering.
