# Claude Reviewer — Iteration 2 (feat-eng-3-20260304-121042)

**Version under review**: v0012 (600 trees, lr=0.03)
**Champion**: v0011 (400 trees, lr=0.05, 34 features)
**Date**: 2026-03-04

## Summary

v0012 achieves its primary objective: recovering EV-VC@500 breadth (+3.5%) while maintaining all gate compliance. The critical 2022-09 tail failure is eliminated (0.0527→0.0720, +36.7%), transforming EV-VC@500 from a binding constraint with razor-thin margins to a comfortable pass on all three layers. The trade is a -5.3% mean EV-VC@100 regression, concentrated in 2022-12 (-21.8%) and 2022-03 (-15.9%), but EV-VC@100 retains +14.2% margin to floor — still the loosest gate in Group A. Spearman (+0.4%), EV-NDCG (+0.2%), C-RMSE (-0.4%), and C-MAE (-0.7%) all improved modestly. The code change is minimal (2 HP defaults + test assertions), correct, and low-risk.

This is a well-executed HP tuning iteration that successfully resolves the most dangerous gate margin identified in iter 1's review. The pipeline is now in a healthier state with no gates near failure thresholds on any layer.

## Gate-by-Gate Analysis (v0012 vs v0011 champion)

### Group A (Hard Gates) — Three-Layer Detail

| Gate | Layer | v0011 | v0012 | Threshold | Status | Margin | Trend |
|------|-------|-------|-------|-----------|--------|--------|-------|
| **EV-VC@100** | L1 (mean) | 0.0801 | 0.0758 | ≥0.0664 | **PASS** | +14.2% | Narrowed from +20.6% |
| | L2 (tail) | 0 fails | 0 fails | ≤1 fail | **PASS** | 0/1 | Stable |
| | L3 (bot2) | 0.0111 | 0.0086 | ≥-0.0089* | **PASS** | +0.0175 | bot2 decreased but margin huge |
| **EV-VC@500** | L1 (mean) | 0.2270 | 0.2348 | ≥0.2179 | **PASS** | **+7.8%** | **Improved from +4.2%** |
| | L2 (tail) | 1 fail | **0 fails** | ≤1 fail | **PASS** | **0/1** | **Tail failure eliminated** |
| | L3 (bot2) | 0.0541 | **0.0698** | ≥0.0341* | **PASS** | **+0.0357** | **Massive improvement from +0.0023** |
| **EV-NDCG** | L1 (mean) | 0.7499 | 0.7518 | ≥0.7137 | **PASS** | +5.3% | Improved from +5.1% |
| | L2 (tail) | 0 fails | 0 fails | ≤1 fail | **PASS** | 0/1 | Stable |
| | L3 (bot2) | 0.6403 | 0.6434 | ≥0.6203* | **PASS** | +0.0231 | Slightly improved |
| **Spearman** | L1 (mean) | 0.3925 | 0.3940 | ≥0.3736 | **PASS** | +5.5% | Improved from +5.1% |
| | L2 (tail) | 0 fails | 0 fails | ≤1 fail | **PASS** | 0/1 | Stable |
| | L3 (bot2) | 0.2678 | 0.2696 | ≥0.2478* | **PASS** | +0.0218 | Slightly improved |

*L3 thresholds = champion bot2 - 0.02

**Verdict: ALL GROUP A GATES PASS ALL THREE LAYERS.**

### Group B (Monitor Gates)

| Gate | v0011 Mean | v0012 Mean | Floor | Status | Delta |
|------|-----------|-----------|-------|--------|-------|
| C-RMSE | 2866.6 | 2855.3 | ≤3062.2 | **PASS** | -0.4% (improved) |
| C-MAE | 1142.5 | 1135.0 | ≤1209.1 | **PASS** | -0.7% (improved) |
| EV-VC@1000 | 0.3040 | 0.3119 | ≥0.3014 | **PASS** | +2.6% (improved) |
| R-REC@500 | 0.0347 | 0.0356 | ≥0.0337 | **PASS** | +2.6% (improved) |

**All Group B gates improved.** Notable: v0011 was the first version to pass C-RMSE and C-MAE; v0012 extends those passes with better margins.

### Binding Gate Constraint Status

EV-VC@500 is **no longer the binding constraint**. With L1 at +7.8%, L2 at 0 failures, and L3 at +0.0357, it has comfortable margins on all layers. The new binding constraint is EV-VC@100 at +14.2% L1 margin — but this is still loose, meaning the pipeline has no tight constraints for the first time.

## Per-Month Breakdown (Group A, v0012 vs v0011)

| Month | EV-VC@100 Δ | EV-VC@500 Δ | EV-NDCG Δ | Spearman Δ |
|-------|-------------|-------------|-----------|------------|
| 2020-09 | +8.8% | -1.1% | +1.4% | +0.4% |
| 2020-11 | +3.8% | -0.3% | +1.0% | +0.0% |
| 2021-01 | +0.0% | -3.2% | -0.5% | +0.3% |
| 2021-03 | -12.9% | +2.6% | +0.4% | -0.1% |
| 2021-05 | **-80.7%** | **+65.2%** | +0.9% | +0.3% |
| 2021-07 | +4.2% | 0.0% | +1.6% | +2.3% |
| 2021-09 | -7.7% | +0.3% | -0.5% | -0.3% |
| 2021-11 | +6.6% | **+31.7%** | +1.1% | +1.3% |
| 2022-03 | -15.9% | -0.7% | -0.3% | -0.1% |
| 2022-06 | -14.2% | +0.9% | -0.1% | +0.0% |
| 2022-09 | +12.9% | **+36.7%** | +1.1% | +0.6% |
| 2022-12 | **-21.8%** | -0.4% | -3.0% | +0.1% |

### Consistency Analysis

**EV-VC@500**: 6/12 months improved, 1 flat, 5 degraded. Improvements concentrated in weak months (2021-05 +65%, 2021-11 +32%, 2022-09 +37%), degradations small (<3.2%) in strong months. This is exactly the desired pattern — lifting the floor without pulling down the ceiling.

**EV-VC@100**: 5/12 improved, 7 degraded. Degradations include 3 significant drops (2021-05 -81% from tiny base, 2022-12 -22%, 2022-03 -16%). Not broadly consistent, but the business impact is muted because (a) the mean is still +14.2% above floor and (b) the large % drops are from low-absolute-value months.

**EV-NDCG & Spearman**: Both show broad consistency — NDCG improved 7/12 months, Spearman improved 8/12 months. These metrics appear robustly improved, not driven by outliers.

### Seasonal Pattern

Weak months remain structurally the same: 2022-06 (summer) and 2022-09 (late summer/fall) are persistently weak across all versions on EV-VC@100/500. 2021-05 (spring) is weak on EV-VC@100. These likely reflect market regime characteristics (lower congestion, fewer binding constraints) rather than model deficiencies. The HP change notably improved the weakest months' EV-VC@500, suggesting more trees + lower LR provides better resolution for sparse-signal regimes.

## Regression Quality (C-RMSE, C-MAE, Spearman)

| Metric | v0011 | v0012 | Delta |
|--------|-------|-------|-------|
| C-RMSE | 2866.6 | 2855.3 | -0.4% |
| C-MAE | 1142.5 | 1135.0 | -0.7% |
| Spearman | 0.3925 | 0.3940 | +0.4% |

All three calibration metrics improved modestly. The more-trees-lower-LR combination provides finer-grained gradient estimation, producing slightly more calibrated predictions. C-RMSE improved despite the lower learning rate not driving toward RMSE minimization — the improvement likely comes from better ensemble averaging reducing variance.

2021-11 remains the worst C-RMSE month (~4767 in v0012 vs ~4798 in v0011), but improved slightly. 2022-06 remains the worst C-MAE month (~2225 in v0012 vs ~2220 in v0011), essentially flat.

## EV Ranking Quality

The core business metric — ranking by P(binding) × predicted_$ — shows a favorable tradeoff:

- **Top-100 precision (EV-VC@100)**: -5.3% mean, within acceptable bounds given +14.2% margin to floor
- **Top-500 breadth (EV-VC@500)**: +3.5% mean, with tail failure eliminated
- **Ranking order (EV-NDCG)**: +0.2%, consistent across months
- **Broader coverage (EV-VC@1000)**: +2.6%, showing the breadth recovery extends beyond top-500

The more-trees-lower-LR change appears to have shifted the ranking function from aggressively separating top-100 from the rest toward providing more granular discrimination across the full top-1000 spectrum. For capital allocation, this means the portfolio can be more confidently diversified across the top-500 positions without losing significant ranking accuracy at the very top.

## Code Review

### Changes Reviewed
1. `ml/config.py`: `n_estimators` 400→600, `learning_rate` 0.05→0.03
2. `ml/tests/test_config.py`: Corresponding assertion updates

**Findings**: None. The code change is correct and minimal. Both files are modified consistently. The test assertions match the new defaults. No other code paths are affected since these are dataclass defaults consumed downstream by XGBoost. The boosting budget went from 20 (400×0.05) to 18 (600×0.03), a slight reduction that provides more ensemble averaging at the cost of slightly less total gradient descent. This is a well-understood tradeoff.

**Feature list**: Confirmed 34 features in config.json, matching v0011. No feature changes.

## Regressor Feature Importance

No feature changes in this iteration (frozen at 34 per batch constraint). However, the HP change may have shifted internal feature importance weights. The improved EV-VC@500 on weak months suggests the longer ensemble (600 trees) is better at leveraging secondary features (density_mean, density_variance, density_entropy, tail_concentration) that help discriminate mid-tier binding severity — these features likely carry subtle signals that need more boosting rounds to extract at a lower learning rate.

**Recommendation for future**: After this batch concludes, consider a feature importance analysis comparing v0011 and v0012 to understand which features benefited most from the HP change. This could inform future feature engineering.

## Unified vs Gated Mode

No change (unified_regressor=false). Given that the gated mode is producing consistent improvements across iterations and all gates now pass comfortably, there is no urgent reason to switch to unified mode. Unified mode would increase training data volume but dilute the signal-to-noise ratio on binding constraints. **Recommendation**: Keep gated mode as default; unified mode is a lower-priority experiment for a future batch when the current gains plateau.

## Statistical Rigor

With 12 eval months:
- **EV-VC@500**: Mean +3.5%, improvement in 6/12 months including all 3 weakest months. Not noise-driven.
- **EV-VC@100**: Mean -5.3%, degradation in 7/12 months. This is a real regression, not noise. Acceptable given floor margin.
- **EV-NDCG**: Mean +0.2%, improved in 7/12 months. Small but directionally consistent.
- **Spearman**: Mean +0.4%, improved in 8/12 months. Small but directionally consistent.

The EV-VC@500 bottom_2_mean jumped from 0.0541 to 0.0698 (+29%) — this is not explainable by noise. The structural improvement in weak months is genuine, driven by the longer ensemble providing better resolution in low-signal regimes.

## Gate Calibration

### Current Assessment (v0012)

| Gate | Floor | v0012 Mean | L1 Margin | Assessment |
|------|-------|-----------|-----------|------------|
| EV-VC@100 | 0.0664 | 0.0758 | +14.2% | Moderate — narrowed from +20.6% but still comfortable |
| EV-VC@500 | 0.2179 | 0.2348 | +7.8% | Healthy — recovered from +4.2% |
| EV-NDCG | 0.7137 | 0.7518 | +5.3% | Moderate |
| Spearman | 0.3736 | 0.3940 | +5.5% | Moderate |

### Repeated Recommendations

1. **L3 noise_tolerance remains non-scale-aware** (0.02 absolute for all metrics). This continues to provide zero protection for EV-VC@100 (bot2 ~0.009) and near-impossible standards for C-RMSE (bot2 ~5283). Recommend `max(abs_floor, pct × |champ_bot2|)` with per-metric tuning. Both reviewers flagged this in prior iterations.

2. **EV-VC@100 tail_floor (0.000135) is non-protective**: v0012's worst EV-VC@100 month is 0.0006 — 4.4x above tail_floor. The tail_floor would need to be ~0.005 to have any discriminative value. Not urgent (no version has ever breached), but the gate adds no safety value in its current form.

3. **No recalibration recommended this iteration**: v0012's improvements are not large enough to warrant gate recalibration, and the gate_calibration.md correctly notes that recalibrating to v0011 would loosen EV-VC@500/NDCG/Spearman floors — undesirable. Keep current gates v4.

## Recommendations for Next Iteration

### Promotion Decision

**RECOMMEND: PROMOTE v0012.** All Group A gates pass all three layers. The primary objective (EV-VC@500 breadth recovery) is achieved. All Group B gates also pass. The EV-VC@100 regression is within acceptable bounds and the overall pipeline health is the best it has been.

### Next Iteration Priorities

1. **Recover EV-VC@100 precision** (secondary priority): v0012 traded -5.3% EV-VC@100 for +3.5% EV-VC@500. The margin is still +14.2%, but two consecutive iterations of EV-VC@100 regression (feature changes in iter 1 concentrated gains, HP changes in iter 2 diluted them) risk compounding. Options:
   - **min_child_weight reduction** (25→15-20): Currently very conservative. Lower MCW allows more granular leaf predictions, potentially improving top-100 discrimination without affecting breadth.
   - **max_depth increase** (5→6): More interaction depth could sharpen top-100 separation. Risk: overfitting. Mitigated by the larger ensemble (600 trees) and low LR.

2. **Explore value_weighted=True**: Weighting training samples by shadow price magnitude would emphasize high-dollar constraints in the loss function. This directly targets top-100 accuracy (high-value constraints should be precisely predicted). Low-risk, easy A/B test.

3. **Do NOT touch features this batch**: The 34-feature set is clean and stable. Feature engineering should be a separate batch with fresh analysis.

4. **Do NOT touch learning_rate or n_estimators further**: The 600/0.03 setting achieved its goal. Further increases in tree count would add training time with diminishing returns.

### What to Screen On

If min_child_weight or value_weighted experiments are run:
- **Weak month**: 2022-06 (weakest EV-VC@100, weakest EV-VC@500)
- **Strong month**: 2021-09 (strong EV-VC@100, used to verify no top-100 regression)
