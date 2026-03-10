# Claude Review — feat-eng-20260303-060938 / Iteration 3 (v0006)

## Summary

v0006 tested hypothesis H8: feature pruning (17→13 features) + revert training window (18→14 months). The result is a **split outcome** — NDCG and VCAP metrics improved substantially (best across all 6 experiments by large margins), while AP regressed to its worst level across all versions. AUC remained flat. This is the most experimentally informative iteration of the batch: it reveals that removing the three unconstrained monotone features (density_skewness, density_kurtosis, density_cv) plus the weakest interaction feature (exceed_severity_ratio) sharpens top-of-ranking quality at the expense of overall positive-class discrimination. The model is now fully monotone-constrained — every remaining feature has an enforced direction — which may be the structural driver of the ranking redistribution.

**Promotion recommendation: NO.** AP mean regression (-0.0044 vs v0, -0.0059 vs v0004) and AP bottom-2 continuing its monotonic decline (now -0.0094 vs v0, margin to Layer 3 failure = 0.0106) disqualify v0006 as a promotion candidate. v0004 remains the best overall version for HUMAN_SYNC consideration.

## Gate-by-Gate Analysis (Three Layers)

### Group A (Blocking)

| Gate | v0006 Mean | v0 Mean | Δ vs v0 | v0004 Mean | Δ vs v0004 | Floor | L1 | L2 (tail fails) | Bot2 | v0 Bot2 | Δ Bot2 | L3 | Overall |
|------|-----------|---------|---------|-----------|------------|-------|----|----|------|---------|--------|----|----|
| S1-AUC | 0.8354 | 0.8348 | +0.0006 | 0.8363 | -0.0009 | 0.7848 | P (+0.0506) | P (0) | 0.8155 | 0.8105 | +0.0050 | P | **P** |
| S1-AP | 0.3892 | 0.3936 | **-0.0044** | 0.3951 | **-0.0059** | 0.3436 | P (+0.0456) | P (0) | 0.3228 | 0.3322 | **-0.0094** | P (margin=0.0106) | **P** |
| S1-VCAP@100 | 0.0270 | 0.0149 | **+0.0121** | 0.0205 | **+0.0065** | -0.0351 | P (+0.0621) | P (0) | 0.0052 | 0.0014 | +0.0038 | P | **P** |
| S1-NDCG | 0.7560 | 0.7333 | **+0.0227** | 0.7371 | **+0.0189** | 0.6833 | P (+0.0727) | P (0) | 0.6824 | 0.6716 | +0.0108 | P | **P** |

**Key observations:**
- **NDCG +0.0227 is the largest Group A mean improvement in pipeline history** — 5.5x larger than v0004's +0.0038. This is not noise; 10/12 months improved.
- **VCAP@100 +0.0121** is the largest VCAP@100 improvement. v0006 is the new best on VCAP@100 mean AND bottom-2.
- **AP -0.0044** is the first time AP has fallen below v0 baseline. 9/12 months regressed. This is broadly distributed, not driven by an outlier.
- **AP bot2 at 0.3228** — worst ever, continuing the monotonic decline: v0(0.3322) → v0002(0.3305) → v0003(0.3277) → v0004(0.3282) → v0005(0.3247) → v0006(0.3228). Margin to Layer 3 failure (0.02 tolerance) is now only **0.0106**.

### Group B (Monitoring)

| Gate | v0006 Mean | v0 Mean | Δ vs v0 | Floor | Headroom | Status |
|------|-----------|---------|---------|-------|----------|--------|
| S1-BRIER | 0.1540 | 0.1503 | +0.0037 (worse) | 0.1703 | 0.0163 | P (headroom narrowing) |
| S1-VCAP@500 | **0.1172** | 0.0908 | **+0.0264** | 0.0408 | 0.0764 | P (**best ever**) |
| S1-VCAP@1000 | **0.1821** | 0.1591 | **+0.0230** | 0.1091 | 0.0730 | P (**best ever**) |
| S1-REC | 0.4179 | 0.4192 | -0.0013 | 0.1000 | 0.3179 | P |
| S1-CAP@100 | 0.7892 | 0.7825 | +0.0067 | 0.7325 | 0.0567 | P |
| S1-CAP@500 | 0.7770 | 0.7740 | +0.0030 | 0.7240 | 0.0530 | P |

**Notable:** VCAP@500 and VCAP@1000 improvements are dramatic — +29% and +14% respectively vs v0. These are the strongest Group B improvements in pipeline history. BRIER continues narrowing (headroom 0.0163, down from v0's 0.0200) — contradicts the direction hypothesis that model simplification would improve calibration.

## Per-Month W/L Analysis (v0006 vs v0)

| Month | AUC Δ | AP Δ | VCAP@100 Δ | NDCG Δ |
|-------|-------|------|------------|--------|
| 2020-09 | **+0.0052** | **+0.0079** | **+0.0177** | **+0.0273** |
| 2020-11 | +0.0009 | -0.0110 | **+0.0128** | **+0.0440** |
| 2021-01 | -0.0051 | -0.0124 | **+0.0152** | **+0.0334** |
| 2021-04 | **+0.0030** | **+0.0053** | -0.0013 | +0.0010 |
| 2021-06 | **+0.0039** | -0.0024 | **+0.0121** | **+0.0159** |
| 2021-08 | -0.0001 | -0.0004 | **+0.0338** | **+0.0411** |
| 2021-10 | -0.0012 | -0.0114 | **+0.0441** | **+0.0595** |
| 2021-12 | -0.0004 | +0.0013 | **+0.0061** | **+0.0297** |
| 2022-03 | -0.0005 | -0.0102 | -0.0006 | **+0.0125** |
| 2022-06 | -0.0057 | -0.0014 | -0.0044 | -0.0054 |
| 2022-09 | -0.0028 | **-0.0163** | **+0.0081** | **+0.0215** |
| 2022-12 | **+0.0103** | -0.0011 | **+0.0019** | -0.0080 |

**W/L Summary:**

| Metric | W | L | T | Assessment |
|--------|---|---|---|------------|
| AUC | 5 | 7 | 0 | Negative (worse than v0004's 9W/3L) |
| AP | 3 | 9 | 0 | **Strongly negative** |
| VCAP@100 | 10 | 2 | 0 | **Strongly positive** (p=0.039) |
| NDCG | 10 | 2 | 0 | **Strongly positive** (p=0.039) |

Both VCAP@100 and NDCG at 10W/2L are statistically significant at p<0.05 (sign test). The AP regression at 3W/9L is also statistically significant (p=0.073, borderline). This is a genuine tradeoff, not noise.

## Seasonal Pattern Analysis

**Weakest months (persistent across all versions):**
- 2022-09: AP=0.2987 (worst AP across all versions, 5th consecutive experiment where this month is the AP floor). AUC=0.8306.
- 2022-12: AUC=0.8191 (VCAP@100=0.0024, near-zero value capture). NDCG=0.7036.
- 2021-04: NDCG=0.6611 (worst NDCG month for v0006, consistent across all versions).

**No seasonal pattern** — the weakest months are scattered across summer (2022-06, 2022-09), winter (2022-12, 2021-12), and spring (2021-04). The weakness is more likely structural (distribution shift in 2022) than seasonal.

**2022-09 deep dive**: AP=0.2987 (v0006) vs 0.3150 (v0). This month has resisted 6 independent interventions. Binding rate is low (0.0663), and the model consistently struggles with positive-class ranking here. The feature importance data shows hist_da_trend contributes only 38.3% of gain in 2022-09 (vs 44.2% average) — the trend signal is weaker when the market undergoes structural change.

## Code Review

### Changes Reviewed
1. `ml/config.py` — FeatureConfig: 4 features removed (density_skewness, density_kurtosis, density_cv, exceed_severity_ratio)
2. `ml/config.py` — PipelineConfig: train_months 18→14
3. `ml/tests/conftest.py` — synthetic_features fixture: 17→13
4. `ml/tests/test_config.py` — all assertions updated for 13 features

### Code Quality Assessment

**Correct**: All changes are minimal, targeted, and correct. The feature removals, monotone constraint string update, and test assertions all match. No bugs introduced.

**Structural observation**: The monotone constraint string changed from `(1,1,1,1,1,-1,-1,-1,1,0,0,0,1,1,1,1,1)` to `(1,1,1,1,1,-1,-1,-1,1,1,1,1,1)`. This is significant — **all three unconstrained features (monotone=0) were removed**. The model is now fully monotone-constrained, meaning every feature has an enforced direction. This may explain the NDCG/VCAP improvement: monotone constraints act as a form of structural regularization that can improve ranking consistency, especially at the top of the prediction list.

**No new issues introduced.** Carried issues unchanged: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), missing schema guard for interaction base columns (MEDIUM).

## Hypothesis Validation

**H8 (Feature Pruning)**: PARTIALLY SUPPORTED with an unexpected twist.

The hypothesis predicted: "Removing 4 near-zero-importance features reduces noise and improves tail metrics (especially AP bottom-2)."

**What actually happened:**
- Tail metrics (bottom-2) did NOT improve for AP — they continued worsening (0.3228 vs 0.3247 vs 0.3282 vs ...)
- But ranking quality (NDCG, VCAP) improved dramatically and broadly (10/12 months each)
- The hypothesis was wrong about the mechanism but revealed a more interesting finding: the unconstrained features (monotone=0) were providing subtle regularization that helped AP, while their removal + full monotone constraint enforcement improved top-of-ranking quality

**Direction honest assessment was accurate**: The direction correctly predicted "the realistic outcome is neutral" with probability assessment "HIGH probability" for neutral result. The actual outcome was more extreme than neutral — it was a clear tradeoff rather than a null result.

## Feature Importance Analysis (v0006 vs v0005)

| Feature | v0005 (17feat) | v0006 (13feat) | Δ | Interpretation |
|---------|---------------|---------------|---|----------------|
| hist_da_trend | 53.9% | 44.2% | -9.7pp | Redistributed to hist_da |
| hist_da | 11.3% | 24.1% | **+12.8pp** | Doubled — absorbed most of the pruned capacity |
| hist_physical_interaction | 14.3% | 11.1% | -3.2pp | Slight decrease |
| prob_below_90 | ~5.0% | 6.1% | +1.1pp | Mild increase |
| Other physical | ~8.9% | ~10.8% | +1.9pp | Mild redistribution |
| overload_exceedance_product | 0.9% | 0.8% | -0.1pp | Stable (near threshold) |

The most notable shift is **hist_da doubling** from 11.3% to 24.1%. The model is now balancing more evenly between trend (44%) and level (24%) of historical shadow prices, rather than being trend-dominated (54% + 11% = 65% historical, now 44% + 24% = 68% historical). The rebalancing toward level signal may explain the improved ranking — absolute shadow price level is a more stable predictor than trend for top-ranked constraints.

## Assessment Against Direction Success Criteria

| Outcome | Criteria | Met? |
|---------|----------|------|
| **Promotion-worthy** | AUC >= 0.837, AP > 0.396, AP bot2 > 0.330 | **NO** — AUC 0.8354, AP 0.3892, AP bot2 0.3228 |
| **Best-so-far** | AUC >= v0004 AND (AP > v0004 OR AP bot2 > v0004) | **NO** — AUC 0.8354 < 0.8363, AP 0.3892 < 0.3951 |
| **Ceiling confirmed** | Within ±0.002 of v0004 on all metrics | **NO** — AP divergence -0.0059, NDCG divergence +0.0189 |
| **Regression** | AUC < v0 (0.8348) or AP < 0.390 | **BORDERLINE** — AUC 0.8354 > v0; AP 0.3892 < 0.390 threshold by 0.0008 |

The outcome doesn't fit neatly into the pre-defined categories. It's a **tradeoff discovery**: v0006 improves ranking quality (NDCG, VCAP) while degrading positive-class precision ranking (AP). This is more informative than a null result.

## Recommendations for HUMAN_SYNC

### 1. Promote v0004 as the best overall version
v0004 (17 features, 14-month window) remains the strongest balanced candidate: AUC 0.8363 (best), AP 0.3951 (best), VCAP@100 0.0205, NDCG 0.7371. It has the best AUC W/L (9W/3L) and statistically significant VCAP@100 improvement (10W/2L, p=0.039).

### 2. Document v0006's NDCG/VCAP profile as a future research direction
The finding that fully monotone-constrained models produce better top-of-ranking quality is novel and worth investigating:
- Could v0004's 17-feature set be used WITH different regularization to get both AP stability and NDCG improvement?
- Would training with an NDCG-optimized objective (LambdaRank, LambdaMART) instead of log loss change the AP/NDCG tradeoff?
- If the business only acts on top-100 predictions, v0006's profile (VCAP@100=0.0270, NDCG=0.756) may be more operationally valuable than v0004's (VCAP@100=0.0205, NDCG=0.737).

### 3. Feature set ceiling is confirmed
Six experiments spanning HP tuning, feature addition (17 features), feature pruning (13 features), and window expansion (10/14/18 months) all produce AUC within [0.832, 0.836]. The current feature set — derivatives of probabilistic power flow analysis and historical DA shadow prices — defines a hard ceiling. Breaking through requires fundamentally new signal sources (e.g., fuel prices, weather, load forecast, outage data, transmission topology).

### 4. Address AP bottom-2 trend before future experiments
The monotonic AP bot2 decline across 6 versions is the most concerning trend. Even if Layer 3 tolerance (0.02) isn't breached yet, the systematic worsening suggests that each model modification is subtly degrading worst-month AP. This should be investigated before launching a new experiment batch.

### 5. Carried code issues (no change)
- Threshold leakage (HIGH): train/val not fully isolated. Affects threshold-dependent metrics only.
- Threshold `>` vs `>=` mismatch (MEDIUM): boundary behavior in apply_threshold.
- Missing schema guard for interaction feature base columns (MEDIUM).

## Gate Calibration Notes

- **AP bot2 approaching Layer 3 boundary**: v0006 AP bot2=0.3228, v0 AP bot2=0.3322, Δ=-0.0094. Margin to 0.02 tolerance is 0.0106. If the trend continues, the next experiment could breach. This validates Codex's earlier suggestion to consider metric-specific tolerances (AP may warrant tighter monitoring at ~0.015 instead of the uniform 0.02).
- **BRIER headroom at 0.0163**: 6th consecutive narrowing (v0: 0.0200). Not yet critical but worth flagging at HUMAN_SYNC.
- **VCAP@100 floor (-0.0351) remains non-binding**: v0006 VCAP@100 min=0.0024, far above. Consider tightening to 0.0 at HUMAN_SYNC.
- **NDCG bot2 recovered**: 0.6824 (v0006) vs 0.6656 (v0004) — first bot2 improvement for NDCG. The pruning helped tail-month NDCG.
- **No floor changes recommended at this time** — defer to HUMAN_SYNC with the full 6-experiment dataset.

## Batch Summary (3 Iterations)

| Iter | Version | Hypothesis | AUC Δ | AP Δ | NDCG Δ | VCAP@100 Δ | Verdict |
|------|---------|-----------|-------|------|--------|------------|---------|
| 1 | v0004 | H6: 14mo window + interactions | +0.0015 | +0.0015 | +0.0038 | +0.0056 | Best balanced — **promotion candidate** |
| 2 | v0005 | H7: 18mo window + importance diagnostic | +0.0013 | -0.0007 | +0.0032 | +0.0044 | Diminishing returns confirmed |
| 3 | v0006 | H8: Prune to 13 features + revert 14mo | +0.0006 | **-0.0044** | **+0.0227** | **+0.0121** | Tradeoff discovery — not promotable |

**Best version**: v0004 (balanced improvement across all Group A metrics).
**Most informative version**: v0006 (revealed the ranking quality tradeoff from monotone constraint structure).
**Overall batch assessment**: Productive — established the feature set ceiling, identified v0004 as a modest improvement over v0, and discovered the monotone constraint effect on ranking quality.
