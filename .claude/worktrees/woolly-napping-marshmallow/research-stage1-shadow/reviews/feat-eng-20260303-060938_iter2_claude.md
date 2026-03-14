# Claude Review — v0005 (Iteration 2, feat-eng-20260303-060938)

## Summary

v0005 tested H7: expanding the training window from 14 to 18 months while collecting the first empirical feature importance data. **Diminishing returns confirmed** — v0005 is marginally worse than v0004 on every Group A metric mean (AUC -0.0002, AP -0.0023, VCAP@100 -0.0012, NDCG -0.0007). The 14→18 expansion provides zero marginal benefit; older 2018-2019 training data dilutes more than it diversifies. Window expansion is exhausted as a lever.

The real value of this iteration is the **feature importance diagnostic**. The model is overwhelmingly driven by historical shadow price signals (79% of gain: hist_da_trend 54%, hist_physical_interaction 14%, hist_da 11%), with physical flow features contributing ~18% and distribution shape features contributing <1.3%. This data is actionable for iter 3: prune the dead-weight features to reduce noise. **Recommendation: do not promote; revert to 14-month window, use feature importance to guide pruning in iter 3.**

## Gate-by-Gate Analysis

### Group A (Blocking) — All PASS

| Gate | v0 Mean | v0004 Mean | v0005 Mean | Δ vs v0 | Δ vs v0004 | Floor | Headroom | Bot2 | v0 Bot2 | Δ Bot2 vs v0 | Pass |
|------|---------|------------|------------|---------|------------|-------|----------|------|---------|--------------|------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | +0.0013 | -0.0002 | 0.7848 | +0.0513 | 0.8156 | 0.8105 | +0.0051 | P |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | -0.0007 | -0.0023 | 0.3436 | +0.0493 | 0.3247 | 0.3322 | **-0.0075** | P |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | +0.0044 | -0.0012 | -0.0351 | +0.0544 | 0.0024 | 0.0014 | +0.0010 | P |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | +0.0032 | -0.0007 | 0.6833 | +0.0532 | 0.6699 | 0.6716 | -0.0017 | P |

**Layer 1 (Mean Quality):** All pass with ~0.05 headroom. No risk of Layer 1 failure.

**Layer 2 (Tail Safety):** Zero months below any tail_floor. All pass.

**Layer 3 (Tail Non-Regression):** Disabled (champion=null). Against v0 reference, all within 0.02 tolerance. **Concern: AP bot2 degradation is accumulating** — v0005 AP bot2=0.3247 vs v0's 0.3322 (Δ=-0.0075). This is the largest Group A bot2 regression observed. It passes the 0.02 tolerance but the trend is unfavorable: v0002(-0.0017) → v0003-window(-0.0045) → v0004(-0.0040) → v0005(-0.0075). Each window expansion worsens the AP tail.

### Group B (Monitor) — All PASS

| Gate | v0 Mean | v0005 Mean | Δ vs v0 | Bot2 | v0 Bot2 | Notes |
|------|---------|------------|---------|------|---------|-------|
| S1-BRIER | 0.1503 | 0.1525 | +0.0022 | 0.1605 | 0.1584 | Headroom=0.0178 (was 0.0187). **5th consecutive narrowing.** |
| S1-VCAP@500 | 0.0908 | 0.0845 | -0.0063 | 0.0449 | 0.0469 | Bot2 improved vs v0004 (0.0387→0.0449) |
| S1-VCAP@1000 | 0.1591 | 0.1481 | -0.0110 | 0.0854 | 0.0890 | Significant mean degradation |
| S1-REC | 0.4192 | 0.4182 | -0.0010 | 0.3391 | 0.3320 | Stable |
| S1-CAP@100 | 0.7825 | 0.7825 | 0.0000 | 0.3750 | 0.3650 | Identical to v0 mean |
| S1-CAP@500 | 0.7740 | 0.7723 | -0.0017 | 0.4740 | 0.4640 | Stable |

**BRIER headroom trend (critical to monitor):** v0(0.0200) → v0003-win(0.0189) → v0004(0.0187) → v0005(0.0178). Monotonically declining. Not yet critical but approaching the point where a feature change might trigger a breach. The 18-month window slightly degraded calibration.

**VCAP@500 bot2 recovery:** Interestingly, v0005 bot2=0.0449 recovered from v0004's 0.0387 (which was dangerously close to floor 0.0408). The 18-month window stabilized the VCAP@500 tail even though the mean didn't improve.

**VCAP@1000 regression:** Mean dropped from 0.1591 (v0) to 0.1481 (v0005), a -0.0110 degradation. The broader ranking continues to trade off against top-100 quality.

## Per-Month Analysis & Seasonal Patterns

### v0005 vs v0004 Per-Month (AUC)

| Month | v0005 | v0004 | Δ | Season |
|-------|-------|-------|---|--------|
| 2020-09 | 0.8458 | 0.8471 | -0.0013 | Fall |
| 2020-11 | 0.8329 | 0.8326 | +0.0003 | Fall |
| 2021-01 | 0.8533 | 0.8532 | +0.0001 | Winter |
| 2021-04 | 0.8349 | 0.8342 | +0.0007 | Spring |
| 2021-06 | 0.8284 | 0.8263 | +0.0021 | Summer |
| 2021-08 | 0.8543 | 0.8538 | +0.0005 | Summer |
| 2021-10 | 0.8496 | 0.8509 | -0.0013 | Fall |
| 2021-12 | 0.8133 | 0.8141 | -0.0008 | Winter |
| 2022-03 | 0.8454 | 0.8453 | +0.0001 | Spring |
| 2022-06 | 0.8249 | 0.8247 | +0.0002 | Summer |
| 2022-09 | 0.8323 | 0.8345 | -0.0022 | Fall |
| 2022-12 | 0.8180 | 0.8186 | -0.0006 | Winter |

**Win/Loss vs v0004:** 7W/5L on AUC (down from v0004's 9W/3L vs v0). Deltas are all <0.003 — pure noise.

### Chronically Weak Months (5 experiments)

| Month | Metric | v0 | v0003 | v0002 | v0004 | v0005 | Trend |
|-------|--------|-----|-------|-------|-------|-------|-------|
| 2022-09 | AP | 0.315 | 0.306 | 0.314 | 0.307 | **0.299** | **Worsening** |
| 2022-12 | AUC | 0.809 | 0.819 | 0.809 | 0.819 | 0.818 | Flat-to-improved |
| 2021-04 | NDCG | 0.660 | 0.662 | 0.663 | 0.663 | 0.666 | Very slowly improving |
| 2021-12 | AUC | 0.812 | 0.814 | 0.812 | 0.814 | 0.813 | Flat |

**2022-09 AP is now at 0.2986 — the worst ever recorded** across all iterations. This month's binding rate is only 6.63% (lowest in the eval set), suggesting it's a genuinely hard month where the model's signal is weakest. Five interventions (HP tuning, interaction features, 14-mo window, combined, 18-mo window) have all failed to improve it. This is structural, not addressable by the current feature set.

## Statistical Rigor

With 12 eval months, I assess v0005 vs v0004:

| Metric | W/L | Mean Δ | Assessment |
|--------|-----|--------|------------|
| AUC | 7W/5L | -0.0002 | Noise (p≈0.77, sign test) |
| AP | 6W/6L | -0.0023 | Noise (p=1.0) |
| VCAP@100 | 6W/6L | -0.0012 | Noise (p=1.0) |
| NDCG | 7W/5L | -0.0007 | Noise (p≈0.77) |

**None of the Group A metrics show any signal vs v0004.** The 14→18 window change is indistinguishable from zero. The improvements vs v0 (AUC +0.0013, NDCG +0.0032) are inherited from the 14-month window established in v0004; the extra 4 months contribute nothing.

## Code Review Findings

### Changes Reviewed
1. **`ml/config.py:95`** — `train_months: 14 → 18`. Correct, minimal change.
2. **`ml/benchmark.py`** — Feature importance extraction (Step A/B/C from direction). Clean implementation:
   - Importance captured after `evaluate_classifier` and before cleanup — correct placement
   - `_feature_importance` key popped from per_month dicts before `aggregate_months()` — critical correctness requirement met
   - Separate file output (`feature_importance.json`) — correctly avoids polluting metrics.json
   - Uses `statistics.mean`/`statistics.stdev` — appropriate for small samples
3. **`ml/tests/test_config.py`** — Assertion updated `14 → 18`. Correct.

### No New Issues
The code changes are minimal and follow the direction faithfully. No bugs, edge cases, or regressions introduced. The feature importance extraction is particularly well done — the `_feature_importance` key convention and pre-aggregation pop are clean patterns.

### Carried Issues (unchanged)
- **Threshold leakage** (HIGH, since smoke-v7): Validation split used for both threshold tuning and evaluation. Affects threshold-dependent metrics only.
- **Threshold `>` vs `>=` mismatch** (MEDIUM, since smoke-v7): PR curve inclusive, apply_threshold exclusive. Affects threshold-dependent metrics.
- **Missing schema guard** for interaction feature base columns (MEDIUM, since v0002).
- These remain deferred to HUMAN_SYNC per D14/D19/D25/D31.

## Feature Importance Analysis (First Empirical Data)

This is the primary deliverable of iter 2. Key findings:

### Importance Tiers

| Tier | Features | % of Gain | Assessment |
|------|----------|-----------|------------|
| **Dominant** | hist_da_trend | 53.9% | Single feature drives >half the model |
| **Strong** | hist_physical_interaction, hist_da | 25.5% | Historical signals collectively 79.4% |
| **Moderate** | prob_below_90, prob_exceed_90, prob_exceed_95 | 10.3% | Core physical flow features |
| **Weak** | prob_below_95 through expected_overload (6 features) | 8.9% | Contributing but minor |
| **Near-zero** | density_kurtosis, density_cv, exceed_severity_ratio, density_skewness | 1.4% | **Pruning candidates** |

### Stability Analysis
Feature importance is remarkably stable across 12 months (hist_da_trend std=0.019 on mean=0.539, CV=3.5%). The ranking is consistent — no month has a radically different importance profile. This makes pruning decisions reliable.

### Implications
1. **The model is essentially a historical trend predictor** augmented by physical flow features. This is not inherently wrong (historical binding is the strongest predictor of future binding), but it means the model is vulnerable to regime changes where historical patterns break.
2. **hist_physical_interaction** (#2 at 14%) validates the iter 1 decision to add interaction features — this one earns its keep.
3. **exceed_severity_ratio** (0.38%) and **overload_exceedance_product** (0.90%) — the other two interaction features — contribute very little. The former should be a pruning candidate.
4. **Distribution shape features** (density_skewness 0.31%, density_kurtosis 0.58%, density_cv 0.40%) contribute 1.3% collectively despite occupying 3 feature slots. These are noise.

## Recommendations for Iteration 3

### Primary: Feature Pruning Experiment (H8)

**Revert train_months to 14** (v0004 was strictly better) and prune the bottom 4 features by importance:
- `density_skewness` (0.31% gain)
- `exceed_severity_ratio` (0.38% gain)
- `density_cv` (0.40% gain)
- `density_kurtosis` (0.58% gain)

These 4 features contribute 1.67% of total gain collectively. Removing them:
- Reduces input noise (fewer features for the model to consider)
- May slightly improve generalization on weak months (less overfitting risk)
- Keeps the proven feature set: 13 features (11 base + hist_physical_interaction + overload_exceedance_product)

**Expected impact**: Small positive or neutral on AUC/AP/NDCG. The effect will be modest — these features are <2% of gain, so removing them won't dramatically change predictions. But if the pruning clears noise that hurts tail months, it could improve bottom-2 metrics.

**Alternative to consider**: If pruning is too conservative, a more aggressive iter 3 could also drop `overload_exceedance_product` (0.9% gain) and `prob_exceed_100` / `prob_exceed_110` (1.1% and 1.1%), going from 17 to 11 features. However, this is riskier and I'd recommend the conservative prune first.

### Secondary: Consider whether window expansion lever has exhausted insights

Five experiments have established the operating envelope:
- AUC: 0.832–0.836 (range: 0.004)
- AP: 0.392–0.395 (range: 0.003)
- NDCG: 0.732–0.737 (range: 0.005)

These ranges are narrow. Feature pruning may not break through the ceiling either. If iter 3 pruning produces another ≤0.002 AUC delta, the batch should consider either: (a) promoting the best available version (v0004, AUC 0.8363) as a modest improvement, or (b) declaring the ceiling reached and pivoting to entirely new signal sources at HUMAN_SYNC.

## Gate Calibration Suggestions

1. **No gate floor changes recommended.** Gates are not blocking valid candidates. Five experiments, all within ±0.004 AUC of v0, all comfortably passing. The issue is the experiments, not the gates.

2. **BRIER headroom (0.0178) continues narrowing.** Trend is monotonic and linked to window/feature expansion. If iter 3 feature pruning improves calibration (as v0003-HP showed — simpler models calibrate better), this may self-correct. Monitor.

3. **Layer 3 tolerance (0.02) remains appropriate** for now. Largest observed bot2 regression is AP -0.0075, well within tolerance. Codex's suggestion to tighten to 0.005-0.01 for AUC/AP/NDCG remains valid for future consideration — after 5 experiments, observed bot2 shifts range ±0.008 for AP (the most volatile), so 0.01 would be appropriate. Defer to HUMAN_SYNC.

4. **VCAP@100 floor (-0.035) remains non-binding** and will likely never bind. Consider tightening to 0.0 at HUMAN_SYNC — no reasonable model should produce negative VCAP@100.

## Verdict

**Do not promote v0005.** It is strictly worse than v0004 on all Group A means and provides no improvement over the iter 1 result. Its value is entirely in the feature importance diagnostic, which should guide iter 3.

**Outcome classification: Diminishing returns** (per direction's criteria). AUC ≤ v0004 (0.8361 ≤ 0.8363), W/L 7W/5L vs v0 (down from 9W/3L). Window expansion is exhausted. Iter 3 should pivot to feature importance-guided pruning.
