# Executive Summary — Batch feat-eng-20260303-060938

**Date**: 2026-03-03
**Iterations**: 3
**Versions tested**: v0004, v0005, v0006
**Champion before**: None (v0 baseline)
**Champion after**: None (v0004 recommended for HUMAN_SYNC)
**Batch type**: Feature engineering + window optimization

## Objective

Test additivity of positive-signal levers (14-month training window + interaction features), then refine configuration through window expansion and feature pruning. This batch built on prior evidence from 3 earlier experiments (HP tuning, interaction features, window expansion) that established an AUC ceiling at ~0.836.

## Results Summary

| Iter | Version | Hypothesis | AUC vs v0 | AP vs v0 | NDCG vs v0 | VCAP@100 vs v0 | Verdict |
|------|---------|-----------|-----------|----------|------------|----------------|---------|
| 1 | v0004 | H6: 14mo window + interactions (combined) | +0.0015 (9W/3L) | +0.0015 (6W/6L) | +0.0038 (7W/5L) | +0.0056 (10W/2L**) | **Best balanced** |
| 2 | v0005 | H7: 18mo window + importance diagnostic | +0.0013 (7W/5L) | -0.0007 (6W/6L) | +0.0032 (8W/4L) | +0.0044 (10W/2L) | Diminishing returns |
| 3 | v0006 | H8: Prune to 13 features + revert 14mo | +0.0006 (5W/7L) | **-0.0044** (3W/9L) | **+0.0227** (10W/2L**) | **+0.0121** (10W/2L**) | Tradeoff discovery |

(**) = statistically significant at p<0.05 (sign test)

## Key Findings

### 1. Feature Set Ceiling Confirmed
Six experiments spanning HP tuning, interaction features, window expansion (10/14/18 months), and feature pruning all produce AUC within [0.832, 0.836]. The current feature set — probabilistic power flow analysis + historical DA shadow prices — defines a hard ceiling. Breaking through requires fundamentally new signal sources.

### 2. v0004 Is the Best Balanced Version
- **Config**: 17 features (14 base + 3 interactions), 14-month training window, v0 HP defaults
- **AUC**: 0.8363 (best mean), 9W/3L (best W/L of any experiment)
- **VCAP@100**: 0.0205, 10W/2L, p=0.039 (first statistically significant improvement in pipeline history)
- **AP**: 0.3951 (best mean across all versions)
- **Recommended for HUMAN_SYNC promotion consideration**

### 3. Monotone Constraint Structure Affects Ranking Quality (Novel Finding)
v0006 (feature pruning) revealed that removing all unconstrained features (monotone=0) makes the model fully monotone-constrained, which:
- **Improves** top-of-ranking quality: NDCG +0.0227 (5.5x any prior improvement), VCAP@100 +0.0121
- **Degrades** positive-class breadth ranking: AP -0.0044 (first regression below v0)
- This is a structural effect, not noise reduction. The unconstrained features provided implicit regularization for AP despite <1% gain contribution.

### 4. AP Bottom-2 Trend Is Concerning
Monotonic decline across all 6 real-data experiments:
- v0: 0.3322 → v0002: 0.3305 → v0003: 0.3277 → v0004: 0.3282 → v0005: 0.3247 → v0006: 0.3228
- Margin to Layer 3 failure (0.02 tolerance) is now only 0.0106
- Every model modification worsens AP in the weakest months — the v0 feature set happens to be well-calibrated for worst-month AP

### 5. Window Expansion Is Exhausted
- 10→14 months: Small positive signal (AUC +0.0013, 7W/4L)
- 14→18 months: Zero marginal benefit (AUC -0.0002 vs v0004)
- Optimal window: 14 months. Older data adds noise, not signal.

### 6. Feature Importance Hierarchy
From v0005 diagnostic (17-feature model):
| Tier | Features | % Gain |
|------|----------|--------|
| Dominant | hist_da_trend | 53.9% |
| Strong | hist_physical_interaction, hist_da | 25.5% |
| Moderate | prob_below_90, prob_exceed_90, prob_exceed_95 | 10.3% |
| Weak | 7 features | 8.9% |
| Near-zero (pruned in v0006) | density_kurtosis, density_cv, exceed_severity_ratio, density_skewness | 1.4% |

The model is fundamentally a historical trend predictor (79% of gain from historical features), augmented by physical flow features (18%).

## Decisions for HUMAN_SYNC

### Promotion
- **Recommend v0004 for promotion consideration** — best balanced improvement across all Group A metrics, first statistically significant result (VCAP@100)
- v0006 not promotable — AP regression disqualifies it under precision-over-recall business objective

### Gate System
1. **Activate Layer 3**: Set champion (v0 or v0004). Currently null → L3 disabled.
2. **Consider metric-specific L3 tolerances**: AUC/NDCG ~0.01, AP ~0.015, VCAP@100 ~0.02
3. **VCAP@100 floor**: Tighten from -0.035 to 0.0 (make informative)
4. **BRIER headroom**: 0.0163 — narrowing trend, monitor

### Code Health
| Issue | Severity | Recommendation |
|-------|----------|----------------|
| Threshold leakage (train=val for threshold tuning) | HIGH | Fix — add holdout split |
| Threshold `>` vs `>=` mismatch | MEDIUM | Fix — align with PR curve convention |
| Silent ptype fallback | MEDIUM | Fix — raise ValueError |
| Missing schema guard for interaction columns | MEDIUM | Fix |

### Next Batch Direction
- **New data sources required**: Fuel prices, weather/load forecasts, outage data, transmission topology changes
- **Ranking objective exploration**: LambdaRank/LambdaMART instead of log loss — may resolve AP/NDCG tradeoff
- **Monotone constraint optimization**: Investigate selective enforcement (some features constrained, others free) vs binary all-or-nothing
- **2022-09 investigation**: Structural weakness at lowest binding rate (6.63%) — 6 independent interventions have failed. May require regime-detection features.

## Hypotheses Tested (This Batch + Prior)

| # | Hypothesis | Result | Key Number |
|---|-----------|--------|------------|
| H3 | HP tuning (deeper + slower) | REFUTED | AUC -0.0025, 0W/11L |
| H4 | Interaction features | NOT SUPPORTED | AUC +0.000, 5W/6L |
| H5 | Window expansion 10→14 | INCONCLUSIVE (weak +) | AUC +0.0013, 7W/4L |
| H6 | Combined window + interactions | PARTIALLY CONFIRMED | AUC +0.0015, 9W/3L; VCAP@100 p=0.039 |
| H7 | Window expansion 14→18 | FAILED | AUC -0.0002 vs v0004 |
| H8 | Feature pruning 17→13 | TRADEOFF | NDCG +0.023 / AP -0.004 |
