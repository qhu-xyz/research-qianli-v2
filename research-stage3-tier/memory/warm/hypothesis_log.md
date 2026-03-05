# Hypothesis Log

## Tested Hypotheses

### H1a: Interaction features + light pruning (iter1, UNTESTED — worker failed)
- **Hypothesis**: Swapping 3 low-importance features (density_skewness, density_cv, prob_exceed_90) for 3 interaction features (hist_physical_interaction, overload_exceedance_product, band_severity) should improve tier 0/1 discrimination and Tier-VC@100
- **Rationale**: Pre-computed interactions between price history and flow exceedance should make compound severity easier to detect at max_depth=5
- **Result**: UNTESTED — worker failed, no artifacts produced
- **Retry**: Queued for iter2 in simplified form

### H1b: Aggressive feature pruning 34→28 (iter1, UNTESTED — worker failed)
- **Hypothesis**: Removing 6 lowest-importance features should improve sampling efficiency and concentrate tree splits on high-signal features
- **Result**: UNTESTED — worker failed

### H2a: Add 3 interaction features → 37 features (iter2, UNTESTED — worker failed)
- **Hypothesis**: Adding hist_physical_interaction, overload_exceedance_product, hist_seasonal_band to the existing 34 features should improve compound severity detection for tier 0/1
- **Rationale**: Pre-computed interactions between price history and flow exceedance provide XGBoost with explicit compound signals at max_depth=5
- **Result**: UNTESTED — worker failed (identical failure mode to iter1)
- **Retry**: Queued as sole hypothesis for iter3 (last iteration)

### H2b: Prune 6 + add 3 interactions → 31 features (iter2, UNTESTED — worker failed)
- **Hypothesis**: Removing 6 lowest-importance features + adding 3 interactions
- **Result**: UNTESTED — worker failed
- **Status**: DROPPED — no iterations remaining for A/B. Iter3 tests only H2a.

### H3a: Add 3 interaction features → 37 features (tier-fe-2 iter1, UNTESTED — worker failed)
- **Hypothesis**: Adding overload_x_hist, prob110_x_recent_hist, tail_x_hist to existing 34 features. These combine top-importance features (recent_hist_da 21.1%, hist_da 13.3%) with physical flow signals. Should help tier 0/1 discrimination and improve Tier-VC@100.
- **Result**: UNTESTED — 3rd consecutive worker failure. Worker wrote handoff without executing any code changes or benchmark.
- **Status**: RETRY in iter2 (same hypothesis, simplified direction)

## Candidate Hypotheses (untested, for future batches)

1. **Increase tier 1 class weight** (5→15 or 20): Should improve Tier-Recall@1 from 0.098 [BLOCKED: FE-only batch]
2. **Reduce to 4 classes**: Drop tier 4 (always 0 samples) to simplify the model [BLOCKED: FE-only batch]
3. **Lower min_child_weight** (25→10): Allow finer splits for rare tier 0/1 samples [BLOCKED: FE-only batch]
4. **Increase n_estimators** (400→800): More capacity for rare class detection [BLOCKED: FE-only batch]
