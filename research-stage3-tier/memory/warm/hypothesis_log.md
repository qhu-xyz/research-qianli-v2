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

## Candidate Hypotheses (untested, for future batches)

1. **Increase tier 1 class weight** (5→15 or 20): Should improve Tier-Recall@1 from 0.098 [BLOCKED: FE-only batch]
2. **Reduce to 4 classes**: Drop tier 4 (always 0 samples) to simplify the model [BLOCKED: FE-only batch]
3. **Lower min_child_weight** (25→10): Allow finer splits for rare tier 0/1 samples [BLOCKED: FE-only batch]
4. **Increase n_estimators** (400→800): More capacity for rare class detection [BLOCKED: FE-only batch]
