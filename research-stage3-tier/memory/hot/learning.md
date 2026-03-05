# Learning

## v0 Baseline Insights

1. **Tier 4 is empty**: No negative shadow prices exist in real MISO data. The model effectively operates as a 4-class classifier. Consider:
   - Removing tier 4 entirely (4-class model)
   - Keeping tier 4 with 0 weight (current approach works but wastes a class)

2. **Tier 1 recall is catastrophic (0.098)**: The model rarely predicts tier 1. This is the most critical improvement area since tier 1 constraints ([1000, 3000)) represent significant value.

3. **Class imbalance dominates**: Even with class weights {0:10, 1:5}, the model overwhelmingly predicts tiers 3-4 (majority classes).

4. **Feature importance (v0)**: Top 5 by gain: recent_hist_da (21.1%), hist_da (13.3%), prob_band_95_100 (6.8%), prob_band_100_105 (6.5%), hist_da_trend (3.8%). Bottom 6 all ~1.1-1.2% (density_skewness, prob_exceed_90, density_cv, density_variance, prob_below_90, prob_exceed_95).

5. **Monthly variance is high**: VC@100 ranges 0.003-0.248. Some months have very few high-value constraints, making metrics noisy.

## Process Learnings

6. **Worker reliability is the critical bottleneck**: Two consecutive identical failures — worker writes handoff claiming "done" but produces zero artifacts. The issue is NOT direction complexity; the worker execution itself is systematically broken.

7. **Leaked state**: Workers increment version_counter.json without producing artifacts. After 2 failures, version_counter is at next_id=3 but only v0 exists in registry/.

8. **Direction simplification alone is insufficient**: Iter2 direction was simpler than iter1 (same hypotheses, clearer instructions) but produced the identical failure. The problem is not in the orchestrator's direction quality.

9. **Screening phase may be a failure amplifier**: Both failed iterations included a 2-month screening step before full benchmark. For iter3, eliminating screening and going directly to full benchmark reduces the number of steps the worker must execute.

## Technical Notes

- XGBoost `predict_proba` drops unseen classes — must pad with zeros via `model.classes_`
- `multi:softprob` objective required for probability outputs
- monotone_constraints must be tuple, not list, for XGBoost
- mem_mb() uses ru_maxrss/1024 on Linux
- Dead interaction features (hist_physical_interaction, overload_exceedance_product, hist_seasonal_band) are already computed by `compute_interaction_features()` — they exist as DataFrame columns but are excluded by `_DEAD_FEATURES` set in config.py
