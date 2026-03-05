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

6. **Worker reliability**: Complex directions with 2 competing hypotheses + detailed overrides increase failure risk. Simpler, single-hypothesis directions with explicit step-by-step instructions are safer.

7. **Leaked state**: Workers can increment version_counter.json without producing artifacts. The orchestrator should be aware of version counter drift after failures.

## Technical Notes

- XGBoost `predict_proba` drops unseen classes — must pad with zeros via `model.classes_`
- `multi:softprob` objective required for probability outputs
- monotone_constraints must be tuple, not list, for XGBoost
- mem_mb() uses ru_maxrss/1024 on Linux
