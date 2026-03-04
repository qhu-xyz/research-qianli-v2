# Hypothesis Log

No hypotheses tested yet. Candidate hypotheses for first batch:

1. **Increase tier 1 class weight** (5→15 or 20): Should improve Tier-Recall@1 from 0.098
2. **Reduce to 4 classes**: Drop tier 4 (always 0 samples) to simplify the model
3. **Lower min_child_weight** (25→10): Allow finer splits for rare tier 0/1 samples
4. **Increase n_estimators** (400→800): More capacity for rare class detection
5. **Feature engineering**: Interaction terms between prob_exceed_110 and recent_hist_da
