# Learning

## v0 Baseline Insights

1. **Tier 4 is empty**: No negative shadow prices exist in real MISO data. The model effectively operates as a 4-class classifier. Consider:
   - Removing tier 4 entirely (4-class model)
   - Keeping tier 4 with 0 weight (current approach works but wastes a class)

2. **Tier 1 recall is catastrophic (0.098)**: The model rarely predicts tier 1. This is the most critical improvement area since tier 1 constraints ([1000, 3000)) represent significant value.

3. **Class imbalance dominates**: Even with class weights {0:10, 1:5}, the model overwhelmingly predicts tiers 3-4 (majority classes).

4. **Feature importance**: Check `registry/v0/feature_importance.json` for top features. Likely dominated by hist_da and prob_exceed features.

5. **Monthly variance is high**: VC@100 ranges 0.008-0.246. Some months have very few high-value constraints, making metrics noisy.

## Technical Notes

- XGBoost `predict_proba` drops unseen classes — must pad with zeros via `model.classes_`
- `multi:softprob` objective required for probability outputs
- monotone_constraints must be tuple, not list, for XGBoost
- mem_mb() uses ru_maxrss/1024 on Linux
