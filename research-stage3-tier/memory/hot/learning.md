# Learning

## v0 Baseline Insights

1. **Tier 4 is empty**: No negative shadow prices exist in real MISO data. The model effectively operates as a 4-class classifier. Consider removing tier 4 entirely or keeping with 0 weight.

2. **Tier 1 recall is catastrophic (0.098)**: The model rarely predicts tier 1. This is the most critical improvement area since tier 1 constraints ([1000, 3000)) represent significant value.

3. **Class imbalance dominates**: Even with class weights {0:10, 1:5}, the model overwhelmingly predicts tiers 3-4 (majority classes).

4. **Feature importance (v0)**: Top 5 by gain: recent_hist_da (21.1%), hist_da (13.3%), prob_band_95_100 (6.8%), prob_band_100_105 (6.5%), hist_da_trend (3.8%). Bottom 6 all ~1.1-1.2% (density_skewness, prob_exceed_90, density_cv, density_variance, prob_below_90, prob_exceed_95).

5. **Monthly variance is high**: VC@100 ranges 0.003-0.248. Some months have very few high-value constraints, making metrics noisy.

## Process Learnings

6. **CRITICAL: Commit all HUMAN-WRITE-ONLY changes before launching pipeline**: The pre-merge guard in `run_single_iter.sh` diffs main working tree vs worktree for evaluate.py and gates.json. If main has uncommitted edits, the worktree (branched from HEAD) has the old committed version, the guard sees a diff, and rejects ALL worker output. This was the root cause of ALL 4 consecutive failures across tier-fe-1 and tier-fe-2. The workers actually ran correctly — their work was silently rejected at merge time.

7. **Leaked state**: Workers increment version_counter.json without producing artifacts when the merge guard rejects. After 4 failures, version_counter reached next_id=5 but only v0 exists in registry/.

8. **Pre-launch checklist**: Before `run_pipeline.sh`: (a) `git status` shows no uncommitted changes to evaluate.py or gates.json, (b) `git diff` on HUMAN-WRITE-ONLY files is empty, (c) all gates have `pending_baseline: false`.

9. **The interaction feature hypothesis remains the right first experiment**: Pre-computed products of top-importance features (recent_hist_da, hist_da) with physical flow signals (expected_overload, prob_exceed_110, tail_concentration) should help tier 0/1 discrimination. This has been the plan since batch tier-fe-1 and remains untested.

## Technical Notes

- XGBoost `predict_proba` drops unseen classes — must pad with zeros via `model.classes_`
- `multi:softprob` objective required for probability outputs
- monotone_constraints must be tuple, not list, for XGBoost
- mem_mb() uses ru_maxrss/1024 on Linux
- Dead interaction features (hist_physical_interaction, overload_exceedance_product, hist_seasonal_band) are already computed by `compute_interaction_features()` — they exist as DataFrame columns but are excluded by `_DEAD_FEATURES` set in config.py
