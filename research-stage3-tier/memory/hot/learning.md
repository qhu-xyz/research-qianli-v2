# Learning

## v0 Baseline Insights

1. **Tier 4 is empty**: No negative shadow prices exist in real MISO data. The model effectively operates as a 4-class classifier. Consider removing tier 4 entirely or keeping with 0 weight.

2. **Tier 1 recall is catastrophic (0.045)**: The model rarely predicts tier 1. This is a structural class weight issue — FE alone cannot fix it. Tier 1 constraints ([1000, 3000)) sit between moderate and heavy binding; the model defaults to predicting adjacent tiers.

3. **Class imbalance dominates**: Even with class weights {0:10, 1:5}, the model overwhelmingly predicts tiers 3-4 (majority classes).

4. **Feature importance (v0)**: Top 5 by gain: recent_hist_da (21.1%), hist_da (13.3%), prob_band_95_100 (6.8%), prob_band_100_105 (6.5%), hist_da_trend (3.8%). Bottom 6 all ~1.1-1.2%.

5. **Monthly variance is high**: VC@100 ranges 0.003-0.249. Some months have very few high-value constraints, making metrics noisy.

## Feature Engineering Learnings

6. **Interaction features work but have a low ceiling**: v0005 added overload_x_hist, prob110_x_recent_hist, tail_x_hist. All metrics improved (+5.4% VC@100) but insufficiently to cross VC@100 floor. XGBoost was already approximating these interactions via multi-level splits.

7. **Pruning hurts weak months**: Hypothesis B (add 3 + prune 5) lost to A (add 3 only). Pruning low-importance features removed marginal signal needed for difficult months like 2022-06.

8. **Bottom_2_mean responds more than mean**: v0005's bottom_2_mean improved 64% while mean improved only 5.4%. Interaction features help most on difficult months.

9. **Improvement distribution matters**: VC@100 improved in 8/12 months (broad) but VC@500/NDCG only in 5/12 (concentrated, driven by 2022-12). Different metrics respond differently to the same feature changes.

## Process Learnings

10. **CRITICAL: Commit all HUMAN-WRITE-ONLY changes before launching pipeline**: The pre-merge guard diffs main working tree vs worktree. Uncommitted edits to evaluate.py/gates.json cause the guard to reject ALL worker output. Root cause of 4 consecutive failures across tier-fe-1 and tier-fe-2.

11. **Leaked state**: Workers increment version_counter.json without producing artifacts when the merge guard rejects. After 4 failures, version_counter reached next_id=5 but only v0 exists in registry/.

12. **Pre-launch checklist**: Before `run_pipeline.sh`: (a) `git status` shows no uncommitted changes to evaluate.py or gates.json, (b) `git diff` on HUMAN-WRITE-ONLY files is empty, (c) all gates have `pending_baseline: false`.

## Technical Notes

- XGBoost `predict_proba` drops unseen classes — must pad with zeros via `model.classes_`
- `multi:softprob` objective required for probability outputs
- monotone_constraints must be tuple, not list, for XGBoost
- mem_mb() uses ru_maxrss/1024 on Linux
- Dead interaction features (hist_physical_interaction, overload_exceedance_product, etc.) are computed but excluded by `_DEAD_FEATURES` in config.py
- noise_tolerance=0.02 is too loose for small-scale metrics like Tier-Recall@1 (bottom_2=0.003) — flag for future recalibration
- Value-QWK is fragile (barely passing by 0.0004) — monitor closely
