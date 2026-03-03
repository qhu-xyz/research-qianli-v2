# Direction — Iteration 2 (feat-eng-20260303-060938, v0005)

## Hypothesis: H7 — Extended Training Window (18 months) + Feature Importance Analysis

**Core question**: Does further window expansion (14→18 months) continue the positive trend, and which features actually drive the model's predictions?

Iteration 1 (v0004) established that 14-month window + 17 features produces the best results so far: AUC 9W/3L, VCAP@100 10W/2L (p=0.039). Window expansion has been the most productive single lever. If the marginal benefit of +4 months hasn't diminished, train_months=18 could push AUC toward 0.838+.

Simultaneously, we need feature importance data to inform iter 3. After 4 experiments, we've confirmed the feature set has a ceiling (~AUC 0.836). Understanding which features contribute most (and which might be noise) is essential for deciding whether to prune or expand in iter 3.

## Specific Changes

### 1. Change train_months from 14 to 18 (MAIN CHANGE)

**File**: `ml/config.py` → `PipelineConfig.train_months`

Change `train_months = 14` to `train_months = 18`.

This gives the model 80% more training data than v0's original 10-month window and 29% more than v0004's 14-month window. The increased diversity should help late-2022 months especially.

**Feasibility check**: The earliest eval month is 2020-09. With train_months=18 + val_months=2, the model needs data starting from 2019-01. Verify that the data loader can provide data from this period. If the earliest available data starts later than 2019-01, reduce train_months to the maximum feasible value and document what was used.

### 2. Keep all 17 features (NO CHANGE)

`ml/config.py` → `FeatureConfig.step1_features` should remain with all 17 features (14 base + 3 interactions). Per D28: v0004 is the best config found so far; changing features simultaneously with window would confound the experiment.

### 3. Keep v0 HP defaults (NO CHANGE)

`ml/config.py` → `HyperparamConfig` should remain at v0 defaults:
- n_estimators=200, max_depth=4, learning_rate=0.1
- subsample=0.8, colsample_bytree=0.8
- reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10

### 4. Export feature importance across all eval months (NEW — DIAGNOSTIC)

**After** the benchmark pipeline completes, add a step to extract and save per-month feature importance:

**Output file**: `registry/${VERSION_ID}/feature_importance.json`

For each of the 12 eval months, extract XGBoost's gain-based feature importance from the trained model. Save as a JSON structure:
```json
{
  "importance_type": "gain",
  "per_month": {
    "2020-09": {"prob_exceed_110": 0.XX, "prob_exceed_105": 0.XX, ...},
    ...
  },
  "aggregate": {
    "mean": {"prob_exceed_110": 0.XX, ...},
    "std": {"prob_exceed_110": 0.XX, ...}
  }
}
```

Implementation approach:
- After `run_benchmark()` completes, the model objects should be accessible (or re-loadable from the gzipped .ubj files in the registry).
- Use `model.get_score(importance_type='gain')` for XGBoost models.
- If models are discarded after evaluation, modify `_eval_single_month()` to return the model along with metrics, or extract importance during the eval loop.
- This is a **diagnostic output only** — it does not affect model training or evaluation. If implementation proves complex (e.g., requires significant refactoring to retain model objects), the worker may skip this step and note it in the handoff. The priority is the train_months=18 experiment.

### 5. No bug fixes needed this iteration

The f2p parsing and dual-default fragility were fixed in v0004. No new HIGH or MEDIUM code issues require immediate attention.

## Execution Order

1. Change `train_months` from 14 to 18 in config.py
2. Run tests: `python -m pytest ml/tests/ -v`
3. Run benchmark pipeline (full 12-month eval)
4. Extract and save feature importance (if feasible without major refactoring)
5. Run validate + compare against v0
6. Commit, then write handoff

## Expected Impact

| Metric | v0 Baseline | v0004 (iter 1) | Expected v0005 (if trend continues) | Expected (if diminishing returns) |
|--------|-------------|----------------|--------------------------------------|-----------------------------------|
| S1-AUC | 0.8348 | 0.8363 | 0.837–0.839 | 0.836–0.837 |
| S1-AP | 0.3936 | 0.3951 | 0.396–0.400 | ~0.395 |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.022–0.028 | ~0.021 |
| S1-NDCG | 0.7333 | 0.7371 | 0.738–0.742 | ~0.737 |
| AUC W/L vs v0 | — | 9W/3L | ≥9/12 | 7-8/12 |

**Success criteria**:
- **Promotion-worthy**: AUC > 0.837 AND ≥8/12 wins AND AP > 0.396 AND no Group A gate failures
- **Encouraging (refine in iter 3)**: AUC > 0.836, VCAP@100 continues to improve, feature importance data collected
- **Diminishing returns (pivot in iter 3)**: AUC ≤ 0.836 or fewer wins than v0004 → window expansion has peaked, pivot to feature importance-guided pruning

## Risk Assessment

1. **Diminishing returns from window expansion (MEDIUM)**: The 10→14 jump gave +0.0013 AUC. The 14→18 jump may give less if the additional 4 months of historical data are less informative. Mitigation: even if AUC gain is zero, the feature importance data makes this iteration worthwhile for informing iter 3.

2. **Data availability for early months (MEDIUM)**: With train_months=18, the 2020-09 eval month needs training data from 2019-01. If data doesn't go back that far, early eval months may fail or use truncated training windows. Mitigation: the worker should check data availability first. If insufficient, reduce to train_months=16 as fallback and document.

3. **VCAP@500 breaching Group B floor (LOW-MEDIUM)**: v0004 bot2=0.0387 is only 0.0021 above the floor. Further window expansion may push this below 0.0408. Mitigation: Group B is non-blocking. Document the breach if it occurs and flag for HUMAN_SYNC.

4. **BRIER regression (LOW)**: BRIER has narrowed to 0.0187 headroom. Adding 4 more training months may further degrade calibration. Mitigation: Group B, non-blocking. Monitor.

5. **2022-09 still stuck (HIGH likelihood, LOW impact)**: This month has resisted 4 interventions. More training data is unlikely to help given the fundamental low-binding-rate issue. Mitigation: expected failure, document. Iter 3 may need different approach.

6. **Feature importance extraction complexity (LOW)**: If the benchmark pipeline doesn't retain trained models, extracting importance requires refactoring. Mitigation: this is marked as optional — worker should prioritize the window expansion experiment.

## What NOT To Do

- Do NOT change hyperparameters (confirmed dead end — 4 experiments)
- Do NOT change threshold_beta (keep 0.7)
- Do NOT change val_months (keep 2)
- Do NOT modify gates.json or evaluate.py
- Do NOT add or remove features (keep all 17)
- Do NOT touch registry/v0/ or any other registry/v*/ except registry/v0005/
- Do NOT invest more than ~30 minutes on feature importance extraction — it's diagnostic, not critical
