# Human Input — First Real-Data Batch

## Objective

Improve ranking quality (AUC, AP, NDCG) via hyperparameter tuning. The v0 baseline used default XGBoost params that were never optimized on real data. This is the lowest-risk, highest-signal first move.

## Business Constraint

**Precision over recall.** Capital is limited. Do NOT lower the threshold, do NOT increase recall at the cost of precision, do NOT use threshold_beta > 1.0. The current threshold (~0.83) and beta (0.7) are intentionally conservative and should remain so.

## What to Change

Hyperparameter tuning focused on ranking quality. Suggested starting config:

| Param | v0 | Proposed | Rationale |
|---|---|---|---|
| `max_depth` | 4 | 6 | Deeper trees capture more complex feature interactions in 270K-row data |
| `n_estimators` | 200 | 400 | More boosting rounds with slower learning |
| `learning_rate` | 0.1 | 0.05 | Slower learning + more trees = better generalization |
| `min_child_weight` | 10 | 5 | Allow finer leaf splits in large dataset |

All other params (subsample, colsample_bytree, reg_alpha, reg_lambda, threshold_beta) stay at v0 defaults.

## What NOT to Change
- Do NOT change threshold_beta (keep 0.7)
- Do NOT change features (iteration 2+ can explore this)
- Do NOT change training window (keep 10+2)
- Do NOT change the threshold scaling factor

## Expected Impact
- Group A ranking metrics (AUC, AP, NDCG) should improve — deeper trees + more rounds = better discrimination
- BRIER may shift slightly — monitor but don't worry unless it exceeds 0.170 ceiling
- Precision should stay stable or improve — ranking improvements help precision at any threshold
- VCAP@100 may improve if the model better separates high-value binding constraints

## Risk Assessment
- Overfitting: deeper trees (max_depth=6) could overfit. Mitigated by lower learning rate (0.05) and existing regularization (subsample=0.8, colsample=0.8, reg_alpha=0.1, reg_lambda=1.0)
- BRIER degradation: unlikely from pure HP changes but has only 0.02 headroom — flag if close
- Compute time: 400 trees × 12 months is ~2x longer than v0 benchmark. Should still fit in worker timeout.

## Success Criteria
- AUC mean > 0.835 (any improvement over v0)
- AP mean > 0.394 (any improvement)
- All Group A gates pass all 3 layers
- No Group A gate regresses on bottom_2_mean
