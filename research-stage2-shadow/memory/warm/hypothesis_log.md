# Hypothesis Log

## Batch: ralph-v2-20260304-031811

### Iter 1 — Hypothesis A: L2 Regularization
- **Config**: reg_lambda=5.0, min_child_weight=25 (all else default)
- **Rationale**: Proven best lever from prior batch (+11% EV-VC@100 in 10/2/24feat). Re-validate in current 6/2/34feat config.
- **Expected**: +5-15% EV-VC@100 mean, improved tail safety
- **Risk**: Low
- **Status**: SCREENING

### Iter 1 — Hypothesis B: L2 + Subsampling Reduction
- **Config**: reg_lambda=5.0, min_child_weight=25, subsample=0.6, colsample_bytree=0.6
- **Rationale**: Different regularization axis (diversity via randomization) should complement L2 (shrinkage). Unlike lr/trees ensemble smoothing, subsampling doesn't compete with L2.
- **Expected**: +3-8% additional on top of L2 alone, especially on weak months
- **Risk**: Medium — may starve trees on small binding subsets
- **Status**: SCREENING
