# Hypothesis Log

## Batch: ralph-v2-20260304-031811

### Iter 1 — Hypothesis A: L2 Regularization ✅ CONFIRMED
- **Config**: reg_lambda=5.0, min_child_weight=25 (all else default)
- **Rationale**: Proven best lever from prior batch (+11% EV-VC@100 in 10/2/24feat). Re-validate in current 6/2/34feat config.
- **Expected**: +5-15% EV-VC@100 mean, improved tail safety
- **Actual**: EV-VC@100 mean +6.5%, bot-2 +23%. EV-VC@500 +5.9%, bot-2 +23%. 9/12 months improved.
- **Risk**: Low — confirmed
- **Status**: CONFIRMED. L2=5/mcw=25 revalidated in 6/2/34feat config. Effect consistent with prior batch.
- **Note**: Spearman slipped -0.0008 (noise, p>>0.05). Gate blocks promotion due to calibration artifact.

### Iter 1 — Hypothesis B: L2 + Subsampling Reduction ❌ FAILED
- **Config**: reg_lambda=5.0, min_child_weight=25, subsample=0.6, colsample_bytree=0.6
- **Rationale**: Different regularization axis (diversity via randomization) should complement L2 (shrinkage). Unlike lr/trees ensemble smoothing, subsampling doesn't compete with L2.
- **Expected**: +3-8% additional on top of L2 alone, especially on weak months
- **Actual**: Screen showed B improved weak month slightly (+6% 2022-06) but severely degraded strong month (-24% 2022-12). Full v1 benchmark confirmed: Spearman collapsed (-12%), 2 months below tail_floor.
- **Risk**: Medium — confirmed
- **Status**: FAILED. subsample=0.6/colsample=0.6 is too aggressive for ~60k binding samples and 34 features. Signal starvation.
- **Diagnosis**: At subsample=0.6, each tree sees ~36k samples — too few for stable regression on 34 features.

### Iter 2 — Hypothesis A: Depth Reduction (depth=5→4) [PLANNED]
- **Config**: max_depth=4, reg_lambda=5.0, min_child_weight=25
- **Rationale**: depth=5 (v0005) showed slight Spearman slip; depth=3 (prior batch v0004) doubled bot-2 but hurt mean. depth=4 is untested middle ground — 16 leaves vs 32 (depth=5) or 8 (depth=3).
- **Expected**: Spearman recovery +0.001-0.003 (fewer spurious interactions), EV-VC@100 possibly flat or slight loss
- **Risk**: Low-medium — prior data brackets the answer
- **Status**: PLANNED

### Iter 2 — Hypothesis B: L1 Regularization (reg_alpha=1.0) [PLANNED]
- **Config**: reg_alpha=1.0, reg_lambda=5.0, min_child_weight=25
- **Rationale**: L1 penalty forces feature weights to zero, performing implicit feature selection. With 34 features (some possibly noise), L1 could zero out noisy features while preserving signal, improving Spearman.
- **Expected**: Spearman improvement if noisy features are hurting rank correlation; EV-VC potentially unchanged
- **Risk**: Medium — reg_alpha=1.0 is a 10x increase from 0.1, could be too aggressive
- **Status**: PLANNED
