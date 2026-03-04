## Status: ITER 1 PLANNED (ralph-v2-20260304-031811)
**Champion**: v0 (baseline, 6/2 train/val, 34 features)
**State**: Orchestrator plan complete, awaiting worker execution

### Current Iteration (1)
- **Hypothesis A**: L2 regularization (reg_lambda=5, mcw=25) — re-validate proven finding in 6/2/34feat config
- **Hypothesis B**: L2 + subsampling (reg_lambda=5, mcw=25, subsample=0.6, colsample=0.6) — test diversity regularization on top of L2
- **Screen months**: 2022-06 (weak) + 2022-12 (strong)
- **Direction**: `memory/direction_iter1.md`

### Iteration History
| Iter | Version | Hypothesis | Result | Status |
|------|---------|-----------|--------|--------|
| 1 | — | L2 vs L2+subsample | — | PLANNED |
