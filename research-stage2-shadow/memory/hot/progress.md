## Status: ITER 2 PLANNED — feat-eng-3-20260304-121042
**Champion**: v0011 (34 features, pruned from v0009's 39 nominal)
**Batch started as**: Feature engineering / selection ONLY
**Batch constraint relaxed for iter 2**: HP changes now allowed

### Iteration 2 Plan
- **Hypothesis A (primary)**: n_estimators=600, lr=0.03 — more trees + lower LR for mid-tier discrimination
- **Hypothesis B (alternative)**: colsample_bytree=0.9, n_estimators=500, lr=0.04 — restore per-tree feature coverage
- **Screen months**: 2022-09 (weak, sole EV-VC@500 tail failure) + 2022-12 (strong, fresh)
- **Objective**: Recover EV-VC@500 breadth without losing EV-VC@100 gains

### Iteration 1 Result
- **Hypothesis A (winner)**: Prune 5 dead features (39→34). Selected over Hyp B (prune + flow_direction).
- **Screen months**: 2022-09 (weak) + 2021-09 (strong)
- **v0011 promoted**: EV-VC@100 +5.2%, Spearman +0.4%. EV-VC@500 -2.5% (precision-vs-breadth tradeoff accepted).
- **Key risk**: EV-VC@500 gate margins very thin (L2 at limit, L3 margin +0.0023)

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | v0011 | **PROMOTED** | EV-VC@100 +5.2%, EV-VC@500 -2.5% (prune dead features) |
| 2 | — | **PLANNED** | HP tuning to recover EV-VC@500 breadth |
| 3 | — | — | — |
