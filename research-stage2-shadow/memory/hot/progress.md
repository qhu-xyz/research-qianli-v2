## Status: ITER 1 COMPLETE, PROMOTED v0011 — feat-eng-3-20260304-121042
**Champion**: v0011 (34 features, pruned from v0009's 39 nominal)
**Batch started as**: Feature engineering / selection ONLY
**Batch constraint relaxed for iter 2**: HP changes now allowed

### Iteration 1 Result
- **Hypothesis A (winner)**: Prune 5 dead features (39→34). Selected over Hyp B (prune + flow_direction).
- **Screen months**: 2022-09 (weak) + 2021-09 (strong)
- **v0011 promoted**: EV-VC@100 +5.2%, Spearman +0.4%. EV-VC@500 -2.5% (precision-vs-breadth tradeoff accepted).
- **Key risk**: EV-VC@500 gate margins very thin (L2 at limit, L3 margin +0.0023)

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | v0011 | **PROMOTED** | EV-VC@100 +5.2%, EV-VC@500 -2.5% (prune dead features) |
| 2 | — | Planned | HP tuning to recover EV-VC@500 breadth |
| 3 | — | — | — |
