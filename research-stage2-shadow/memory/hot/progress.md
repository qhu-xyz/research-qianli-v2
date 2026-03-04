## Status: ITER 1 PLANNED — feat-eng-3-20260304-121042
**Champion**: v0009 (39 features, 34 effective) — promoted in prior batch
**Batch constraint**: Feature engineering / selection ONLY (no HP changes)

### Iteration 1 Plan
- **Hypothesis A**: Prune 5 zero-filled features (39→34). Conservative cleanup.
- **Hypothesis B**: Prune 5 zero-filled + add flow_direction (39→35). Cleanup + new signal.
- **Screen months**: 2022-09 (weak, worst EV-VC@500) + 2021-09 (strong, best EV-VC@100/500)
- **Direction file**: `memory/direction_iter1.md` (written)
- **Status**: AWAITING WORKER

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | — | Planned | Zero-fill pruning ± flow_direction |
| 2 | — | — | — |
| 3 | — | — | — |
