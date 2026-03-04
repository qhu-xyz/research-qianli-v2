## Status: ITER 2 PLANNED — feat-eng-3-20260304-102111
**Champion**: v0009 (39 features, 34 effective) — promoted iter 1
**Previous champion**: v0007 (34 features, 29 effective)
**Batch constraint**: Feature engineering / selection ONLY (no HP changes)

### Iteration 1 Result
- **Hypothesis**: Add 5 unused raw columns → 39 features (Hypothesis B won screen)
- **Outcome**: PROMOTED. EV-VC@100 +9.0%, EV-VC@500 +1.5%, EV-NDCG +0.5%, Spearman -0.6%
- **Process issue**: v0009 is duplicate of v0008 (byte-identical configs/metrics)
- **Key insight**: Distributional shape features (skewness, kurtosis, CV) contribute signal. season_hist_da_3 and prob_below_85 also selected by screen.

### Iteration 2 Plan
- **Hypothesis A**: Prune 5 zero-filled features (39→34). Conservative cleanup.
- **Hypothesis B**: Prune 5 zero-filled + add flow_direction (39→35). Cleanup + new signal.
- **Screen months**: 2021-11 (weak, worst Spearman) + 2020-11 (strong, best Spearman)
- **Key constraint**: Previous direction_iter2.md violated batch constraint (used value_weighted=True). Corrected to FE-only hypotheses.
- **Direction file**: `memory/direction_iter2.md` (written)
- **Status**: AWAITING WORKER

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | v0009 | **PROMOTED** | EV-VC@100 +9.0% |
| 2 | — | Planned | Zero-fill pruning ± flow_direction |
| 3 | — | — | — |
