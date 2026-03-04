## Status: ITER 2 PLANNED — AWAITING WORKER (ralph-v2-20260304-031811)
**Champion**: v0 (baseline, 6/2 train/val, 34 features)
**Best candidate**: v0005 (reg_lambda=5, mcw=25) — clear improvement but blocked by Spearman gate calibration artifact

### Iteration 2 Plan
- **Hypothesis A** (primary): depth=5→4 + L2 base — reduce interaction complexity to recover Spearman
- **Hypothesis B** (alternative): reg_alpha=0.1→1.0 + L2 base — L1 feature selection to improve rank correlation
- **Screen months**: 2021-11 (worst Spearman, 0.2635) + 2022-12 (best EV-VC@100, 0.1988)
- **Winner criteria**: Higher mean Spearman across screen months, with EV-VC@100 regression protection
- **Direction file**: `memory/direction_iter2.md`

### Blocking Issue (carried from iter 1)
**Gate calibration**: All Group A floors are set at v0's exact mean, making promotion impossible unless every metric simultaneously improves. v0 itself fails its own L1 gates on 2 metrics. HUMAN_SYNC required to fix.

### Iteration History
| Iter | Version | Hypothesis | EV-VC@100 Δ | Spearman Δ | Promoted | Blocker |
|------|---------|-----------|-------------|-----------|----------|---------|
| 1 | v0005 | L2 reg (λ=5, mcw=25) | +6.5% | -0.2% | No | Spearman L1 (gate calibration) |
| 2 | TBD | depth=4 or reg_alpha=1.0 | TBD | TBD | TBD | TBD |

### Next Steps
1. Worker screens both hypotheses on 2021-11 and 2022-12
2. Worker picks winner per criteria, implements code changes, runs full 12-month benchmark
3. If promoted: update champion. If not: analyze for iter 3 direction.
