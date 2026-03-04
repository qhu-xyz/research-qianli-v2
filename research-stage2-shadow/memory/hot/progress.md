## Status: BATCH COMPLETE — ralph-v2-20260304-031811
**Champion**: v0007 (reg_lambda=1.0, mcw=25) — promoted from iter 3
**Previous champion**: v0 (baseline)

### Batch Summary
| Iter | Version | Hypothesis | EV-VC@100 Δ | EV-VC@500 Δ | Spearman Δ | Promoted | Blocker |
|------|---------|-----------|-------------|-------------|-----------|----------|---------|
| 1 | v0005 | L2 reg (λ=5, mcw=25) | +6.5% | +5.9% | -0.2% | No | Spearman L1 (gate calibration) |
| 2 | v0006 | L1 reg (α=1.0) — CONFIG BUG | = v0005 | = v0005 | = v0005 | No | Config bug + Spearman L1 |
| 3 | v0007 | MCW-Only (λ=1.0, mcw=25) | +1.3% | +6.2% | +0.1% | **YES** | — |

### Key Findings
1. **mcw=25 is a pure EV-VC lever** — improves value capture without affecting Spearman
2. **reg_lambda > 1.0 compresses Spearman** — L2 helps EV-VC but hurts rank correlation
3. **Optimal point found**: reg_lambda=1.0, mcw=25
4. **Gate calibration blocked 2 of 3 iterations** — HUMAN_SYNC urgently needed

### Next Batch Priorities
1. Gate recalibration (HUMAN_SYNC)
2. Wire value_weighted through pipeline.py
3. mcw fine-tuning sweep (15-35 range)
4. Feature importance pipeline
5. Tail robustness investigation (2021-11, 2022-06)
