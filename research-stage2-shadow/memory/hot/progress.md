## Status: ITER 3 DIRECTION WRITTEN — FINAL ITERATION (ralph-v2-20260304-031811)
**Champion**: v0 (baseline, 6/2 train/val, 34 features)
**Best candidate**: v0005 (reg_lambda=5, mcw=25) — clear improvement but blocked by Spearman gate calibration artifact

### Iteration 3 Plan (FINAL)
- **Hypothesis A** (primary): value_weighted=True + L2 base — orthogonal lever, weights training by shadow price magnitude
- **Hypothesis B** (alternative): reg_lambda=2.0, mcw=15 — moderate L2 relaxation to recover Spearman while retaining some EV-VC
- **Screen months**: 2021-11 (worst Spearman, 0.2635) + 2022-12 (best EV-VC, 0.1988)
- **Winner criteria**: Primary is Spearman recovery (need ≥ 0.3928 mean), with EV-VC@100 floor protection
- **Direction file**: `memory/direction_iter3.md`

### Blocking Issue (carried from iter 1, reiterated iter 2)
**Gate calibration**: All Group A floors at v0 exact mean. v0005 blocked despite +6.5% EV-VC@100 / +5.9% EV-VC@500. HUMAN_SYNC urgently needed.

### Iteration History
| Iter | Version | Hypothesis | EV-VC@100 Δ | Spearman Δ | Promoted | Blocker |
|------|---------|-----------|-------------|-----------|----------|---------|
| 1 | v0005 | L2 reg (λ=5, mcw=25) | +6.5% | -0.2% | No | Spearman L1 (gate calibration) |
| 2 | v0006 | L1 reg (α=1.0) — CONFIG BUG | +0.0% (= v0005) | +0.0% (= v0005) | No | Config bug + Spearman L1 |
| 3 | TBD | value_weighted or L2 relaxation | TBD | TBD | TBD | TBD |

### Key Constraint for Iter 3
This is the FINAL iteration. Regularization axis is exhausted (L2, L1, depth, subsampling all tested without Spearman recovery). Must explore orthogonal levers: value-weighted training or accept a moderate L2 relaxation that trades some EV-VC for Spearman.
