## Status: ITER 3 PLANNING COMPLETE — FINAL ITERATION (ralph-v2-20260304-031811)
**Champion**: v0 (baseline, 6/2 train/val, 34 features)
**Best candidate**: v0005 (reg_lambda=5, mcw=25) — clear improvement but blocked by Spearman gate calibration artifact

### Iteration 3 Plan (FINAL)
- **Hypothesis A** (primary): L2-Only decomposition — reg_lambda=3.0, mcw=10. Isolates L2 effect by reverting mcw to v0's default.
- **Hypothesis B** (alternative): MCW-Only decomposition — reg_lambda=1.0, mcw=25. Isolates mcw effect by reverting L2 to v0's default.
- **Screen months**: 2021-11 (worst Spearman, 0.2635) + 2022-12 (best EV-VC, 0.1988)
- **Winner criteria**: Higher mean Spearman across screen months, with EV-VC@100 < 0.065 override protection
- **Strategy**: Decompose v0005's 2-parameter change (L2=5+mcw=25) to find which parameter caused the Spearman compression. The answer tells us the minimum config for promotion.
- **Direction file**: `memory/direction_iter3.md`

### Why Not value_weighted?
value_weighted was identified as the top orthogonal lever, but it's NOT WIRED in pipeline.py. Testing it requires modifying pipeline.py, which is out of scope. Decomposition is the best available strategy.

### Blocking Issue (carried from iter 1, reiterated iter 2)
**Gate calibration**: All Group A floors at v0 exact mean. v0005 blocked despite +6.5% EV-VC@100 / +5.9% EV-VC@500. HUMAN_SYNC urgently needed.

### Iteration History
| Iter | Version | Hypothesis | EV-VC@100 Δ | Spearman Δ | Promoted | Blocker |
|------|---------|-----------|-------------|-----------|----------|---------|
| 1 | v0005 | L2 reg (λ=5, mcw=25) | +6.5% | -0.2% | No | Spearman L1 (gate calibration) |
| 2 | v0006 | L1 reg (α=1.0) — CONFIG BUG | +0.0% (= v0005) | +0.0% (= v0005) | No | Config bug + Spearman L1 |
| 3 | TBD | L2/mcw decomposition | TBD | TBD | TBD | TBD |

### Key Constraint for Iter 3
This is the FINAL iteration. The decomposition approach answers: "which of the 2 parameters in v0005 (L2 or mcw) caused the 0.0008 Spearman drop?" The answer determines the minimum config that can pass ALL gates.
