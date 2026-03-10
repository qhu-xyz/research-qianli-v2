# Archive Index

## ralph-v1-20260304-003317
- **Date**: 2026-03-04
- **Iterations**: 3
- **Result**: 0 promotions (infra failure); 2 gate-passing versions on worktree branches (v0003, v0004)
- **Best finding**: L2 regularization (reg_lambda=5, mcw=25) → +11% EV-VC@100 mean
- **Action needed**: Cherry-pick v0003 (commit `01c22af`) to main
- **Summary**: [executive_summary.md](ralph-v1-20260304-003317/executive_summary.md)

## ralph-v2-20260304-031811
- **Date**: 2026-03-04
- **Iterations**: 3
- **Result**: 1 promotion — v0007 (reg_lambda=1.0, mcw=25) promoted as new champion
- **Best finding**: mcw=25 is a pure EV-VC lever (+6.2% EV-VC@500) with no Spearman cost; L2 > 1.0 is anti-Spearman
- **Key issue**: Gate calibration at v0 exact mean blocked 2 of 3 iterations; HUMAN_SYNC urgently needed
- **Summary**: [executive_summary.md](ralph-v2-20260304-031811/executive_summary.md)
