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

## feat-eng-3-20260304-102111
- **Date**: 2026-03-04
- **Iterations**: 1
- **Result**: 1 promotion — v0009 (39 features, +5 distributional/raw features)
- **Best finding**: Distributional shape features (skewness, kurtosis, CV) carry real signal → EV-VC@100 +9.0%
- **Note**: Single-iteration batch; v0009 byte-identical to v0008 from partial prior run
- **Summary**: (inline in experiment log, no separate archive)

## feat-eng-3-20260304-121042
- **Date**: 2026-03-04
- **Iterations**: 3 (2 successful, 1 worker failure)
- **Result**: 2 promotions — v0011 (prune dead features), v0012 (HP optimization 600t/lr=0.03)
- **Exit champion**: v0012 (34 features, n_est=600, lr=0.03, mcw=25)
- **Best finding**: Feature cleanup + ensemble optimization → eliminated all tail failures, pipeline at healthiest state
- **Untested**: mcw=15 and value_weighted=True (worker failed on iter 3) — carry forward
- **Summary**: [executive_summary.md](feat-eng-3-20260304-121042/executive_summary.md)
