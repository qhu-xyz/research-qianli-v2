## Status: ITER 1 COMPLETE — NOT PROMOTED (ralph-v2-20260304-031811)
**Champion**: v0 (baseline, 6/2 train/val, 34 features)
**Best candidate**: v0005 (reg_lambda=5, mcw=25) — clear improvement but blocked by Spearman gate calibration artifact

### Iteration 1 Result
- **v0005**: L2 reg (reg_lambda=5, mcw=25) — EV-VC@100 +6.5%, EV-VC@500 +5.9%, C-RMSE -7.2%
- **Screening**: Hypothesis A (L2 only) beat Hypothesis B (L2 + subsample 0.6) decisively. B starved signal.
- **Gate result**: Spearman L1 fails by 0.0008 (floor at v0 exact mean, calibration artifact)
- **Promotion**: BLOCKED — requires gate recalibration (HUMAN_SYNC)
- **Reviewer consensus**: v0005 is genuinely better; Spearman failure is noise, not regression

### Blocking Issue
**Gate calibration**: All Group A floors are set at v0's exact mean, making promotion impossible unless every metric simultaneously improves. v0 itself fails its own L1 gates on 2 metrics. HUMAN_SYNC required to fix.

### Iteration History
| Iter | Version | Hypothesis | EV-VC@100 Δ | Spearman Δ | Promoted | Blocker |
|------|---------|-----------|-------------|-----------|----------|---------|
| 1 | v0005 | L2 reg (λ=5, mcw=25) | +6.5% | -0.2% | No | Spearman L1 (gate calibration) |

### Next: Iteration 2
- **Hypothesis A**: depth=4 + L2 (reduce interaction complexity, possibly recover Spearman)
- **Hypothesis B**: reg_alpha=1.0 + L2 (L1 feature selection, zero out noisy features)
- **Screen months**: 2021-11 (worst Spearman) + 2022-12 (regression canary)
