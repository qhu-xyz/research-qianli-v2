# Hypothesis Log

## Batch: ralph-v2-20260304-031811

### Iter 1 — Hypothesis A: L2 Regularization ✅ CONFIRMED
- **Config**: reg_lambda=5.0, min_child_weight=25 (all else default)
- **Rationale**: Proven best lever from prior batch (+11% EV-VC@100 in 10/2/24feat). Re-validate in current 6/2/34feat config.
- **Expected**: +5-15% EV-VC@100 mean, improved tail safety
- **Actual**: EV-VC@100 mean +6.5%, bot-2 +23%. EV-VC@500 +5.9%, bot-2 +23%. 9/12 months improved.
- **Risk**: Low — confirmed
- **Status**: CONFIRMED. L2=5/mcw=25 revalidated in 6/2/34feat config. Effect consistent with prior batch.
- **Note**: Spearman slipped -0.0008 (noise, p>>0.05). Gate blocks promotion due to calibration artifact.

### Iter 1 — Hypothesis B: L2 + Subsampling Reduction ❌ FAILED
- **Config**: reg_lambda=5.0, min_child_weight=25, subsample=0.6, colsample_bytree=0.6
- **Rationale**: Different regularization axis (diversity via randomization) should complement L2 (shrinkage). Unlike lr/trees ensemble smoothing, subsampling doesn't compete with L2.
- **Expected**: +3-8% additional on top of L2 alone, especially on weak months
- **Actual**: Screen showed B improved weak month slightly (+6% 2022-06) but severely degraded strong month (-24% 2022-12). Full v1 benchmark confirmed: Spearman collapsed (-12%), 2 months below tail_floor.
- **Risk**: Medium — confirmed
- **Status**: FAILED. subsample=0.6/colsample=0.6 is too aggressive for ~60k binding samples and 34 features. Signal starvation.
- **Diagnosis**: At subsample=0.6, each tree sees ~36k samples — too few for stable regression on 34 features.

### Iter 2 — Hypothesis A: Depth Reduction (depth=5→4) ❌ FAILED (screen only)
- **Config**: max_depth=4, reg_lambda=5.0, min_child_weight=25
- **Rationale**: depth=5 (v0005) showed slight Spearman slip; depth=3 (prior batch v0004) doubled bot-2 but hurt mean. depth=4 is untested middle ground.
- **Expected**: Spearman recovery +0.001-0.003, EV-VC@100 flat or slight loss
- **Actual (screen)**: Spearman 2021-11=0.2626 (-0.3%), 2022-12=0.3872 (+0.4%). EV-VC@100 2021-11=0.0282 (-42%), 2022-12=0.2001 (+0.7%).
- **Status**: FAILED. Depth=4 catastrophically degrades EV-VC@100 on weak month (-42%) with negligible Spearman effect. Eliminated by override rule before full benchmark.
- **Diagnosis**: 16 leaves (depth=4) can't capture the complex constraint interactions needed for weak months. Depth reduction is not a viable Spearman lever.
- **Key number**: mean Spearman=0.3249 (screen), mean EV-VC@100=0.1142 (screen)

### Iter 2 — Hypothesis B: L1 Regularization (reg_alpha=1.0) ⚠️ INCONCLUSIVE (config bug)
- **Config**: reg_alpha=1.0, reg_lambda=5.0, min_child_weight=25
- **Rationale**: L1 sparsity to zero out noisy features, improving Spearman.
- **Expected**: Spearman improvement from implicit feature selection; EV-VC potentially unchanged
- **Actual (screen)**: Spearman 2021-11=0.2627 (-0.3%), 2022-12=0.3867 (+0.3%). EV-VC@100 2021-11=0.0375 (-23%), 2022-12=0.1932 (-2.8%).
- **Actual (full benchmark)**: ⚠️ INVALID — config.json records reg_alpha=0.1, metrics identical to v0005. Benchmark ran with old default.
- **Status**: INCONCLUSIVE for full-12-month. Screen suggests L1=1.0 has negligible Spearman impact and small negative EV-VC@100 impact. The "all features survive L1 sparsity" conclusion cannot be confirmed from full benchmark but is consistent with screen behavior.
- **Key number**: mean Spearman=0.3247 (screen), mean EV-VC@100=0.1154 (screen)

### Iter 2 — META-FINDING: Regularization Axis Exhausted for Spearman
- **Evidence**: 4 regularization approaches tested across 2 batches: L2 (λ=5), L1 (α=1.0), depth reduction (5→4 and 5→3), subsampling (0.6). None recovered the 0.0008 Spearman gap caused by L2=5/mcw=25.
- **Mechanism**: L2 compression improves top-K value capture (EV-VC) by reducing prediction variance, but slightly hurts rank correlation (Spearman) by compressing the prediction distribution. This is a fundamental trade-off, not noise.
- **Implication**: Iter 3 must explore non-regularization axes (value-weighted training, loss function changes) or accept the trade-off and seek gate recalibration.
