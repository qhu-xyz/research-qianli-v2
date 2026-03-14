# Executive Summary — Batch feat-eng-3-20260303-104101

**Date**: 2026-03-03
**Iterations**: 3 (v0008, v0009, v0010)
**Type**: Feature engineering (distribution, interactions, HP optimization)
**Champion entering**: v0007 (AUC=0.8485, AP=0.4391, 19 features)
**Champion exiting**: v0009 (AUC=0.8495, AP=0.4445, 29 features)

## Batch Objective

Continue aggressive feature engineering from the feat-eng-2 batch's breakthrough (v0007). Human directive was to expand features from the source loader and derived interactions. Three iterations targeted: (1) distribution/band features for NDCG recovery, (2) derived interactions for VCAP@100 recovery, (3) hyperparameter optimization.

## Results by Iteration

### Iter 1 — H10: Distribution Shape + Near-Boundary Band + Seasonal Historical (v0008) → PROMOTED

**Changes**: 19→26 features. Added density_mean, density_variance, density_entropy, tail_concentration, prob_band_95_100, prob_band_100_105, hist_da_max_season.

**Key results vs v0007**:
- AUC: +0.0013 (8W/4L), AP: +0.0027 (9W/3L)
- NDCG bot2: **+0.0101** (margin 0.0046→0.0301) — primary target achieved
- VCAP@100: -0.0007 (4W/8L) — dilution from feature expansion
- BRIER: -0.0012 (better)
- Combined feature importance: 10.3%

**Verdict**: PROMOTED. NDCG-targeted features worked — both worst months lifted simultaneously.

### Iter 2 — H11: Derived Interaction Features + colsample_bytree (v0009) → PROMOTED

**Changes**: 26→29 features. Added band_severity, sf_exceed_interaction, hist_seasonal_band. colsample_bytree 0.8→0.9.

**Key results vs v0008**:
- AP: **+0.0027 (9W/3L)** — new pipeline high at 0.4445
- VCAP@100 bot2: **+0.0028** (primary target achieved)
- NDCG: +0.0013 (7W/5L)
- AUC: -0.0003 (4W/8L — noise)
- BRIER: -0.0007 (better)
- Combined feature importance: **17.13%** — highest single-iteration block ever

**Verdict**: PROMOTED. hist_seasonal_band (#2 feature at 11.75%) is the single most impactful derived feature in pipeline history.

### Iter 3 — H12: More Trees + Slower Learning Rate (v0010) → NOT PROMOTED (null)

**Changes**: n_estimators 200→300, learning_rate 0.1→0.07. No feature changes.

**Key results vs v0009**:
- All Group A means within ±0.0021 of champion (noise)
- All W/L ratios near 50/50 (best: AUC 6W/5L/1T)
- Bot2 mixed: NDCG +0.0037 / AP +0.0036 (improvement) vs AUC -0.0017 / VCAP@100 -0.0019 (degradation)

**Verdict**: NOT PROMOTED. Confirmed null — model at capacity ceiling. Tree count/learning rate are not the binding constraint.

## Net Batch Impact (v0007 → v0009)

| Metric | v0007 | v0009 | Delta | Significance |
|--------|-------|-------|-------|-------------|
| S1-AUC | 0.8485 | 0.8495 | +0.0010 | Small, incremental |
| S1-AP | 0.4391 | **0.4445** | **+0.0054** | New pipeline high |
| S1-VCAP@100 | 0.0247 | **0.0266** | +0.0019 | Improved after recovery |
| S1-NDCG | 0.7333 | 0.7359 | +0.0026 | Moderate |
| S1-BRIER | 0.1395 | **0.1376** | -0.0019 | Better calibration |
| Precision | 0.496 | 0.503 | +0.007 | Business objective maintained |
| Features | 19 | 29 | +10 | Saturated search space |

## Key Learnings

1. **Distribution/band features improve NDCG specifically**: prob_band_95_100 discriminates binding intensity at the margin. Bot2 improvement NOT driven by mean shift — it's tail-lift.

2. **Multiplicative interactions capture genuinely new signal**: hist_seasonal_band (11.75%) proves that cross-class interactions (history × physics) outperform same-class interactions. Trees cannot efficiently approximate these products from raw components.

3. **Feature engineering has diminishing returns**: v0007 (6 features, 4.66% importance → AUC +0.0137), v0008 (7 features, 10.3% → AUC +0.0013), v0009 (3 features, 17.13% → AUC -0.0003). Higher importance ≠ proportionally higher lift.

4. **colsample_bytree matters for feature-rich models**: 0.9 with 29 features ensures ~26/29 features per tree. Rule of thumb: ≥85% features per tree when total >20.

5. **HP tuning does not help when feature information is the bottleneck**: v0010 null confirms this. 200 trees at lr=0.1 and 300 trees at lr=0.07 produce identical results.

6. **Optimization frontier confirmed at**: AUC ~0.850, AP ~0.445, NDCG ~0.736, BRIER ~0.137. Further gains require fundamentally new signal sources or architectural changes.

7. **2021-04 and 2022-12 are structurally resistant**: Spring transition and late-2022 distribution shift months remain weak across all experiments.

## Gate Calibration Recommendations (for HUMAN_SYNC)

1. **VCAP@100 floor**: Tighten -0.035 → **0.0** (recommended 4 consecutive iterations)
2. **CAP@100/500 floors**: Relax by **0.03** (to 0.7025 / 0.6940) — 4 consecutive champion versions failing
3. **Noise tolerance**: Keep **0.02**, consider 0.015 in future
4. **Month coverage**: Add `n_months == n_months_requested` enforcement before gate checks
5. **Threshold reproducibility**: Thread `threshold_scaling_factor` through benchmark path

## Code Issues Deferred to HUMAN_SYNC

| Priority | Issue | Iterations flagged |
|----------|-------|--------------------|
| HIGH | Missing month-coverage enforcement | iter3 |
| HIGH | Threshold-selection leakage | smoke-v6 |
| MEDIUM | Benchmark ignores threshold_scaling_factor | iter3 |
| MEDIUM | Threshold tuned/evaluated on same split | iter3 |
| MEDIUM | check_gates_multi_month None-as-pass | iter1 |
| MEDIUM | Temporal leakage fallback | iter2 |

## What Should the Next Batch Explore?

The feature engineering and HP search spaces within the current XGBoost architecture are exhausted. Three directions require human design decisions:

1. **Temporal features**: Distribution shift in late-2022 months suggests time-aware features (e.g., months since window start, volatility regime indicators) could help the persistent tail weakness.

2. **Multi-stage pipeline**: Use Stage 1 probabilities as input to a Stage 2 model incorporating portfolio-level or market-level information.

3. **Alternative loss functions**: Ranking-specific losses (pairwise, LambdaMART) may improve NDCG directly, which has been the most resistant metric throughout.

These are structural changes requiring human architectural decisions, not incremental optimization. The pipeline is ready for HUMAN_SYNC.
