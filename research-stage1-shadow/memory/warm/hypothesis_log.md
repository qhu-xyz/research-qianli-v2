# Hypothesis Log

> Previous hypotheses (H1-H8) archived in memory/archive/feat-eng-20260303-060938/
> See memory/hot/learning.md for distilled learnings from 6 real-data experiments.

## H10: Distribution Shape + Near-Boundary Band + Seasonal Historical Features — CONFIRMED

**Hypothesis**: Adding 7 features from distribution shape, near-boundary bands, and seasonal historical signal will improve NDCG ranking quality while maintaining AUC/AP gains.

**Result**: **CONFIRMED.** NDCG bot2 +0.0101 (margin 0.0046→0.0301). AUC +0.0013, AP +0.0027, BRIER -0.0012, Precision +0.007.

**Key Numbers**:
- AUC: 0.8498 (+0.0013, 8W/4L), AP: 0.4418 (+0.0027, 9W/3L)
- NDCG: 0.7346 (+0.0013, 8W/4L); bot2: 0.6663 (+0.0101)
- VCAP@100: 0.0240 (-0.0007, 4W/8L); bot2: 0.0061 (-0.0033)
- Feature importance: 10.3% combined (prob_band_95_100 #5 at 3.82%, hist_da_max_season #7 at 2.60%)

**What worked**: Near-boundary bands directly improved NDCG by discriminating binding intensity. Bot2 lifted in both worst months simultaneously. prob_band_95_100 became #5 feature overall. Precision improved without recall sacrifice.

**What didn't work**: VCAP@100 regressed (4W/8L) — feature dilution at top-100. Improvement magnitudes modest vs v0007 — diminishing returns from additive feature engineering. 2021-04 remains structurally worst NDCG month (0.651).

**Implication**: NDCG-targeted features work. Additive feature engineering reaching diminishing returns. Future gains require derived interactions or regularization tuning. VCAP@100 dilution needs investigation.

## H9: Shift Factor + Constraint Metadata Features — STRONGLY CONFIRMED

**Hypothesis**: Adding 6 new features from entirely new signal categories (network topology via shift factors + constraint structural metadata) will break the AUC ceiling at ~0.836 because the model is feature-starved, not complexity-starved.

**Result**: **STRONGLY CONFIRMED.** AUC +0.0137 (12W/0L, p≈0.0002), AP +0.0455 (11W/1L, p≈0.006). AUC ceiling broken decisively: 0.8485 vs prior ceiling of ~0.836.

**Key Numbers**:
- AUC: 0.8348 → 0.8485 (+0.0137) — largest single-experiment AUC improvement
- AP: 0.3936 → 0.4391 (+0.0455) — 3x larger than any prior AP delta
- NDCG: 0.7333 → 0.7333 (+0.0000) — flat; 5W/7L; bot2 regressed -0.0154
- BRIER: 0.1503 → 0.1395 (-0.0108) — unexpected improvement
- Feature importance: 4.66% combined gain, but massive generalization impact

**What worked**:
1. Network topology is an entirely orthogonal signal class — confirmed feature-starvation hypothesis
2. Even low-importance features can have high generalization value (auxiliary discriminators)
3. AP bot2 6-experiment decline reversed to +0.0363
4. 2022-09 (structurally broken for 5 interventions) improved: AUC +0.019, AP +0.032

**What didn't work**:
1. NDCG not helped (5W/7L) — shift factors help separate binders from non-binders but may add noise to relative ordering among binders
2. CAP@100/500 degraded ~0.05 — higher threshold reduces predicted positive count
3. VCAP@1000 degraded -0.019 — broader value capture worsened

**Implication**: The model needs more feature diversity, not more model complexity. Continue adding new signal categories. NDCG requires specific attention — may need monotone constraint tuning or NDCG-targeted feature design.
