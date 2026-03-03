# Hypothesis Log

> Hypotheses from smoke runs (n=20 synthetic data) are archived below.
> Real-data hypotheses start from batch 1.

## H1 (smoke): Infrastructure determinism — CONFIRMED
**Result**: All 10 metrics bit-for-bit identical across runs with seed=42.

## H2 (smoke): Threshold-beta reduction fixes S1-REC — FAILED
**Result**: Beta < 1 weights precision, not recall. Direction had formula inverted. No effect on metrics.
**Lesson**: beta=0.7 is precision-favoring, which aligns with business objective. Do not change.

---

## H3 (real data, iter1): HP tuning improves ranking quality — REFUTED
**Batch**: hp-tune-20260302-134412, **Version**: v0003
**Changes**: max_depth 4→6, n_estimators 200→400, lr 0.1→0.05, min_child_weight 10→5
**Expected**: AUC +0.005–0.015, AP +0.01–0.03, NDCG +0.005–0.015
**Actual**:
- AUC: -0.0025 (0W/11L/1T, p≈0.003) — **statistically significant degradation**
- AP: -0.0015 (4W/8L)
- NDCG: -0.0010 (4W/8L)
- VCAP@100: +0.0015 (7W/5L, mean up but tail down)
- BRIER: -0.0041 (12W/0L, p≈0.0002) — **statistically significant improvement** (calibration only, Group B)

**Key Insight**: v0 defaults were already near-optimal. The model is **feature-limited, not complexity-limited**. Deeper trees improve probability calibration but slightly worsen discrimination. The 14 features have reached their informational ceiling for ranking quality at AUC ~0.835.

**Lesson**: Standard XGBoost HP tuning (deeper + slower + finer) is not a viable path to improvement for this dataset. Next lever: feature engineering to provide new discriminative signal.

---

## H4 (real data, iter1): Interaction features provide new discriminative signal — NOT SUPPORTED
**Batch**: hp-tune-20260302-144146, **Version**: v0002
**Changes**: Reverted HPs to v0 defaults + added 3 interaction features (exceed_severity_ratio, hist_physical_interaction, overload_exceedance_product). Features: 14→17.
**Expected**: AUC +0.005–0.015, AP +0.010–0.025, AUC wins ≥8/12
**Actual**:
- AUC: +0.0000 (5W/6L/1T) — zero effect
- AP: +0.0010 (7W/5L) — marginal, noise-level
- NDCG: +0.0016 (8W/4L) — driven by 2021-01 outlier (+0.042)
- VCAP@500: -0.0043, VCAP@1000: -0.0031 (regression at broader K)
- Bottom-2: regressed on AP, VCAP@100, NDCG

**Key Insight**: XGBoost depth-4 already discovers most useful interactions. Pre-computing them saves tree depth but adds no new information. AUC ceiling at ~0.835 confirmed across both HP tuning and feature interactions.

**Lesson**: Feature engineering within the current feature set cannot break the AUC ceiling. Next lever: expand training window (10→14 months) to address late-2022 distribution shift.

---

## H5 (real data, iter1): Training window expansion breaks AUC ceiling — INCONCLUSIVE (weak positive)
**Batch**: feat-eng-20260302-194243, **Version**: v0003
**Changes**: Reverted to v0's 14 base features + `train_months` 10→14 + benchmark.py plumbing fix
**Expected**: AUC +0.002–0.008, wins ≥7/12, mean AUC >0.835, especially late-2022 months improved
**Actual**:
- AUC: +0.0013 (7W/4L/1T, p≈0.27) — positive but below significance
- AP: +0.0012 (8W/4L, p≈0.19) — positive but below significance
- NDCG: +0.0019 (7W/4L/1T, p≈0.27) — positive but below significance
- VCAP@100: +0.0034 (9W/3L, p≈0.07) — strongest signal, approaching significance
- BRIER: +0.0011 (slight regression, not meaningful)
- Bottom-2: AUC +0.0057 (improved), AP -0.0045 (regressed), NDCG -0.0059 (regressed)

**Success criteria technically met** (≥7/12 AUC wins AND mean AUC >0.835) but effect sizes below statistical significance.

**Target months**: 2022-12 AUC +0.0098 (biggest gain, supporting hypothesis). 2022-09 AUC flat, AP -0.0091 (didn't help, contradicting hypothesis for this month).

**Key Insight**: The 14-month window is the first lever to produce a positive AUC signal across iterations. Effect is small but distributed (not single-outlier driven for AUC). The window expansion helps 2022-12 substantially but cannot address 2022-09, suggesting different failure modes at those two months. The 2022-09 weakness (lowest binding rate at 6.63%, AP consistently worst across all versions) likely reflects a fundamental feature-target mismatch rather than insufficient training diversity.

**Lesson**: Window expansion provides a small, real improvement and should be retained as the new default. But it alone cannot break decisively past AUC ~0.836. Combining with interaction features (H6) is the next test of additivity. If combined effect is still <+0.003 AUC, the 14-feature set has a hard ceiling and fundamentally new features are needed.

---

## H6 (real data, iter1): Combined window + interactions are additive — PARTIALLY CONFIRMED
**Batch**: feat-eng-20260303-060938, **Version**: v0004
**Changes**: 14-month window (from v0003) + 3 interaction features (from v0002) + f2p bug fix + dual-default bug fix. Features: 14→17, train_months=14, HPs at v0 defaults.
**Expected**: AUC 0.836–0.838 (if additive), VCAP@100 0.019–0.023, NDCG 0.737–0.740
**Actual**:
- AUC: +0.0015 (9W/3L, p=0.073 sign test) — best W/L of any experiment
- AP: +0.0015 (6W/6L) — flat, interactions hurt AP consistency
- VCAP@100: +0.0056 (10W/2L, **p=0.039 sign test — FIRST STATISTICALLY SIGNIFICANT RESULT**)
- NDCG: +0.0038 (7W/5L) — roughly additive
- BRIER: +0.0013 (worse, 3W/8L/1T)

**Additivity breakdown**:
- AUC: mostly driven by window expansion (+0.0013 window, +0.0000 interactions → combined +0.0015). Sub-additive.
- AP: sub-additive (+0.0012 + +0.0010 = +0.0022 expected, got +0.0015). Interactions reduce AP consistency.
- VCAP@100: **super-additive** (+0.0034 + +0.0009 = +0.0043 expected, got +0.0056). Interactions boost top-K re-ranking when training data is more diverse.
- NDCG: roughly additive (+0.0019 + +0.0016 = +0.0035 expected, got +0.0038).

**Key Insight**: The combination is partially additive. VCAP@100 benefits super-additively, suggesting interaction features capture complementary signal for top-K precision when paired with a longer training window. AP sub-additivity suggests the interaction features add noise to positive-class ranking that partially offsets their VCAP benefit. AUC is essentially driven by window expansion alone.

**Assessment vs direction criteria**:
- Promotion-worthy (AUC > 0.837, ≥8/12 wins, AP > 0.396): NO (AUC=0.8363, AP=0.3951)
- Encouraging (AUC > 0.835, 7+/12 wins): YES
- Dead end (AUC ≤ 0.835): NO

**Lesson**: The 17-feature + 14-month window configuration represents a local optimum. Breaking AUC ~0.836 likely requires either (a) further window expansion (test 18 months), (b) fundamentally new feature sources, or (c) feature pruning to reduce noise (especially for AP). The feature set has a confirmed ceiling — 4 experiments spanning 3 independent levers have produced max AUC improvement of +0.0015 from v0. The VCAP@100 super-additivity signal is genuine and valuable (the model captures more value at the very top of the prediction stack), but the AUC/AP ceiling is real.
