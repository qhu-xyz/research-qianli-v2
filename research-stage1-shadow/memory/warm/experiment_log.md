# Experiment Log

> **NOTE**: Experiments below smoke-v6 and smoke-v7 used SMOKE_TEST data (n=20).
> Gate floors have since been recalibrated from real v0 (12 months, ~270K rows/month).
> See memory/hot/gate_calibration.md for current floors.

## Iteration 1 — v0001 (Infrastructure Validation)

**Batch**: smoke-v6-20260227-190225
**Date**: 2026-02-28
**Hypothesis**: Running pipeline with identical v0 config produces identical metrics (determinism check)
**Changes**: None — exact v0 config replay
**Result**: PASS — all 10 metrics bit-for-bit identical to v0 (zero delta)

| Gate | v0 | v0001 | Delta | Status |
|------|-----|-------|-------|--------|
| S1-AUC | 0.7500 | 0.7500 | 0.0 | PASS |
| S1-AP | 0.5909 | 0.5909 | 0.0 | PASS |
| S1-VCAP@100 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-VCAP@500 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-VCAP@1000 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-NDCG | 0.5044 | 0.5044 | 0.0 | PASS |
| S1-BRIER | 0.2021 | 0.2021 | 0.0 | PASS |
| S1-REC (B) | 0.0000 | 0.0000 | 0.0 | FAIL |
| S1-CAP@100 (B) | 0.0000 | 0.0000 | 0.0 | PASS |
| S1-CAP@500 (B) | 0.0000 | 0.0000 | 0.0 | PASS |

**Overall Pass**: NO (S1-REC fails Group B floor)
**Promoted**: No
**Key Learnings**: Pipeline is fully deterministic (seed=42 for data + XGBoost). Infrastructure components all work correctly. 66/66 tests pass.
**Code Issues Found**: from_phase broken (Codex HIGH), threshold leakage on same split (Codex HIGH), Group B policy not enforced (both reviewers), gzip gap in pipeline.py (Claude minor), dead config scale_pos_weight_auto (Codex MEDIUM), version allocator not wired (Codex MEDIUM)

---

## Iteration 1 — v0001 (H2: threshold_beta=0.3 + bug fixes)

**Batch**: smoke-v7-20260227-191851
**Date**: 2026-02-28
**Hypothesis**: H2 — Lowering threshold_beta from 0.7 to 0.3 will weight recall more in F-beta, producing positive predictions and fixing S1-REC.
**Changes**: (1) from_phase guard, (2) Group B pass policy fix, (3) model gzip, (4) pipeline run with threshold_beta=0.3
**Result**: **HYPOTHESIS FAILED** — All metrics bit-for-bit identical to v0. Threshold remained at 0.8203, pred_binding_rate=0.0.

**Root cause**: The direction specification had the F-beta formula backwards. Beta < 1 weights precision more (not recall). Beta=0.3 made optimization MORE precision-oriented. Need beta > 1 to favor recall.

| Gate | v0 | v0001 | Delta | Status |
|------|-----|-------|-------|--------|
| S1-AUC | 0.7500 | 0.7500 | 0.0 | PASS |
| S1-AP | 0.5909 | 0.5909 | 0.0 | PASS |
| S1-VCAP@100 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-VCAP@500 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-VCAP@1000 | 1.0000 | 1.0000 | 0.0 | PASS |
| S1-NDCG | 0.5044 | 0.5044 | 0.0 | PASS |
| S1-BRIER | 0.2021 | 0.2021 | 0.0 | PASS |
| S1-REC (B) | 0.0000 | 0.0000 | 0.0 | FAIL |
| S1-CAP@100 (B) | 0.0000 | 0.0000 | 0.0 | PASS |
| S1-CAP@500 (B) | 0.0000 | 0.0000 | 0.0 | PASS |

**Overall Pass**: YES (B:NO) — Group A all pass, Group B S1-REC still fails.
**Promoted**: No
**Bug fixes validated**: from_phase guard works, Group B policy correctly separates groups, model gzip functional. 70/70 tests pass (4 new).
**Key Learnings**:
- F-beta formula: beta < 1 → precision; beta > 1 → recall. Direction had this inverted.
- Worker correctly diagnosed the error post-hoc in changes_summary.md.
- Codex found new HIGH issue: threshold `>` vs `>=` mismatch (PR curve inclusive, apply_threshold exclusive) may suppress positives.
- All code changes clean, no regressions.
**Open Issues**: threshold `>` vs `>=` mismatch (Codex HIGH, new), threshold leakage (Codex HIGH, carried), dead config scale_pos_weight_auto (MEDIUM, carried), version allocator (MEDIUM, carried), misleading beta docstring in threshold.py (Claude LOW, new), DRY opportunity in compare.py (Claude LOW, new), missing-metric pass disagreement between table/JSON (Codex MEDIUM, new)

---

## Iteration 1 — v0003 (H3: Hyperparameter tuning — first real-data experiment)

**Batch**: hp-tune-20260302-134412
**Date**: 2026-03-02
**Hypothesis**: H3 — Deeper trees (max_depth=6), slower learning (lr=0.05), more trees (n_estimators=400), finer splits (min_child_weight=5) improve ranking quality over v0 defaults.
**Changes**: 4 hyperparameter adjustments in `ml/config.py` → `HyperparamConfig`. No feature, threshold, or pipeline changes.
**Result**: **HYPOTHESIS REFUTED** — All Group A ranking metrics regressed slightly. AUC degradation statistically significant (0W/11L/1T, p≈0.003).

| Gate | v0 Mean | v0003 Mean | Delta | v0 Bot2 | v0003 Bot2 | Δ Bot2 | Pass |
|------|---------|------------|-------|---------|------------|--------|------|
| S1-AUC | 0.8348 | 0.8323 | **-0.0025** | 0.8105 | 0.8089 | -0.0016 | YES |
| S1-AP | 0.3936 | 0.3921 | **-0.0015** | 0.3322 | 0.3299 | -0.0023 | YES |
| S1-VCAP@100 | 0.0149 | 0.0164 | +0.0015 | 0.0014 | 0.0007 | -0.0007 | YES |
| S1-NDCG | 0.7333 | 0.7323 | **-0.0010** | 0.6716 | 0.6675 | -0.0041 | YES |
| S1-BRIER | 0.1503 | 0.1462 | **-0.0041** ✓ | — | — | — | YES |

**Overall Pass**: YES (all 3 layers pass for all Group A gates)
**Promoted**: No — no Group A improvement over v0
**Per-month consistency**: AUC worse in 11/12 months, AP worse in 8/12, NDCG worse in 8/12. BRIER improved in 12/12 months.
**Weakest months unchanged**: 2022-09 (AP=0.309), 2022-12 (AUC=0.809), 2021-04 (NDCG=0.662)
**Key Learnings**:
- v0 HP defaults were already near-optimal. Standard "deeper + slower" XGBoost tuning did not improve ranking quality.
- Model is **feature-limited, not complexity-limited** — deeper trees can't extract more signal from these 14 features.
- Deeper trees DO improve calibration (BRIER 12/12 months better) but hurt discrimination (AUC worse).
- Late-2022 distribution shift not addressable via tree complexity.
**Code Issues**: Clean implementation, no new bugs. Carried: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), dead config (LOW), Layer 3 disabled when champion=null (MEDIUM).

---

## Iteration 1 — v0002 (H4: Interaction features — second real-data experiment)

**Batch**: hp-tune-20260302-144146
**Date**: 2026-03-02
**Hypothesis**: H4 — Pre-computed interaction features (3 new) provide discriminative signal that depth-4 trees cannot efficiently discover through single-feature splits.
**Changes**: (1) Reverted HPs to v0 defaults (isolating feature effect from v0003), (2) Added 3 interaction features: exceed_severity_ratio, hist_physical_interaction, overload_exceedance_product. Total features: 14→17. All monotone +1.
**Result**: **HYPOTHESIS NOT SUPPORTED** — AUC unchanged, ranking metrics marginally positive but within noise.

| Gate | v0 Mean | v0002 Mean | Delta | v0 Bot2 | v0002 Bot2 | Δ Bot2 | Pass |
|------|---------|------------|-------|---------|------------|--------|------|
| S1-AUC | 0.8348 | 0.8348 | +0.0000 | 0.8105 | 0.8105 | +0.0000 | YES |
| S1-AP | 0.3936 | 0.3946 | +0.0010 | 0.3322 | 0.3305 | -0.0017 | YES |
| S1-VCAP@100 | 0.0149 | 0.0158 | +0.0009 | 0.0014 | 0.0006 | -0.0008 | YES |
| S1-NDCG | 0.7333 | 0.7349 | +0.0016 | 0.6716 | 0.6703 | -0.0013 | YES |
| S1-BRIER | 0.1503 | 0.1505 | +0.0002 | — | — | — | YES |

**Overall Pass**: YES (all 3 layers pass for all Group A gates)
**Promoted**: No — no meaningful improvement over v0
**Per-month consistency**: AUC 5W/6L/1T, AP 7W/5L, NDCG 8W/4L
**Weakest months unchanged**: 2022-09 (AP=0.314), 2022-12 (AUC=0.809, VCAP@100=0.000), 2021-04 (NDCG=0.663)
**Notable**: VCAP@500 -0.0043 and VCAP@1000 -0.0031 (interactions help top-100 but hurt broader ranking). 2021-01 NDCG outlier (+0.042) drives most of the mean improvement.
**Key Learnings**:
- AUC ceiling at ~0.835 is confirmed across two independent levers: HP tuning (v0003) and interaction features (v0002)
- XGBoost depth-4 can already discover most useful feature interactions — pre-computing them saves tree depth but doesn't unlock new patterns
- Interaction features help ranking (NDCG 8W/4L) more than discrimination (AUC 5W/6L) — consistent with improving ordering of already-positive cases, not improving the separation boundary
- Early months (2020–2021H1) benefit more than late months (2022) — distribution shift is the real bottleneck
- Bottom-2 regressed on 3/4 Group A metrics: gains concentrated in middle of distribution, not tails
**Code Issues**: No new bugs introduced. New MEDIUM: missing schema guard for interaction feature base columns (Codex). Stale docstrings (14→17 mismatch, both reviewers). Carried: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), dead config (LOW), Layer 3 disabled (MEDIUM).

---

## Iteration 1 — v0003 (H5: Training window expansion 10→14 months — third real-data experiment)

**Batch**: feat-eng-20260302-194243
**Date**: 2026-03-02
**Hypothesis**: H5 — Expanding training window from 10 to 14 months addresses late-2022 distribution shift by providing 40% more training examples with greater seasonal diversity.
**Changes**: (1) Reverted to v0's 14 base features (removed 3 interaction features from v0002), (2) Changed `train_months` 10→14 in PipelineConfig, (3) Fixed benchmark.py train_months/val_months plumbing (bug fix — params weren't threaded through), (4) Added schema guard for feature columns in features.py, (5) Updated docstrings and tests.
**Result**: **SMALL POSITIVE BUT NOT SIGNIFICANT** — Directionally best result of 3 real-data iterations, but effect sizes below statistical significance.

| Gate | v0 Mean | v0003 Mean | Delta | W/L/T | v0 Bot2 | v0003 Bot2 | Δ Bot2 | Pass |
|------|---------|------------|-------|-------|---------|------------|--------|------|
| S1-AUC | 0.8348 | 0.8361 | **+0.0013** | 7W/4L/1T | 0.8105 | 0.8162 | +0.0057 | YES |
| S1-AP | 0.3936 | 0.3948 | **+0.0012** | 8W/4L | 0.3322 | 0.3277 | -0.0045 | YES |
| S1-VCAP@100 | 0.0149 | 0.0183 | **+0.0034** | 9W/3L | 0.0014 | 0.0016 | +0.0002 | YES |
| S1-NDCG | 0.7333 | 0.7352 | **+0.0019** | 7W/4L/1T | 0.6716 | 0.6657 | -0.0059 | YES |
| S1-BRIER | 0.1503 | 0.1514 | +0.0011 | — | — | — | — | YES |

**Overall Pass**: YES (all 3 layers pass for all Group A gates)
**Promoted**: No — improvements directionally positive but not statistically significant
**Per-month consistency**: AUC 7W/4L/1T, AP 8W/4L, NDCG 7W/4L/1T, VCAP@100 9W/3L
**Success criteria**: Technically met (≥7/12 AUC wins, mean AUC >0.835) but effect too small to justify promotion
**Best month**: 2022-12 (AUC +0.0098, AP +0.0142) — the weakest month improved most, supporting seasonal diversity hypothesis
**Target month 2022-09**: AUC flat (+0.0000), AP regressed (-0.0091) — distribution shift there requires different approach
**Statistical significance**: AUC δ=0.087σ (p≈0.27), AP δ=0.029σ (p≈0.19), NDCG δ=0.046σ (p≈0.27), VCAP@100 δ=0.27σ (p≈0.07). None significant at p<0.05.
**Group B notable**: VCAP@500 -0.0063, CAP@100 -0.0117, CAP@500 -0.0107 (broader ranking degraded while top-100 improved).
**Key Learnings**:
- 14-month window is the first lever to produce a positive AUC signal (7W, up from 5W interactions and 0W HP tuning)
- Effect is small but distributed (not driven by a single outlier month)
- 2022-12 benefited most (+0.0098 AUC) but 2022-09 unchanged — distribution shift at 2022-09 is not addressable by window expansion alone
- Broader ranking (CAP, VCAP@500) traded for top-100 improvement — consistent with business objective
- Bottom-2 mixed: AUC tail improved, AP/NDCG tails slightly regressed (within 0.02 tolerance)
**Code Issues**: Dual-default fragility in benchmark.py — train_months=14 hardcoded in function signatures AND PipelineConfig (Claude MEDIUM). f2p parsing crash: `int("2p")` fails for cascade stage-3 (Codex HIGH, new). Test fixture 17-wide vs 14 features (Codex LOW). Carried: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), dead config (LOW), Layer 3 disabled (MEDIUM).

---

## Iteration 1 — v0004 (H6: Combined 14-month window + interaction features — fourth real-data experiment)

**Batch**: feat-eng-20260303-060938
**Date**: 2026-03-03
**Hypothesis**: H6 — Are the two positive-signal levers (14-month window from v0003, 3 interaction features from v0002) additive? Expected AUC ~0.836–0.838 if additive.
**Changes**: (1) Added 3 interaction features back (exceed_severity_ratio, hist_physical_interaction, overload_exceedance_product — features 14→17), (2) Kept train_months=14 from v0003, (3) Fixed f2p parsing crash (regex-based horizon extraction), (4) Fixed dual-default fragility (None sentinel with PipelineConfig fallback), (5) Updated tests for 17 features.
**Result**: **PARTIALLY ADDITIVE — Encouraging but not promotion-worthy**

| Gate | v0 Mean | v0004 Mean | Delta | W/L/T | v0 Bot2 | v0004 Bot2 | Δ Bot2 | Pass |
|------|---------|------------|-------|-------|---------|------------|--------|------|
| S1-AUC | 0.8348 | 0.8363 | **+0.0015** | **9W/3L** | 0.8105 | 0.8164 | +0.0059 | YES |
| S1-AP | 0.3936 | 0.3951 | **+0.0015** | 6W/6L | 0.3322 | 0.3282 | -0.0040 | YES |
| S1-VCAP@100 | 0.0149 | 0.0205 | **+0.0056** | **10W/2L** | 0.0014 | 0.0011 | -0.0003 | YES |
| S1-NDCG | 0.7333 | 0.7371 | **+0.0038** | 7W/5L | 0.6716 | 0.6656 | -0.0060 | YES |
| S1-BRIER | 0.1503 | 0.1516 | +0.0013 | 3W/8L/1T | — | — | — | YES |

**Overall Pass**: YES (all 3 layers pass for all Group A gates)
**Promoted**: No — falls in "encouraging" band (AUC=0.8363 < 0.837 threshold, AP 6W/6L)
**Per-month consistency**: AUC 9W/3L (best W/L of any experiment), VCAP@100 10W/2L (first stat-sig improvement, sign test p=0.039), AP 6W/6L (flat), NDCG 7W/5L

**Additivity assessment**:
| Metric | Window (v0003) | Interactions (v0002) | Combined (v0004) | Sum | Assessment |
|--------|----------------|----------------------|-------------------|-----|------------|
| AUC Δ | +0.0013 | +0.0000 | +0.0015 | +0.0013 | Mostly window |
| AP Δ | +0.0012 | +0.0010 | +0.0015 | +0.0022 | Sub-additive |
| VCAP@100 Δ | +0.0034 | +0.0009 | +0.0056 | +0.0043 | **Super-additive** |
| NDCG Δ | +0.0019 | +0.0016 | +0.0038 | +0.0035 | ~Additive |

**Statistical significance**:
- VCAP@100 10W/2L: sign test p=0.039 — **first statistically significant improvement in pipeline history**
- AUC 9W/3L: sign test p=0.073 — approaching significance
- NDCG 7W/5L: p=0.39 — not significant
- AP 6W/6L: p=1.0 — no signal

**Per-month highlights**:
- Best month: 2022-12 AUC +0.0098 (consistent with v0003-window, interaction features added nothing here)
- Worst month: 2022-09 AP=0.3072 (4th consecutive failure across all levers, binding rate 6.63%)
- No tail_floor breaches

**Group B notable**:
- VCAP@500: -0.0065 (3rd consecutive regression, bot2=0.0387 approaching floor 0.0408)
- BRIER: 0.1516 (headroom 0.0187, 4th consecutive narrowing)
- CAP@100: +0.0025, CAP@500: +0.0010 (slight recovery from v0003's regression)

**Key Learnings**:
- Combining levers is partially additive — VCAP@100 benefits most from the combination (super-additive)
- AP is the one Group A metric that doesn't respond to any lever tested so far (6W/6L when combined)
- The 17-feature set + 14-month window defines a ceiling of ~AUC 0.836, NDCG 0.737
- Top-100 precision vs broader ranking is a consistent tradeoff across all experiments
- 2022-09 has resisted 4 independent interventions — likely structural

**Code Issues**: Both bug fixes correct (f2p regex, dual-default sentinel). No new issues introduced. Codex flagged silent fallback in ptype parser (MEDIUM). Carried: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), missing schema guard for interaction base columns (MEDIUM).

---

## Iteration 2 — v0005 (H7: 18-month window + feature importance diagnostic — fifth real-data experiment)

**Batch**: feat-eng-20260303-060938
**Date**: 2026-03-03
**Hypothesis**: H7 — Does further window expansion (14→18 months) continue the positive AUC trend, and which features actually drive predictions?
**Changes**: (1) `train_months` 14→18 in PipelineConfig, (2) Added gain-based feature importance extraction to benchmark.py (captures per-month importance, saves to feature_importance.json), (3) Updated test assertion for train_months.
**Result**: **DIMINISHING RETURNS CONFIRMED** — v0005 is marginally worse than v0004 on every Group A metric mean. Feature importance diagnostic successfully collected.

| Gate | v0 Mean | v0004 Mean | v0005 Mean | Δ vs v0 | Δ vs v0004 | v0 Bot2 | v0005 Bot2 | Δ Bot2 vs v0 | Pass |
|------|---------|------------|------------|---------|------------|---------|------------|--------------|------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | +0.0013 | -0.0002 | 0.8105 | 0.8156 | +0.0051 | YES |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | -0.0007 | -0.0023 | 0.3322 | 0.3247 | **-0.0075** | YES |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | +0.0044 | -0.0012 | 0.0014 | 0.0024 | +0.0010 | YES |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | +0.0032 | -0.0007 | 0.6716 | 0.6699 | -0.0017 | YES |
| S1-BRIER | 0.1503 | 0.1516 | 0.1525 | +0.0022 | +0.0009 | — | 0.1605 | — | YES |

**Overall Pass**: YES (all 3 layers pass for all Group A gates)
**Promoted**: No — strictly worse than v0004 on all Group A means
**Per-month consistency vs v0004**: AUC 7W/5L (noise), AP 6W/6L (noise), VCAP@100 6W/6L (noise), NDCG 7W/5L (noise)
**Per-month consistency vs v0**: AUC 7W/5L, VCAP@100 10W/2L, NDCG 8W/4L, AP 6W/6L
**Weakest months**: 2022-09 AP=0.2986 (**worst ever recorded**), 2021-12 AUC=0.8133, 2022-12 AUC=0.8180

**Feature importance (first empirical data)**:
| Rank | Feature | % Gain | Assessment |
|------|---------|--------|------------|
| 1 | hist_da_trend | 53.9% | Dominant — single feature > half the model |
| 2 | hist_physical_interaction | 14.3% | Strong — validates interaction features from iter 1 |
| 3 | hist_da | 11.3% | Strong — historical collective = 79.4% of gain |
| 4-6 | prob_below_90, prob_exceed_90, prob_exceed_95 | 10.3% | Moderate — core physical flow features |
| 7-13 | 7 features | 8.9% | Weak — contributing but minor |
| 14-17 | density_kurtosis, density_cv, exceed_severity_ratio, density_skewness | 1.4% | Near-zero — **pruning candidates** |

**Key Learnings**:
- Window expansion exhausted: 14→18 months provides zero marginal benefit. Older 2018-2019 data dilutes more than it diversifies.
- The model is essentially a historical trend predictor (79% of gain from hist_da_trend + hist_physical_interaction + hist_da), augmented by physical flow features (18%).
- Distribution shape features (skewness, kurtosis, CV) contribute <1.3% collectively — clear noise.
- hist_physical_interaction (#2 at 14%) validates iter 1's decision to add interaction features. exceed_severity_ratio (#16 at 0.38%) does not.
- AP bot2 trend is worsening: v0002(-0.0017) → v0003(-0.0045) → v0004(-0.0040) → v0005(-0.0075). Each window expansion degrades the AP tail.
- 2022-09 AP now at 0.2986 — worst across all 5 experiments. Structural, not addressable by current features.
- VCAP@500 bot2 recovered to 0.0449 (from v0004's 0.0387) — 18-month window stabilized VCAP@500 tail despite not improving mean.

**Code Issues**: Clean implementation, no new issues introduced. New LOW: feature importance output has no test coverage for file existence/schema (Codex). Carried: threshold leakage (HIGH), threshold `>` vs `>=` (MEDIUM), missing schema guard (MEDIUM).
