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
