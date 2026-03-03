# Experiment Log

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
