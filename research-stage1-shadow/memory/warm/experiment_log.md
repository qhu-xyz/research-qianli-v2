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
