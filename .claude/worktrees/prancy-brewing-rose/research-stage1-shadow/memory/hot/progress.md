# Progress

## Current Batch: smoke-v7-20260227-191851

| Field | Value |
|-------|-------|
| Batch ID | smoke-v7-20260227-191851 |
| Iteration | 1 of 3 — planning complete |
| State | ORCHESTRATOR_PLANNING → direction_iter1 written |
| Champion | None (v0 baseline) |

### Prior Batch Context (smoke-v6)
- Iter 1 confirmed pipeline determinism (v0001 = v0, zero delta, 66/66 tests pass)
- Iter 2 was planned (threshold_beta=0.3 + bug fixes) but never executed
- Key bugs found: from_phase broken (HIGH), threshold leakage (HIGH, deferred), Group B policy (MEDIUM)

### Iteration 1 Plan
- **Hypothesis**: H2 — Lowering threshold_beta from 0.7 to 0.3 produces positive predictions, fixing S1-REC
- **Additional changes**: Fix from_phase crash recovery, Group B pass policy in compare.py, model gzip
- **Key risk**: S1-BRIER has only 0.02 headroom — threshold changes may flip it
- **Status**: Direction written, awaiting worker pickup

### v0 Baseline Summary
- 20 test samples, 2 positive (binding_rate=0.1)
- Group A gates: all pass
- Group B gates: S1-REC fails (0.0 vs floor 0.4) — model predicts no positives at threshold 0.82
