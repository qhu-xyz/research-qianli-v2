# Progress

## Current Batch: smoke-v6-20260227-190225

| Field | Value |
|-------|-------|
| Batch ID | smoke-v6-20260227-190225 |
| Iteration | 1 of 3 — synthesis complete |
| State | ORCHESTRATOR_SYNTHESIZING → direction_iter2 written |
| Champion | None (v0 baseline) |

### Iteration 1 Results
- **Hypothesis**: Default hyperparams reproduce v0 metrics (determinism check)
- **Version**: v0001
- **Outcome**: PASS — all 10 metrics identical to v0 (zero delta)
- **Promoted**: No (no improvement, S1-REC Group B still fails)
- **Code issues found**: from_phase broken (HIGH), threshold leakage (HIGH), Group B policy gap (MEDIUM)
- **Tests**: 66/66 passed

### Iteration 2 Plan
- **Hypothesis**: Lowering threshold_beta from 0.7 to 0.3 will produce positive predictions, fixing S1-REC
- **Additional**: Fix from_phase crash recovery, implement Group B pass policy
- **Key risk**: S1-BRIER has only 0.02 headroom — threshold changes may flip it
- **Status**: Direction written, awaiting worker pickup

### v0 Baseline Summary
- 20 test samples, 2 positive (binding_rate=0.1)
- Group A gates: all pass
- Group B gates: S1-REC fails (0.0 vs floor 0.4) — expected, model predicts no positives at threshold 0.82
