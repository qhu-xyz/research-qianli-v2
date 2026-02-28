# Progress

## Current Batch: smoke-v6-20260227-190225

| Field | Value |
|-------|-------|
| Batch ID | smoke-v6-20260227-190225 |
| Iteration | 1 of 3 |
| State | ORCHESTRATOR_PLANNING → direction written |
| Objective | Infrastructure validation (SMOKE_TEST) |
| Champion | None (v0 baseline) |

### Iteration 1 Plan
- **Hypothesis**: Default hyperparams reproduce v0 metrics (determinism check)
- **Direction**: No ML changes — run pipeline with identical v0 config
- **Expected outcome**: Metrics match v0 within noise tolerance (0.02)
- **Status**: Direction written, awaiting worker pickup

### v0 Baseline Summary
- 20 test samples, 2 positive (binding_rate=0.1)
- Group A gates: all pass
- Group B gates: S1-REC fails (0.0 vs floor 0.4) — expected, model predicts no positives at threshold 0.82
