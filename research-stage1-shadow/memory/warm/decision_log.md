# Decision Log

## Iteration 1 Synthesis — 2026-02-28

### D1: No promotion for v0001
**Rationale**: v0001 is identical to v0 (zero delta on all metrics). No improvement, S1-REC (Group B) still fails. This was by design — iteration 1 was infrastructure validation only.

### D2: No gate floor changes
**Rationale**: Both reviewers agree — too early to recalibrate. Only SMOKE_TEST data (n=20) available. Codex notes noise_tolerance=0.02 is not statistically meaningful at this sample size (one AUC pairwise swap = ~0.028). Keep current floors and recalibrate after real-data run.

### D3: Iteration 2 target — fix S1-REC via threshold adjustment
**Rationale**: S1-REC is the only failing gate. Root cause is threshold=0.82 too high for binding_rate=0.1, resulting in pred_binding_rate=0.0 (no positives predicted). Approach: lower threshold_beta from 0.7 to ~0.3 to favor recall over precision. Risk: S1-BRIER has only 0.02 headroom — monitor closely.

### D4: Iteration 2 should also fix from_phase and Group B policy
**Rationale**: Codex flagged from_phase as HIGH severity — broken crash recovery. Group B policy gap flagged by both reviewers. These are low-risk code fixes that improve pipeline robustness. Threshold leakage (Codex HIGH) is deferred — impractical to fix at n=20 (SMOKE_TEST), will address when real data is available.

### D5: Defer threshold leakage fix to real-data iteration
**Rationale**: At n=20, a train/tune/test 3-way split would leave too few samples per partition. The concern is valid but only actionable with larger datasets.
