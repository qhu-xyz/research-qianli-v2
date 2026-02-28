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

---

## Iteration 1 Synthesis (smoke-v7) — 2026-02-28

### D6: No promotion for v0001
**Rationale**: v0001 metrics are bit-for-bit identical to v0. The H2 hypothesis failed — threshold_beta=0.3 had zero effect because beta < 1 weights precision, not recall (direction had the formula inverted). Bug fixes (from_phase, Group B policy, gzip) are valuable infrastructure improvements but do not change model performance. S1-REC still 0.0.

### D7: No gate floor changes
**Rationale**: Both reviewers agree — still too early. All deltas are exactly zero. n=20 is too small for stable inference. Keep floors unchanged and revisit after real data.

### D8: Iteration 2 — fix beta direction (beta=2.0) + fix threshold `>` vs `>=` mismatch
**Rationale**: Two independent root causes for S1-REC=0.0 identified:
1. **Beta direction error** (both reviewers): beta < 1 favors precision. Must use beta > 1 to favor recall. beta=2.0 is the starting point (F_2 = 5PR/(4P+R) → heavily weights recall).
2. **Threshold `>` vs `>=` mismatch** (Codex HIGH, new): PR curve thresholds are inclusive (>=), but `apply_threshold` uses strict `>`. At discrete n=20, samples exactly at the threshold boundary get excluded. Fixing this may independently help produce positive predictions.

Both fixes should be applied in iter2 to maximize the chance of non-zero recall. If S1-BRIER flips due to the threshold drop, document the actual value and assess whether the floor needs recalibration at HUMAN_SYNC.

### D9: Fix misleading beta docstring (LOW priority)
**Rationale**: Claude found that threshold.py:8 says "0.7 = moderate recall/precision balance" which is misleading — beta=0.7 moderately favors precision. This likely contributed to the inverted hypothesis. Low priority but should be corrected to prevent future misdirection.

### D10: Continue deferring threshold leakage
**Rationale**: Codex reiterated this as HIGH. Still impractical at n=20. Remains deferred to real data.
