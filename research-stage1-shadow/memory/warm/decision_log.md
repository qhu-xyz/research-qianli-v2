# Decision Log

> **NOTE**: Decisions D1–D10 below were made during SMOKE_TEST runs (n=20).
> Gate floors have since been recalibrated from real v0 (12 months, ~270K rows/month).
> D2/D7 (no floor changes) are superseded — floors are now calibrated from real data.

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

---

## Iteration 1 Synthesis (hp-tune-20260302-134412) — 2026-03-02

### D11: No promotion for v0003
**Rationale**: v0003 shows no improvement on any Group A ranking metric. AUC degraded in 11/12 months (statistically significant, p≈0.003). AP and NDCG degraded in 8/12 months. Only positive signal is BRIER -0.004 (Group B calibration, not ranking). Gates pass due to 0.05 headroom, not improvement. Both reviewers independently recommend against promotion. H3 refuted.

### D12: No gate calibration changes
**Rationale**: Only 1 real-data iteration beyond v0. Premature to tighten floors. Codex suggests metric-specific noise_tolerance — valid but needs 3-4 data points. Layer 3 non-regression effectively disabled when champion=null — acceptable since v0 is reference baseline. Revisit after iter2/iter3.

### D13: Revert HP changes, pivot to feature engineering for iter2
**Rationale**: Model is feature-limited, not complexity-limited. Evidence: (1) deeper trees didn't improve discrimination despite 2x more rounds; (2) AUC degraded systematically across 11/12 months; (3) BRIER improved (better calibration) while AUC worsened (worse discrimination) — ranking signal from 14 features is saturated. Both reviewers agree features are the bottleneck. Revert to v0 defaults, add interaction features.

### D14: Continue deferring threshold methodology fixes
**Rationale**: Threshold leakage (HIGH) and `>` vs `>=` mismatch (MEDIUM) affect threshold-dependent metrics only (REC, CAP, precision, BRIER). All Group A blocking gates are threshold-independent ranking metrics. Structural fix best done at HUMAN_SYNC. Issues affect v0 and v0003 equally.

---

## Iteration 1 Synthesis (hp-tune-20260302-144146) — 2026-03-02

### D15: No promotion for v0002
**Rationale**: v0002 (interaction features) shows no meaningful improvement over v0 on any Group A metric. AUC unchanged (+0.0000, 5W/6L/1T). AP +0.0010 and NDCG +0.0016 are within noise (<0.05σ). Bottom-2 regressed on 3/4 Group A metrics. VCAP@500 and VCAP@1000 both regressed. The 2021-01 NDCG outlier (+0.042) drives the mean improvement. Both reviewers independently recommend against promotion.

### D16: No gate calibration changes
**Rationale**: 2 real-data iterations (v0003, v0002), both neutral-to-negative vs v0. Premature to tighten floors. Layer 3 remains effectively disabled (champion=null). Revisiting metric-specific noise_tolerance deferred to after 3-4 iterations per Codex suggestion. VCAP@100 effectively non-binding (negative floor) — noted but premature to act.

### D17: Pivot to training window expansion for iter2
**Rationale**: Two iterations have exhausted the within-feature-set levers:
- v0003 (HP tuning): AUC 0W/11L — complexity not the bottleneck
- v0002 (interaction features): AUC 5W/6L/1T — interactions don't break the ceiling
The persistent late-2022 weakness and pattern that early months benefit more than late months suggests the 10-month rolling window may be too short for distribution shifts. Expanding train_months 10→14 is the next logical lever: more diverse training examples without requiring new data sources. Keep 3 interaction features (marginally positive, computationally cheap).

### D18: Keep interaction features in iter2
**Rationale**: The 3 interaction features showed marginally positive signal for ranking (NDCG 8W/4L, AP 7W/5L). Computationally cheap, don't hurt discrimination (AUC neutral), and may interact positively with longer training windows. Remove only if iter2 shows they actively harm results.

### D19: Continue deferring threshold methodology fixes
**Rationale**: Same as D14 — threshold-dependent metrics only. Group A gates are threshold-independent. Fix at HUMAN_SYNC.

---

## Iteration 1 Synthesis (feat-eng-20260302-194243) — 2026-03-02

### D20: No promotion for v0003
**Rationale**: v0003 (14-month training window) shows small, directionally positive improvements on all Group A metrics: AUC +0.0013 (7W/4L/1T), AP +0.0012 (8W/4L), NDCG +0.0019 (7W/4L/1T), VCAP@100 +0.0034 (9W/3L). All 3 gate layers pass. However:
- No delta is statistically significant (all <1σ, best is VCAP@100 at p≈0.07)
- Gains partly outlier-driven (removing 2022-12 collapses AUC/AP gains)
- Target month 2022-09 didn't improve (AP regressed -0.0091)
- Bottom-2 mixed (AP -0.0045, NDCG -0.0059 vs v0)
- CAP metrics regressed (CAP@100 -0.0117, CAP@500 -0.0107)
Both reviewers independently recommend against promotion. This is the strongest of 3 real-data iterations but not sufficient to change the production baseline.

### D21: No gate calibration changes
**Rationale**: 3 real-data iterations (v0003-HP, v0002, v0003-window), all producing deltas within ±0.003 of v0 on Group A means. Gates are not blocking valid candidates — there simply haven't been meaningful improvements. Codex suggestion for metric-specific Layer 3 tolerances remains valid but still premature. Layer 3 remains disabled (champion=null). BRIER headroom narrowing (0.0189 vs 0.0200 initially) is worth monitoring but not actionable.

### D22: Combine 14-month window + interaction features for iter2 (H6)
**Rationale**: Three iterations have isolated individual levers:
- v0003-HP (HP tuning): AUC 0W/11L — not the bottleneck
- v0002 (interaction features): AUC 5W/6L/1T, NDCG 8W/4L — marginal positive ranking signal
- v0003-window (14-month window): AUC 7W/4L/1T, VCAP@100 9W/3L — best AUC signal so far
The natural next step is combining the two positive-signal levers. If effects are additive: AUC ~+0.0013, NDCG ~+0.0035, VCAP@100 ~+0.0043. Even partial additivity would produce the strongest result yet. This also addresses D18's intent (keep interaction features) while building on the window expansion. Reverting interactions in v0003 was methodologically correct to isolate the window effect; now we can combine.

### D23: Fix f2p parsing bug in iter2
**Rationale**: Codex (HIGH) identified that `int(ptype[1:])` crashes for "f2p" in data_loader.py. This blocks cascade stage-3 evaluation. Fix with an explicit mapping or regex. Not blocking for f0/f1 evaluation but should be resolved for future cascade work.

### D24: Fix dual-default fragility in benchmark.py in iter2
**Rationale**: Claude (MEDIUM) identified that `train_months=14` is hardcoded in both `_eval_single_month()` and `run_benchmark()` signatures AND in `PipelineConfig`. Future experiments that change PipelineConfig must manually update benchmark signatures in lockstep. Fix by using `None` sentinel with fallback to `PipelineConfig().train_months`.

### D25: Continue deferring threshold methodology fixes
**Rationale**: Same as D14/D19 — threshold-dependent metrics only. Group A gates are all threshold-independent ranking metrics. Fix at HUMAN_SYNC.
