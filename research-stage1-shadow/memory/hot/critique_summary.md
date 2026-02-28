# Critique Summary — Iteration 1

**Batch**: smoke-v7-20260227-191851 | **Version**: v0001

## Claude Review — PASS on code, FAIL on hypothesis

- H2 hypothesis failed: beta=0.3 weights precision, not recall (direction had F-beta formula inverted)
- Worker correctly diagnosed the error in changes_summary.md
- All 3 bug fixes implemented cleanly, no blocking issues
- 4 new tests, 70/70 passing, zero regressions
- Gate analysis: S1-BRIER still tightest (0.02 headroom), will likely flip when threshold drops in iter2
- NEW finding: misleading beta docstring in threshold.py:8 ("0.7 = moderate recall/precision balance" — wrong, beta=0.7 favors precision)
- NEW finding: DRY opportunity — build_comparison_table re-implements Group A/B tracking instead of calling evaluate_overall_pass
- Recommends beta=2.0 for iter2 (or beta=3.0 if 2.0 insufficient)
- No gate floor changes

## Codex Review — Zero progress on performance, structural issues found

- Confirmed hypothesis failure, zero delta on all metrics
- NEW **HIGH**: Threshold `>` vs `>=` mismatch — PR curve thresholds are inclusive but apply_threshold uses strict `>`, making realized predictions more conservative than optimized point. This can suppress positives and directly impact S1-REC.
- Reiterated **HIGH**: Threshold leakage (tune and eval on same validation split)
- NEW **MEDIUM**: Markdown pass column can disagree with JSON pass logic on missing metrics (value=None handling differs between build_comparison_table and evaluate_overall_pass)
- Reiterated **MEDIUM**: Memory scaling risk (DataFrames + arrays both alive across phases)
- Reiterated **LOW**: Dead config scale_pos_weight_auto, duplicated pass aggregation logic
- Recommends: align threshold semantics (`>=` or subtract epsilon), remove threshold leakage, refactor pass logic
- No gate floor changes

## Synthesized Assessment

**Consensus**: H2 failed due to inverted F-beta understanding. Infrastructure fixes are clean. No gate recalibration justified. Must use beta > 1 for iter2.

**Key divergence**:
- Codex found the `>` vs `>=` threshold mismatch (new HIGH) that Claude did not identify. This is a potentially important contributing factor to zero positives at n=20 — if the optimal threshold sits exactly at a sample's probability, strict `>` excludes it while `>=` would include it.
- Claude found the misleading docstring that likely contributed to the hypothesis error. Codex did not mention this.
- Both agree on beta > 1 being the fix direction.

**Priority for iter2**:
1. Fix threshold `>` vs `>=` semantics (Codex HIGH, new — may independently enable positive predictions)
2. Use threshold_beta=2.0 (both reviewers — correct the F-beta direction)
3. Fix beta docstring in threshold.py (Claude LOW)
4. Monitor S1-BRIER closely — 0.02 headroom, almost certain to shift

**Carried issues (deferred)**:
- Threshold leakage (both HIGH) — impractical at n=20
- Dead config scale_pos_weight_auto (Codex MEDIUM)
- Version allocator not wired (Codex MEDIUM)
- Missing-metric pass disagreement (Codex MEDIUM)
- Memory scaling (Codex MEDIUM)
