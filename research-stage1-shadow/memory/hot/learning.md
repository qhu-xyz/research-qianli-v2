# Learning

## From Iteration 1 (Infrastructure Validation — smoke-v6)

### Pipeline
- Pipeline is fully deterministic: seed=42 for data + XGBoost produces bit-for-bit identical metrics across runs
- Version registry structure works: config.json, metrics.json, meta.json, changes_summary.md, comparison.md, model/classifier.ubj.gz
- 70/70 tests pass (4 new in smoke-v7 iter1)

### Gate Insights
- S1-BRIER is the tightest gate (0.02 headroom) — binding constraint for any threshold changes
- S1-AP floor (0.12) is very generous — 0.47 headroom
- VCAP@K gates are untested at n=20 (saturated to 1.0)
- S1-REC fails because threshold=0.82 yields zero positive predictions

### Methodological
- Threshold leakage: threshold is optimized and evaluated on same validation split — overstates threshold-dependent metrics. Not fixable at n=20 but must address for real data.
- noise_tolerance=0.02 is not statistically meaningful at n=20 (one AUC swap ≈ 0.028)

### Reviewer Dynamics
- Codex found deeper structural bugs; Claude had better practical gate analysis
- Reading reviews independently is valuable — they catch different things

## From Iteration 1 (H2 threshold_beta — smoke-v7)

### F-beta Formula (CRITICAL — got this wrong)
- **beta < 1 → weights PRECISION more** (higher threshold, fewer positives)
- **beta > 1 → weights RECALL more** (lower threshold, more positives)
- **beta = 1 → standard F1** (equal weighting)
- F_β = (1 + β²) × P × R / (β² × P + R)
- The direction_iter1.md had this inverted, causing hypothesis H2 to have zero effect
- Worker correctly diagnosed the error post-hoc — good safety net

### Threshold Semantics Mismatch (Codex HIGH, new)
- `precision_recall_curve` thresholds are inclusive (>=), but `apply_threshold` in threshold.py uses strict `>`
- At discrete n=20, samples exactly at the boundary get excluded
- This makes realized predictions strictly more conservative than the optimized operating point
- Could be a contributing factor to zero positives — must fix in iter2

### Bug Fixes Validated
- from_phase guard: `NotImplementedError` for phases > 1 prevents silent crashes
- Group B pass policy: compare.py now correctly separates Group A (blocking) from Group B (informational)
- Model gzip: automatic compression after save, uncompressed file deleted
- All three changes were clean, well-tested, no regressions

### Code Quality Observations
- Misleading docstring in threshold.py:8 ("0.7 = moderate recall/precision balance") — wrong, beta=0.7 favors precision. Likely contributed to the inverted hypothesis.
- DRY issue: build_comparison_table re-implements Group A/B tracking instead of calling evaluate_overall_pass
- Missing-metric handling differs between markdown table and JSON output (Codex MEDIUM)

### Open Issues (carried forward)
- Threshold leakage (both HIGH) — deferred to real data
- Dead config scale_pos_weight_auto (MEDIUM)
- Version allocator not wired (MEDIUM)
- Memory scaling for real data (MEDIUM)
