# Learning

## From Iteration 1 (Infrastructure Validation)

### Pipeline
- Pipeline is fully deterministic: seed=42 for data + XGBoost produces bit-for-bit identical metrics across runs
- Version registry structure works: config.json, metrics.json, meta.json, changes_summary.md, comparison.md, model/classifier.ubj.gz
- 66/66 tests pass
- Model gzip in pipeline.py is fragile — relies on caller to compress, not automatic

### Code Bugs Found (not yet fixed)
- **from_phase crash recovery is broken**: phases >1 reference uninitialized variables (train_df, val_df, model, etc.)
- **Group B pass policy not implemented**: compare.py treats all gates equally in pass aggregation — Group B failures block overall pass even when they shouldn't
- **scale_pos_weight_auto is dead config**: stored in config but training always auto-computes regardless
- **version_counter.json not wired in**: pipeline requires explicit --version-id, allocator unused

### Gate Insights
- S1-BRIER is the tightest gate (0.02 headroom) — binding constraint for any threshold changes
- S1-AP floor (0.12) is very generous — 0.47 headroom
- VCAP@K gates are untested at n=20 (saturated to 1.0)
- S1-REC fails because threshold=0.82 yields zero positive predictions

### Methodological
- Threshold leakage: threshold is optimized and evaluated on same validation split — overstates threshold-dependent metrics (S1-REC, S1-CAP@K). Not fixable at n=20 but must address for real data.
- noise_tolerance=0.02 is not statistically meaningful at n=20 (one AUC swap ≈ 0.028)

### Reviewer Dynamics
- Codex found deeper structural bugs; Claude had better practical gate analysis
- Reading reviews independently is valuable — they catch different things
