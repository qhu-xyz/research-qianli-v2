# Critique Summary — Iteration 3 (feat-eng-3-20260303-104101, v0010) — FINAL

## Reviewer Agreement (high confidence)
1. **Do NOT promote v0010** — unanimous; confirmed null result; all metrics within noise of champion v0009
2. **v0009 remains champion** — both explicitly recommend keeping v0009's configuration (29 features, n_estimators=200, lr=0.1, colsample_bytree=0.9)
3. **Model capacity ceiling confirmed** — 10 experiments have converged; performance bounded by feature information, not tree capacity
4. **All Group A gates pass all 3 layers** — technically promotable, but no evidence of improvement
5. **Code changes are minimal and correct** — 2 HP values + 2 test assertions, no logic changes
6. **CAP@100/500 Group B failures persist** — 4th consecutive version; structural model profile shift
7. **Pipeline ready for HUMAN_SYNC** — feature engineering and HP tuning exhausted

## Claude-Specific Insights
- Detailed per-month redistribution analysis: 4 months improved on all metrics, 4 degraded on all, 4 mixed = noise signature
- 2022-12 persistent weakest month — regressed across all 4 Group A metrics (deltas all < 0.004 but uniformly negative)
- 2022-03 NDCG +0.0083 — largest single positive signal; finer splits helped this spring transition month
- Feature importance redistribution: interactions 17.13%→15.8% is mechanical (more splitting opportunities), not information loss
- Recommendations for next batch: temporal features, multi-stage pipeline, alternative ranking-specific loss
- Gate calibration: tighten VCAP@100 to 0.0; relax CAP@100/500 by 0.03; consider noise_tolerance 0.015

## Codex-Specific Insights
- Code findings (new in iter 3):
  - (HIGH) `check_gates_multi_month` does not enforce full primary-month coverage — a version can skip hard months and pass
  - (MEDIUM) Benchmark ignores `threshold_scaling_factor` — threshold-dependent metrics not reproducible between pipeline and benchmark
  - (MEDIUM) Threshold tuned and evaluated on same validation split — optimistic bias for Group B
  - (LOW) Feature prep uses `.fill_null(0)` but not NaN-aware — potential silent propagation
- Gate calibration: keep Group A floors unchanged; tighten VCAP@100 to 0.0; relax CAP@100/500; consider noise_tolerance 0.015

## Synthesis Assessment
- **Core agreement**: Identical on non-promotion, risk identification, and HUMAN_SYNC readiness
- **Complementary strengths**: Claude provides deeper seasonal/trend analysis and next-batch strategy; Codex provides precise code-level findings and reproducibility concerns
- **No material disagreement**: Both recommend same action on every dimension
- **New code finding from Codex is significant**: Month-coverage gap could allow a version to game Group A by skipping hard months — should be priority fix at HUMAN_SYNC

## Open Code Issues (cumulative — for HUMAN_SYNC)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Missing month-coverage enforcement in gate checks | HIGH | Codex (iter3) | NEW — priority fix |
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| check_gates_multi_month None-as-pass | MEDIUM | Codex (iter1) | Deferred to HUMAN_SYNC |
| Benchmark ignores threshold_scaling_factor | MEDIUM | Codex (iter3) | NEW |
| Threshold tuned/evaluated on same split | MEDIUM | Codex (iter3) | NEW — statistical bias |
| Temporal leakage fallback (row split) | MEDIUM | Codex (iter2) | Recommend hard assertion |
| Interaction computation all-or-nothing | MEDIUM | Codex (iter2) | Tech debt |
| Missing source features warn not fail-fast | MEDIUM | Codex (feat-eng-2 iter1) | Low priority |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| NaN vs null handling gap | LOW | Codex (iter3) | NEW |
| Source-feature list duplication | LOW | Codex (iter1) | Tech debt |
| Missing formula correctness tests for interactions | LOW | Codex (iter2) | Tech debt |
| Threshold scaling unbounded | LOW | Codex (iter1) | Low priority |
| sf_nonzero_frac smoke data can exceed 1.0 | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | Low priority |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
