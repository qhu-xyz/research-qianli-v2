# Critique Summary — Iteration 1 (feat-eng-3-20260303-104101, v0008)

## Reviewer Agreement (high confidence)
1. **Promote v0008** — unanimous; all Group A gates pass all 3 layers; NDCG bot2 improved meaningfully (+0.0101)
2. **AUC/AP improvements are consistent** — 8W/4L and 9W/3L respectively; not outlier-driven
3. **VCAP@100 is the primary concern** — both flag 4W/8L regression; closest L3 risk (margin +0.0167)
4. **Code changes are clean** — both confirm correctness of feature additions, monotone constraints, smoke data, and test updates
5. **CAP@100/500 Group B failures expected** — both agree these are floor calibration artifacts, not model deficiency
6. **Precision-first strategy strengthened** — precision improved +0.007 while recall held steady
7. **Diminishing returns acknowledged** — 10.3% feature importance but modest metric lifts vs v0007's breakthrough

## Claude-Specific Insights
- Detailed per-month seasonal analysis: spring transition months (2021-04, 2022-03) remain NDCG weak spots
- 2022-03 got the largest single-month NDCG lift (+0.018); 2021-08 showed largest NDCG regression (-0.009)
- NDCG 8W/4L — improvement consistent across majority of months, not driven by 1-2 outliers
- Suggests: derived interactions (density_mean*prob_exceed_100, tail_concentration*expected_overload), investigate VCAP@100 dilution via top-100 feature importance analysis
- Recommends colsample_bytree 0.8→0.9 or 1.0 with 26 features
- Do NOT change HPs yet beyond colsample_bytree

## Codex-Specific Insights
- Code findings: (a) MEDIUM — threshold selection and evaluation on same validation split inflates threshold-dependent metrics, (b) MEDIUM — check_gates_multi_month treats None as pass, (c) LOW — threshold scaling not bounded, (d) LOW — source-feature lists duplicated across modules
- Precise gate margin calculations across all 3 layers
- Recommends: split threshold tuning from reporting, fix check_gates_multi_month, centralize source-feature metadata
- Tighten VCAP@100 floor from -0.035 to 0.0; relax CAP@100/500 floors by 0.02

## Synthesis Assessment
- **Core agreement**: Identical on promotion, risk identification (VCAP@100), and direction (interactions for iter 2)
- **Complementary strengths**: Claude provides deeper seasonal/trend analysis; Codex provides precise gate math and code-level findings
- **No material disagreement**: Both recommend promotion; both flag VCAP@100 as primary future risk

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| check_gates_multi_month None-as-pass | MEDIUM | Codex (feat-eng-3 iter1) | NEW — recommend fix at HUMAN_SYNC |
| Missing source features warn not fail-fast | MEDIUM | Codex (feat-eng-2 iter1) | Low priority |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Layer 3 disabled when champion=null | MEDIUM | Codex (hp-tune-134412) | Fixed (champion now set) |
| Source-feature list duplication | LOW | Codex (feat-eng-3 iter1) | NEW — tech debt |
| Threshold scaling unbounded | LOW | Codex (feat-eng-3 iter1) | NEW — low priority |
| sf_nonzero_frac smoke data can exceed 1.0 | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Missing source feature ValueError not tested | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | Low priority |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
