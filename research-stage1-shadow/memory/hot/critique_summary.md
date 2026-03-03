# Critique Summary — Iteration 1 (feat-eng-2-20260303-092848, v0007)

## Reviewer Agreement (high confidence)
1. **Promote v0007** — unanimously; first version to satisfy all human-input success criteria (AUC>0.840, AP>0.400, 12W/12>8/12)
2. **AUC/AP improvements are statistically robust** — 12W/0L (p≈0.0002) and 11W/1L (p≈0.006) respectively; not outlier-driven
3. **NDCG is the primary risk** — both flag bot2 regression -0.0154 (margin 0.0046 to L3 tolerance); 5W/7L with late-2022 concentration
4. **Code changes are clean** — both confirm correctness of feature additions, monotone constraints, smoke data, and test updates
5. **Activate Layer 3** — both recommend setting champion (currently null → L3 auto-pass)
6. **Precision-first strategy unchanged** — both agree; strong AUC/AP profile is aligned with business objective
7. **Feature importance paradox acknowledged** — 4.66% combined gain yet +0.0137 AUC; features are auxiliary discriminators with high generalization value

## Claude-Specific Insights
- Detailed per-month tables for all Group A metrics with sign test p-values
- AP bot2 trend reversal analysis: 6-experiment monotonic decline reversed to +0.0363 vs v0
- BRIER trend reversal: 6-experiment narrowing ended; topology features improve calibration
- CAP@100/500 at 0.002 and 0.004 headroom from Group B floors — model profile shifted from threshold-dependent capture to ranking quality
- NDCG decomposition: losses concentrated in 2022-03 (-0.031), 2022-06 (-0.017), 2022-09 (-0.017), 2022-12 (-0.018); gains in 2020-09 (+0.032), 2021-06 (+0.039), 2021-10 (+0.036)
- Recommends: NDCG-targeted monotone constraint sensitivity analysis + distribution shape features + HP tuning on 19-feature base
- Suggests relaxing CAP@100/500 floors by 0.02 if v0007 promoted

## Codex-Specific Insights
- Code findings: (a) MEDIUM — missing source features warned not failed fast in data_loader.py, (b) LOW — sf_nonzero_frac smoke data can exceed 1.0, (c) LOW — missing source feature ValueError not unit-tested
- Precise gate margin calculations for all 10 gates across 3 layers
- Layer 3 effectively disabled when champion=null — operational gap
- Noise tolerance 0.02 is loose for AUC but just adequate for NDCG
- VCAP@100 floor (-0.035) remains non-binding; recommends tightening to 0.0
- Recommends fixing threshold methodology debt before interpreting Group B trends

## Synthesis Assessment
- **Core agreement**: Identical on promotion decision, risk identification (NDCG), and strategic direction
- **Complementary strengths**: Claude provides deeper statistical/trend analysis; Codex provides precise gate math and code-level findings
- **No material disagreement**: Both recommend promotion; both flag NDCG as primary future risk

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Missing source features warn not fail-fast | MEDIUM | Codex (feat-eng-2 iter1) | Low priority |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Layer 3 disabled when champion=null | MEDIUM | Codex (hp-tune-134412) | Recommend fixing at HUMAN_SYNC |
| sf_nonzero_frac smoke data can exceed 1.0 | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Missing source feature ValueError not tested | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | Low priority |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
