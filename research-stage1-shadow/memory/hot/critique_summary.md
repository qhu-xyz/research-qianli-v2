# Critique Summary — Iteration 2 (feat-eng-3-20260303-104101, v0009)

## Reviewer Agreement (high confidence)
1. **Promote v0009** — unanimous; all Group A gates pass all 3 layers; VCAP@100 recovery achieved (primary target)
2. **AP at new pipeline high** — 0.4445 (9W/3L); both note this is approaching statistical significance
3. **Feature importance validates signal** — 17.13% combined from 3 interactions; hist_seasonal_band (#2 at 11.75%) is exceptional
4. **Code changes are clean** — both confirm correctness of interaction formulas, monotone constraints, smoke data, test updates
5. **CAP@100/500 Group B failures persist** — both reaffirm this is a floor calibration issue (3 consecutive versions failing)
6. **AUC flat within noise** — both correctly identify 4W/8L as noise (all monthly deltas ≤ ±0.003)
7. **colsample_bytree 0.9 validated** — no overfitting signal; BRIER continued to improve

## Claude-Specific Insights
- Detailed seasonal analysis of VCAP@100 gains: concentrated in spring/summer months (2021-08 +0.0239, 2022-09 +0.0060, 2021-06 +0.0049) where hist_seasonal_band adds most signal
- Early-period months (2020-09, 2020-11, 2021-01) show small VCAP@100 regressions — shorter history reduces interaction signal
- NDCG improvements in structurally weak months (2021-08 +0.0125, 2021-10 +0.0113, 2021-06 +0.0064)
- Recommends: n_estimators 300, learning_rate 0.07 for iter 3. Do NOT add more interactions (saturation). colsample_bytree 0.9 is optimal.
- Gate calibration: tighten VCAP@100 floor to 0.0; relax CAP@100/500 by 0.03 (reaffirming iter 1)
- Calculated new L3 floors if v0009 promoted: AUC ≥ 0.7989, AP ≥ 0.3512, VCAP@100 ≥ -0.0111, NDCG ≥ 0.6448

## Codex-Specific Insights
- Notes VCAP@100 mean gain partially concentrated in 2021-08 (+0.0239); without it, net gain smaller — but bot2 improvement is the more meaningful signal
- Code findings:
  - (MEDIUM) Temporal leakage fallback in data_loader.py — row-proportion split when auction_month missing
  - (MEDIUM) Interaction computation all-or-nothing — computes all interactions even if only some are in config
  - (LOW) New interaction feature tests validate presence/count but not formula correctness
- Gate calibration: keep Group A floors unchanged; tighten VCAP@100 to 0.0; relax CAP@100/500 by 0.02-0.03; keep noise_tolerance 0.02
- Notes noise_tolerance 0.02 is generous relative to observed bot2 deltas (~0.001-0.003)

## Synthesis Assessment
- **Core agreement**: Identical on promotion, risk identification, and iter 3 direction
- **Complementary strengths**: Claude provides deeper seasonal/trend analysis; Codex provides precise gate math and code-level findings
- **No material disagreement**: Both recommend promotion; both recommend same iter 3 direction (more trees, lower LR)
- **Mild divergence on gate calibration**: Claude recommends tightening Group A headroom language; Codex says keep unchanged. Difference is cosmetic — both pass comfortably.

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| check_gates_multi_month None-as-pass | MEDIUM | Codex (feat-eng-3 iter1) | Deferred to HUMAN_SYNC |
| Temporal leakage fallback (row split) | MEDIUM | Codex (feat-eng-3 iter2) | NEW — recommend hard assertion |
| Interaction computation all-or-nothing | MEDIUM | Codex (feat-eng-3 iter2) | NEW — tech debt |
| Missing source features warn not fail-fast | MEDIUM | Codex (feat-eng-2 iter1) | Low priority |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Source-feature list duplication | LOW | Codex (feat-eng-3 iter1) | Tech debt |
| Missing formula correctness tests for interactions | LOW | Codex (feat-eng-3 iter2) | NEW — tech debt |
| Threshold scaling unbounded | LOW | Codex (feat-eng-3 iter1) | Low priority |
| sf_nonzero_frac smoke data can exceed 1.0 | LOW | Codex (feat-eng-2 iter1) | Low priority |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | Low priority |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
