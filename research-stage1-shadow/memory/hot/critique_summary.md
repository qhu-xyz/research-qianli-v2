# Critique Summary — Iteration 1 (feat-eng-20260303-060938, v0004)

## Reviewer Agreement (high confidence)
1. **Do not promote v0004** — all gates pass but effect sizes too small for production baseline change; "encouraging" not "promotion-worthy"
2. **Best result of any experiment** — AUC 9W/3L (best W/L), VCAP@100 10W/2L (first statistically significant improvement, sign test p=0.039)
3. **Partially additive** — VCAP@100 super-additive, NDCG roughly additive, AUC mostly window-driven, AP sub-additive
4. **Clean code changes** — both bug fixes (f2p parsing, dual-default) correct and well-implemented. No new issues introduced.
5. **AP is the lagging metric** — 6W/6L, the one Group A metric that hasn't responded to any lever in 4 experiments
6. **2022-09 remains the hardest month** — AP=0.307, 4th consecutive failure across all levers. Binding rate 6.63% (lowest). Structural feature-target mismatch.
7. **VCAP@500 regression growing** — 3rd consecutive experiment showing VCAP@500 decline; bot2=0.0387 approaching floor 0.0408
8. **Precision preserved** — pred_binding_rate 0.0749 (essentially identical to v0's 0.075), precision 0.4449 vs 0.4423

## Claude-Specific Insights
- Thorough additivity analysis showing VCAP@100 super-additivity (combined > sum of parts)
- Sign test analysis: VCAP@100 p=0.039 (significant), AUC p=0.073 (approaching). First statistically significant result in pipeline history.
- Recommends feature selection/importance analysis to understand which of 17 features contribute
- AP weakness analysis: interactions reduce AP consistency (6W/6L vs 8W/4L for window alone)
- BRIER headroom (0.0187) continuing monotonic decline: v0(0.020) → v0003(0.0189) → v0004(0.0187)
- Recommends exploring new feature sources if current set has hard ceiling
- No gate calibration changes recommended

## Codex-Specific Insights
- Confirms all 3 layers pass cleanly for all gates, closest margin to Layer 3 failure is NDCG (+0.0140)
- Bottom-2 analysis: AP -0.0040, NDCG -0.0060, VCAP@100 -0.0003 vs v0 bottom-2 (all within tolerance)
- VCAP@500 bot2 approaching floor: 0.0387 vs 0.0408 (margin only 0.0021)
- Code findings: threshold boundary mismatch (HIGH, carried), threshold leakage (HIGH, carried), silent ptype fallback (MEDIUM, new)
- Recommends tightening Layer 3 tolerance to metric-specific values: AUC/AP/NDCG ~0.005-0.01, VCAP@100 looser due to high variance
- Recommends keeping current feature set for one more iteration, emphasizing Group A + bottom_2 behavior

## Synthesis Assessment
- **Core agreement**: Both reach identical promotion decision and identify AP as the problem metric
- **Complementary strengths**: Claude provides statistical significance testing and additivity framework; Codex provides tighter gate analysis (margin-to-failure) and code correctness
- **Key divergence**: Claude recommends feature selection (potentially drop some interactions); Codex recommends keeping all 17 features for another round. Resolution: keep all 17 for iter 2 (window expansion test) but export feature importance to make data-driven decision for iter 3
- **Both reviewers reiterate**: threshold methodology debt is the largest outstanding code issue, affects threshold-dependent metrics, deferred to HUMAN_SYNC

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Silent ptype fallback (horizon=3) | MEDIUM | Codex (feat-eng-060938) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Layer 3 disabled when champion=null | MEDIUM | Codex (hp-tune-134412) | Accepted (v0 is reference) |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
