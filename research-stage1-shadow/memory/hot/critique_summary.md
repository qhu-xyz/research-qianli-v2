# Critique Summary — Iteration 3 (feat-eng-20260303-060938, v0006)

## Reviewer Agreement (high confidence)
1. **Do not promote v0006** — unanimously; AP regressed below v0 baseline (first time ever), AP bot2 worst at 0.3228
2. **v0004 remains the best version** — both reviewers independently identify v0004 as the strongest balanced candidate for HUMAN_SYNC
3. **NDCG/VCAP improvements are real and significant** — both note the 10W/2L consistency and statistical significance (p=0.039)
4. **AP regression is broadly distributed** — 3W/9L, not outlier-driven. This is a genuine tradeoff, not noise.
5. **Clean code changes** — both confirm feature removal, monotone constraint update, and test updates are all correct and minimal
6. **Full monotone constraint effect** — both identify the structural observation that removing all unconstrained features (monotone=0) is the likely driver of the ranking quality improvement
7. **Feature set ceiling confirmed** — 6 experiments spanning 3 independent levers have exhausted within-feature-set improvements
8. **Threshold methodology debt unchanged** — threshold leakage (HIGH), `>` vs `>=` (MEDIUM) persist. Both agree to defer to HUMAN_SYNC.

## Claude-Specific Insights
- Detailed per-month W/L table with all 12 months x 4 Group A metrics
- Sign test significance: VCAP@100 and NDCG both p=0.039 (10W/2L), AP p=0.073 (3W/9L, borderline significant regression)
- Seasonal pattern analysis: weakest months scattered (2022-09, 2022-12, 2021-04) — structural, not seasonal
- 2022-09 deep dive: AP=0.2987 (worst ever), hist_da_trend contributes only 38.3% of gain vs 44.2% average — trend signal weaker during structural change
- Feature importance redistribution: hist_da doubled (11.3% to 24.1%), creating more balanced level vs trend signal
- NDCG +0.0227 is the largest Group A mean improvement in pipeline history (5.5x larger than v0004's +0.0038)
- BRIER headroom narrowing to 0.0163 — 6th consecutive, contradicts model simplification expectation
- Recommends v0004 for promotion at HUMAN_SYNC; v0006's NDCG/VCAP profile as a future research direction
- Suggests investigating LambdaRank objective and whether top-100-only operational use changes the value assessment

## Codex-Specific Insights
- Three-layer gate detail with exact margins: Layer 3 closest to fail is AP (margin 0.0106, within tolerance but narrowing)
- Champion is null — Layer 3 non-regression effectively disabled. Recommends setting champion immediately (operational change)
- Noise tolerance 0.02 is loose relative to observed shifts (~0.01 for AUC/AP/NDCG). Recommends metric-specific tolerances: AUC ~0.01, AP ~0.015, NDCG ~0.01, VCAP@100 ~0.02
- VCAP@100 floor (-0.035) non-binding; discuss tightening to 0.0 at HUMAN_SYNC
- Emphasizes AP stability for weak months (2022-09 priority) as the key constraint on promotion
- Recommends fixing threshold methodology debt before interpreting Group B trend lines

## Synthesis Assessment
- **Core agreement**: Identical on promotion decision, best version identification, and ceiling diagnosis
- **Complementary strengths**: Claude provides deeper statistical analysis and feature importance interpretation; Codex provides precise gate margin calculations and operational recommendations (champion activation)
- **The v0006 finding**: Both agree this is the most experimentally informative iteration — the AP/NDCG tradeoff from monotone constraint structure is a novel, actionable insight

## Open Code Issues (cumulative, unchanged)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Layer 3 disabled when champion=null | MEDIUM | Codex (hp-tune-134412) | Recommend fixing at HUMAN_SYNC |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | Low priority |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
