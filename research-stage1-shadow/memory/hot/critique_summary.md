# Critique Summary — Iteration 2 (feat-eng-20260303-060938, v0005)

## Reviewer Agreement (high confidence)
1. **Do not promote v0005** — unanimously; strictly worse than v0004 on all Group A means
2. **Diminishing returns confirmed** — 14→18 month window adds zero marginal benefit; all vs-v0004 deltas are noise
3. **Feature importance is the real deliverable** — first empirical data on feature contributions: hist_da_trend dominates (54% gain), distribution shape features are noise (<1.3%)
4. **Window expansion exhausted as a lever** — both reviewers explicitly state this; the productive range was 10→14 months
5. **Clean code changes** — feature importance extraction well-implemented, `_feature_importance` key convention and pre-aggregation pop are correct
6. **AP bot2 trend is concerning** — worsening across iterations: -0.0017 → -0.0045 → -0.0040 → -0.0075 (vs v0). Still within tolerance but monotonically degrading.
7. **2022-09 AP at all-time low** — 0.2986, worst across 5 experiments. Structural feature-target mismatch.
8. **VCAP@500 bot2 recovered** — 0.0449 (from v0004's dangerous 0.0387). 18-month window stabilized the tail.
9. **BRIER headroom at 0.0178** — 5th consecutive narrowing

## Claude-Specific Insights
- Per-month win/loss analysis: all AUC deltas vs v0004 are <0.003, pure noise
- Feature importance tier analysis: Dominant (54%), Strong (25%), Moderate (10%), Weak (9%), Near-zero (1.4%)
- hist_physical_interaction validated at #2 (14%), exceed_severity_ratio and density shape features identified as pruning candidates
- Recommends iter 3: revert to 14-month window, prune bottom 4 features (density_skewness, exceed_severity_ratio, density_cv, density_kurtosis)
- Suggests aggressive alternative: also drop overload_exceedance_product and prob_exceed_100/110 (17→11 features), but recommends conservative prune first
- Gate calibration: VCAP@100 floor to 0.0 at HUMAN_SYNC
- VCAP@1000 mean degradation noted: -0.0110 vs v0

## Codex-Specific Insights
- Three-layer gate detail with exact margins: Layer 3 closest to fail is AP (margin +0.0125)
- Layer 3 effectively disabled when champion=null — accepted with v0 as reference
- New LOW: feature importance output has no test coverage for file existence/schema
- Recommends metric-specific Layer 3 tolerances (AUC/AP/NDCG ~0.01) after champion is set
- Emphasis on AP stability for weak months (2022-09 priority)
- BRIER as closest mean gate to boundary (0.0178 headroom)

## Synthesis Assessment
- **Core agreement**: Identical promotion decision; window expansion exhausted
- **Complementary strengths**: Claude provides statistical rigor and feature pruning specifics; Codex provides gate margin analysis and code correctness
- **Iter 3 direction**: Both point toward feature pruning. Prune bottom 4 features, revert to 14-month window.

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Silent ptype fallback | MEDIUM | Codex (feat-eng-060938 iter1) | Low priority |
| Missing schema guard for interaction base columns | MEDIUM | Codex (hp-tune-144146) | Low priority |
| Layer 3 disabled when champion=null | MEDIUM | Codex (hp-tune-134412) | Accepted |
| Feature importance no test coverage | LOW | Codex (feat-eng-060938 iter2) | New |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
