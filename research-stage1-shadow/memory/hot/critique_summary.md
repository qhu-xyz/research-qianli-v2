# Critique Summary — Iteration 1 (hp-tune-20260302-134412, v0003)

## Reviewer Agreement (high confidence)
1. **Do not promote v0003** — no Group A improvement, AUC degraded systematically
2. **Model is feature-limited, not complexity-limited** — HP tuning is the wrong lever
3. **Clean code implementation** — no new bugs introduced, diff matches direction exactly
4. **Late-2022 weakness persists** — distribution shift not addressed by tree complexity
5. **Threshold methodology issues are real but pre-existing** — don't invalidate v0003 vs v0 comparison

## Claude-Specific Insights
- AUC 0W/11L is statistically significant (p≈0.003) — not noise, a real systematic effect
- BRIER 12W/0L also significant (p≈0.0002) — deeper trees genuinely improve calibration
- Recommends feature engineering: interaction features, temporal features, feature selection
- Recommends full HP revert to v0 defaults before iter2
- Gate calibration: appropriate as-is, no changes needed with only 1 data point

## Codex-Specific Insights
- Layer 3 non-regression disabled when champion=null (compare.py:219) — de facto no regression protection
- noise_tolerance=0.02 is loose for AUC (std=0.015) — suggests metric-specific tolerances after more data
- Threshold-selection leakage (HIGH): benchmark evaluates on same split used for threshold tuning
- Dead config `scale_pos_weight_auto` misleads experiment interpretation
- Recommends month-level consistency checks (≥8/12 non-negative deltas) before accepting mean gains

## Synthesis Assessment
- Claude's statistical analysis is compelling — the month-level win/loss counts provide stronger evidence than mean deltas alone
- Codex's structural code findings (leakage, Layer 3 disabled) are valid but deferred to HUMAN_SYNC
- Both reviewers converge on the same fundamental insight: these 14 features have reached their ranking ceiling at AUC ~0.835
- Divergence on HP strategy is minor: Claude says revert fully, Codex says small local steps — reverting is correct since v0 is proven and any local variation has shown zero upside

## Open Code Issues (carried forward)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Layer 3 disabled when champion=null | MEDIUM | Codex (iter1) | Accepted (v0 is reference) |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
