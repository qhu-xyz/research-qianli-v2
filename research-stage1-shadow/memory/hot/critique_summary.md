# Critique Summary — Iteration 1 (hp-tune-20260302-144146, v0002)

## Reviewer Agreement (high confidence)
1. **Do not promote v0002** — no meaningful Group A improvement; statistically neutral
2. **AUC ceiling confirmed at ~0.835** — two independent levers (HP tuning, interactions) both failed to move it
3. **Clean code implementation** — no new bugs introduced; feature computation correct and idiomatic
4. **Late-2022 weakness persists** — 2022-09 (AP=0.314), 2022-12 (AUC=0.809, VCAP@100=0.000)
5. **Next direction: temporal/seasonal features or longer training windows** — both recommend new approaches beyond current feature set
6. **Threshold methodology debt remains** — leakage (HIGH) and `>` vs `>=` (MEDIUM) are pre-existing

## Claude-Specific Insights
- Deep statistical analysis: AUC 5W/6L/1T is noise; 2021-01 NDCG outlier (+0.042) drives mean improvement
- Bottom-2 regressed on 3/4 Group A metrics — gains concentrated in middle, not tails
- VCAP@500/VCAP@1000 regression: interactions help top-100 but hurt broader ranking
- Recommends longer training windows (12-14 months) as simpler lever before new features
- Also suggests constraint-level features and cross-constraint context as longer-term options
- Gate calibration appropriate — VCAP@100 effectively non-binding but premature to tighten

## Codex-Specific Insights
- Treats iteration as statistically neutral; emphasizes not robustly positive
- New finding (MEDIUM): missing schema guard for interaction feature base columns
- Layer 3 tolerance (0.02) very loose relative to observed bottom_2 deltas (0.001-0.002)
- Recommends seasonality/temporal-shift features targeting 2022-09 and 2022-12

## Synthesis Assessment
- Both reviewers independently reach identical conclusion: don't promote, AUC ceiling is real
- Claude provides stronger statistical reasoning; Codex adds actionable code improvements
- Key convergent insight: within-feature-set engineering cannot break through
- Late-2022 pattern (early months improve, late months don't) is the strongest signal for next focus

## Open Code Issues (carried forward)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Missing schema guard for interactions | MEDIUM | Codex (iter1/v0002) | Include in iter2 |
| Layer 3 disabled when champion=null | MEDIUM | Codex (iter1/v0003) | Accepted (v0 is reference) |
| Stale docstrings (14→17) | LOW | Both (iter1/v0002) | Include in iter2 |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
