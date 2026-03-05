# Gate Calibration

## v0 Baseline Gates (calibrated 2026-03-04)

All metrics higher-is-better. Floors = v0 mean, tail floors = v0 min (offset=0).

| Gate | Floor | Tail Floor | Group |
|------|-------|------------|-------|
| Tier-VC@100 | 0.075 | 0.008 | A |
| Tier-VC@500 | 0.217 | 0.047 | A |
| Tier0-AP | 0.306 | 0.114 | A |
| Tier01-AP | 0.311 | 0.193 | A |
| Tier-NDCG | 0.767 | 0.629 | B |
| QWK | 0.359 | 0.184 | B |
| Macro-F1 | 0.369 | 0.288 | B |
| Value-QWK | 0.391 | 0.180 | B |
| Tier-Recall@0 | 0.374 | 0.076 | B |
| Tier-Recall@1 | 0.098 | 0.026 | B |

## Gate System
- Group A (hard/blocking): Tier-VC@100, Tier-VC@500, Tier0-AP, Tier01-AP
- Group B (monitor): Tier-NDCG, QWK, Macro-F1, Value-QWK, Tier-Recall@0, Tier-Recall@1
- Three layers: mean quality, tail safety (max 1 failure), tail non-regression (bottom_2_mean)
- noise_tolerance: 0.02

## Calibration Observations (from iter1 results)

### noise_tolerance may be too loose for small-scale metrics
- Codex review flagged: noise_tolerance=0.02 is a global absolute value
- For Tier-Recall@1 (champion bottom_2=0.003), tolerance of 0.02 effectively guarantees L3 pass even under major degradation
- For Tier-VC@100 (champion bottom_2=0.010), tolerance is ~2x the metric value — quite loose
- **Recommendation for future**: Consider metric-scaled tolerance (e.g., `min(0.01, 0.5 * champion_bottom_2)` or a relative fraction)
- **Action**: Do NOT change during this FE-only batch (gates.json is frozen). Flag for next non-FE batch.

### Tier-VC@100 floor is borderline appropriate
- v0005 missed by 0.0004 (0.6%). This is within measurement noise for 12 months.
- However, the gate IS doing its job — it's saying "show me you're actually better on the money metric"
- Keep floor unchanged. If iter2 doesn't cross it, FE alone may be insufficient.

### Value-QWK is fragile
- v0005: 0.3918 vs floor 0.3914 — passing by 0.0004
- Any change that hurts value-weighted ordinal consistency could flip this gate
- Monitor closely in iter2
