# Gate Calibration

## v0 Baseline Gates (calibrated 2026-03-04)

All metrics higher-is-better. Floors = v0 mean, tail floors = v0 min (offset=0).

| Gate | Floor | Tail Floor | Group |
|------|-------|------------|-------|
| Tier-VC@100 | 0.075 | 0.008 | A |
| Tier-VC@500 | 0.217 | 0.047 | A |
| Tier-NDCG | 0.767 | 0.629 | A |
| QWK | 0.359 | 0.184 | A |
| Macro-F1 | 0.369 | 0.288 | B |
| Tier-Accuracy | 0.943 | 0.931 | B |
| Adjacent-Accuracy | 0.975 | 0.961 | B |
| Tier-Recall@0 | 0.374 | 0.076 | B |
| Tier-Recall@1 | 0.098 | 0.026 | B |

## Gate System
- Group A (hard/blocking): Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK
- Group B (monitor): Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1
- Three layers: mean quality, tail safety (max 1 failure), tail non-regression (bottom_2_mean)
- noise_tolerance: 0.02
