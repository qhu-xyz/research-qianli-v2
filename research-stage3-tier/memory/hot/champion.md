# Champion

**Current champion: v0** (baseline, no iterations yet)

## Key Metrics (12-month means)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Tier-VC@100 | 0.075 | 0.065 | 0.008 | 0.246 |
| Tier-VC@500 | 0.217 | 0.102 | 0.047 | 0.413 |
| Tier-NDCG | 0.767 | 0.067 | 0.629 | 0.858 |
| QWK | 0.359 | 0.080 | 0.184 | 0.485 |
| Macro-F1 | 0.369 | 0.055 | 0.288 | 0.458 |
| Tier-Recall@0 | 0.374 | 0.176 | 0.076 | 0.680 |
| Tier-Recall@1 | 0.098 | 0.058 | 0.026 | 0.197 |

## Critical Observations

- **Tier 4 has 0 samples** in ALL months (no negative shadow prices in real data)
- **Tier-Recall@1 is catastrophically low** (mean 0.098) — missing most strongly binding constraints
- **Tier-VC@100 is very low** (mean 0.075) — top-100 ranking quality poor
- **Worst months**: 2021-11 (QWK=0.184), 2022-06 (NDCG=0.629, VC@500=0.047), 2021-05 (VC@100=0.008)
- **Best month**: 2020-09 (Recall@0=0.680, Recall@1=0.197) and 2021-09 (VC@100=0.246)

## v0 Config

- n_estimators=400, max_depth=5, learning_rate=0.05
- class_weights: {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}
- 34 features, monotone constraints active
- reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
