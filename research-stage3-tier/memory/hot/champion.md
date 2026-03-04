# Champion

**Current champion: v0** (baseline, no iterations yet)

## Key Metrics (12-month means)

| Metric | Mean | Std | Min | Max | Bot2 |
|--------|------|-----|-----|-----|------|
| Tier-VC@100 | 0.0708 | 0.0652 | 0.0029 | 0.2483 | 0.0103 |
| Tier-VC@500 | 0.2296 | 0.1008 | 0.0450 | 0.3806 | 0.0564 |
| Tier-NDCG | 0.7711 | 0.0653 | 0.6430 | 0.8583 | 0.6585 |
| QWK | 0.3698 | 0.0644 | 0.2439 | 0.4526 | 0.2584 |
| Macro-F1 | 0.3560 | 0.0440 | 0.2897 | 0.4216 | 0.2975 |
| Tier-Accuracy | 0.9460 | 0.0082 | 0.9304 | 0.9590 | 0.9338 |
| Adjacent-Accuracy | 0.9759 | 0.0081 | 0.9597 | 0.9857 | 0.9621 |
| Tier-Recall@0 | 0.4389 | 0.1522 | 0.1874 | 0.6425 | 0.2041 |
| Tier-Recall@1 | 0.0467 | 0.0501 | 0.0000 | 0.1825 | 0.0030 |
| Value-QWK | 0.3914 | 0.0971 | 0.1804 | 0.5103 | 0.1934 |

## Critical Observations

- **Tier 4 has 0 samples** in ALL months (no negative shadow prices in real data)
- **Tier-Recall@1 is catastrophically low** (mean 0.047) — missing most strongly binding constraints
- **Tier-VC@100 is very low** (mean 0.071) — top-100 ranking quality poor
- **Worst months**: 2021-11 (QWK=0.244, VC@100=0.003, Recall@1=0.000), 2022-06 (NDCG=0.643, VC@500=0.045)
- **Best month**: 2021-09 (VC@100=0.248, QWK=0.449) and 2020-09 (Recall@0=0.633)
- **Early stopping** effective: models use 87-285 trees out of 400

## v0 Config

- n_estimators=400, max_depth=5, learning_rate=0.05
- class_weights: {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}
- 34 features, monotone constraints active
- reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- early_stopping_rounds=50 on val set
