# Champion

**Current champion: v0** (baseline, no iterations yet)

## Key Metrics (12-month means)

### Group A — Blocking Gates (tier-count invariant)

| Metric | Mean | Std | Min | Max | Bot2 | Floor |
|--------|------|-----|-----|-----|------|-------|
| Tier-VC@100 | 0.0708 | 0.0652 | 0.0029 | 0.2483 | 0.0103 | 0.075 |
| Tier-VC@500 | 0.2296 | 0.1008 | 0.0450 | 0.3806 | 0.0564 | 0.217 |
| Tier0-AP | 0.3062 | 0.1443 | 0.1144 | 0.5942 | 0.1146 | 0.306 |
| Tier01-AP | 0.3110 | 0.0663 | 0.1935 | 0.4003 | 0.1941 | 0.311 |

### Group B — Monitor (no hard gates)

| Metric | Mean | Std | Min | Max | Bot2 |
|--------|------|-----|-----|-----|------|
| Tier-NDCG | 0.7711 | 0.0653 | 0.6430 | 0.8583 | 0.6585 |
| QWK | 0.3698 | 0.0644 | 0.2439 | 0.4526 | 0.2584 |
| Macro-F1 | 0.3560 | 0.0440 | 0.2897 | 0.4216 | 0.2975 |
| Value-QWK | 0.3914 | 0.0971 | 0.1804 | 0.5103 | 0.1934 |
| Tier-Recall@0 | 0.4389 | 0.1522 | 0.1874 | 0.6425 | 0.2041 |
| Tier-Recall@1 | 0.0467 | 0.0501 | 0.0000 | 0.1825 | 0.0030 |

## Critical Observations

- **Tier 4 has 0 samples** in ALL months (no negative shadow prices in real data)
- **Tier0-AP has high variance** (0.114 to 0.594) — worst months are 2022-06 and 2022-09
- **Tier01-AP worst months** 2022-06 (0.195) and 2022-12 (0.194) — late 2022 is hard
- **Tier-Recall@1 is catastrophically low** (mean 0.047) — missing most strongly binding constraints
- **Worst months**: 2021-11 (QWK=0.244, VC@100=0.003), 2022-06 (NDCG=0.643, VC@500=0.045, Tier0-AP=0.114)
- **Best month**: 2020-09 (Tier0-AP=0.594, Recall@0=0.633)
- **Early stopping** effective: models use 87-285 trees out of 400

## v0 Config

- n_estimators=400, max_depth=5, learning_rate=0.05
- class_weights: {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}
- 34 features, monotone constraints active
- reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- early_stopping_rounds=50 on val set
