# Champion

**Current champion: v0** (baseline, no successful promotions yet)
**Best candidate: v0005** (iter1) — NOT promoted, Tier-VC@100 fails L1 by 0.0004

## Key Metrics (12-month means)

### Group A — Blocking Gates (tier-count invariant)

| Metric | v0 Mean | v0005 Mean | Delta | Floor | Status |
|--------|---------|------------|-------|-------|--------|
| Tier-VC@100 | 0.0708 | 0.0746 | +0.0038 | 0.075 | **FAIL L1** |
| Tier-VC@500 | 0.2296 | 0.2329 | +0.0033 | 0.217 | PASS |
| Tier0-AP | 0.3062 | 0.3126 | +0.0064 | 0.306 | PASS |
| Tier01-AP | 0.3110 | 0.3132 | +0.0022 | 0.311 | PASS |

### Group B — Monitor (no hard gates)

| Metric | v0 Mean | v0005 Mean | Delta |
|--------|---------|------------|-------|
| Tier-NDCG | 0.7711 | 0.7751 | +0.0040 |
| QWK | 0.3698 | 0.3706 | +0.0008 |
| Macro-F1 | 0.3560 | 0.3560 | +0.0000 |
| Value-QWK | 0.3914 | 0.3918 | +0.0004 |
| Tier-Recall@0 | 0.4389 | 0.4403 | +0.0014 |
| Tier-Recall@1 | 0.0467 | 0.0447 | -0.0020 |

## Critical Observations

- **Tier 4 has 0 samples** in ALL months (no negative shadow prices in real data)
- **Tier-VC@100 gap is tiny** (0.0004) — log transforms in iter2 should close it
- **Tier-Recall@1 is catastrophically low** (0.045) — structural, FE cannot fix
- **Value-QWK barely passing** (0.3918 vs 0.3914) — monitor closely
- **Worst months**: 2021-11 (VC@100=0.008), 2022-06 (NDCG=0.636, VC@500=0.048, Tier0-AP=0.106)
- **Best month**: 2020-09 (Tier0-AP=0.605, Recall@0=0.633)

## v0 Config

- n_estimators=400, max_depth=5, learning_rate=0.05
- class_weights: {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}
- 34 features, monotone constraints active
- reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- early_stopping_rounds=50 on val set

## v0005 Config

- Same hyperparams as v0
- 37 features (+3 interaction: overload_x_hist, prob110_x_recent_hist, tail_x_hist)
- Monotone constraints: all 3 new features = +1
