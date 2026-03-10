# v0008 Changes Summary — Iter 1 (feat-eng-3)

## Hypothesis H10: Distribution Shape + Near-Boundary Band + Seasonal Historical Features

### What Changed

Added 7 new source-loader features (19 → 26 total features), expanding the model's input beyond shift factors and constraint metadata:

| Feature | Monotone | Category | Avg Importance |
|---------|----------|----------|----------------|
| `density_mean` | 1 | Distribution shape | 0.81% |
| `density_variance` | 0 | Distribution shape | 0.91% |
| `density_entropy` | 0 | Distribution shape | 0.72% |
| `tail_concentration` | 1 | Near-boundary band | 0.37% |
| `prob_band_95_100` | 1 | Near-boundary band | **3.82%** |
| `prob_band_100_105` | 1 | Near-boundary band | 1.07% |
| `hist_da_max_season` | 1 | Historical enrichment | **2.60%** |

**Total new feature importance: ~10.3%** — significantly more than v0007's shift factors (4.66% combined).

### Files Modified

1. **`ml/config.py`** — Added 7 features to `FeatureConfig.step1_features` with monotone constraints
2. **`ml/features.py`** — Expanded `source_features` set to include all 13 source-loader features
3. **`ml/data_loader.py`** — Updated `_load_smoke()` with specialized synthetic generators for 7 new features; updated `_load_real()` diagnostic check to include new columns
4. **`ml/tests/test_config.py`** — Updated feature count (19 → 26), monotone constraint string, and expected feature name list

### What Did NOT Change

- Hyperparameters: kept v0 defaults (n_estimators=200, max_depth=4, etc.)
- train_months: kept 14
- threshold_beta: kept 0.7
- All existing 19 features retained
- gates.json and evaluate.py untouched

### Results (12-month rolling benchmark, f0, onpeak)

#### Group A Metrics (blocking)

| Metric | v0007 (champion) | v0008 | Delta | W/L |
|--------|-----------------|-------|-------|-----|
| S1-AUC mean | 0.8485 | 0.8498 | +0.0013 | — |
| S1-AP mean | 0.4391 | 0.4418 | +0.0027 | — |
| S1-NDCG mean | 0.7333 | 0.7346 | +0.0013 | — |
| S1-VCAP@100 mean | 0.0247 | 0.0240 | -0.0007 | — |

#### Bot2 (worst-2-month average)

| Metric | v0007 bot2 | v0008 bot2 | Delta | L3 Floor | Margin |
|--------|-----------|-----------|-------|----------|--------|
| S1-AUC | 0.8188 | 0.8199 | +0.0011 | 0.7988 | 0.0211 |
| S1-AP | 0.3685 | 0.3726 | +0.0041 | 0.3485 | 0.0241 |
| S1-NDCG | 0.6562 | **0.6663** | **+0.0101** | 0.6362 | **0.0301** |
| S1-VCAP@100 | 0.0094 | 0.0061 | -0.0033 | -0.0106 | 0.0167 |

#### Group B Metrics (monitor)

| Metric | v0007 | v0008 | Delta |
|--------|-------|-------|-------|
| S1-BRIER mean | 0.1395 | 0.1383 | -0.0012 (better) |
| S1-REC mean | 0.4237 | 0.4228 | -0.0009 |
| Precision mean | 0.5020 | 0.5091 | +0.0071 |

### Key Findings

1. **NDCG bot2 improved by +0.0101** — the primary target. Margin to L3 floor expanded from 0.0046 (dangerously tight) to 0.0301 (comfortable). The near-boundary band features (prob_band_95_100, prob_band_100_105) help discriminate binding intensity among binders.

2. **prob_band_95_100 is the #5 feature overall (3.82%)** — the highest-importance new feature. It captures mass in the near-binding band, directly discriminating "on the edge" constraints.

3. **hist_da_max_season is #7 overall (2.60%)** — captures extreme historical events that the mean (hist_da) misses.

4. **2022-09 continued to improve**: AUC 0.853→0.857, AP maintained at 0.348. The structurally difficult month benefits from richer features.

5. **Precision improved (+0.007)** while recall held steady — the new features improve discrimination quality without changing the threshold profile.

6. **All L3 floors pass** with comfortable margins.
