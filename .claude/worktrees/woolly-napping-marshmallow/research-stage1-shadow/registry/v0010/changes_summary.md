# v0010 Changes Summary — Iter 3 (H12: More Trees + Slower Learning Rate)

## Hypothesis

With 29 features producing 17.13% interaction importance, the model may be under-treed at n_estimators=200 with learning_rate=0.1 (unchanged since v0). Increasing to 300 trees with learning_rate=0.07 maintains similar total gradient magnitude (300×0.07=21 vs 200×0.1=20) while allowing finer-grained splits.

## Changes Made

### `ml/config.py`
- `HyperparamConfig.n_estimators`: 200 → 300
- `HyperparamConfig.learning_rate`: 0.1 → 0.07

### `ml/tests/test_config.py`
- Updated assertions for `n_estimators` (200→300) and `learning_rate` (0.1→0.07)

### What was NOT changed
- All 29 features preserved exactly as-is
- colsample_bytree=0.9, subsample=0.8, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10
- threshold_beta=0.7, train_months=14
- No changes to features.py, data_loader.py, evaluate.py, or gates.json

## Results (12-month benchmark, f0, onpeak)

### Aggregate Comparison (v0009 → v0010)

| Metric | v0009 mean | v0010 mean | Delta | W/L/T |
|--------|-----------|-----------|-------|-------|
| S1-AUC | 0.8495 | 0.8496 | +0.0001 | 6W/5L/1T |
| S1-AP | 0.4445 | 0.4424 | -0.0021 | 6W/6L/0T |
| S1-NDCG | 0.7359 | 0.7359 | +0.0000 | 5W/6L/1T |
| S1-VCAP@100 | 0.0266 | 0.0254 | -0.0012 | 5W/7L/0T |
| S1-BRIER | 0.1376 | 0.1374 | -0.0002 | 6W/5L/1T |
| S1-REC | 0.4280 | 0.4289 | +0.0009 | — |

### Bottom-2 Comparison

| Metric | v0009 bot2 | v0010 bot2 | Delta |
|--------|-----------|-----------|-------|
| S1-AUC | 0.8189 | 0.8172 | -0.0017 |
| S1-AP | 0.3712 | 0.3748 | +0.0036 |
| S1-NDCG | 0.6648 | 0.6685 | +0.0037 |
| S1-VCAP@100 | 0.0089 | 0.0070 | -0.0019 |
| S1-BRIER | 0.1452 | 0.1448 | -0.0004 |

### L3 Floor Check (all PASS)

| Metric | v0010 bot2 | L3 floor | Margin |
|--------|-----------|----------|--------|
| S1-AUC | 0.8172 | 0.7989 | +0.0183 |
| S1-AP | 0.3748 | 0.3512 | +0.0236 |
| S1-VCAP@100 | 0.0070 | -0.0111 | +0.0181 |
| S1-NDCG | 0.6685 | 0.6448 | +0.0237 |

### BRIER Overfitting Check
- BRIER mean: 0.1374 (threshold: 0.140) — **OK, no overfitting signal**

### Per-Month Weaknesses

| Month | Metric | v0009 | v0010 | Delta |
|-------|--------|-------|-------|-------|
| 2022-12 | AUC | 0.8181 | 0.8161 | -0.0020 |
| 2022-12 | AP | 0.4070 | 0.4039 | -0.0031 |
| 2022-12 | NDCG | 0.6991 | 0.6952 | -0.0039 |
| 2022-09 | AP | 0.3482 | 0.3511 | +0.0029 |
| 2021-04 | NDCG | 0.6529 | 0.6519 | -0.0010 |
| 2022-03 | NDCG | 0.6768 | 0.6851 | +0.0083 |

### Feature Importance
- 29 features confirmed (unchanged from v0009)
- Top-5 by gain: hist_da_trend (39.7%), hist_seasonal_band (10.9%), hist_physical_interaction (10.5%), hist_da (6.6%), prob_band_95_100 (4.2%)
- Derived interaction features total: 15.8% (vs 17.13% in v0009 — slight redistribution with more trees)

## Interpretation

**Result: Null — model has reached capacity ceiling.** All metrics are within noise of the champion (v0009). The largest single-metric delta is AP mean at -0.0021, well within the 0.02 noise tolerance. W/L ratios are near 50/50 across all Group A metrics.

The only noteworthy signals:
1. **NDCG bot2 improved by +0.0037** — the tightest L3 constraint got more comfortable
2. **AP bot2 improved by +0.0036** — worst-month ranking quality slightly better
3. **2022-03 NDCG improved by +0.0083** — one of the structurally weak months showed the largest single improvement
4. **2022-12 regressed slightly across all metrics** — but stays well above L3 floors

This confirms the prediction from the direction: with 29 features and strong interaction signal, the model is at its optimization ceiling. More trees with slower learning rate neither helps nor hurts meaningfully. The pipeline is ready for HUMAN_SYNC.
