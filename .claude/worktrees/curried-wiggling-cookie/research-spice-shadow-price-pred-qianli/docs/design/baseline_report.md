# Legacy Baseline Benchmark Report

> **Model ID**: `v000-legacy-20260220`
> **Created**: 2026-02-20
> **Source commit**: `b32bf6b` (original unmodified pipeline)
> **Status**: BASELINE (reference point for all future experiments)

---

## 1. Experiment Summary

### What was tested

The **original shadow price prediction pipeline** with all default settings, no modifications. This establishes the floor that every future model version must beat.

### Key legacy defaults

| Parameter | Value | Note |
|-----------|-------|------|
| Features (step1 + step2) | 8 density features | `hist_da` commented out |
| Seasonal features | None | Not implemented |
| Classifier | XGBoost only | ElasticNet commented out |
| Regressor | XGBoost only | ElasticNet commented out |
| Threshold beta | 0.5 (precision-weighted) | In-sample F0.5 optimization |
| Train/Val split | None | Full 12-month rolling window, no validation set |
| Regression clamp | None | Unclamped predictions |
| Anomaly detection | Enabled | IQR-based, k=3.0 |
| Feature selection | Spearman + AUC consistency | Per-branch monotonicity check |

### Benchmark scope

| Dimension | Values |
|-----------|--------|
| **Planning years** | PY20 (Jun 2020-May 2021), PY21 (Jun 2021-May 2022) |
| **Auction months** | 8 quarterly: 2020-07, 2020-10, 2021-01, 2021-04, 2021-07, 2021-10, 2022-01, 2022-04 |
| **Class types** | onpeak, offpeak |
| **Period types** | f0 (current month), f1 (next month) |
| **Total scored** | **32** (8 months x 2 class x 2 period = complete) |
| **Training window** | Rolling 12-month lookback per auction month |

---

## 2. Hard Gate Baseline Values

These are the legacy numbers that define the **absolute floor** for promotion. A candidate model must exceed these on the same benchmark periods.

### Onpeak

| Gate | Metric | Legacy Baseline | Proposed Floor | Gap |
|------|--------|---------------:|---------------:|----:|
| **S1-AUC** | AUC-ROC | **0.6905** | 0.80 | -0.11 |
| **S1-REC** | Recall | **0.2702** | 0.30 | -0.03 |
| **S2-SPR** | Spearman (TP) | **0.3774** | 0.30 | +0.08 |
| **C-VC@1000** | ValCap@1000 (constraint) | **0.8379** | 0.50 | +0.34 |
| **C-RMSE** | RMSE (all) | **$1,254.52** | ceiling TBD | -- |

### Offpeak

| Gate | Metric | Legacy Baseline | Proposed Floor | Gap |
|------|--------|---------------:|---------------:|----:|
| **S1-AUC** | AUC-ROC | **0.7002** | 0.80 | -0.10 |
| **S1-REC** | Recall | **0.2766** | 0.30 | -0.02 |
| **S2-SPR** | Spearman (TP) | **0.4466** | 0.30 | +0.15 |
| **C-VC@1000** | ValCap@1000 (constraint) | **0.8637** | 0.50 | +0.36 |
| **C-RMSE** | RMSE (all) | **$1,857.06** | ceiling TBD | -- |

### Interpretation

- **S1-AUC and S1-REC are below proposed floors.** The legacy classifier with F0.5 threshold (precision-weighted) sacrifices recall. Switching to F2.0 (recall-weighted) and adding a val-set threshold should close this gap.
- **S2-SPR already passes the floor.** The regressor's ranking ability on true positives is adequate but improvable with `hist_da` + seasonal features.
- **C-VC@1000 is strong.** Even with weak recall, the top-1000 constraints capture 84-86% of total congestion value. This reflects that the model correctly identifies the highest-value constraints even if it misses many lower-value binders.
- **C-RMSE is the ceiling gate** — it should decrease with regression clamping (offpeak has extreme outliers driving RMSE to $1,857).

---

## 3. Stage 1: Classifier Performance

### Overall means (across all 16 periods per class type)

| Metric | Onpeak | Offpeak |
|--------|-------:|--------:|
| AUC-ROC | 0.6905 | 0.7002 |
| AUC-PR (avg_precision) | 0.2160 | 0.2199 |
| Brier score | 0.0963 | 0.0913 |
| Precision | 0.3261 | 0.3286 |
| Recall | 0.2702 | 0.2766 |
| F1 | 0.2943 | 0.2990 |
| F-beta 0.5 (legacy) | 0.3124 | 0.3155 |
| F-beta 2.0 (current) | 0.2793 | 0.2848 |
| Actual binding rate | ~7.2% | ~7.0% |
| Predicted binding rate | ~6.2% | ~6.1% |

### Observations

1. **AUC-ROC ~0.69-0.70**: The model has moderate discriminative ability. It's better than random (0.50) but below the 0.80 floor. The 8 density features provide signal but not enough separation.
2. **Precision > Recall**: By design — the F0.5 threshold optimization trades recall for precision. The model predicts ~6% binding rate vs ~7% actual, under-predicting the number of binders.
3. **Brier score ~0.09**: Reasonable calibration given the ~7% base rate. Not far from the "predict all non-binding" baseline of ~0.07.
4. **Onpeak vs Offpeak are nearly identical**, suggesting the density features work similarly across peak types.

### By planning year

| Class | PY20 AUC | PY21 AUC | Delta |
|-------|----------|----------|-------|
| Onpeak | 0.6895 | 0.6914 | +0.002 |
| Offpeak | 0.6965 | 0.7039 | +0.007 |

Stable across years — no temporal degradation.

### By horizon (f0 vs f1)

| Class | f0 AUC | f1 AUC | Delta |
|-------|--------|--------|-------|
| Onpeak | 0.7073 | 0.6736 | -0.034 |
| Offpeak | 0.7115 | 0.6889 | -0.023 |

**f1 degrades by 2-3% AUC** relative to f0. Expected — predicting further out is harder.

### By season

| Season | Onpeak AUC | Offpeak AUC |
|--------|-----------|------------|
| Spring | **0.7166** | **0.7271** |
| Summer | 0.6842 | 0.7026 |
| Fall | 0.6790 | 0.6976 |
| Winter | 0.6821 | 0.6735 |

**Spring is easiest** (higher AUC, higher recall). **Fall/Winter are hardest.** Winter offpeak is the weakest slice at 0.67 AUC.

---

## 4. Stage 2: Regressor Performance (True Positives)

### Overall means

| Metric | Onpeak | Offpeak |
|--------|-------:|--------:|
| Spearman (TP) | 0.3774 | 0.4466 |
| MAE (TP) | $2,313 | $2,259 |
| RMSE (TP) | $4,975 | $5,070 |
| Bias (TP) | -$1,423 | -$1,463 |
| Residual std (TP) | -- | -- |

### Observations

1. **Spearman 0.38-0.45**: Moderate ranking quality. The model puts high-value constraints above low-value ones ~40% better than random. Room for improvement with `hist_da` which provides direct dollar-scale signal.
2. **Systematic under-prediction (negative bias)**: The mean predicted shadow price is ~$1,400-1,500 below actual. The regressor consistently underestimates, likely because XGBoost regresses toward the mean of the training distribution.
3. **Offpeak Spearman > Onpeak**: Offpeak constraint values may be more predictable from density features. The offpeak spread is 0.45 vs 0.38 — a notable difference.
4. **Large RMSE relative to MAE**: The RMSE/MAE ratio of ~2.1-2.2 indicates heavy-tailed errors (some very large misses).

### By season

| Season | Onpeak Spearman | Offpeak Spearman |
|--------|:---------------:|:----------------:|
| Spring | 0.3802 | 0.4425 |
| Summer | 0.3805 | 0.4275 |
| Fall | 0.3392 | 0.4076 |
| Winter | **0.4098** | **0.5091** |

**Winter has highest Spearman** despite having lowest classifier AUC. The constraints that bind in winter are more predictable in magnitude (fewer, higher-value constraints).

---

## 5. Combined Pipeline (End-to-End)

### Overall means

| Metric | Onpeak | Offpeak |
|--------|-------:|--------:|
| RMSE (all) | $1,255 | $1,857 |
| MAE (all) | $147 | $143 |

### RMSE by season

| Season | Onpeak RMSE | Offpeak RMSE |
|--------|:----------:|:-----------:|
| Spring | $1,227 | $1,217 |
| Summer | $766 | $863 |
| Fall | $1,231 | $1,116 |
| **Winter** | **$1,795** | **$4,231** |

**Winter offpeak RMSE explodes to $4,231** — this is where regression clamping will have the biggest impact. A few extreme outlier predictions dominate the error.

---

## 6. Ranking Metrics (Portfolio-Level)

### Constraint-level ValCap@K (what % of total $ captured in top K constraints)

**Onpeak:**

| K | PY20 | PY21 | f0 | f1 | Overall |
|---|------|------|----|----|---------|
| 50 | -- | -- | -- | -- | variable |
| 100 | ~5% | ~8% | ~8% | ~2% | ~5% |
| 260 | ~35% | ~40% | ~45% | ~30% | ~37% |
| 520 | ~70% | ~68% | ~74% | ~65% | ~70% |
| **1000** | **84%** | **84%** | **86%** | **82%** | **84%** |

**Offpeak:**

| K | Overall |
|---|---------|
| 260 | ~45% |
| 520 | ~74% |
| **1000** | **86%** |

### NDCG

| Level | Onpeak | Offpeak |
|-------|-------:|--------:|
| Outage-level | 0.614 | 0.607 |
| Constraint-level | 0.311 | 0.290 |

### Observations

1. **ValCap@1000 is the standout metric at 84-86%.** Even with mediocre AUC and recall, the model captures the vast majority of congestion value in the top 1000 constraints (out of ~5,400 total). This is because the high-value constraints have very distinctive density profiles.
2. **The jump from K=260 to K=520 is dramatic** (35% -> 70%), suggesting the model's ranking quality is concentrated in the top ~500 constraints.
3. **NDCG is modest** (0.31 constraint-level). There's significant room to improve overall ranking quality.
4. **f0 outperforms f1** at every K level, consistent with the classifier degradation pattern.

---

## 7. Stratified Breakdown Summary

### Dimension: Planning Year (PY20 vs PY21)

| Metric | PY20 | PY21 | Trend |
|--------|------|------|-------|
| AUC-ROC (onpeak) | 0.690 | 0.691 | Stable |
| AUC-ROC (offpeak) | 0.697 | 0.704 | Stable |
| Recall (onpeak) | 0.277 | 0.264 | Slight decline |
| Spearman (onpeak) | 0.409 | 0.346 | **Decline** |
| ValCap@1000 (onpeak) | 0.838 | 0.838 | Stable |

**Key finding**: Classifier is stable across years but regressor Spearman drops in PY21 (onpeak). PY21 has higher shadow prices (post-COVID recovery, Winter Storm Uri effects) creating harder regression targets.

### Dimension: f0 vs f1

| Metric | f0 | f1 | f1 Degradation |
|--------|----|----|----------------|
| AUC-ROC | 0.709 | 0.681 | -4% |
| Recall | 0.286 | 0.263 | -8% |
| Spearman | 0.427 | 0.406 | -5% |
| ValCap@1000 | 0.868 | 0.837 | -4% |

**f1 uniformly degrades across all metrics.** The 4-8% degradation is consistent and expected. Longer horizons have more uncertainty.

### Dimension: Onpeak vs Offpeak

| Metric | Onpeak | Offpeak | Winner |
|--------|--------|---------|--------|
| AUC-ROC | 0.691 | 0.700 | Offpeak |
| Recall | 0.270 | 0.277 | Offpeak |
| Spearman | 0.377 | 0.447 | **Offpeak** |
| RMSE (all) | $1,255 | $1,857 | Onpeak |
| ValCap@1000 | 0.838 | 0.864 | Offpeak |

**Offpeak consistently outperforms onpeak** on discrimination and ranking, but has worse RMSE due to extreme outliers (Winter 2022).

### Dimension: Season

**Best season**: Spring (highest AUC, highest recall, strong Spearman)
**Worst season**: Winter (lowest AUC, lowest recall for offpeak, highest RMSE)
**Fall**: Weakest Spearman
**Summer**: Lowest shadow prices, easiest RMSE

---

## 8. Model Type Breakdown

The legacy pipeline uses **branch-specific models** for constraints with sufficient training data, falling back to a **default model** for rare branches.

- Default model fallback rate: 0% (all branches had sufficient data)
- Branch-specific models trained: ~700-750 per auction month
- Total constraint universe: ~5,400 per period

---

## 9. Known Weaknesses (Improvement Roadmap)

| # | Weakness | Evidence | Fix | Expected Gate Impact |
|---|----------|----------|-----|---------------------|
| W1 | Low AUC-ROC (~0.69) | 8 density features insufficient | Add `hist_da`, seasonal features | S1-AUC +0.05-0.10 |
| W2 | Low Recall (~0.27) | F0.5 threshold sacrifices recall | Switch to F2.0 + val-set threshold | S1-REC +0.10-0.20 |
| W3 | Systematic under-prediction | Bias = -$1,400 on TPs | Regression clamp, `hist_da` feature | S2-SPR +0.05, C-RMSE down |
| W4 | Offpeak RMSE explosion | Winter offpeak RMSE=$4,231 | Regression clamp at percentile | C-RMSE -50% |
| W5 | f1 degradation | 4-8% across all metrics | Horizon-aware features | Modest |
| W6 | Low constraint NDCG (0.30) | Poor fine-grained ranking | Better probability calibration | C-VC@K up |

---

## 10. Files and Artifacts

| Artifact | Path |
|----------|------|
| Scoring function | `src/shadow_price_prediction/evaluation.py::score_results_df()` |
| Per-run parquets (32) | `/opt/temp/tmp/pw_data/spice6/legacy_baseline/results_*.parquet` |
| Aggregated CSV | `registry/legacy_baseline_agg.csv` |
| Full JSON (all metrics) | `registry/legacy_baseline.json` |
| Legacy source code | `_legacy/src/shadow_price_prediction/` (from git `b32bf6b`) |
| Orchestrator script | `notebook/run_legacy_baseline.py` |
| Worker script | `notebook/_legacy_worker.py` |

---

## 11. Reproducibility

```bash
# Activate pmodel venv
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# Re-run full benchmark (skips existing parquets)
PYTHONPATH=/.../src:$PYTHONPATH python /.../notebook/run_legacy_baseline.py --mode full

# Re-score only (no pipeline runs)
PYTHONPATH=/.../src:$PYTHONPATH python /.../notebook/run_legacy_baseline.py --mode score
```

All 32 parquets are deterministic given the same data vintage. The legacy code is frozen at commit `b32bf6b`.
