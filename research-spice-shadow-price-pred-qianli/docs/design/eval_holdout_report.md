# Train/Val/Holdout Evaluation Report

**Date**: 2026-02-18
**Experiment**: F0 smoke test, auction month 2025-01, MISO
**Config**: Signal 6.7B v2, XGB+LR classifiers, XGB+ElasticNet regressors
**Data split**: 6 mo fit / 3 mo val / 3 mo holdout (2024-01 through 2024-11)

---

## 1. Experiment Setup

### Run 3: +seasonal features, +constraint-level metrics, +feature selection diagnostics (current)

| Parameter | Value |
|-----------|-------|
| Auction month | 2025-01 |
| Period type | f0 |
| Class types | onpeak, offpeak |
| Fit period | 2024-01 to 2024-06 (786,659 rows) |
| Val period | 2024-07 to 2024-09 (403,671 rows) |
| Holdout period | 2024-10 to 2024-12 (257,386 rows) |
| Label modification | `prob_exceed_90 < 1e-5` => label=0 (831K labels modified) |
| Threshold optimized (onpeak) | 0.524 (F-beta=0.497, beta=2.0) |
| Threshold optimized (offpeak) | 0.649 (F-beta=0.522, beta=2.0) |
| Branch models (onpeak) | 615 clf / 641 reg (of 2,782 test branches) |
| Branch models (offpeak) | 557 clf / 558 reg (of 2,782 test branches) |
| Never-binding branches | 3,756 (onpeak) / 3,832 (offpeak) |

**Changes vs Run 2:**
- `season_hist_da_1`, `season_hist_da_2`, `season_hist_da_3` added to Step 2 (regression) features
- Constraint-level top-K metrics added to evaluation
- Feature selection verbose diagnostics added
- Defensive handling for missing seasonal feature columns

**Key discovery**: `hist_da` was **KEPT** by feature selection in this run (AUC=0.835-0.851, Spearman=0.292-0.304). `density_skewness` was **DROPPED** instead (AUC=0.431-0.440, Spearman=-0.047 to -0.052 — fails both checks with constraint=+1). This differs from Run 2 where `hist_da` was dropped — the difference may be due to BUG-3 fix changing the training data subset.

---

## 2. Feature Selection Diagnostics (Classifier, Default Model)

### Onpeak

| Feature | Constraint | AUC | Spearman | Status |
|---------|-----------|-----|----------|--------|
| prob_exceed_110 | +1 | 0.847 | 0.277 | KEEP |
| prob_exceed_105 | +1 | 0.851 | 0.278 | KEEP |
| prob_exceed_100 | +1 | 0.856 | 0.281 | KEEP |
| prob_exceed_95 | +1 | 0.860 | 0.283 | KEEP |
| prob_exceed_90 | +1 | 0.863 | 0.285 | KEEP |
| prob_below_95 | -1 | 0.141 | -0.303 | KEEP |
| prob_below_90 | -1 | 0.137 | -0.300 | KEEP |
| **density_skewness** | +1 | **0.440** | **-0.047** | **DROP** |
| hist_da | +1 | 0.835 | 0.292 | KEEP |

### Offpeak

| Feature | Constraint | AUC | Spearman | Status |
|---------|-----------|-----|----------|--------|
| prob_exceed_110 | +1 | 0.850 | 0.270 | KEEP |
| prob_exceed_105 | +1 | 0.853 | 0.271 | KEEP |
| prob_exceed_100 | +1 | 0.858 | 0.274 | KEEP |
| prob_exceed_95 | +1 | 0.862 | 0.276 | KEEP |
| prob_exceed_90 | +1 | 0.864 | 0.277 | KEEP |
| prob_below_95 | -1 | 0.138 | -0.295 | KEEP |
| prob_below_90 | -1 | 0.136 | -0.292 | KEEP |
| **density_skewness** | +1 | **0.431** | **-0.052** | **DROP** |
| hist_da | +1 | 0.851 | 0.304 | KEEP |

**`density_skewness` was dropped because:**
- Expected constraint=+1 (higher skewness → more binding), but both AUC (<0.5) and Spearman (negative) show the **opposite** relationship
- AUC=0.431-0.440 < threshold 0.5 → fails AUC check
- Spearman=-0.047 to -0.052 < threshold 0.0 → fails Spearman check
- Method="both" requires passing both checks → DROP

---

## 3. Classification Metrics

| Split | Class | Threshold | Prec | Recall | F1 | F2 | AUC-ROC | AUC-PR |
|-------|-------|-----------|------|--------|-----|-----|---------|--------|
| **Val** | onpeak | 0.524 | 0.296 | **0.783** | 0.430 | 0.589 | 0.846 | 0.216 |
| **Holdout** | onpeak | 0.524 | 0.313 | **0.561** | 0.401 | 0.484 | 0.845 | 0.276 |
| **Val** | offpeak | 0.649 | 0.327 | **0.781** | 0.461 | 0.612 | 0.857 | 0.239 |
| **Holdout** | offpeak | 0.649 | 0.340 | **0.602** | 0.434 | 0.521 | 0.861 | 0.281 |

Classification metrics are unchanged from Run 2 — the seasonal features were only added to the regressor, and `hist_da` vs `density_skewness` swap had negligible impact.

---

## 4. Regression Metrics (True Positives Only)

| Split | Class | n_TP | MAE ($) | RMSE ($) | Spearman |
|-------|-------|------|---------|----------|----------|
| **Val** | onpeak | 13,874 | 842 | 1,789 | 0.267 |
| **Holdout** | onpeak | 8,291 | 1,091 | 2,282 | **0.484** |
| **Val** | offpeak | 13,743 | 695 | 1,603 | 0.386 |
| **Holdout** | offpeak | 8,398 | 1,913 | **12,034** | **0.552** |

### Comparison across runs

| Metric | Run 1 | Run 2 | Run 3 | Change (2→3) |
|--------|-------|-------|-------|-------------|
| Offpeak holdout RMSE | $90,911 | $12,034 | $12,034 | 0% |
| Onpeak holdout Spearman | 0.462 | 0.476 | **0.484** | **+2%** |
| Offpeak holdout Spearman | 0.521 | 0.540 | **0.552** | **+2%** |

The seasonal features (`season_hist_da_1/2/3`) provided a small but consistent improvement in Spearman ranking (+2% on both classes).

---

## 5. Top-K Ranking Metrics (per-outage level)

### Holdout — Onpeak

| K | Prec@K | Rec@K | Lift | ValCap@K | MeanVal@K |
|---|--------|-------|------|----------|-----------|
| 100 | 21.0% | 0.1% | 3.7x | 0.0% | $65 |
| 250 | 20.8% | 0.4% | 3.6x | 0.1% | $70 |
| 500 | 22.4% | 0.8% | 3.9x | 0.3% | $104 |
| 1,000 | 23.1% | 1.6% | **4.0x** | 0.9% | $138 |
| 2,000 | 21.2% | 2.9% | 3.7x | 1.8% | $139 |

### Holdout — Offpeak

| K | Prec@K | Rec@K | Lift | ValCap@K | MeanVal@K |
|---|--------|-------|------|----------|-----------|
| 100 | 25.0% | 0.2% | **4.6x** | 0.1% | $169 |
| 250 | 25.2% | 0.4% | 4.7x | 0.2% | $135 |
| 500 | 22.8% | 0.8% | 4.2x | 0.5% | $142 |
| 1,000 | 21.4% | 1.5% | 4.0x | 0.8% | $125 |
| 2,000 | 21.2% | 3.0% | 3.9x | 1.7% | $132 |

Per-outage NDCG: 0.60-0.62 across all splits.

---

## 6. Constraint-Level Top-K Metrics (NEW — IMP-14)

These metrics aggregate per-outage predictions to the constraint level (by `branch_name` + `flow_direction`) before computing top-K. This matches how the signal is consumed: a portfolio manager evaluates ~1,300 constraints, not ~400,000 per-outage rows.

### Holdout — Onpeak (5,771 constraints, 579 binding, 10.0% rate)

| K | Prec@K | Rec@K | Lift | ValCap@K | MeanVal@K |
|---|--------|-------|------|----------|-----------|
| 50 | 10.0% | 0.9% | 1.0x | 0.2% | $535 |
| 100 | 7.0% | 1.2% | 0.7x | 0.2% | $275 |
| 260 | 13.9% | 6.2% | 1.4x | 2.1% | $1,205 |
| 520 | 17.9% | 16.1% | 1.8x | **17.2%** | $5,061 |
| 1,000 | **30.0%** | **51.8%** | **3.0x** | **74.3%** | $11,368 |

### Holdout — Offpeak (5,769 constraints, 503 binding, 8.7% rate)

| K | Prec@K | Rec@K | Lift | ValCap@K | MeanVal@K |
|---|--------|-------|------|----------|-----------|
| 50 | 14.0% | 1.4% | 1.6x | 0.2% | $554 |
| 100 | 13.0% | 2.6% | 1.5x | 0.5% | $792 |
| 260 | 13.9% | 7.2% | 1.6x | 2.0% | $1,174 |
| 520 | 16.9% | 17.5% | 1.9x | **19.6%** | $5,830 |
| 1,000 | **28.1%** | **55.9%** | **3.2x** | **76.7%** | $11,856 |

Constraint NDCG: 0.27-0.34 across all splits.

### Interpretation

The constraint-level view reveals a fundamentally different picture from per-outage:

1. **Top 1,000 constraints capture 74-77% of total congestion value** — this is strong. A portfolio focused on the top 1K (out of ~5,800) would capture most of the value.

2. **Lift builds with K**: At K=1000, Lift=3.0-3.2x. At K=50-100, Lift is near random (0.7-1.6x). The model's constraint-level signal is moderate in concentration but excellent in value capture at medium K.

3. **Value is concentrated**: MeanVal@1000 = $11-12K per constraint, vs MeanVal@50 = $500-800. The model doesn't reliably identify the single most valuable constraints, but the top 1K batch is extremely productive.

4. **Constraint NDCG (0.27-0.34) is much lower than per-outage NDCG (0.60-0.62)** — this means the per-outage ranking looks good because it correctly ranks easy non-binding rows, but the constraint-level ranking (where it matters) is weaker.

---

## 7. Key Findings (Run 3)

### What improved
- **Spearman ranking**: +2% on both classes from seasonal features (0.476→0.484 onpeak, 0.540→0.552 offpeak)
- **hist_da now included in classifier**: Feature selection kept it (AUC=0.835-0.851, well above 0.5 threshold)
- **Constraint-level metrics available**: First quantitative view of signal quality at the portfolio construction level

### What was discovered
- **`density_skewness` has wrong-sign relationship**: AUC<0.5, Spearman<0 — it should either have constraint=-1 or be removed from step1_features
- **Constraint-level value capture is strong**: Top 1K constraints capture 74-77% of total congestion value
- **Constraint-level NDCG is low** (0.27-0.34): The model ranks constraints poorly at the very top but captures value well at medium K

### What still needs improvement
- **Top-of-ranking quality at constraint level**: Lift@50-100 near random — need better top-of-funnel ranking
- **density_skewness**: Should be investigated — maybe constraint should be -1 or 0 instead of +1
- **Multi-month evaluation**: All results from single auction month (2025-01) — need confidence intervals
- **Probability calibration**: Platt scaling or isotonic regression

### Next steps
1. Fix `density_skewness` constraint (currently +1, should be -1 or removed)
2. Multi-month experiment for confidence intervals
3. Probability calibration (Platt scaling / isotonic regression)
4. Tier lift table on holdout data
5. Investigate constraint-level top-of-ranking quality
