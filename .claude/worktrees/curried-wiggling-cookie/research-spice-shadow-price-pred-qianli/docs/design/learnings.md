# Learnings: Evaluation Metrics for Shadow Price Prediction

**Last updated**: 2026-02-18

---

## 1. Bug Found: NaN Poisoning in Threshold Optimizer

### The bug

`find_optimal_threshold()` in `models.py` computes F-beta across the precision-recall curve:

```python
fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
optimal_idx = np.argmax(fbeta)
```

When `p + r = 0` (which happens at very high thresholds where the model predicts nothing), the division produces `NaN`. `np.argmax` on an array containing `NaN` has **undefined behavior** — in practice it returns the index of the first `NaN`, which corresponds to the highest threshold on the curve.

### Impact

With `threshold_beta=2.0` (recall-favoring), the optimizer should find a **low** threshold (~0.5) to maximize recall. Instead, it returned **0.930** — a near-maximum threshold that aggressively suppressed recall.

| Metric | Before fix (threshold=0.93) | After fix (threshold=0.51) | Change |
|--------|---------------------------|---------------------------|--------|
| **Recall (val)** | 0.382 | **0.743** | +94% |
| **F2 (val)** | 0.337 | **0.555** | +65% |
| **F1 (val)** | 0.287 | **0.403** | +40% |
| **Precision (val)** | 0.230 | **0.276** | +20% |
| **Recall (holdout)** | 0.381 | **0.567** | +49% |
| **F2 (holdout)** | 0.358 | **0.483** | +35% |
| **F1 (holdout)** | 0.329 | **0.395** | +20% |
| **TPs (holdout)** | 5,640 | **8,383** | +49% |

Precision **also improved** because the lower threshold captures enough additional true positives to offset the additional false positives.

### Fix

One line:

```python
fbeta = np.nan_to_num(fbeta, nan=0.0)  # before argmax
```

### Lesson

Always sanitize metric arrays before `argmax`/`argmin`. NaN comparisons in numpy are silently undefined — they don't raise errors, they just return wrong answers.

---

## 2. Why Threshold-Based Metrics Are Unreliable Here

### Class imbalance breaks threshold optimization

With 4-6% binding rate, the precision-recall curve has pathological properties:

1. **Sawtooth pattern**: Small threshold changes toggle hundreds of predictions, causing wild P/R oscillations.
2. **Degenerate regions**: At high thresholds, both precision and recall collapse to 0/0 (the NaN bug above).
3. **F-score surface is non-smooth**: Multiple local optima; the "optimal" threshold is sensitive to the specific validation set.
4. **Threshold doesn't transfer**: A threshold optimized on validation data may not be optimal on test data because the score distribution shifts.

### AUC-ROC is deceptive for imbalanced data

Our AUC-ROC is 0.844-0.861. This looks strong, but AUC-ROC evaluates the **entire** FPR range. With ~95% negatives, the true negative count is huge, making FPR insensitive. A model that poorly ranks the top 1% can still have excellent AUC-ROC because it correctly handles the easy 90%.

### AUC-PR is more honest but not actionable

AUC-PR of 0.21-0.28 (vs 0.04-0.06 random baseline) shows the model has real signal. But AUC-PR is a single number averaging over all thresholds — it doesn't tell you how the model performs at *your* operating point.

### What to use instead

**Rank-based, threshold-free metrics** that evaluate what actually matters for portfolio construction: "If I pick the top K constraints, how much value do I capture?"

---

## 3. Recommended Metrics for FTR Signal Evaluation

### 3.1 Value Capture@K (Primary economic metric)

```
ValueCapture@K = sum(actual_shadow_price[top-K]) / sum(actual_shadow_price[all])
```

Directly answers: "If I allocate capital to the top-K constraints from this model, what fraction of total congestion rent do I capture?"

- **K = portfolio size** (capital-constrained). For our signal with ~1300 constraints in 5 tiers, K = 260 (tier 1) and K = 520 (tiers 1-2) are natural choices.
- **Random baseline**: K/N (e.g., 260/1300 = 20%)
- **Perfect foresight**: The maximum achievable VC@K given the value distribution.

### 3.2 NDCG (Ranking quality)

```
DCG@K  = sum_{i=1}^{K} relevance_i / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

With `relevance_i = actual_shadow_price`, NDCG captures whether high-value constraints are ranked above low-value ones. Unlike Value Capture, it also penalizes poor ordering *within* the top-K.

### 3.3 Precision@K and Lift@K (Binary detection quality)

```
Precision@K = |{top-K} ∩ {actual binding}| / K
Lift@K      = Precision@K / base_binding_rate
```

Answers: "Of the top-K ranked constraints, how many actually bind?" Lift normalizes by the base rate so it's comparable across different datasets. Lift@K = 5x means the top-K is 5x more concentrated with binding constraints than a random selection.

### 3.4 Spearman Rank Correlation (Magnitude ordering among binding)

```
rho_s = SpearmanCorr(predicted_score[binding], actual_SP[binding])
```

Computed only on actually-binding constraints. Captures whether the model correctly ranks binding constraints by severity — important for position sizing.

### 3.5 Tier Lift Table (Operational diagnostic)

Since the signal groups constraints into 5 tiers, compute per-tier:

| Tier | Binding Rate | Lift | Mean SP (binding) | Value Share |
|------|-------------|------|-------------------|-------------|

**Requirements for a useful model:**
- Binding rate monotonically decreasing across tiers
- Tier 1 should capture >50% of total congestion value
- Tier 5 should have near-zero binding rate

This is the table a portfolio manager reviews — it maps directly to trading decisions.

### Metrics NOT recommended

| Metric | Why not |
|--------|---------|
| F1 at fixed threshold | Threshold-dependent, unstable, NaN-prone on imbalanced data |
| AUC-ROC | Too forgiving — evaluates the irrelevant FPR>5% region |
| MAE/RMSE on all constraints | Dominated by the 95% of non-binding rows where both actual and predicted are ~0 |
| Brier Score | Measures calibration; irrelevant when signal is used for ranking, not probability |
| KS statistic | Single-point summary; the tier table is strictly more informative |

---

## 4. Current Results (2025-01 f0, after NaN fix)

### Classification

| Split | Class | Threshold | Prec | Recall | F1 | F2 | AUC-ROC | AUC-PR |
|-------|-------|-----------|------|--------|-----|-----|---------|--------|
| Val | onpeak | 0.513 | 0.276 | 0.743 | 0.403 | 0.555 | 0.845 | 0.208 |
| Holdout | onpeak | 0.513 | 0.304 | 0.567 | 0.395 | 0.483 | 0.844 | 0.268 |
| Val | offpeak | 0.507 | 0.295 | 0.759 | 0.424 | 0.577 | 0.857 | 0.234 |
| Holdout | offpeak | 0.507 | 0.325 | 0.627 | 0.428 | 0.529 | 0.861 | 0.275 |

- AUC-ROC and AUC-PR are unchanged from before the fix (they're threshold-independent).
- Recall nearly doubled with the corrected threshold.
- Val-to-holdout drop in recall (0.74→0.57 onpeak) is expected — val was used for threshold optimization.

### Regression on TPs

| Split | Class | n_TP | MAE ($) | RMSE ($) | Spearman |
|-------|-------|------|---------|----------|----------|
| Val | onpeak | 13,161 | 846 | 1,805 | 0.261 |
| Holdout | onpeak | 8,383 | 1,085 | 2,306 | 0.462 |
| Val | offpeak | 13,354 | 720 | 1,648 | 0.397 |
| Holdout | offpeak | 8,748 | 6,979 | 90,911 | 0.521 |

- More TPs now (13K vs 6-7K before) because the lower threshold catches more binding constraints.
- Spearman is positive (0.26-0.52): the model has real ranking signal among TPs.
- Offpeak holdout RMSE ($90K) inflated by outlier predictions — confirms the need for clamping `expm1()`.

### Top-K Ranking (per-outage level)

| Split | K | Prec@K | Lift | ValCap@K | MeanVal@K |
|-------|---|--------|------|----------|-----------|
| **Holdout onpeak** | 100 | 32.0% | 5.6x | 0.2% | $256 |
| | 500 | 20.4% | 3.5x | 0.5% | $148 |
| | 1,000 | 19.3% | 3.4x | 0.9% | $129 |
| | 2,000 | 21.6% | 3.8x | 1.8% | $141 |
| **Val onpeak** | 100 | 3.0% | 0.7x | 0.0% | $6 |
| | 500 | 6.6% | 1.5x | 0.1% | $34 |
| | 1,000 | 7.2% | 1.6x | 0.3% | $50 |

NDCG: 0.60-0.61 across all splits.

### Key finding: Val top-K is poor, holdout is moderate

The val top-K metrics show the model's highest-probability predictions are mostly wrong (Lift < 1.7x, barely above random). The holdout top-K is better (Lift 3.4-5.6x) but Value Capture is still under 2% at K=2000.

**Why the discrepancy?** The top-K metrics are computed at the **per-outage level** (400K rows, ~17K positives). K=2000 is only 0.5% of all rows. The model's probability ordering at the very top is unreliable — many never-binding branches get high default-model probabilities. The overall classification (Recall=74%, AUC=0.85) works by correctly handling the majority, not by perfectly ranking the absolute top.

**Implication**: The model is a decent detector (finds most binding constraints) but a weak ranker (doesn't reliably place the highest-value constraints at the very top). For signal generation, the aggregation step (across outage dates, into tiers) partially compensates — the constraint-level aggregation smooths out per-outage noise. But improving the top-of-ranking is the highest-priority improvement area.

---

## 5. Run 2: +hist_da Features + Regression Clamping

### Changes applied
1. **IMP-13**: Clamped regression output — `np.clip(raw_pred, None, 12)` at all 3 `expm1` sites
2. **IMP-1**: Added `hist_da` to Step 1 (classification) features
3. **IMP-2**: Added `hist_da` + `recent_hist_da` to Step 2 (regression) features

### Two new bugs fixed en route
- **BUG-5**: `spearmanr()` returns array instead of scalar — `np.isnan(corr)` failed. Fixed with `float(np.asarray(corr).flat[0])`.
- **Duplicate column bug**: Smoke test had `step2_features = step1_features + [('hist_da', 1), ...]` which duplicated `hist_da`. XGBoost crashes on DataFrames with duplicate column names. Fixed in smoke test + added dedup guard in `train_regressors`.

### Key results

| Metric | Run 1 (baseline) | Run 2 (+hist_da +clamp) | Change |
|--------|-------------------|------------------------|--------|
| Offpeak holdout RMSE (TPs) | **$90,911** | **$12,034** | **-87%** |
| Onpeak holdout Spearman | 0.462 | 0.476 | +3% |
| Offpeak holdout Spearman | 0.521 | 0.540 | +4% |
| Offpeak holdout Lift@100 | 3.1x | 4.6x | +48% |
| Clf metrics | — | — | ~unchanged |
| NDCG | 0.60-0.61 | 0.60-0.62 | ~0% |

### Lesson: `hist_da` was dropped by feature selection for classification

Feature selection checks Spearman correlation and AUC for each feature against the binary target (`label > 0`). Out of 9 candidate features, 8 were selected — `hist_da` was dropped.

**Why?** `hist_da = log1p(sum of all historical DA shadow prices)` conflates two signals:
- **Frequency**: Has this constraint ever bound? (useful for classification)
- **Magnitude**: How large were historical shadow prices? (useful for regression but noisy for binary detection)

A constraint that bound once for $100K and a constraint that bound 50 times for $2K each both get similar `hist_da` values, but they have very different binding probabilities. The Spearman/AUC check may have failed because the magnitude-weighted signal doesn't cleanly predict the binary outcome.

**Implication**: For classification, a **binary** or **frequency-based** historical feature (e.g., "number of months this constraint bound in the last 12 months") would be more appropriate than the magnitude-weighted `hist_da`.

---

## 6. Run 3: +Seasonal Features + Constraint-Level Metrics + Feature Selection Diagnostics

### Changes applied
1. **IMP-7**: Added `season_hist_da_1`, `season_hist_da_2`, `season_hist_da_3` to Step 2 (regression) features
2. **IMP-14**: Constraint-level top-K metrics in `evaluation.py`
3. **IMP-17**: Feature selection verbose diagnostics in `models.py`
4. Defensive handling for missing seasonal feature columns

### Key results

| Metric | Run 2 | Run 3 | Change |
|--------|-------|-------|--------|
| Onpeak holdout Spearman | 0.476 | **0.484** | **+2%** |
| Offpeak holdout Spearman | 0.540 | **0.552** | **+2%** |
| Clf metrics | — | — | ~unchanged |
| Per-outage NDCG | 0.60-0.62 | 0.60-0.62 | 0% |
| **Constraint NDCG** | — | **0.27-0.34** | (new metric) |
| **Constraint ValCap@1000** | — | **74-77%** | (new metric) |

### Lesson 1: `hist_da` was actually kept in Run 3

In Run 2, `hist_da` was dropped by feature selection. In Run 3, it was **kept** (AUC=0.835-0.851, Spearman=0.292-0.304). The feature that was dropped instead was `density_skewness` (AUC=0.431-0.440, Spearman=-0.047 to -0.052 — fails both checks with constraint=+1 because the relationship is negative).

The difference from Run 2 is likely due to the BUG-3 fix (validation-based threshold optimization) which changed the training data flow — feature selection now runs on the 6-month fit split rather than the full 12-month training set. The `hist_da` vs binary target relationship is cleaner on the shorter subset.

### Lesson 2: `density_skewness` has wrong-sign constraint

`density_skewness` is configured with `constraint=+1` (expecting positive monotonicity — higher skewness → more binding). But empirically, AUC<0.5 and Spearman<0 show the opposite. This means either:
- The constraint direction should be `-1` (negative skewness → more binding)
- Or the feature is too weak to be useful (|Spearman| < 0.05)

### Lesson 3: Constraint-level metrics reveal different picture from per-outage

| Level | NDCG | Lift@1000 | ValCap@1000 |
|-------|------|-----------|-------------|
| Per-outage (~257K rows) | 0.61 | 4.0x | 0.9% |
| Constraint (~5,800 rows) | 0.29 | 3.0x | 74.3% |

Per-outage NDCG (0.61) is inflated by correctly ranking the ~95% of non-binding rows. Constraint-level NDCG (0.29) is a more honest measure of signal quality for portfolio construction.

However, constraint-level **Value Capture@1000 = 74%** is excellent — the top 1K constraints (out of ~5,800) capture three-quarters of total congestion value. The model is a good "value concentrator" even if it doesn't perfectly order the top 50.

---

## 7. Next Steps

### Investigation
1. **Fix `density_skewness`**: Change constraint to `-1` or `0`, or remove entirely.
2. **Why are val top-K metrics so poor?** Inspect top-100 highest-probability predictions on val data.
3. **Probability calibration**: Reliability diagrams (predicted probability vs actual binding rate in bins).

### Metric infrastructure
4. **Multi-month evaluation**: Run across 6+ auction months for confidence intervals.
5. **Tier lift table**: Per-tier binding rate, lift, value share on holdout data.
6. **Baseline comparisons**: Value Capture@K for "historical binding frequency" baseline.
