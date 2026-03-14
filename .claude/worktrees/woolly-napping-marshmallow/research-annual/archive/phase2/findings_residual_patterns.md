# Residual Pattern Analysis: nodal_f0 Baseline

**Date:** 2026-02-18
**Data:** `aq{1,2,3,4}_all_baselines.parquet` (baselines), `all_residuals_v2.parquet` (11.5M rows)
**Script:** `scripts/explore_residuals.py`

---

## Executive Summary

The nodal_f0 baseline has a **persistent positive bias** (mean residual +220 $/MWh for aq1) and residuals are **strongly autocorrelated** across planning years (r = 0.41-0.71 lag-1). Three actionable improvement opportunities:

1. **Scale correction (alpha = 1.45):** Multiplying nodal_f0 by 1.45 reduces MAE by 5-13% across all PYs.
2. **Persistent path-level bias:** 58% of paths with 4+ years of history have residuals with the same sign >= 80% of years. A path-level bias correction using prior-year residuals is exploitable.
3. **Blending with f0_path_corr:** Optimal blend at w=0.9 (90% nodal_f0, 10% f0_path_corr) provides marginal MAE improvement on the overlap subset.

---

## 1. Overall Residual Distribution (aq1, mcp_mean - nodal_f0)

| Metric | Value |
|--------|-------|
| N (valid rows) | 147,315 |
| Mean (bias) | +219.6 |
| Median | +45.0 |
| Std | 1,667.9 |
| MAE | 798.2 |
| RMSE | 1,682.3 |
| P5 / P95 | -1,742 / +2,445 |

**Key observation:** The distribution is right-skewed (mean >> median), indicating fat-tailed positive residuals. The model systematically underestimates MCP.

### By Planning Year

| PY | N | Bias | MAE | RMSE |
|----|---|------|-----|------|
| 2020 | 17,692 | +85.7 | 407.8 | 742.2 |
| 2021 | 22,702 | +158.4 | 808.2 | 1,499.1 |
| 2022 | 22,695 | +452.9 | 1,260.8 | 2,764.9 |
| 2023 | 29,133 | +140.7 | 711.4 | 1,534.8 |
| 2024 | 26,879 | +280.1 | 819.1 | 1,588.1 |
| 2025 | 28,214 | +189.2 | 732.9 | 1,280.3 |

PY 2022 is the worst year (high gas prices / congestion). Bias is **always positive** -- nodal_f0 consistently underestimates clearing prices.

### By Class Type

| Class | N | Bias | MAE | RMSE |
|-------|---|------|-----|------|
| offpeak | 71,769 | +217.5 | 746.6 | 1,655.2 |
| onpeak | 75,546 | +221.7 | 847.3 | 1,707.7 |

Onpeak has ~13% higher MAE than offpeak. Bias is similar in both.

### By |nodal_f0| Magnitude

| Bin | N | Bias | MAE | Bias Ratio |
|-----|---|------|-----|------------|
| |f0| < 1 | 2,477 | -20.4 | 89.0 | -66.2x |
| 1 <= |f0| < 5 | 4,814 | +14.6 | 157.0 | 5.1x |
| 5 <= |f0| < 20 | 12,184 | +12.2 | 217.8 | 1.0x |
| 20 <= |f0| < 50 | 17,187 | +26.9 | 298.0 | 0.79x |
| |f0| >= 50 | 110,653 | +286.7 | 983.6 | 0.48x |

**Critical finding:** 75% of rows have |f0| >= 50, and those carry the bulk of the error. The bias ratio (bias / |f0|) of 0.48 for the large-magnitude bin means mcp is on average ~48% larger than the f0 prediction, suggesting a scale factor correction.

### Residual Magnitude Distribution

| |residual| bin | N | % | Mean resid |
|----------------|---|---|------------|
| < 0.5 | 688 | 0.5% | -0.003 |
| 0.5 - 2 | 1,091 | 0.7% | -0.015 |
| 2 - 5 | 1,839 | 1.2% | -0.190 |
| 5 - 10 | 2,482 | 1.7% | -0.330 |
| 10 - 25 | 6,692 | 4.5% | -1.147 |
| 25 - 50 | 9,580 | 6.5% | -1.898 |
| >= 50 | 124,943 | 84.8% | +259.2 |

**85% of residuals exceed $50 in magnitude.** The small-residual paths have slight negative bias (model slightly overestimates), while the dominant large-residual paths have massive positive bias. This is consistent with "nodal_f0 gets the direction right but underestimates magnitude."

---

## 2. Autocorrelation of Residuals Across Planning Years

### Lag-1 Correlation (Same Path, Consecutive PYs)

**From baselines file (aq1):**

| PY pair | Correlation | N paths |
|---------|-------------|---------|
| 2020 -> 2021 | 0.546 | 5,501 |
| 2021 -> 2022 | 0.547 | 5,410 |
| 2022 -> 2023 | 0.436 | 6,941 |
| 2023 -> 2024 | 0.405 | 6,837 |
| 2024 -> 2025 | 0.712 | 7,806 |

**From all_residuals_v2 (aq1, round 1):**

| PY pair | Correlation | N paths |
|---------|-------------|---------|
| 2019 -> 2020 | 0.479 | 4,087 |
| 2020 -> 2021 | 0.537 | 5,550 |
| 2021 -> 2022 | 0.572 | 5,458 |
| 2022 -> 2023 | 0.701 | 7,010 |
| 2023 -> 2024 | 0.610 | 6,939 |
| 2024 -> 2025 | 0.667 | 7,888 |

### Lag-2 Correlation

| PY pair | Correlation | N paths |
|---------|-------------|---------|
| 2020 -> 2022 | 0.452 | 3,757 |
| 2021 -> 2023 | 0.401 | 4,887 |
| 2022 -> 2024 | 0.601 | 4,796 |
| 2023 -> 2025 | 0.365 | 6,514 |

### Pooled Autocorrelation

| Segment | Lag-1 r | N |
|---------|---------|---|
| Overall | 0.410 | 32,495 |
| Onpeak | 0.394 | 16,684 |
| Offpeak | 0.428 | 15,811 |

**This is highly actionable.** Residuals are persistent enough that last year's residual is predictive of this year's residual. A simple correction of `nodal_f0_adjusted = nodal_f0 + w * prior_year_residual` should improve accuracy.

### Persistent Bias Paths

Among 5,895 paths with >= 4 years of history (round 1, aq1):
- **3,392 (58%)** have positive residuals >= 80% of years (model underestimates)
- **320 (5%)** have negative residuals >= 80% of years (model overestimates)
- **2,183 (37%)** are neutral (mixed sign)

For consistently positive paths: mean residual = +913, std = 721
For consistently negative paths: mean residual = -1,028, std = 918

**The 58% persistent-positive fraction is the main opportunity.** These paths have a structural feature (likely congestion patterns) that the f0 forward systematically underprices.

---

## 3. Correlation with Available Features

### Raw Feature Correlations

| Feature | corr(residual, feature) | N |
|---------|------------------------|---|
| mcp_mean | +0.927 | 147,315 |
| mcp_minus_mtm1 | +0.918 | 147,315 |
| mtm_1st_mean | +0.502 | 147,315 |
| prior_r1_path | +0.470 | 36,932 |
| nodal_f0 | +0.438 | 147,315 |
| prior_r3_path | +0.443 | 44,633 |
| prior_r2_path | +0.443 | 41,840 |
| f1_path | +0.362 | 38,414 |
| f0_path_corr | +0.355 | 67,291 |
| abs_nodal_f0 | +0.202 | 147,315 |
| nodal_correction | -0.009 | 67,291 |

**Key findings:**
- The residual is most correlated with mcp_mean itself (trivially) and mtm_1st_mean (+0.50).
- nodal_f0 magnitude (|nodal_f0|) has modest correlation (+0.20): larger f0 values have larger residuals, consistent with the scale factor finding.
- The nodal correction (nodal_f0 - f0_path_corr) has essentially zero correlation with the residual (-0.009), meaning the nodal adjustment itself is not introducing systematic error.
- Prior round paths (r1, r2, r3) all have ~0.44-0.47 correlation, confirming that historical cleared prices are informative.

---

## 4. Baseline Comparison

### Overall (aq1)

| Baseline | MAE | RMSE | Bias | N |
|----------|-----|------|------|---|
| **nodal_f0** | 798.2 | 1,682.3 | +219.6 | 147,315 |
| f0_path_corr | 572.9 | 1,240.4 | +126.4 | 67,291 |
| f1_path | 538.5 | 1,135.6 | +134.7 | 38,414 |
| prior_r3_path | 658.9 | 1,486.8 | +242.1 | 44,633 |
| prior_r2_path | 703.4 | 1,575.2 | +280.8 | 41,840 |
| prior_r1_path | 740.2 | 1,609.2 | -32.3 | 36,932 |
| mtm_1st_mean | 922.1 | 1,811.2 | +424.0 | 147,315 |

**Note:** f0_path_corr and f1_path have better MAE but lower coverage (46% and 26% respectively vs 100% for nodal_f0). Direct comparison requires restricting to the same rows. The prior R1 path is interesting for having near-zero bias.

### Cross-Quarter (all 4 quarters)

| Quarter | Offpeak MAE | Onpeak MAE | Offpeak Bias | Onpeak Bias |
|---------|-------------|------------|--------------|-------------|
| aq1 | 746.6 | 847.3 | +217.5 | +221.7 |
| aq2 | 932.5 | 959.8 | +329.1 | +350.7 |
| aq3 | 774.7 | 818.6 | +216.5 | +233.9 |
| aq4 | 659.7 | 659.6 | +200.1 | +180.3 |

The positive bias is consistent across all quarters. aq2 (summer) has the highest errors, aq4 (winter) the lowest.

---

## 5. Improvement Opportunities

### 5a. Scale Correction (Shrinkage)

Testing `prediction = alpha * nodal_f0`:

| Alpha | MAE | RMSE | Improvement |
|-------|-----|------|-------------|
| 1.00 (baseline) | 798.2 | 1,682.3 | -- |
| 1.10 | 779.3 | 1,641.1 | 2.4% |
| 1.20 | 763.8 | 1,604.7 | 4.3% |
| **1.45** (optimal) | ~690-788 | -- | **5-13%** |

**Optimal alpha by PY:**

| PY | Optimal Alpha | Improvement |
|----|---------------|-------------|
| 2020 | 1.45 | 7.3% |
| 2021 | 1.45 | 5.6% |
| 2022 | 1.45 | 8.1% |
| 2023 | 1.20 | 2.9% |
| 2024 | 1.45 | 11.8% |
| 2025 | 1.45 | 12.8% |

**Optimal alpha by class_type:** Both onpeak and offpeak have optimal alpha at 1.45.

**This is the simplest win:** multiply nodal_f0 by ~1.45 uniformly. However, the optimal alpha is remarkably stable across PYs (1.45 for 5 of 6 years), suggesting this is a structural underscaling in the f0 forward curve rather than an accident.

### 5b. Blending nodal_f0 with f0_path_corr

For the 67,291 rows where both are available:

| Weight (w for nodal_f0) | MAE | RMSE |
|--------------------------|-----|------|
| 0.0 (pure f0_path_corr) | 572.9 | 1,240.4 |
| 0.5 | 565.7 | 1,229.3 |
| 0.9 (optimal) | 564.1 | 1,226.1 |
| 1.0 (pure nodal_f0) | 564.3 | 1,226.1 |

The improvement from blending is marginal (~1.5% MAE reduction vs pure f0_path). On this overlap set, nodal_f0 and f0_path_corr are already close in quality.

### 5c. Volume as Signal

| Volume Decile | Mean Vol | MAE | Bias |
|---------------|----------|-----|------|
| 0 (lowest) | 0.15 | 1,211.5 | +702.4 |
| 4 | 1.88 | 761.8 | +300.9 |
| 7 | 5.08 | 608.0 | +244.9 |
| 8 | 7.76 | 585.6 | +286.4 |
| 9 (highest) | 28.22 | 699.6 | +410.7 |

Low-volume paths have the highest errors and bias. MAE decreases with volume until the top decile (which includes extreme paths). **Volume could be used as a confidence signal** -- low-volume paths need wider bands.

---

## 6. Round Comparison (from all_residuals_v2)

| Round | N | Mean Bias | MAE | RMSE |
|-------|---|-----------|-----|------|
| R1 | 2,388,474 | +420.7 | 850.5 | 1,707.3 |
| R2 | 4,179,459 | +21.3 | 70.1 | 141.8 |
| R3 | 4,925,847 | +11.3 | 56.2 | 114.2 |

R2 and R3 have dramatically lower residuals (MAE ~70 and ~56 vs ~850 for R1). This is expected since R2/R3 baselines use M-only (monthly round price) which is much closer to actual clearing.

---

## Actionable Next Steps

1. **Implement alpha scaling (1.45x):** Test in-sample on aq1, validate on aq2-4. Provides ~5-13% MAE improvement with zero complexity cost.

2. **Path-level bias correction using prior-year residuals:** Given lag-1 autocorrelation of 0.41-0.71, a simple `nodal_f0_adj = alpha * nodal_f0 + beta * prior_residual` should capture persistent path-level mispricing. Requires joining prior-PY residuals.

3. **Volume-based confidence weighting for band calibration:** Use cleared volume as a signal for band width -- widen bands for low-volume paths where MAE is 2x higher.

4. **Investigate the 58% persistent-positive paths:** These paths where the model always underestimates likely have structural congestion characteristics (wind zones, constrained interfaces) that could be modeled explicitly.
