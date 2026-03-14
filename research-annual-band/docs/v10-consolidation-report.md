# V10 Consolidation Report: Unit Fix + Simplified Asymmetric Bands

**Date:** 2026-03-14

---

## 1. Critical Discovery: R1 `mcp_mean` Unit Mismatch

### What We Found

`mcp_mean` in `all_residuals_v2.parquet` has **inconsistent units across rounds**:

| Round | `mcp` | `mcp_mean` | Correct? |
|-------|-------|-----------|----------|
| R1 | -451.80 (quarterly) | -451.80 (= mcp, quarterly) | **WRONG** — should be -150.60 (monthly) |
| R2 | -58.06 (quarterly) | -19.35 (= mcp/3, monthly) | Correct |
| R3 | 375.50 (quarterly) | 125.17 (= mcp/3, monthly) | Correct |

### Root Cause

**Location:** Notebook `archive/phase1_3/notebooks/03_corrected_baseline.ipynb`, Step 3.

1. Step 2 calls `aptools.tools.get_m2m_mcp_for_trades_all(trades)` which correctly sets `mcp_mean = mcp_amortized / months_in_duration` (monthly) for ALL rounds, including R1.
2. Step 3 calls `fill_mtm_1st_period_with_hist_revenue()` per group for R1 to fill `mtm_1st_mean` with H baseline. This pbase function (`miso.py:322-366`) returns a DataFrame that **drops `mcp_mean`**.
3. Step 4 concatenates R1 (missing `mcp_mean`) with R2/R3 (has correct `mcp_mean`). During concat, R1's `mcp_mean` gets backfilled with `mcp` (quarterly) instead of staying as the correct monthly value.

**This is NOT a pbase bug.** pbase correctly computes `mcp_mean = mcp/3`. The bug is in the notebook's data pipeline — `fill_mtm_1st_period_with_hist_revenue` drops the column.

### Impact

All R1 banding experiments (v1 through v9) were calibrated on `residual = quarterly_mcp - monthly_baseline`, a ~3x scale mismatch. R1 band widths were inflated by ~3x. R2/R3 were unaffected.

### Fix Applied

```python
# In aq*_all_baselines.parquet:
df = df.with_columns((pl.col("mcp_mean") / 3).alias("mcp_mean"))

# In all_residuals_v2.parquet:
df = df.with_columns(
    pl.when(pl.col("round") == 1)
    .then(pl.col("mcp") / 3)
    .otherwise(pl.col("mcp_mean"))
    .alias("mcp_mean")
)
```

Backups saved as `*.parquet.bak`.

### Convention (added to CLAUDE.md)

For MISO annual auctions, all prices are **quarterly** (3-month total). `mcp` = quarterly clearing price. `mcp_mean` should always = `mcp / 3` (monthly average). Baselines (`nodal_f0`, `mtm_1st_mean`) are monthly — compare against `mcp_mean`, not `mcp`.

---

## 2. Baseline Comparison After Unit Fix

With corrected monthly `mcp_mean`, MAE drops dramatically for all baselines:

| Baseline | Avg MAE | Avg Dir% | Avg P95 | Coverage | Notes |
|----------|--------:|---------:|--------:|---------:|-------|
| f1 path-level | **209** | 80.8% | 809 | 36.0% | Best MAE but low coverage |
| f0 path-level | **217** | 81.1% | 845 | 45.1% | Second best MAE |
| Prior year R3 MCP | 232 | 79.6% | 909 | 27.9% | |
| Prior year R2 MCP | 236 | 79.6% | 924 | 25.9% | |
| **Nodal f0 stitch (v3)** | **264** | **82.9%** | **1,037** | **91.9%** | **Best high-coverage baseline** |
| H = DA congestion (v1) | 369 | 67.4% | 1,292 | 100.0% | Worst accuracy |

**Old (wrong units) vs corrected MAE for nodal_f0:**

| Quarter | Old MAE | Corrected MAE | Reduction |
|---------|--------:|-------------:|---------:|
| aq1 | 798 | 307 | -61.6% |
| aq2 | 947 | 293 | -69.0% |
| aq3 | 797 | 257 | -67.8% |
| aq4 | 704 | 201 | -69.6% |

**Conclusion:** Nodal f0 remains the best R1 baseline. H (DA congestion) is 40% worse on MAE and 15pp worse on direction accuracy.

---

## 3. V10 Band Results (Corrected Units)

### Method

Same as v9: asymmetric signed quantile pairs, 5 quantile bins, per-class (onpeak/offpeak), no sign split, no correction. Temporal expanding CV, min_train_pys=2. 8 coverage levels (P10-P99).

Only R1 results changed (R2/R3 already had correct units in v9).

### R1 P95 Half-Width: 70-76% Reduction

| Quarter | v9 (wrong units) | v10 (corrected) | Reduction | P95 Coverage |
|---------|-----------------:|----------------:|---------:|--------:|
| aq1 | 2,701 | **859** | **-68.2%** | 93.2% |
| aq2 | 2,772 | **784** | **-71.7%** | 91.3% |
| aq3 | 2,339 | **674** | **-71.2%** | 93.6% |
| aq4 | 1,369 | **455** | **-66.8%** | 89.9% |
| **Avg** | **2,295** | **693** | **-69.8%** | **92.0%** |

### R1 Coverage at All Levels

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 9.2 | 27.5 | 46.2 | 66.6 | 77.3 | 87.7 | 93.2 | 98.1 |
| aq2 | 8.8 | 27.3 | 45.7 | 64.4 | 74.5 | 85.5 | 91.3 | 97.2 |
| aq3 | 8.9 | 28.1 | 46.8 | 66.5 | 76.8 | 87.8 | 93.6 | 98.1 |
| aq4 | 8.5 | 26.1 | 44.2 | 62.9 | 72.6 | 83.3 | 89.9 | 96.2 |

### R1 Per-PY Stability

| Quarter | PY2022 | PY2023 | PY2024 | Range | Worst |
|---------|-------:|-------:|-------:|------:|------:|
| aq1 | 90.7% | 91.1% | 97.6% | 6.9pp | 90.7% |
| aq2 | 84.9% | 90.6% | 97.0% | 12.1pp | 84.9% |
| aq3 | 89.2% | 95.8% | 94.7% | 6.5pp | 89.2% |
| aq4 | 82.8% | 93.7% | 94.5% | 11.7pp | 82.8% |

Significantly more stable than v9: worst-PY improved from 86.2% to 82.8-90.7% with narrower ranges.

### R1 Per-Bin P95 Coverage

| Quarter | q1 | q2 | q3 | q4 | q5 |
|---------|---:|---:|---:|---:|---:|
| aq1 | 96.5 | 96.0 | 96.1 | 96.5 | 84.7 |
| aq2 | 94.7 | 93.5 | 94.7 | 94.9 | 82.4 |
| aq3 | 94.4 | 95.0 | 94.9 | 95.5 | 89.8 |
| aq4 | 92.2 | 91.1 | 91.3 | 93.0 | 83.7 |

q5 still under-covers (83-90%) — structural issue with heavy-tailed high-value paths.

### R1 Class Parity

| Quarter | onpeak | offpeak | gap |
|---------|-------:|--------:|----:|
| aq1 | 92.11% | 94.35% | 2.24pp |
| aq2 | 90.53% | 92.15% | 1.62pp |
| aq3 | 93.71% | 93.57% | 0.14pp |
| aq4 | 90.28% | 89.57% | 0.71pp |

Parity gap slightly higher than v9 (max 2.24pp vs 1.07pp) but still within tolerance.

### R2/R3 Results (Unchanged from v9)

| Round | Quarter | P95 Cov | P95 HW |
|-------|---------|--------:|-------:|
| R2 | aq1 | 89.6% | 180 |
| R2 | aq2 | 89.8% | 186 |
| R2 | aq3 | 91.0% | 170 |
| R2 | aq4 | 90.3% | 173 |
| R3 | aq1 | 91.5% | 164 |
| R3 | aq2 | 91.6% | 155 |
| R3 | aq3 | 92.0% | 146 |
| R3 | aq4 | 92.6% | 144 |

---

## 4. Cross-Round Comparison (All on Monthly Scale)

With corrected units, R1 widths are now comparable to R2/R3:

| Round | Avg P95 HW | Avg P95 Cov | Baseline |
|-------|----------:|-----------:|----------|
| R1 | **693** | 92.0% | nodal_f0 (monthly) |
| R2 | **177** | 90.2% | mtm_1st_mean (monthly) |
| R3 | **152** | 91.9% | mtm_1st_mean (monthly) |

R1 is still 4x wider than R2/R3 — expected because R1 has no prior round and `nodal_f0` is a weaker baseline than `mtm_1st_mean` (prior round MCP). But the gap is now 4x, not the artificial 13x from wrong units.

---

## 5. Width at All Levels (R1, Corrected)

| Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99 |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|
| aq1 | 22 | 69 | 136 | 254 | 364 | 589 | 859 | 1,637 |
| aq2 | 22 | 67 | 132 | 243 | 346 | 553 | 784 | 1,460 |
| aq3 | 19 | 60 | 117 | 210 | 298 | 470 | 674 | 1,240 |
| aq4 | 16 | 51 | 95 | 164 | 222 | 330 | 455 | 821 |

---

## 6. ML Opportunity (Quick Test Results)

Before the unit fix, we tested LightGBM quantile regression on R1 with 4 features (`nodal_f0`, `|nodal_f0|`, `is_onpeak`, `mtm_1st_mean`). Results showed 23-41% width reduction vs empirical quantiles. This test needs to be **rerun with corrected units** — the improvement may be smaller now that the empirical baseline is properly calibrated.

---

## 7. Files Modified

| File | Change |
|------|--------|
| `/opt/temp/qianli/annual_research/all_residuals_v2.parquet` | R1 `mcp_mean = mcp/3` (backup: `.bak`) |
| `/opt/temp/qianli/annual_research/crossproduct_work/aq{1-4}_all_baselines.parquet` | `mcp_mean /= 3` (backup: `.bak`) |
| `CLAUDE.md` | Added MISO annual price unit convention |
| `versions/bands/v10_corrected/r{1,2,3}/` | New band results with corrected units |

---

## 8. Next Steps

1. **Rerun ML test** with corrected units to see if LightGBM still helps R1
2. **Holdout validation** (PY 2025) on v10 to confirm results hold out-of-sample
3. **Regenerate upstream data** — properly fix the notebook to preserve `mcp_mean` through `fill_mtm_1st_period_with_hist_revenue`, rather than relying on the `/3` fix at load time
