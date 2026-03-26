# Scaling Changes Log — R1 mcp_mean Unit Fix

**Date:** 2026-03-14
**Issue:** R1 `mcp_mean` was quarterly (`= mcp`) instead of monthly (`= mcp/3`)

---

## Files Modified

### 1. `/opt/temp/qianli/annual_research/all_residuals_v2.parquet`
- **Change:** R1 rows (round==1): `mcp_mean = mcp / 3`
- **Rows affected:** 2,388,474 (all R1 rows)
- **R2/R3 rows:** unchanged (already correct)
- **Recomputed:** `residual = mcp_mean - mtm_1st_mean`, `abs_residual = |residual|`
- **Backup:** `all_residuals_v2.parquet.bak`

### 2. `/opt/temp/qianli/annual_research/crossproduct_work/aq1_all_baselines.parquet`
- **Change:** `mcp_mean /= 3` (all rows are R1)
- **Rows:** 165,898
- **Backup:** `aq1_all_baselines.parquet.bak`

### 3. `/opt/temp/qianli/annual_research/crossproduct_work/aq2_all_baselines.parquet`
- **Change:** `mcp_mean /= 3`
- **Rows:** 166,535
- **Backup:** `aq2_all_baselines.parquet.bak`

### 4. `/opt/temp/qianli/annual_research/crossproduct_work/aq3_all_baselines.parquet`
- **Change:** `mcp_mean /= 3`
- **Rows:** 152,522
- **Backup:** `aq3_all_baselines.parquet.bak`

### 5. `/opt/temp/qianli/annual_research/crossproduct_work/aq4_all_baselines.parquet`
- **Change:** `mcp_mean /= 3`
- **Rows:** 151,210
- **Backup:** `aq4_all_baselines.parquet.bak`

## Code That Uses These Files

| Script | Column Used | Status |
|--------|-------------|--------|
| `scripts/run_v9_bands.py` | `mcp_mean` as target | Uses corrected data (v10 run) |
| `scripts/run_phase3_bands.py` | `mcp_mean` as target | Needs rerun if results matter |
| `scripts/run_v3_bands.py` through `run_v8_bands.py` | `mcp_mean` as target | All band versions v1-v8 had inflated R1 widths |
| `scripts/run_aq{1-4}_experiment.py` | `mcp_mean` for baseline comparison | Corrected data flows through |

## NO Silent Fallbacks

The fix is explicit: `mcp_mean = mcp / 3` applied only to R1 rows. No defaults, no `.get()` with fallback values, no silent corrections. The original quarterly `mcp` column is preserved unchanged.

## How to Verify

```python
import polars as pl

# all_residuals_v2: R1 mcp_mean should be mcp/3
df = pl.read_parquet("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")
r1 = df.filter(pl.col("round") == 1)
r2 = df.filter(pl.col("round") == 2)
assert (r1["mcp"] / r1["mcp_mean"]).mean() == pytest.approx(3.0, abs=0.01)  # R1 fixed
assert (r2["mcp"] / r2["mcp_mean"]).mean() == pytest.approx(3.0, abs=0.01)  # R2 was already correct

# aq*_all_baselines: mcp_mean should be ~1/3 of original
df_new = pl.read_parquet(".../aq1_all_baselines.parquet")
df_old = pl.read_parquet(".../aq1_all_baselines.parquet.bak")
assert (df_old["mcp_mean"] / df_new["mcp_mean"]).mean() == pytest.approx(3.0, abs=0.01)
```
