# Annual FTR Bid Pricing -- Project Memory

Quick-reference. For full experiment results see `findings.md`. For domain rules see `runbook.md`.

---

## Project Overview

Design bid price strategies for MISO annual FTR auctions (baseline, bands, clearing_prob, optimizer). Production currently uses ftr23 (parametric). Target: migrate to ftr24/v1 (conformal bands).

---

## Repo Structure

```
research-annual/
  mem.md                          <- this file
  runbook.md                      <- domain reference
  findings.md                     <- all experiment results (Phase 1 + Phase 2)
  design-planning.md              <- pipeline design spec (Rev 3)
  .gitignore
  pipeline/
    __init__.py
    pipeline.py                   <- metrics, versioning, gates, CLI
  versions/
    baseline/
      v1/                         <- H baseline (0.85 * DA congestion)
      v2/                         <- Pure DA (no shrinkage)
      v3/                         <- Nodal f0 stitch (current promoted)
      promoted.json               <- points to v3
    bands/
      r1/
        v1/                       <- Empirical quantile bins
        v2/                       <- Width reduction via quantile bins
        v3/                       <- Per-class stratified bands (current promoted)
        promoted.json             <- points to v3
      r2/
        v1/                       <- R2 quantile 6-bin
        v2/                       <- R2 per-class stratified (current promoted)
        promoted.json             <- points to v2
      r3/
        v1/                       <- R3 quantile 6-bin
        v2/                       <- R3 per-class stratified (current promoted)
        promoted.json             <- points to v2
  scripts/
    baseline_utils.py             <- shared eval/print/replacement utilities
    run_aq{1,2,3,4}_experiment.py <- per-quarter R1 baseline experiments
    run_phase2_improvement.py     <- α-scaling + prior-year residual LOO
    run_phase3_bands.py           <- R1 band calibration (symmetric quantile bins)
    run_phase3_v2_bands.py        <- R1 band width reduction experiments
    run_r2r3_bands.py             <- R2/R3 band calibration
    run_v3_bands.py               <- Per-class stratified bands (all rounds)
  archive/
    phase1_3/                     <- Phase 1-3 findings, notebooks, old scripts
    phase2/                       <- Phase 2 dead-end scripts + old findings
    baseline_formula/             <- formula derivation notes
    learning.md, research_plan.md, stage2_design.md
```

---

## Status

| Phase | Status |
|-------|--------|
| Phase 1: R1 baseline comparison (all 4 quarters) | DONE |
| Phase 2a: α-scaling + prior-year residual | DONE |
| Phase 2b: Signal blending (Ridge + convex + de-biased) | DONE (dead end) |
| Phase 3: R1 band calibration | DONE |
| Phase 3 v2: Band width reduction | DONE |
| Phase 3 R2/R3: Band calibration | DONE |
| Phase 3 v3: Per-class stratified bands | DONE |

---

## Key Bugs Found & Fixed

1. **Year mapping PY-2 vs PY-1:** Correct: `dy = PY-1` for months >= 6, `dy = PY` for months < 6.
2. **Nodal replacement needs BFS + date filtering:** Forward + reverse chains, filtered to target PY delivery date.
3. **`get_mcp_df()` column 0 = f0, column -1 = annual R1.** Column count varies per month.
4. **Replacement dates are tz-aware (EST).** Must `tz_localize()` before comparison.
5. **`cast_category_to_str` required** after `get_all_cleared_trades()`.

---

## Key Findings

### Phase 1: Nodal f0 is the best R1 baseline

| Tier | Source | Coverage | Purpose |
|------|--------|----------|---------|
| 1 | Nodal f0 (3-month avg) | 98.8-100% | Primary |
| 2 | H (DA congestion) | 0-1.2% | Fallback |

- Ranking identical across all quarters: **Nodal f0 ≈ f0 path < f1 < R3 ≈ R2 < R1 < QF < H**
- 2-tier cascade is optimal. f0 path as Tier 1 always hurts (+4-5 MAE)
- All baselines have persistent positive bias (+200-400)

### Phase 2a: α-scaling + prior-year residual

| Quarter | Raw MAE | α-scaled MAE | Combined MAE | vs raw |
|---------|--------:|-------------:|-------------:|-------:|
| aq1 | 798 | 736 (-8%) | 663 (-10%) | -10% |
| aq2 | 947 | 815 (-14%) | 650 (-17%) | -17% |
| aq3 | 797 | 668 (-16%) | 570 (-23%) | -23% |
| aq4 | 704 | 566 (-20%) | 459 (-31%) | -31% |

- α-scaling (`α × nodal_f0`, α=1.55-1.60) corrects persistent positive bias while preserving Dir%
- Prior-year residual (`+ β × (PY-1 mcp - PY-1 f0)`, β=0.15-0.40) exploits lag-1 autocorrelation (r=0.41-0.71) on ~22% of recurring paths

### Phase 2b: Signal blending — dead end

- **Convex combination** (w × f0 + (1-w) × H): H adds zero information. LOO-optimal w = 0.95-1.00.
- **Ridge regression**: overfitting disguised as blending (learns f0 weight 1.9-2.3×)
- **De-biased averaging/median**: fixes bias but destroys Dir% (drops 10pp)
- **Why**: all other signals (H, f0_path, f1, priors, quarterlies) are noisier versions of the same congestion signal. No independent directional information. Only α-scaling preserves direction perfectly.

---

## Data Locations

```
/opt/temp/qianli/annual_research/
  all_residuals_v2.parquet         # 11.5M rows
  f0p_cleared_all.parquet          # 12.7M rows (f0, f1, q4)
  crossproduct_work/
    aq{1,2,3,4}_all_baselines.parquet  # per-quarter experiment outputs
```
---

## Pipeline & Versioning

Each experiment gets a versioned directory under `versions/{part}/` with `config.json`, `metrics.json`, and `NOTES.md`. Versions are promoted via gate checks.

```bash
# CLI commands
python pipeline/pipeline.py list baseline          # list all versions
python pipeline/pipeline.py list bands/r1          # list R1 band versions
python pipeline/pipeline.py compare baseline v3    # compare v3 vs promoted
python pipeline/pipeline.py promote baseline v4    # promote (checks gates)
python pipeline/pipeline.py validate baseline      # schema validation
python pipeline/pipeline.py validate bands/r1      # validate R1 bands
python pipeline/pipeline.py create baseline v4 "description"
```

**Current promoted:** baseline v3 (nodal f0 stitch), bands/r1 v3 (per-class stratified), bands/r2 v2 (per-class stratified), bands/r3 v2 (per-class stratified).

**Baseline gate system:** 3 HARD gates (MAE, Dir%, coverage), 3 SOFT (bias, stability, worst-PY), 3 ADVISORY (class parity, win rate, tail risk).

**Band gate system:** 4 HARD gates (BG0 baseline promoted, BG1 P95 accuracy, BG2 P50 accuracy, BG3 per-bin uniformity), 2 SOFT (BG4 width narrower, BG5 per-PY stability), 2 ADVISORY (BG6 width monotonicity, BG7 class parity coverage).

Both parts independently versioned under `versions/baseline/` and `versions/bands/{r1,r2,r3}/`.

**Primary gate metric:** Temporal expanding window (train on PYs < test PY) with `min_train_pys=3` filter to exclude cold-start folds. LOO by PY retained as secondary diagnostic.

### Phase 3: R1 band calibration

| Quarter | P50 cov | P95 cov | Worst PY P95 | P95 width CV |
|---------|--------:|--------:|:------------:|-----------:|
| aq1 | 50.6% | 94.7% | 87.7% (2022) | 0.086 |
| aq2 | 50.1% | 94.7% | 85.5% (2022) | 0.082 |
| aq3 | 50.0% | 94.7% | 87.3% (2022) | 0.085 |
| aq4 | 49.8% | 94.9% | 93.2% (2022) | 0.030 |

- Symmetric bands via empirical quantile bins by |nodal_f0|, LOO by PY
- All HARD gates pass (P95/P50 accuracy within tolerance, per-bin uniformity)
- BG5 SOFT fail: PY 2022 worst coverage 85-88% for aq1-aq3 (known outlier year)
- Promoted as bands v1

---

### Phase 3 v2: Band width reduction

Tested 5 bin configs to reduce P95 widths. Winner: `quantile_bins` (data-driven percentile boundaries).

| Quarter | v1 P95 width | v2 P95 width | Change |
|---------|-------------:|-------------:|-------:|
| aq1 | 3,318 | 2,646 | -20.3% |
| aq2 | 3,897 | 3,122 | -19.9% |
| aq3 | 3,104 | 2,496 | -19.6% |
| aq4 | 2,776 | 2,217 | -20.1% |

- Coverage accuracy equivalent to v1 (P95 error -0.13 to -0.34pp)
- All HARD gates pass, BG4 (width narrower) PASS, BG5 SOFT fail (PY 2022, same as v1)
- `split_large` failed: splitting at 3k exposed large_hi widths 11-16k, raising mean
- `six_bins` failed: extreme bin had <100 rows
- Asymmetry diagnostic CV 0.05-0.44 justifies symmetric bands
- Script: `scripts/run_phase3_v2_bands.py`

---

### Phase 3 R2/R3: Band calibration

R2/R3 use M baseline (prior round MCP), which is much closer to actuals. Tested `fixed_4bin`, `quantile_4bin`, `quantile_6bin`. Winner: `quantile_6bin` for both rounds.

**P95 mean width by round ($/MWh):**

| Quarter | R1 (v2) | R2 | R3 | R2/R1 | R3/R1 |
|---------|--------:|---:|---:|------:|------:|
| aq1 | 2,646 | 217 | 181 | 8.2% | 6.9% |
| aq2 | 3,122 | 228 | 181 | 7.3% | 5.8% |
| aq3 | 2,496 | 202 | 162 | 8.1% | 6.5% |
| aq4 | 2,217 | 211 | 162 | 9.5% | 7.3% |

- R2 bands ~12x narrower than R1, R3 ~15x narrower
- All HARD gates pass for both rounds
- BG5 SOFT fail: PY 2022 worst-PY coverage 81-87% (same systematic issue)
- 6 bins feasible with ~1M-1.2M rows/quarter (vs R1's ~135K)
- Script: `scripts/run_r2r3_bands.py`

---

### Phase 3 v3: Per-class stratified bands

Pooled bands under-cover onpeak and over-cover offpeak because onpeak has up to 20% higher MAE. v3 calibrates separate widths per (bin, class_type). Width-neutral, but dramatically reduces class parity gap.

**Class parity gap (P95, |onpeak - offpeak| coverage):**

| Quarter | R1 pooled | R1 per-class | R2 pooled | R2 per-class | R3 pooled | R3 per-class |
|---------|----------:|-------------:|----------:|-------------:|----------:|-------------:|
| aq1 | 0.09pp | 0.01pp | 0.43pp | 0.07pp | 1.02pp | 0.14pp |
| aq2 | 0.45pp | 0.10pp | 0.04pp | 0.09pp | 0.26pp | 0.08pp |
| aq3 | 0.47pp | 0.02pp | 0.58pp | 0.00pp | 0.08pp | 0.06pp |
| aq4 | 0.05pp | 0.07pp | 0.15pp | 0.07pp | 0.13pp | 0.01pp |

- BG7 (class parity) passes for the first time on all rounds
- All HARD gates pass, widths neutral (R1 aq4 is 6.5% narrower)
- BG5 SOFT fail: PY 2022 (same systematic issue)
- Script: `scripts/run_v3_bands.py`

---

## Next Steps
1. **Production integration:** Migrate to ftr24/v1 conformal prediction framework
2. **Signal data:** SPICE_ANNUAL / DA_ANNUAL (deferred; needs production env)

## baseline
- now we are only done with aq1 right?
- what about aq2 - aq4? what was the production code using? DA congestion or mtm_1st_mean?
  - make use of our modeling pipeline so we know what research have been conducted