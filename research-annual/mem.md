# Annual FTR Bid Pricing -- Project Memory

Quick-reference. For full experiment results see `findings_all_quarters.md`. For domain rules see `runbook.md`.

---

## Project Overview

Design bid price strategies for MISO annual FTR auctions (baseline, bands, clearing_prob, optimizer). Production currently uses ftr23 (parametric). Target: migrate to ftr24/v1 (conformal bands).

---

## Repo Structure

```
research-annual/
  mem.md                          <- this file
  runbook.md                      <- domain reference (condensed)
  findings_all_quarters.md        <- consolidated experiment results (all 4 quarters)
  .gitignore
  scripts/
    baseline_utils.py             <- shared eval/print/replacement utilities
    run_aq{1,2,3,4}_experiment.py <- per-quarter R1 baseline experiments
  archive/
    phase1_3/                     <- earlier findings, notebooks, old scripts
    baseline_formula/             <- formula derivation notes
    learning.md, research_plan.md, stage2_design.md  <- early planning docs
```

---

## Status

| Phase | Status |
|-------|--------|
| Phase 1-3 (data, residuals, legacy) | DONE |
| Phase 3.5 (R1 improvement experiments) | DONE |
| Phase 4 (nodal f0 stitching, aq1/aq2) | DONE |
| Phase 5 (aq3/aq4 experiments) | DONE |
| Phase 6 (band calibration) | PLANNED |

---

## Key Bugs Found & Fixed

1. **Year mapping PY-2 vs PY-1:** Correct: `dy = PY-1` for months >= 6, `dy = PY` for months < 6.
2. **Nodal replacement needs BFS + date filtering:** Forward + reverse chains, filtered to target PY delivery date.
3. **`get_mcp_df()` column 0 = f0, column -1 = annual R1.** Column count varies per month.
4. **Replacement dates are tz-aware (EST).** Must `tz_localize()` before comparison.
5. **`cast_category_to_str` required** after `get_all_cleared_trades()`.

---

## Key Findings

**Optimal R1 baseline: 2-tier cascade for all 4 quarters**

| Tier | Source | Coverage | Purpose |
|------|--------|----------|---------|
| 1 | Nodal f0 (3-month avg) | 98.8-100% | Primary |
| 2 | H bias-corrected (LOO) | 0-1.2% | Fallback |

- Head-to-head win rate: Nodal f0 wins 51.8% vs f0 path 48.0% (identical across all quarters)
- f0 path is NOT a production tier (adding it as Tier 1 always hurts)
- Per-month stitching verification: 99-100% exact match across all quarters

**R2/R3: M-only is optimal.** Band width signals: MTM drift (5.4x), volume (1.7x).

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

## Next Steps

1. **Band calibration (Phase 6):** |baseline| magnitude bins for R1, MTM drift for R3, volume for R2
2. **Signal data:** SPICE_ANNUAL / DA_ANNUAL (deferred; needs production env)
3. **Production integration:** Migrate to ftr24/v1 conformal prediction framework
