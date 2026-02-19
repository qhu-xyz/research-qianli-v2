# MISO Annual FTR Bid Pricing — Research Runbook

**Repo:** `research-qianli-v2/research-annual` | **Updated:** 2026-02-18

Practical domain reference for MISO annual FTR research. For experiment results, see `findings_all_quarters.md`.

---

## 1. Auction Structure

3 rounds (R1, R2, R3), each auctioning all 4 quarters simultaneously. ~April 8 / April 22 / May 5.

| Quarter | Delivery | Notes |
|---------|----------|-------|
| aq1 | Jun, Jul, Aug | Summer |
| aq2 | Sep, Oct, Nov | Fall |
| aq3 | Dec, Jan, Feb | Crosses calendar year |
| aq4 | Mar, Apr, May | Spring; H gets only 1 month DA data (March) |

Class types: `onpeak`, `offpeak`. MTM: R1 has none (uses H); R2 uses R1 MCP; R3 uses R2 MCP.

---

## 2. Planning Year & Year Mapping

PY = starting year: PY 2024 = Jun 2024 through May 2025.

**Year mapping for prior data** (most critical formula):

```
delivery_month >= 6  →  dy = PY - 1
delivery_month < 6   →  dy = PY
```

| PY | aq1 (6,7,8) | aq2 (9,10,11) | aq3 (12,1,2) | aq4 (3,4,5) |
|----|-------------|---------------|--------------|-------------|
| 2024 | dy=2023 | dy=2023 | Dec:2023, Jan/Feb:2024 | dy=2024 |

**Bug found:** Previous code used `dy = PY - 2` (2 years stale). Corrected in all experiment scripts.

---

## 3. Quarterly Forward Schedule

| Product | Auction Months | Delivery | In f0p parquet? |
|---------|---------------|---------|-----------------|
| q1 | Does not exist | — | — |
| q2 | July | Sep/Oct/Nov | No (Ray only) |
| q3 | July, October | Dec/Jan/Feb | No (Ray only) |
| q4 | July, October, January | Mar/Apr/May | Yes |

---

## 4. Baselines by Round

**R1 "H" baseline:** Historical DA congestion from `fill_mtm_1st_period_with_hist_revenue()` in `pbase/analysis/tools/miso.py:322`. Uses prior-year DA settlement congestion with 0.85 shrinkage. Coverage ~100%.

**R2/R3 "M" baseline:** Prior round's clearing price via `aptools.tools.get_m2m_mcp_for_trades_all()`. M-only is optimal — any blend with H is strictly worse (even 5% H weight raises p95 by 5%).

---

## 5. R1 Residual Statistics

PY 2019-2025, from `all_residuals_v2.parquet`:

| Quarter | n | Bias | MAE | p95 | Dir% |
|---------|---|------|-----|-----|------|
| aq1 | 618K | +411 | 838 | 3,171 | 66% |
| aq2 | 629K | +489 | 943 | 3,712 | 67% |
| aq3 | 572K | +402 | 816 | 3,211 | 67% |
| aq4 | 569K | +375 | 797 | 3,107 | 62% |

R1 is 12.7x worse than R2 by p95. Bias ranges +148 (PY 2020) to +679 (PY 2022).

**By |H| bin:** Dir% ranges from 50% (tiny <50) to 90% (large 1k+). Band widths must scale with |H|.

---

## 6. R1 Experiment Results (Summary)

Full results with per-quarter tables in **`findings_all_quarters.md`**.

**Key finding: 2-tier cascade is optimal for all 4 quarters.**

| Tier | Source | Coverage | Purpose |
|------|--------|----------|---------|
| 1 | Nodal f0 (3-month avg) | 98.8-100% | Primary baseline |
| 2 | H bias-corrected (LOO) | 0-1.2% | Fallback for missing nodes |

Head-to-head win rate: Nodal f0 wins 51.8% vs f0 path 48.0% — invariant across all quarters.

f0 path is NOT a production tier. Adding it as Tier 1 always hurts (+4-5 MAE, -0.3-0.6pp dir).

**Early-phase experiments** (prior-year MCP, shrinkage sweep, bias correction, multi-year averaging) are archived in `archive/phase1_3/findings_baseline_improvement.md`. Summary:
- Per-quarter LOO bias correction: -4.5% p95, +3.5pp dir (universally applicable)
- Prior-year MCP: 32% coverage, -41% p95 on matched paths (now superseded by nodal f0 at 99%+ coverage)
- Shrinkage sweep: negligible effect, not worth changing
- Multi-year averaging: hurts, not recommended

---

## 7. R2/R3 Research (Summary)

M-only is optimal. Band width signals: volume decile (1.7x range) and MTM drift (5.4x range for R3).

f2p-style binning is confirmed feasible: all |M| bins have >21K rows per PY.

---

## 8. Nodal Replacement & Stitching

Nodes get renamed over time. `MisoNodalReplacement().load_data()` has ~920 records spanning 2013-2025.

**Key rule:** Filter replacements to the **target PY delivery date** using `build_py_fwd_map()`, then register MCP under all aliases via `get_all_aliases()` (BFS through forward+reverse chains). Both functions are in `scripts/baseline_utils.py`.

**Target dates by quarter:**

| Quarter | Target | Notes |
|---------|--------|-------|
| aq1 | `{PY}-08` | Same calendar year |
| aq2 | `{PY}-11` | Same calendar year |
| aq3 | `{PY+1}-02` | Crosses year boundary |
| aq4 | `{PY+1}-05` | Crosses into PY+1 |

**Stitching:** `path_mcp = sink_node_f0 - source_node_f0`, averaged across 3 delivery months. Use `get_mcp_df()` **column 0** (f0 monthly forward). Column -1 is annual R1 — wrong product.

**Verification:** Per-month stitching matches path f0 at 99-100% ($0.015 tolerance). The ~37% "exact match" in 3-month averages is expected: paths traded in <3 months average fewer months than nodal.

---

## 9. Legacy System (ftr23)

Production annual pricing: `pmodel/base/ftr23/`. Key issue: **all 3 rounds use identical parameters** — same bid spread for R1 (p95=3,307) and R3 (p95=202), a 16x calibration error.

Other problems: `price_change_cap=2000` too small for R1 large paths (actual p95=8,772); positive bias uncorrected; parametric bounds too narrow.

Detailed analysis archived in prior runbook versions.

---

## 10. Data Locations

All at `/opt/temp/qianli/annual_research/`:

| File | Description |
|------|-------------|
| `annual_cleared_all_v2.parquet` | 11.5M rows, PY 2019-2025, all rounds/quarters |
| `all_residuals_v2.parquet` | 11.5M rows, with residuals (100% coverage) |
| `r1_filled_v2.parquet` | 2.4M R1 trades with H baseline |
| `f0p_cleared_all.parquet` | 12.7M rows, f0/f1/q4 period types |
| `crossproduct_work/aq{1,2,3,4}_all_baselines.parquet` | Per-quarter experiment outputs |

Old files (`*_v1`, non-`_v2`): superseded, do not use.

---

## 11. Code Locations

**pbase:**
- `analysis/tools/miso.py:322` — `fill_mtm_1st_period_with_hist_revenue()` (R1 H baseline)
- `analysis/tools/miso.py` — `annual_round_day_map` (auction calendar)
- `data/m2m/calculator.py` — `MisoCalculator.get_mcp_df()` (nodal MCPs)
- `data/dataset/replacement.py` — `MisoNodalReplacement` (node rename table)
- `analysis/tools/all_positions.py` — `MisoApTools.get_all_cleared_trades()`, `get_m2m_mcp_for_trades_all()`, `cast_category_to_str()`

**Research scripts:**
- `scripts/baseline_utils.py` — Shared eval/print/replacement utilities
- `scripts/run_aq{1,2,3,4}_experiment.py` — Per-quarter R1 baseline experiments

**Legacy (ftr23):**
- `pmodel/base/ftr23/v1/base.py:94` — `_set_bid_price()`
- `pmodel/base/ftr23/v1/miso_base.py:111` — `_set_bid_curve()`
- `pmodel/base/ftr23/v3/base.py:3086` — `_add_price_related_columns()`

---

## 12. Environment

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

Ray (required for all pbase data loaders):
```python
from pbase.config.ray import init_ray
import pmodel
init_ray(address='ray://10.8.0.36:10001', extra_modules=[pmodel])
```

Memory rules: see `CLAUDE.md`. Key points: use polars, lazy scan, `del + gc.collect()`, print `mem_mb()`, shutdown Ray after use.

---

## 13. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Coverage % | Fraction of paths where baseline is non-null |
| Bias | mean(actual - baseline). Positive = underestimate |
| MAE | mean(\|actual - baseline\|) |
| p95 AE | 95th percentile absolute error (drives band width) |
| Dir Acc | sign(baseline) == sign(actual). Excludes zeros |

**Warning:** Baselines with different coverage are NOT comparable via overall metrics. Always use head-to-head on matched paths.

---

## 14. Known Issues

1. **Year mapping bug:** `dy = PY - 2` (stale) vs correct `PY - 1`/`PY`. Always verify.
2. **aq4 gets 1 month DA data.** April cutoff → March only → Dir% ~62-64%.
3. **q1 does not exist.** aq1 has no quarterly forward.
4. **f0p parquet has f0, f1, q4 only.** q2/q3 must come from cleared trades via Ray.
5. **Replacement target dates differ by quarter.** aq1: `{PY}-08`, aq2: `{PY}-11`, aq3: `{PY+1}-02`, aq4: `{PY+1}-05`.
6. **Replacement dates are tz-aware (EST).** Must `tz_localize()` before comparison.
7. **`get_mcp_df()` column 0 = f0, column -1 = annual R1.** Column count varies per month.
8. **`cast_category_to_str()` required** after `get_all_cleared_trades()`.
9. **Selection bias:** Low-coverage baselines appear better (cover easier paths). Use head-to-head only.
10. **PY 2019 has no prior-year data.** Exclude from comparisons.
11. **R1 bias is not stable** (+148 to +679 by PY). Use LOO correction, not fixed offset.
12. **NEVER run long scripts via `claude -r`.** OOM crash loops.
