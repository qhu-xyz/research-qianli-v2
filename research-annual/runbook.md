# MISO Annual FTR Bid Pricing — Research Runbook

**Repo:** `research-qianli-v2/research-annual` | **Updated:** 2026-02-22

Practical domain reference for MISO annual FTR research. For full experiment tables, see `findings.md`.

---

## 1. Auction Structure

Timeline: `PY 2022 R1 -> R2 -> R3 -> f0 monthly -> PY 2023 R1 -> R2 -> R3 -> f0 -> ...`

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
delivery_month >= 6  ->  dy = PY - 1
delivery_month < 6   ->  dy = PY
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

## 5. R1 Research Results

Full per-quarter tables in **`findings.md`**.

### Phase 1: Baseline Comparison

**Nodal f0 stitch is the best R1 baseline across all 4 quarters.** `path_mcp = sink_f0 - source_f0` from `get_mcp_df(market_month=PY-1)` col 0, averaged over 3 delivery months.

| Quarter | H MAE | Nodal f0 MAE | H Dir% | Nodal Dir% | Improvement |
|---------|------:|-------------:|-------:|-----------:|------------:|
| aq1 | 934 | 798 | 67.7% | 80.9% | -15% MAE, +13pp Dir |
| aq2 | 1,070 | 947 | 69.0% | 82.1% | -11% MAE, +13pp Dir |
| aq3 | 920 | 797 | 69.1% | 83.8% | -13% MAE, +15pp Dir |
| aq4 | 893 | 704 | 64.3% | 84.8% | -21% MAE, +21pp Dir |

Ranking identical across all quarters: **Nodal f0 ≈ f0 path < f1 < R3 ≈ R2 < R1 < QF < H**.

2-tier cascade (Nodal f0 -> H fallback) is optimal. f0 path as Tier 1 always hurts (+4-5 MAE).

### Phase 2: Improving Nodal f0

**Alpha scaling** (`pred = alpha * nodal_f0`, LOO by PY):
- Nodal f0 has persistent positive bias (+200-340) — it underestimates MCP by ~50%.
- Multiplying by alpha=1.55-1.60 corrects this. Direction accuracy is unchanged (scaling preserves sign).

| Quarter | Raw MAE | Alpha-scaled MAE | Improvement |
|---------|--------:|-----------------:|------------:|
| aq1 | 798 | 736 | -8% |
| aq2 | 947 | 815 | -14% |
| aq3 | 797 | 668 | -16% |
| aq4 | 704 | 566 | -20% |

**Alpha + prior-year residual** (`pred = alpha * f0 + beta * (PY-1 mcp - PY-1 f0)`):
- Exploits residual autocorrelation (lag-1 r = 0.41-0.71).
- Only ~22% of paths have prior-year data (recurring paths from PY-1's R1).
- Gives further 2-11% on top of alpha-scaling on those paths.

| Quarter | Alpha-only MAE | Combined MAE | vs raw |
|---------|---------------:|-------------:|-------:|
| aq1 | 669 | 663 | -10% |
| aq2 | 674 | 650 | -17% |
| aq3 | 620 | 570 | -23% |
| aq4 | 536 | 459 | -31% |

### Phase 2 Dead Ends

**Signal blending does not work.** Tested: convex combination (w * f0 + (1-w) * H), Ridge regression, de-biased averaging/median, inverse-MAE weighting, best-available cascade. All lose to alpha-scaled nodal_f0.

- **Convex combination of f0 + H:** LOO-optimal weight on f0 = 0.95-1.00. H adds zero information.
- **Ridge regression:** Learns f0 weight = 1.9-2.3x — overfitting disguised as blending. Catastrophic on PY 2023.
- **De-biased blending:** Fixes bias but destroys Dir% (drops 10pp). Averaging in other signals flips signs.

**Why:** All available signals (H, f0_path, f1, priors, quarterlies) are noisier versions of the same congestion signal. They carry no independent directional information. Alpha-scaling is the only transformation that corrects bias while preserving direction perfectly.

### Recommended Production Cascade

| Tier | Method | Coverage | Notes |
|------|--------|----------|-------|
| 1a | `alpha * nodal_f0 + beta * prior_residual` | ~22% | Where path recurs from PY-1 R1 |
| 1b | `alpha * nodal_f0` | ~77% | Remaining paths with nodal coverage |
| 2 | H (fallback) | ~1% | Missing nodes only |

Optimal parameters (LOO): alpha = 1.40-1.60, beta = 0.15-0.40.

---

## 6. R2/R3 Research (Summary)

M-only is optimal. Band width signals: volume decile (1.7x range) and MTM drift (5.4x range for R3).

f2p-style binning is confirmed feasible: all |M| bins have >21K rows per PY.

---

## 7. Nodal Replacement & Stitching

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

**Verification:** Per-month stitching matches path f0 at 99-100% ($0.015 tolerance).

---

## 8. Legacy System (ftr23)

Production annual pricing: `pmodel/base/ftr23/`. Key issue: **all 3 rounds use identical parameters** — same bid spread for R1 (p95=3,307) and R3 (p95=202), a 16x calibration error.

Other problems: `price_change_cap=2000` too small for R1 large paths (actual p95=8,772); positive bias uncorrected; parametric bounds too narrow.

---

## 9. Data Locations

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

## 10. Code Locations

**pbase:**
- `analysis/tools/miso.py:322` — `fill_mtm_1st_period_with_hist_revenue()` (R1 H baseline)
- `analysis/tools/miso.py` — `annual_round_day_map` (auction calendar)
- `data/m2m/calculator.py` — `MisoCalculator.get_mcp_df()` (nodal MCPs)
- `data/dataset/replacement.py` — `MisoNodalReplacement` (node rename table)
- `analysis/tools/all_positions.py` — `MisoApTools.get_all_cleared_trades()`, `get_m2m_mcp_for_trades_all()`, `cast_category_to_str()`

**Research scripts:**
- `scripts/baseline_utils.py` — Shared eval/print/replacement utilities
- `scripts/run_aq{1,2,3,4}_experiment.py` — Per-quarter R1 baseline experiments
- `scripts/run_phase2_improvement.py` — Alpha-scaling + prior-year residual LOO
- `scripts/run_phase3_v2_bands.py` — Band width reduction experiments (5 configs)

**Legacy (ftr23):**
- `pmodel/base/ftr23/v1/base.py:94` — `_set_bid_price()`
- `pmodel/base/ftr23/v1/miso_base.py:111` — `_set_bid_curve()`
- `pmodel/base/ftr23/v3/base.py:3086` — `_add_price_related_columns()`

---

## 11. Environment

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

## 12. Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| Coverage % | Fraction of paths where baseline is non-null |
| Bias | mean(actual - baseline). Positive = underestimate |
| MAE | mean(\|actual - baseline\|) |
| p95 AE | 95th percentile absolute error (drives band width) |
| Dir Acc | sign(baseline) == sign(actual). Excludes zeros |

**Warning:** Baselines with different coverage are NOT comparable via overall metrics. Always use head-to-head on matched paths.

---

## 13. Known Issues

1. **Year mapping bug:** `dy = PY - 2` (stale) vs correct `PY - 1`/`PY`. Always verify.
2. **aq4 gets 1 month DA data.** April cutoff -> March only -> Dir% ~62-64% for H.
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

---

## 14. Pipeline & Versioning

### Overview

Experiments are versioned under `versions/{part}/` (e.g., `versions/baseline/v3/`). Each version has:
- `config.json` — method description, parameters, data sources, environment
- `metrics.json` — structured evaluation results (overall, per-PY, per-class, stability, matched)
- `NOTES.md` — human-readable explanation and decision rationale

A `promoted.json` file in each part directory tracks the current best version.

### CLI

```bash
python pipeline/pipeline.py create baseline v4 "description"
python pipeline/pipeline.py list baseline
python pipeline/pipeline.py compare baseline v4         # vs promoted
python pipeline/pipeline.py compare baseline v4 v2      # vs explicit version
python pipeline/pipeline.py promote baseline v4          # gate check + promote
python pipeline/pipeline.py promote baseline v4 --force  # override HARD failures
python pipeline/pipeline.py validate baseline            # schema checks
```

### Promotion Gates

| # | Gate | Severity | Check |
|---|------|----------|-------|
| G1 | MAE improvement | HARD | candidate matched MAE <= promoted matched MAE, all 4 quarters |
| G2 | Direction preserved | HARD | candidate matched Dir% >= promoted - 1.0pp, all 4 quarters |
| G3 | Coverage floor | HARD | candidate coverage >= 95%, all 4 quarters |
| G4 | Bias sign | SOFT | candidate bias >= 0, all 4 quarters |
| G5 | Per-PY stability | SOFT | mae_cv < 0.30, all 4 quarters |
| G6 | Worst-PY bound | SOFT | worst_py_mae < 1.5 * median_py_mae, all 4 quarters |
| G7 | Class parity | ADVISORY | |onpeak - offpeak| / avg < 0.40 |
| G8 | Win rate | ADVISORY | win_rate >= 50% on matched rows |
| G9 | Tail risk | ADVISORY | p99 <= promoted p99 * 1.10 |

- **HARD:** Must pass to promote (unless `--force`).
- **SOFT:** Failure needs written justification in NOTES.md.
- **ADVISORY:** Informational warnings only.
- G1/G2/G8/G9 use `matched` section (apples-to-apples on same paths), not `overall`.

### Staleness

If a candidate's `matched.compared_against` doesn't match the current promoted version, the `compare` command warns and skips comparative gates. `promote` refuses unless `--force`.

### Running a New Experiment

1. `python pipeline/pipeline.py create baseline v4 "description"`
2. Edit `config.json` to fill in method, parameters, data_sources
3. Run experiment script, at the end call:
   ```python
   from pipeline.pipeline import compute_full_evaluation, save_metrics
   metrics = compute_full_evaluation(df, candidate_col="prediction", promoted_col="nodal_f0", quarters=["aq1","aq2","aq3","aq4"])
   metrics["version"] = "v4"
   metrics["evaluated_at"] = "2026-..."
   metrics["compared_against"] = "v3"
   metrics["matched"]["compared_against"] = "v3"
   save_metrics(metrics, Path("versions/baseline/v4"))
   ```
4. `python pipeline/pipeline.py compare baseline v4`
5. `python pipeline/pipeline.py promote baseline v4`
6. Write NOTES.md with results and decision

---

## 15. Band Calibration

### Overview

Prediction intervals ("bands") around the baseline: given our f0 prediction, what range captures X% of true MCPs?

**Method (v1):** Symmetric empirical quantile bins.
1. Bin paths by `|nodal_f0|` using boundaries `[0, 50, 250, 1000, inf]` → tiny/small/medium/large
2. For each bin, compute quantile of `|mcp - nodal_f0|` at each coverage level
3. Apply symmetric bands: `nodal_f0 ± width`
4. Validate via leave-one-PY-out (train on 5 PYs, test on 1)

Coverage levels: P50, P70, P80, P90, P95.

### Current Promoted Results (per-class stratified)

All rounds now use per-class stratified bands: separate widths for onpeak/offpeak within each |baseline| bin. Coverage and widths are equivalent to pooled, but class parity gap is reduced from 0.05-1.02pp to 0.00-0.14pp.

**R1 v3** (4 quantile bins, per-class):

| Quarter | P95 cov | P95 error | P95 mean width |
|---------|--------:|----------:|---------------:|
| aq1 | 94.66% | -0.34pp | 2,664 |
| aq2 | 94.65% | -0.35pp | 3,126 |
| aq3 | 94.71% | -0.29pp | 2,497 |
| aq4 | 94.36% | -0.64pp | 2,073 |

**R2 v2** (6 quantile bins, per-class):

| Quarter | P95 cov | P95 error | P95 mean width |
|---------|--------:|----------:|---------------:|
| aq1 | 94.63% | -0.37pp | 217 |
| aq2 | 94.71% | -0.29pp | 228 |
| aq3 | 94.60% | -0.40pp | 202 |
| aq4 | 94.64% | -0.36pp | 211 |

**R3 v2** (6 quantile bins, per-class):

| Quarter | P95 cov | P95 error | P95 mean width |
|---------|--------:|----------:|---------------:|
| aq1 | 94.57% | -0.43pp | 181 |
| aq2 | 94.79% | -0.21pp | 182 |
| aq3 | 94.54% | -0.46pp | 162 |
| aq4 | 94.66% | -0.34pp | 163 |

### Cross-Round P95 Width Summary

| Quarter | R1 | R2 | R3 |
|---------|---:|---:|---:|
| aq1 | 2,664 | 217 | 181 |
| aq2 | 3,126 | 228 | 182 |
| aq3 | 2,497 | 202 | 162 |
| aq4 | 2,073 | 211 | 163 |

### Band Promotion Gates

| # | Gate | Severity | Check |
|---|------|----------|-------|
| BG0 | Baseline still promoted | HARD | band's baseline_version == current promoted baseline |
| BG1 | P95 coverage accuracy | HARD | \|actual - 95.0\| < 3.0pp for all 4 quarters |
| BG2 | P50 coverage accuracy | HARD | \|actual - 50.0\| < 5.0pp for all 4 quarters |
| BG3 | Per-bin uniformity (P95) | HARD | all 4 bins within 5pp of target, all 4 quarters |
| BG4 | Width narrower or equal | SOFT | candidate P95 width <= promoted width (if exists) |
| BG5 | Per-PY stability | SOFT | p95_worst_py_coverage >= 90.0 for all 4 quarters |
| BG6 | Width monotonicity | ADVISORY | p50 < p70 < p80 < p90 < p95 overall |
| BG7 | Class parity coverage | ADVISORY | \|onpeak - offpeak\| < 5pp at P95 |

### CLI

```bash
python pipeline/pipeline.py validate bands/r1     # R1
python pipeline/pipeline.py validate bands/r2     # R2
python pipeline/pipeline.py validate bands/r3     # R3
python pipeline/pipeline.py compare bands/r1 v3
python pipeline/pipeline.py compare bands/r2 v2
python pipeline/pipeline.py compare bands/r3 v2
python pipeline/pipeline.py list bands/r1
```

### Band Methodology

Symmetric empirical quantile bins with per-class (onpeak/offpeak) stratification. Bin boundaries computed from |baseline| percentiles on training set; separate widths calibrated per (bin, class). R1 uses 4 bins (nodal_f0 baseline, ~135K rows/quarter). R2/R3 use 6 bins (M baseline, ~1M rows/quarter).

**CV method:** Temporal expanding window (train on PYs < test PY) with `min_train_pys=3` filter. LOO by PY retained as secondary diagnostic in `loo_validation` section of metrics.

### Running Band Experiments

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
# R1 v1 (original)
python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_phase3_bands.py
# R1 v2 (width reduction)
python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_phase3_v2_bands.py
# R2/R3 v1 (pooled)
python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_r2r3_bands.py
# v3 per-class stratified (all rounds)
python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_v3_bands.py
```

Scripts output to `versions/bands/{r1,r2,r3}/{version}/metrics.json` directly. Then validate/compare/promote as above.
