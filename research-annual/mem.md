# Annual FTR Bid Pricing — Project Memo

## Task
- Understand our team's annual framework (both historical and recent)
- Design bid price strategies for annual (baseline → bands → clearing_prob → optimizer)
- The ftr24/v1 code pattern should remain; only annual-specific pieces change

## Corrected Structure (MISO)
- **3 rounds** (R1, R2, R3), NOT 4. Verified from code + data PY 2019-2025.
- **4 quarters** (aq1-aq4), each round auctions ALL quarters simultaneously.
- R1 has **no MTM for any quarter**. R2/R3 have previous round's MCP as MTM.
- PJM has 4 rounds and has MTMs for all rounds including R1.

## Key Constraints (Do Not Forget)
1. **R1 is the hard problem.** H (historical DA congestion) gives p95 errors of 3,300 — 6.7x worse than f0p. R1 bands must be very wide. Don't assume annual = narrow.
2. **R2/R3 are tractable.** Previous round MCP gives p95 ~600-700, comparable to f0p. f2p-style binning confirmed feasible (35k+ rows per bin).
3. **Limited data for R1.** Annual has 3 rounds/year vs f0p's monthly auctions. Cannot fit complex models. But R2/R3 pooled data is plentiful.
4. **Volume is moderate.** Annual avg MW/row = 4.24 vs f0p = 2.95 (1.44x). Not the dramatic difference originally assumed.

## f0p Baseline Reference
```
baseline = mtm_weight * mtm_1st_mean + rev_weight * 1(rev)
```
- f0: 0.77/0.23, f1: 0.85/0.15, f2: 0.94/0.06, f3: 0.93/0.07, q2-q4: ~0.92/0.08
- f0/f1: LightGBM bands. f2p: rule-based binned bands (by |mtm| size).
- 10 bid levels from band pairs at coverage targets 10/30/50/70/99.
- Width caps: low segment p90 (hard 1000), high segment p99 (hard 3000).

## Why Separate Annual Model
- R1 has no mtm_1st_mean → can't use f0p baseline formula directly
- 1(rev) in April has wrong season for Jun-Aug delivery → signal is weak
- Annual `aq1-aq4` are not wired in v1 band_generator (crashes with ValueError)
- Much less training data → LightGBM residual approach is not viable

## Data Access
```python
from pbase.analysis.tools.all_positions import MisoApTools, PjmApTools

# Cleared trades — MUST use full PY range to get all 4 quarters
miso_aptools = MisoApTools()
trades = miso_aptools.get_all_cleared_trades(
    start_date=pd.Timestamp("2024-06-01"),
    end_date=pd.Timestamp("2025-06-01")  # exclusive end, covers Jun-May
)
annual = trades[trades["period_type"].isin(["aq1","aq2","aq3","aq4"])]

# Ray init (local):
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

# R1 baseline reconstruction:
miso_tools = miso_aptools.tools
filled = miso_tools.fill_mtm_1st_period_with_hist_revenue(r1_trades_group)
# Groups by (auction_date, period_type, class_type) — fills mtm_1st_mean column
```

## Cached Data
```
/opt/temp/qianli/annual_research/
  annual_cleared_all.parquet       # 11.5M rows, PY 2019-2025 (old, category dtype)
  annual_cleared_all_v2.parquet    # 11.5M rows, PY 2019-2025, dtype fixed
  annual_with_mcp_v2.parquet       # 11.5M rows, + full MISO MCP data (corrected pipeline)
  r1_filled_v2.parquet             # 2.4M rows, R1 trades with H baseline filled
  all_residuals_v2.parquet         # 11.5M rows, all rounds with residuals (100% coverage)
  f0p_cleared_all.parquet          # 12.7M rows, PY 2019-2025, f0/f1/q4
  r1_filled.parquet                # (old, superseded by v2)
  all_residuals.parquet            # (old, superseded by v2)
```

## Key Code Locations
- `pmodel/base/ftr24/v1/band_generator.py` — f0p bands, BASELINE_CONFIG, clearing probs
- `pmodel/base/ftr24/v1/miso_base.py` — `_fill_mtm` (historical DA congestion proxy)
- `pmodel/base/ftr24/v1/autotuning.py` — round/period definitions
- `pmodel/base/ftr24/v1/params/prod/auc2603/miso_a_offpeak.py` — annual params
- `pbase/analysis/tools/miso.py` — `annual_round_day_map`, `fill_mtm_1st_period_with_hist_revenue`
- Notebooks: `/home/xyz/workspace/pmodel/notebook/hz/2025-planning-year/2025-26-annual/`

## Repo Structure
```
research-annual/
  mem.md                      ← this file
  stage1.md                   ← working doc with research plan + assumption tracking
  findings_data_profile.md    ← Phase 1 results: data profile tables
  findings_residuals.md       ← Phase 2 results: residual analysis tables (CORRECTED 2026-02-11)
  findings_legacy_pricing.md  ← Phase 3 results: legacy ftr23 code analysis + recommendations (UPDATED)
  findings_baseline_improvement.md ← Phase 3.5 results: R1 baseline improvement + R2/R3 supplementary features (2026-02-12)
  r1-research.md                ← R1 baseline research: all experiments, DA feature engineering, combined results (2026-02-12)
  notebooks/
    01_data_profile.ipynb     ← data loading + profiling code
    02_residual_analysis.ipynb ← residual computation code (superseded by 03)
    03_corrected_baseline.ipynb ← corrected pipeline with full MISO MCP data
  archive/                    ← previous analysis (learning.md, stage2_design.md, etc.)
```

## Current Stage: Stage 1 — Phase 1+2+3+3.5 COMPLETED (updated 2026-02-18), Phase 4 next
- Phase 1 (data exploration): DONE. See `findings_data_profile.md`.
- Phase 2 (residual analysis): **CORRECTED** 2026-02-11. R2/R3 baselines now use full MISO MCP data via `get_m2m_mcp_for_trades_all()`. R2/R3 coverage now 100% (was 68%/75%). R2/R3 residuals much smaller than previously reported. See `findings_residuals.md`.
- Phase 3 (legacy pricing analysis): **UPDATED** 2026-02-11. Added worked example and per-round×quarter grid with corrected numbers. See `findings_legacy_pricing.md`.
- Phase 3.5 (baseline improvement research): **COMPLETED** 2026-02-12. Six experiments on R1 improvement and R2/R3 supplementary features. See `findings_baseline_improvement.md`.
- Phase 4 (band calibration): Next. Design annual-specific band widths.
- A6 resolved: production uses ftr23 (v1 parametric approach, not ftr24/v1).

### Phase 3.5 Key Conclusions (Baseline Improvement Research, 2026-02-18 updated)

**200-word summary of all R1 research findings:**

R1's baseline H (historical DA congestion, 0.85 shrinkage) has p95=3,307 and 66% direction accuracy — 6.7× worse than f0p. We tested 7 experiments across two phases. **Phase 1 (baseline tuning):** Per-quarter bias correction (leave-one-year-out) reduced p95 by 4.5% to 3,157. Prior-year R1 MCP achieved p95=2,624 on matched paths (32% coverage), but this advantage partly reflects selection bias — persistent paths are inherently easier. Shrinkage tuning (0.85→1.0) gave only -2.3%. **Phase 2 (DA feature engineering):** Tested 5 theories on whole-market data (2019-2022): volatility weighting, middle-month emphasis, percentile clipping, positive-only filtering, and trend extraction. All failed to beat the simple mean — DA congestion is structurally disconnected from auction clearing prices. **Phase 3 (nodal stitching):** Reconstructed prior-year path MCPs from per-node data (MisoCalculator), boosting coverage from 32% to 92%. Direction accuracy improved +6.7pp (74.3% vs 67.6%), but p95 worsened (3,923) due to selection bias removal. Optimal blend: 25% nodal + 75% H (p95=3,639, dir acc=71.6%). **Bottom line:** backward-looking DA features have a ceiling. The highest-potential untested avenue is forward-looking signals (SPICE/DA constraint forecasts). For R2/R3, M-only baseline is definitively optimal; no non-leaking feature improves it.

**R1 improvements (ranked by impact):**
1. Per-quarter bias correction (LOO): p95 3,307→3,157 (-4.5%), dir acc 66%→70%. Low effort, universally applicable.
2. Prior-year R1 MCP: p95 2,624 vs H's 3,045 (-14%) on matched paths, but only 32% path coverage. 50/50 blend with corrected H yields p95=1,796 on matched paths.
3. Nodal MCP stitching: 32%→92% coverage, dir acc +6.7pp. Best blend 25% nodal + 75% H. p95 worse due to selection bias removal.
4. Shrinkage 0.85→1.0: p95 3,307→3,230 (-2.3%). Negligible.
5. DA feature engineering (volatility, middle-month, percentile, positive-only, trend): ALL failed. No signal.
6. Signal data (SPICE/DA): not accessible for testing, highest-potential future avenue.

**Direction accuracy by |MCP| threshold:**
- |MCP| >= 0: 66.2% (all paths). |MCP| >= 100: 70.3%. |MCP| >= 500: 79.2%. |MCP| >= 1000: 85.1%.
- H is directionally useful only for large MCPs; near-zero paths are coin flips.

**R2/R3 conclusions:**
- M-only baseline is definitively optimal. No non-leaking feature improves it.
- split_month_mcp was identified as data leakage (r=0.999 with outcome).
- MTM drift (|mtm_now_0 - mtm_1st_mean|) is excellent for R3 band width scaling: 5.4× range in p95 by drift bin.
- Volume is useful for band width scaling: 1.7-1.8× range in p95 by volume decile.

## Stage 2: Band Analysis
(Blocked on Stage 1 Phase 3+4 completion)


## current conversation — Detailed Research Report

### Phase 1+2: Data Loading and Residual Analysis

We loaded 11.5 million annual cleared trade rows and 12.7 million f0p (monthly forward) cleared trade rows from PY 2019–2025 using `MisoApTools.get_all_cleared_trades()`. The annual data covers all 3 rounds × 4 quarters × 2 class types across 7 planning years. The f0p data covers f0, f1, q4 period types for the same years and is used purely as a benchmark for comparison.

**What is the "baseline" for each round?**

The baseline is the system's best guess of what the MCP (market clearing price) will be for each path, before any bid adjustment. The residual (error) is: `residual = actual_MCP - baseline`. The smaller the residual, the better the baseline predicts the clearing price, and the narrower our bid bands can be.

- **R1 baseline ("H" = Historical DA Congestion):** For Round 1, there is no prior auction clearing price to reference. The only information available is historical day-ahead (DA) congestion from the prior year. The function `fill_mtm_1st_period_with_hist_revenue()` (`pbase/analysis/tools/miso.py:322`) computes H by:
  1. Taking each path's source and sink nodes
  2. Loading monthly-average DA congestion prices for the delivery months (e.g., for aq1=Jun-Aug, it loads Jun/Jul/Aug of the prior year)
  3. Only using months before the April cutoff date (this is why aq4 gets only 1 month of data instead of 3)
  4. Applying a 0.85 shrinkage factor to congestion prices in the "profitable direction" (positive sink congestion × 0.85, negative source congestion × 0.85) as a conservative adjustment
  5. Averaging across planning years, then across nodes
  6. Setting `mtm_1st_mean = sink_congestion - source_congestion` as the baseline

  This gives a single number per path that represents "what DA congestion looked like on this path last year." **Coverage: 100% of R1 trades** (2,388,474 out of 2,393,574 rows; 5,100 lost to node coverage gaps).

- **R2 baseline ("M" = previous round's MCP):** For Round 2, the system looks up what each path's MCP was in Round 1. This is loaded via `get_m2m_mcp_for_trades_all()` which matches on (planning_year, period_type, class_type, source_id, sink_id). **Coverage: 68%** — the remaining 32% are paths that first appeared in R2 (not traded in R1), so there's no R1 MCP to match against.

- **R3 baseline:** Same as R2 but matches to R2's MCP. **Coverage: 75%** — more paths overlap between R2 and R3 than between R1 and R2.

**How bad is each baseline? (Residual analysis results)**

The critical finding is that R1's baseline (H) is a very weak predictor of the actual clearing price, while R2/R3's baseline (M) is reasonably good:

| Metric | R1 (H baseline) | R2 (M baseline) | R3 (M baseline) | f0p (MTM baseline) |
|--------|-----------------|-----------------|-----------------|-------------------|
| Mean |residual| | 849 | 191 | 157 | 133 |
| Median |residual| | 360 | 89 | 75 | 59 |
| p95 |residual| | 3,300 | 693 | 568 | 492 |
| p99 |residual| | 7,346 | 1,540 | 1,244 | 1,157 |
| Direction accuracy | 65% | 91% | 93% | ~95% |

"Direction accuracy" means: does the baseline correctly predict whether the MCP will be positive or negative? For R1, the baseline gets the sign right only 65% of the time (barely better than a coin flip for some quarters). For R2/R3, it's 91–93% correct.

The p95 |residual| means: 95% of the time, the actual MCP is within this distance of the baseline. For R1, that distance is 3,300 — meaning our bid bands need to span at least ±3,300 around H to cover 95% of outcomes. For f0p, the same metric is only 492. **R1 is 6.7× harder to predict than f0p.**

R2/R3 residuals (p95 = 568–693) are 15–40% larger than f0p's 492, but in the same ballpark. This means R2/R3 can use a similar band approach to f0p (which uses rule-based binning by |MTM| magnitude).

**Quarter-level findings:**
- aq4 (Mar-May) residual magnitudes are NOT worse than other quarters — actually slightly easier than aq2 (Sep-Nov), which is the hardest quarter by magnitude (p95 = 3,712).
- However, aq4 has notably worse direction accuracy: 62% vs 66–67% for other quarters. This is consistent with the structural issue that aq4 only gets 1 month of DA data (March) instead of 3, because the April cutoff filters out April and May data.
- Conclusion: same formula for all 4 quarters is justified (magnitudes are similar), but aq4 might benefit from wider symmetric bands to compensate for worse direction prediction.

**R2/R3 training data feasibility for f2p-style binning:**
We binned R2/R3 trades by |M| magnitude into tiny (<50), small (50-250), medium (250-1k), large (1k+). Every bin has >35,000 rows per planning year — far exceeding the minimum threshold. f2p-style binned band calibration is completely feasible for R2/R3.

### Phase 3: Legacy Code Analysis (ftr23 Bid Pricing Pipeline)

We read the full production codebase for annual bid pricing at `/home/xyz/workspace/pmodel/src/pmodel/base/ftr23/`. The key files are:

- `v1/miso_models_a_prod_r1.py` — MISO annual production model (used for R1)
- `v3/params/prod/auc25annual/miso_models_a_prod_r2.py` and `r3.py` — MISO annual R2/R3
- `v1/base.py:94` — `_set_bid_price()`: the core formula that converts baseline into bid prices
- `v1/miso_base.py:111` — `_set_bid_curve()`: converts price levels into a multi-point bid curve
- `v3/base.py:3086` — `_add_price_related_columns()`: entry point that selects H or M baseline
- `v1/pjm_models_a_prod_r1.py` — PJM annual for comparison

**How the legacy system converts the baseline into actual bid prices:**

The full pipeline for every annual trade is:

1. **Baseline selection** (`_add_price_related_columns`, v3/base.py:3086):
   - If MISO annual R1: force `mtm_1st_mean = NaN`, then call `_fill_mtm()` to compute H from DA congestion
   - If R2/R3: call `get_m2m_mcp_for_trades_all()` to get M = previous round's MCP
   - Since `prediction_class_instance = None` in ALL annual params files: `mcp_pred = mtm_1st_mean`. There is no machine learning prediction layer. The baseline IS the bid center.

2. **Signal-based path ranking** (within `generate_trades_one_auc_period_round`):
   - Three signal families are loaded for each round:
     - `SPICE_ANNUAL_V4.4/V4.5` — congestion price forecast from SPICE model
     - `DA_ANNUAL_V1.4` — day-ahead LMP forecast
     - `TMP_VIOLATED_{tag}` — transmission constraint violation indicators
   - These signals produce per-node constraint exposure scores at different tiers (tier0 = most important constraints, tier4 = least important)
   - The exposure score for each path = sink_exposure - source_exposure on the top-tier constraints

3. **Bid price computation** (`_set_bid_price`, v1/base.py:94):
   - Trades are grouped by (trade_type, mtm_flow_type) — that is, buy/sell × prevailing/counter flow direction
   - Within each group, paths are ranked by their `exposure_tier1_bid_price` score (percentile rank from 0 to 1)
   - The rank is mapped to a bid adjustment through a parametric power function:
     ```
     price_change = |mcp_pred| × change_pct + offset
     ```
     where `change_pct` is mapped from `[change_lb, change_ub]` = `[-0.5, 1.5]` and `offset` from `[offset_lb, offset_ub]` = `[-50, 150]` based on the path's rank
   - The mapping uses `power=2` with `thres=0.2`, meaning: the bottom 20% of paths (by exposure) get roughly zero adjustment, and the adjustment ramps up quadratically for higher-ranked paths
   - `price_change` is clipped at `price_change_cap = 2000`
   - Two bid levels are set:
     - `predicted_bid_1 = mcp_pred + price_change` (the aggressive, signal-adjusted price)
     - `predicted_bid_2 = mcp_pred` (the raw baseline itself)
   - Buy bids are clipped at 5000

4. **Bid curve construction** (`_set_bid_curve`, v1/miso_base.py:111):
   - Takes the predicted_bid columns and applies ×3 quarterly scaling (because aq1-aq4 are quarterly periods)
   - For sell trades, signs are flipped
   - Prices are sorted from highest to lowest
   - Volumes are interpolated linearly: `bid_volume_1 = 0` (at the highest/most aggressive price), `bid_volume_k = total_bid_volume` (at the lowest/most conservative price), intermediate points linearly interpolated
   - Pads to 10 bid points with NaN

**Concrete example of what this produces for an R1 buy/prevailing trade with H=500:**

| Path Rank (by signal exposure) | change_pct | offset | price_change | bid_1 (before ×3) |
|-------------------------------|-----------|--------|-------------|-------------------|
| Bottom (rank=0, weak signals) | -0.5 | -50 | -300 | 200 |
| 20th percentile (≈thres) | ~0 | ~0 | ~0 | ~500 |
| Median (rank=0.5) | ~0.5 | ~50 | +300 | 800 |
| Top (rank=1.0, strong signals) | +1.5 | +150 | +900 | 1,400 |

After ×3 quarterly scaling: bid prices become 600, 1,500, 2,400, 4,200 respectively.
The second bid level (predicted_bid_2) = H = 500 (×3 = 1,500).
So the final bid curve for the top-ranked path spans from ~4,200 (aggressive) down to ~1,500 (conservative).

**The critical problem: these params are IDENTICAL for R1, R2, and R3.**

In the code (`miso_models_a_prod_r1.py:90-108`), the round-specific parameter overrides are:
```python
@property
def _a_r1_params(self):
    params = deepcopy(self.params)
    update_params = {}          # <-- EMPTY. No R1-specific changes.
    params.update(update_params)
    return params

@property
def _a_r2_params(self):
    params = deepcopy(self.params)
    update_params = {}          # <-- EMPTY. No R2-specific changes.
    params.update(update_params)
    return params
```

All three rounds use the exact same base `self.params` dictionary. This means the bid spread is the same for R1 (where p95 residual is 3,300) and R3 (where p95 residual is 568). The system treats all rounds as equally uncertain, which is empirically wrong by a factor of 6×.

**How the legacy R1 bid range compares to what's actually needed:**

For a typical R1 path with H=500 (roughly median):
- Legacy max bid range: from (H×0.5 - 50) to (H + min(H×1.5 + 150, 2000)) = 200 to 1,400. Total range = 1,200.
- Actual MCP p95 range: H ± 3,300 = from -2,800 to +3,800. Total range = 6,600.
- **Legacy covers ~18% of the actual p95 range for a median-H path.**

For a large R1 path with H=2,000:
- Legacy max: 2,000 + min(3,150, 2,000) = 4,000. Min: 2,000×0.5 - 50 = 950. Range = 3,050.
- Actual p95 range for large-|H| paths: ±8,772. Total range = 17,544.
- **Legacy covers ~17% of the actual p95 range for a large-H path.**

The `price_change_cap = 2000` is the binding constraint — it prevents the system from bidding more than 2,000 above H, regardless of how strong the signal is. But for large paths, the actual MCP can be 8,000+ away from H.

**Positive bias is not corrected:**

Across all R1 trades PY 2019-2025, actual MCPs clear on average +420 above H (the mean signed residual is +420). This bias is consistent across quarters (ranging from +375 for aq4 to +489 for aq2). The legacy system sets the bid center at exactly H, without adding any bias correction. This means we are systematically bidding below the expected clearing level.

However, the bias is not stable across planning years: it ranges from +149 (PY 2020) to +679 (PY 2022). A fixed correction of +400 would help on average but wouldn't be reliable year to year.

### R2/R3 Legacy Assessment

R2/R3 use the same parametric framework as R1, but with the much stronger M baseline (previous round MCP). The assessment:

- The M-only baseline is confirmed optimal: we tested blending M with H at ratios of 95/5, 90/10, and 80/20. Every blend is strictly worse than M-only (p95 goes from 632 up to 725+ as H weight increases). Adding historical DA congestion to the previous round's MCP adds noise, not signal.
- The parametric bid spread (±2,000 max range) is too wide for R2/R3. With p95 residuals of 600-700, the system is spreading bids over a range 3× wider than necessary, which wastes bid stack space on prices that will almost never clear.
- The f2p-style approach (binning paths by |M| magnitude, computing historical residual quantiles per bin) is confirmed feasible with >35k rows per bin per PY.

### PJM Annual Comparison

PJM annual uses the same ftr23 parametric framework but with several differences:

- **4 rounds** (R1–R4) instead of MISO's 3
- **PJM R1 HAS real MTM** — PJM's R1 can reference prior forward clearing prices, so the baseline is fundamentally more informative than MISO's year-old DA congestion proxy
- **PJM differentiates rounds by scaling params ×2/3** in later rounds — at least some round-specific adjustment, unlike MISO's identical params
- **PJM is more conservative:** CPR (cost-per-round target) = 0.65 vs MISO's 1.1; counter cost path UB = 300 vs MISO's 500
- **Bid curve scaling:** PJM uses ×12 for annual products vs MISO's ×3 for quarterly products
- Same signal framework (SPICE/DA/TMP_VIOLATED) but with PJM-specific versions

### Baseline Improvement Experiments

**R1 multi-year averaging (n_years = 1, 2, 3):**
The default H baseline uses 1 year of lookback DA congestion data. We tested whether averaging over 2 or 3 years improves the baseline:

| n_years | p95 |residual| | Mean |residual| | Verdict |
|---------|-------------------|-----------------|---------|
| 1 (default) | 3,307 | 848 | Baseline |
| 2 | 3,326 | 863 | No improvement (+0.6%) |
| 3 | 3,412 | 891 | Slightly worse (+3.2%) |

Multi-year averaging does not help. Year-over-year DA congestion patterns are not stable enough for averaging to reduce variance. The fundamental problem is that last year's DA congestion is a weak predictor of this year's auction clearing price, and adding more years of old data doesn't fix that.

**R1 residual structure by |H| magnitude:**

| |H| bin | n trades | Mean |Residual| | p95 |Residual| | Direction Accuracy |
|---------|---------|----------------|----------------|-------------------|
| tiny (<50) | 788k | 349 | 1,279 | 50% (coin flip) |
| small (50-250) | 559k | 590 | 2,249 | 68% |
| medium (250-1k) | 570k | 929 | 3,430 | 77% |
| large (1k+) | 471k | 1,653 | 8,772 | 90% |

Key insight: when H is near zero (tiny bin), the baseline tells us almost nothing about the MCP — direction accuracy is 50% (random). When |H| is large (>1,000), the baseline is highly directionally accurate (90%) but the absolute error is massive (p95 = 8,772). This means residuals scale strongly with |H| magnitude, which argues for |H|-dependent bid widths rather than fixed parametric bounds.

### Recommendations

**For R1 (immediate, for upcoming auction):**
1. Create R1-specific parameter overrides (the `update_params = {}` dictionaries should not be empty)
2. Raise `price_change_cap` from 2,000 to at least 4,000 to cover the p95 residual of 3,300
3. Add approximately +400 bias correction to the bid center: `mcp_pred = H + 400`
4. Widen the parametric bounds: `change_ub` from 1.5 to ~2.5, `offset_ub` from 150 to ~300
5. Keep the signal-ranked adjustment — it's the most valuable component in the system
6. Consider |H|-dependent adjustments: tiny-H paths need tighter bands (~1,300) while large-H paths need much wider bands (~8,800)

**For R2/R3 (immediate):**
1. Create separate R2 and R3 parameter overrides with tighter bounds than R1
2. Use M-only baseline (do NOT blend with H — confirmed to always make things worse)
3. Consider migrating to f2p-style binned width tables using historical annual R2/R3 residuals

**For medium term:**
Migrate annual pricing into the ftr24/v1 conformal prediction band framework, which generates bands at explicit coverage levels (10/30/50/70/99%) calibrated to the actual residual distribution. This is fundamentally better than the ftr23 parametric approach because it's data-driven rather than hand-tuned.

See `findings_legacy_pricing.md` for the complete code analysis with line references.

Apart from what you said:
# the summarization part is quite well written. 
- one purpose of our research is to understand how effective the old pricing model is
    - summarize the old model for me with an example in the report.
    - and for each round, each aq type, find me the key findings, the stats, and your suggestions.

## on your report
- do more research on R1's baseline to see how we can beat current production
  - on improving r1: so your source - sink are both averages right? 
  - for aq1, jun-aug, what about we do some kind of normalization/ clipping/ percentage extraction / time series related prediction
  - for instance: for two nodes both with average historical LMP in da in the same period, if node A is more spiky, maybe I'd preferred this, as spiky means more revenue
  - but I do believe there are many other rules we can try to extract here. build theories, and test them ONE BY ONE. 
  - use the whole market path data not just from a few competitors. test on data from 2019-2022

- maybe the idea is right just the format, features used, or the params needs tuning
- for r2/r2: u saying that pure mtm_1st_mean is the best? but if there is anything we can add to it, what would it be?

another theory: for annual trade, maybe the middle bump in price is more favorable.
- example: nodeA has bump similar to node b, but one happens early in june whereas one in mid july. I'd prefer mid july price bump.

## testing r1 theories
- 89.5% node overlap what does this mean?

- add in the signals and see if they mean anything. find a way to use them and load them. 

## long-term round 5 pjm (year1 round 5)
  - check how close it is. 