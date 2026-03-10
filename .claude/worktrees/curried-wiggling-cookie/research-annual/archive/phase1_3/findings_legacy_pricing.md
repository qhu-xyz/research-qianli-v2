# Findings: Legacy Annual FTR Bid Pricing (ftr23)

**Source:** Code analysis of `/home/xyz/workspace/pmodel/src/pmodel/base/ftr23/` (v1, v3, v8)
**Date:** 2026-02-11
**Scope:** MISO annual R1/R2/R3, PJM annual, bid pricing pipeline end-to-end

---

## 1. How the Legacy System Works (End-to-End Pipeline)

### 1.1 The Pipeline

The ftr23 annual bid pricing pipeline has 4 stages:

```
Stage 1: Baseline Computation
  R1: mtm_1st_mean = H (historical DA congestion via _fill_mtm)
  R2/R3: mtm_1st_mean = M (previous round's MCP via get_m2m_mcp_for_trades_all)

Stage 2: MCP Prediction
  prediction_class_instance = None for ALL annual rounds
  => mcp_pred = mtm_1st_mean (no ML overlay)

Stage 3: Bid Price Setting (_set_bid_price_v1)
  Groups by (trade_type × mtm_flow_type) = (buy/sell) × (prevail/counter)
  Ranks paths by constraint exposure signal score
  Maps rank → price_change via parametric power function
  predicted_bid_1 = mcp_pred + price_change  (signal-adjusted)
  predicted_bid_2 = mcp_pred                 (raw baseline)

Stage 4: Bid Curve (_set_bid_curve)
  Sorts predicted_bid_1 and predicted_bid_2 into price levels
  Applies ×3 quarterly scaling for aq1-aq4
  Interpolates volumes linearly between bid points
  Outputs 2-10 (price, volume) pairs per trade
```

### 1.2 The Bid Price Formula (Stage 3 Detail)

For each path in a (trade_type, flow_type) group:

```python
# 1. Rank paths by constraint exposure from signals
rank = percentile_rank(exposure_tier1_bid_price)  # 0 to 1

# 2. Map rank to adjustment via power function
#    Uses scipy.fsolve to find breakpoint, then MinMaxScaler
#    power=2 means most paths cluster near the conservative end
change_pct = nonlinear_map(rank, change_lb, change_ub, power=2, thres=0.2)
offset     = nonlinear_map(rank, offset_lb, offset_ub, power=2, thres=0.2)

# 3. Compute price change
price_change = |mcp_pred| * change_pct + offset
price_change = clip(price_change, upper=price_change_cap)

# 4. Set bid prices
predicted_bid_1 = mcp_pred + price_change   # aggressive (signal-adjusted)
predicted_bid_2 = mcp_pred                   # conservative (raw baseline)
```

**What this means concretely (buy/prevail, using production params):**

| Path Rank | change_pct | offset | Bid (if H=500) | Bid (if H=2000) |
|-----------|-----------|--------|----------------|-----------------|
| Bottom (rank=0) | -0.5 | -50 | 500 + (-300) = 200 | 2000 + (-1050) = 950 |
| 20th pctile (rank≈thres) | ~0 | ~0 | 500 + 0 = 500 | 2000 + 0 = 2000 |
| Median (rank=0.5) | ~0.5 | ~50 | 500 + 300 = 800 | 2000 + min(1050, 2000) = 3050 |
| Top (rank=1.0) | 1.5 | 150 | 500 + 900 = 1400 | 2000 + min(3150, 2000) = 4000 |

The `price_change_cap = 2000` clips the adjustment for large |H| paths.

### 1.3 Production Parameters

**MISO Annual (v1/miso_models_a_prod_r1.py — same for R1, R2, R3):**

| Parameter | buy_prevail | buy_counter |
|-----------|------------|-------------|
| change_lb | -0.5 | -0.5 |
| change_ub | 1.5 | 1.5 |
| offset_lb | -50 | -50 |
| offset_ub | 150 | 150 |
| power | 2 | 2 |
| thres | 0.2 | 0.2 |
| price_change_cap | 2000 | 2000 |

Other key params:
- `exposure_reduction_mtm_lb/ub = ±3500` (exposure reduction only active within this MTM range)
- `lsr_mtm_neg_shrink_ratio = 0.6` (shrinks negative-MTM paths' contribution in optimizer)
- `lsr_mtm_pos_shrink_ratio = 1.0` (no shrink for positive-MTM paths)
- `shadow_price_upper_clip = 1,000,000` (effectively uncapped)
- Volume targets: R1=8100, R2=8600, R3=9100 MW total

### 1.4 Signal Inputs

Three signal families used for path ranking (same signals for all stages):

```
TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R{round}   — congestion price forecast
TEST.Signal.MISO.DA_ANNUAL_V1.4.R{round}       — DA LMP forecast
TEST.Signal.MISO.TMP_VIOLATED_{tag}.R{round}   — constraint violation indicator
```

The signals produce per-path exposure scores at different constraint tiers. `exposure_tier1_bid_price` (from the top 1 tier of constraints) drives the bid price ranking.

---

## 2. Critical Assessment: Does the Method Make Sense?

### 2.1 R1: Partially Correct Architecture, Poor Calibration

**What works:**

1. **H as center is the only option.** For R1, there's no prior MCP. Historical DA congestion is the best available proxy. The 0.85 shrinkage on profitable-direction congestion is a sensible conservative bias. (Our testing confirmed: multi-year averaging (n=2,3) doesn't improve p95 at all — stays at ~3,300.)

2. **Signal-ranked adjustment is the most valuable component.** SPICE, DA, and TMP_VIOLATED signals provide forward-looking constraint information that H doesn't have. Using these to differentiate bid aggressiveness per path is correct. Paths on likely-to-bind constraints should get more aggressive bids.

**What's wrong:**

3. **R1, R2, R3 use IDENTICAL parameters — wrong.**
   - `_a_r1_params`, `_a_r2_params`, `_a_r3_params` all return `deepcopy(self.params)` with zero round-specific overrides.
   - But R1 p95 residual = 3,300; R2 = 693; R3 = 568 (6.7x to 4.8x ratio).
   - Using the same bid spread for all rounds means R1 bands are systematically too narrow and R2/R3 bands are too wide.

4. **price_change_cap = 2000 is too low for R1.**
   - R1 p95 |residual| = 3,300. The cap limits maximum bid adjustment to 2000.
   - For any path with |H| > ~1,200, the cap binds: max bid = H + 2000, regardless of how high the path ranks.
   - Our data shows: for large |H| paths (>1000), p95 residual is 8,772. Cap of 2000 covers only 23% of the actual range.

5. **The +420 positive bias is unaddressed.**
   - MCPs systematically clear above H (mean bias = +420 across all R1 trades).
   - The bid center is H itself — no bias correction.
   - This means we're systematically bidding below the expected clearing level.

6. **No calibration to actual residual distribution.**
   - The bounds (change_lb=-0.5, change_ub=1.5, offset_lb=-50, offset_ub=150) appear hand-tuned.
   - They're not derived from historical residual quantiles.
   - They don't adapt to the magnitude of |H| (the same percentage bounds apply to H=50 and H=5,000).

7. **No clearing probability model.**
   - The system creates a bid curve between two prices (predicted_bid_1 and predicted_bid_2) with linear volume interpolation.
   - This doesn't reflect the actual uncertainty about clearing prices. The ftr24/v1 approach with conformal prediction bands at explicit coverage levels (10/30/50/70/99%) is fundamentally better.

8. **The ×3 quarterly scaling is applied AFTER bid pricing.**
   - `_set_bid_curve` multiplies all bid prices by 3 for quarterly periods.
   - But the parametric bounds (±50 offset, ±0.5 change) are calibrated for monthly-scale prices.
   - After ×3 scaling, the effective offset range becomes ±150 and change range ±1.5 on the quarterly-scaled price.
   - Whether this is appropriate depends on whether the quarterly MCP residuals also scale by 3× — unclear.

### 2.2 R2/R3: Structurally Sound, Calibration Issues

**What works:**

1. **M (previous round MCP) is a strong baseline.** R2/R3 direction accuracy = 90-93%, p95 residual = 568-693. This is comparable to f0p.

2. **Same signal-ranked framework applies.** The adjustment logic is the same, and SPICE/DA/TMP signals help differentiate paths.

**What's wrong:**

3. **Same parameters as R1.** R2/R3 need tighter bounds. The current params produce bid spreads appropriate for p95 ~2,000-3,000 residuals, but R2/R3 only have p95 ~600-700.

4. **The v3/auc25annual R2/R3 files are structurally identical to v1.** Despite being in the v3 directory, they don't use the v3 ElasticNet bid pricing approach. `prediction_class_instance = None` and no `bid_price_training_data_sv_file` key. They use the same v1 parametric approach.

5. **The v3 ElasticNet approach was never deployed for annual.** The more sophisticated `_set_bid_price` at v3/base.py:526 (which trains an ElasticNet on historical auction outcomes) exists but is only used for f0p monthly auctions where training data is plentiful. For annual, the v1 parametric approach is used.

### 2.3 PJM Annual: Better Differentiated

PJM's approach (from v1/pjm_models_a_prod_r1.py and v3/auc25annual/pjm_models_a_prod_r3/r4.py):

**Key differences from MISO:**

| Dimension | PJM | MISO |
|-----------|-----|------|
| Rounds | 4 (R1-R4) | 3 (R1-R3) |
| R1 MTM | Has prior forward clearing MCP | No MTM — uses H (DA congestion) |
| CPR target | 0.65 (conservative) | 1.1 (aggressive) |
| Counter cost path UB | 300 (differentiated) | 500 (same as prevail) |
| Offset/cap scaling | ×2/3 in later rounds | No round scaling |
| Period structure | Single "a" period | 4 quarters (aq1-aq4) |
| Bid curve scaling | ×12 (annual) | ×3 (quarterly) |

**PJM's approach is structurally better for R1** because PJM R1 actually has MTM from prior forward clearing. The baseline is fundamentally more informative than MISO's year-old DA congestion proxy.

PJM also applies ×2/3 parameter scaling in later rounds, which is a (crude) form of round differentiation. MISO doesn't differentiate rounds at all.

---

## 3. Experiment Setup: What We Verified Against

### 3.1 Data

All experiments used historical cleared trades from PY 2019-2025:
- **Annual:** 11.5M rows from `MisoApTools.get_all_cleared_trades()`, filtered to period_type ∈ {aq1, aq2, aq3, aq4}
- **f0p:** 12.7M rows for comparison (f0, f1, q4 periods)
- Cached at `/opt/temp/qianli/annual_research/`

### 3.2 Baseline Reconstruction

**R1 (H baseline):**
- Used `fill_mtm_1st_period_with_hist_revenue()` from `pbase/analysis/tools/miso.py:322`
- Groups trades by (auction_date, period_type, class_type)
- Computes sink - source DA congestion from prior year's delivery months (before April cutoff)
- Applies 0.85 shrinkage to profitable-direction congestion
- Result: `mtm_1st_mean` = H for each trade
- **Coverage: 100.0%** (2,388,474 of 2,393,574 R1 rows)
- Cached: `/opt/temp/qianli/annual_research/r1_filled.parquet`

**R2/R3 (M baseline):**
- Matched each R2 trade to R1's MCP on same (PY, period_type, class_type, source_id, sink_id)
- Matched each R3 trade to R2's MCP
- **R2 coverage: 68%**, R3 coverage: 75% (unmatched = new paths entering in later rounds)
- Only matched rows used for residual analysis

### 3.3 Residual Definition

```
residual = mcp_mean - baseline
```

Where:
- `mcp_mean` = actual clearing price from historical cleared trades
- `baseline` = H (for R1) or M (for R2/R3) or mtm_1st_mean (for f0p)

This is the error of the legacy system's bid center. The residual distribution tells us how wide the bid bands need to be.

### 3.4 Variants Tested

**R1 baseline variants:**

| Variant | Description | p95 |residual| | Verdict |
|---------|------------|-------------------|---------|
| H (n_years=1) | Default: 1 year lookback | 3,307 | Baseline |
| H (n_years=2) | 2 year lookback | 3,326 | No improvement |
| H (n_years=3) | 3 year lookback | 3,412 | Slightly worse |
| H + bias correction | Leave-one-out PY-level bias subtracted | 3,160 | -4.4% (marginal) |

Multi-year averaging doesn't help because year-over-year congestion patterns are not stable enough.

**R1 residual by |H| bin:**

| |H| bin | n | Mean |Res| p95 |Res| Dir Accuracy |
|---------|---:|--------:|--------:|-------------|
| tiny (<50) | 788k | 349 | 1,279 | 50% (random) |
| small (50-250) | 559k | 590 | 2,249 | 68% |
| medium (250-1k) | 570k | 929 | 3,430 | 77% |
| large (1k+) | 471k | 1,653 | 8,772 | 90% |

Key insight: H is informative about **direction** for large paths (90% accuracy) but nearly random for tiny paths. The absolute error scales strongly with |H|.

**R2/R3 blend test:**

| Blend (M weight, H weight) | R2 p95 | R3 p95 | Verdict |
|---------------------------|--------|--------|---------|
| M=1.0, H=0.0 | 632 | 568 | **Best** |
| M=0.95, H=0.05 | 674 | — | Worse |
| M=0.90, H=0.10 | 725 | — | Worse |

Adding ANY H to the R2/R3 baseline makes it worse. M alone is definitively best.

### 3.5 Key Comparison: Legacy R1 Bid Range vs Actual MCP Range

The legacy system's maximum bid range for a buy/prevail trade (with production params):

```
Min bid  = mcp_pred + mcp_pred * (-0.5) + (-50) = 0.5*H - 50
Max bid  = mcp_pred + min(mcp_pred * 1.5 + 150, 2000) = H + min(1.5*H + 150, 2000)
Range    = max - min = min(2*H + 200, 2000 + 0.5*H + 50)
```

For a typical path (H=500): range = 1,200. But actual MCP can be anywhere in H ± 3,300 (p95).
For a large path (H=2,000): range = 2,000 (cap binds). But actual p95 residual is 8,772.

**The legacy bid range covers roughly 35-55% of the actual MCP uncertainty range for R1.**

After ×3 quarterly scaling, the range becomes 3× wider in dollar terms, but the MCP residual (which is already on quarterly-scaled values) is also at the quarterly scale. So the coverage ratio doesn't change.

---

## 4. Recommendations

### 4.1 R1: Accept Wide Uncertainty, Maximize Signal Value

R1 is inherently hard. H is a weak predictor (p95 error = 3,300). No tested improvement to H reduces this significantly. The recommendations:

1. **Keep H as bid center** — no better option exists. Don't waste effort trying to improve H.

2. **Add bias correction of +400 to bid center.** MCPs systematically clear above H. Shift: `center = H + 400`. This doesn't reduce the spread but centers the bid stack better. (Per-PY bias ranges 149-679, so 400 is a stable middle ground.)

3. **Raise price_change_cap from 2000 to at least 4000.** The cap should cover p95 residual (3,300). Current cap of 2000 clips the bid range for all paths with |H| > ~1,200.

4. **Create R1-specific parameters.** R1 must not share params with R2/R3. R1 needs:
   - Wider change_ub (e.g., 2.5 instead of 1.5)
   - Wider offset_ub (e.g., 300 instead of 150)
   - Higher price_change_cap (≥4000)

5. **Keep signal-ranked adjustment.** This is the most valuable component. SPICE/DA/TMP_VIOLATED signals are the only forward-looking information available.

6. **Consider |H|-dependent bid ranges.** Our data shows residuals scale strongly with |H|. tiny-|H| paths need ±1,300; large-|H| paths need ±8,800. A fixed percentage/offset doesn't work well across this range. Use |H|-binned width tables (like f2p) instead of parametric bounds.

### 4.2 R2/R3: Migrate to Calibrated Bands

R2/R3 are tractable (p95 ~600-700, direction accuracy >90%). Recommendations:

1. **Use M-only as baseline.** Adding H makes things worse. Confirmed by blend testing.

2. **Create round-specific parameters.** R3 residuals are ~18% smaller than R2. Different bounds.

3. **Migrate to f2p-style binned bands.** We confirmed >35k rows per |M| bin per PY. Use historical residual quantiles per bin to set bid widths, rather than parametric change/offset bounds.

4. **The ftr24/v1 conformal prediction framework is the right target.** It's calibrated to actual residual distributions, uses explicit coverage targets (10/30/50/70/99%), and has clearing probability estimation. The annual R2/R3 problem has enough data to use this framework.

### 4.3 PJM: Structurally Better Starting Point

PJM R1 has real MTM (prior forward clearing prices), so the baseline is fundamentally stronger than MISO R1. The same improvements (round-specific params, calibrated widths) apply but with less urgency.

### 4.4 Migration Path

**Phase 1 (Quick Win):** Update ftr23 params for the upcoming auction:
- Create R1-specific params: wider bounds, higher cap, +400 bias
- Tighten R2/R3 params to match their actual residual scale

**Phase 2 (Medium Term):** Integrate annual into ftr24/v1:
- Add aq1-aq4 to BASELINE_CONFIG in band_generator.py
- R2/R3: Use f2p-style binned bands with annual-specific width tables
- R1: Use rule-based bands calibrated to |H| bins (not ML, since training data is sparse)

---

## 5. Summary Table

| Dimension | R1 Legacy | R1 Assessment | R2/R3 Legacy | R2/R3 Assessment |
|-----------|-----------|--------------|--------------|-----------------|
| Baseline | H (DA congestion) | Only option; p95 error=3,300 | M (prev round MCP) | **Excellent**; p95=202-260 (corrected) |
| ML overlay | None | Correct (too few data for ML) | None | Not needed (M is near-optimal) |
| Bid pricing | Parametric power fn | Reasonable architecture, bad calibration | Same params as R1 | Should be round-specific, much tighter |
| Bid range | ≤2,000 adjustment | Too narrow by ~2× | ≤2,000 adjustment | **Too wide by ~8×** (corrected: p95=260) |
| Signals | SPICE/DA/TMP | Most valuable component | Same signals | Good |
| Clearing prob | None (linear volume interp) | Missing | None | Missing |
| Round differentiation | None (R1=R2=R3 params) | Critical gap | None | Critical gap |

**Note on corrected R2/R3 numbers:** Previous analysis reported R2/R3 p95=568-693. Corrected analysis using full MISO market MCP data shows p95=202-260 — a ~63% reduction. See `findings_residuals.md` Section 1 for details on the correction.

---

## 6. Worked Example: End-to-End Pricing of a Single Path

### 6.1 The Path

We trace the full legacy pricing pipeline for one concrete trade from the data:

- **Path:** `NSP.JEFFERS2 → ALTW.BROOKE1` (source → sink)
- **Planning Year:** 2021
- **Round:** R1 (first round, no prior MCP)
- **Period:** aq1 (Jun-Aug delivery)
- **Class:** onpeak
- **Trade type:** buy / obligation

### 6.2 Step 1: Baseline Computation (H)

The function `fill_mtm_1st_period_with_hist_revenue()` is called. For PY 2021, aq1, onpeak:

1. **Identify delivery months:** aq1 covers June, July, August of the planning year (2021).

2. **Load prior-year DA congestion:** The system loads monthly-average DA congestion prices from the prior year for those months (June 2020, July 2020, August 2020, September 2020). The cutoff is April 2021, so all months before April are eligible.

3. **Compute per-node congestion:**
   - `sink_congestion` = DA congestion price at ALTW.BROOKE1 (averaged over those months)
   - `source_congestion` = DA congestion price at NSP.JEFFERS2 (averaged over those months)

4. **Apply 0.85 shrinkage:** If the profitable direction congestion is positive (i.e., sink congestion > source congestion for a buy), multiply by 0.85 as a conservative bias. This reduces the baseline slightly toward zero.

5. **Result:** `H = mtm_1st_mean = sink_congestion - source_congestion ≈ 499.88`

This means the system estimates that this path's FTR will clear at approximately $499.88/MW for the quarterly period, based on what DA congestion looked like on this path in the prior year.

### 6.3 Step 2: MCP Prediction

Since `prediction_class_instance = None` for all annual rounds:

```
mcp_pred = mtm_1st_mean = H = 499.88
```

There is no ML overlay. The baseline IS the bid center.

### 6.4 Step 3: Signal Ranking and Bid Price

The path is grouped with all other `(buy, prevailing)` paths in this auction period. Signal exposure scores are computed from SPICE_ANNUAL_V4.4, DA_ANNUAL_V1.4, and TMP_VIOLATED signals:

1. **Exposure score:** `exposure_tier1_bid_price` is computed for each path based on the constraint violation signals. Suppose this path has a moderately high exposure score, placing it at the **60th percentile** (rank = 0.60) among all buy/prevailing paths.

2. **Map rank to adjustment:** With production params `change_lb=-0.5, change_ub=1.5, power=2, thres=0.2`:

   The power function with `thres=0.2` means:
   - Paths below the 20th percentile get roughly zero adjustment (near `change_lb`)
   - The adjustment ramps up quadratically for higher-ranked paths
   - At rank 0.60 (above the threshold):
     - `change_pct ≈ 0.62` (interpolated between 0 and 1.5 with power=2 curve)
     - `offset ≈ 62` (interpolated between 0 and 150)

3. **Compute price_change:**
   ```
   price_change = |mcp_pred| × change_pct + offset
                = |499.88| × 0.62 + 62
                = 309.93 + 62
                = 371.93
   ```

   Check cap: `clip(371.93, upper=2000)` → 371.93 (no clipping)

4. **Set bid prices:**
   ```
   predicted_bid_1 = mcp_pred + price_change = 499.88 + 371.93 = 871.81  (signal-adjusted)
   predicted_bid_2 = mcp_pred = 499.88                                     (raw baseline)
   ```

### 6.5 Step 4: Bid Curve

1. **Quarterly scaling (×3):** Both prices are multiplied by 3:
   ```
   bid_price_1 = 871.81 × 3 = 2,615.43
   bid_price_2 = 499.88 × 3 = 1,499.64
   ```

2. **Sort prices** (highest to lowest): [2,615.43, 1,499.64]

3. **Assign volumes** (linear interpolation):
   ```
   bid_volume_1 = 0 MW          (at the aggressive price)
   bid_volume_2 = total_bid_vol (at the conservative price)
   ```

   This means: "We are willing to buy up to `total_bid_vol` MW at $1,499.64, but at $2,615.43 we want 0 MW." The volume ramps linearly between these two price levels.

4. **Pad to 10 points** with NaN.

### 6.6 What Actually Happened

- **Actual MCP for this path:** `mcp_mean = 783.74` (quarterly-scale)
- **Monthly-scale MCP:** 783.74 / 3 ≈ 261.25
- **H baseline was:** 499.88
- **Residual:** 783.74 - 499.88 = +283.86 (MCP cleared above H)

**Assessment of the bid:**

| Question | Answer |
|----------|--------|
| Was the bid center (H=499.88) close to the actual MCP (783.74)? | Moderate — residual of +284 is above median but below p90 for R1 |
| Did the bid range capture the actual MCP? | **Yes** — the bid range was [1,499.64, 2,615.43] after ×3 scaling, and the actual MCP ×3 = 783.74 × 3 = 2,351. This is within the bid range. |
| Was this path easy or hard to predict? | Moderate difficulty — H correctly predicted the positive direction and magnitude was in the right ballpark |

However, this is a **median-H path** where the system works reasonably. For large-|H| paths (>1,000), the price_change_cap of 2,000 would bind, and the actual MCP could be 8,000+ away from H — far outside any bid range.

### 6.7 What This Example Illustrates

1. **The ×3 quarterly scaling is applied after bid pricing.** The parametric bounds are computed on the "raw" (monthly-scale) baseline, then everything is scaled by 3. Whether this is appropriate depends on whether quarterly MCPs also scale linearly with monthly MCPs.

2. **The signal-ranked adjustment (price_change) is moderate for a median-ranked path.** It adds +372 to a baseline of 500, which is a +74% adjustment. For top-ranked paths, it would be +900 (180%). For bottom-ranked paths, it would be -300 (-60%).

3. **The cap of 2,000 doesn't bind here** because |mcp_pred| × change_pct + offset = 372, well below 2,000. But for |mcp_pred| > 1,200, the cap starts binding: |1,200| × 1.5 + 150 = 1,950 ≈ cap.

4. **The two-price bid curve is very simple.** It doesn't model the actual uncertainty distribution — it's just a linear ramp between the baseline and the signal-adjusted price. The ftr24/v1 approach with 10 price levels at calibrated coverage quantiles is fundamentally richer.

---

## 7. Per-Round × Per-Quarter Analysis Grid (Corrected)

This section provides a detailed breakdown for each of the 12 cells in the 3 rounds × 4 quarters grid, using corrected data from `notebooks/03_corrected_baseline.ipynb`. All numbers use the full MISO market MCP data.

### 7.1 R1 × aq1 (Jun-Aug, Round 1)

| Metric | Value |
|--------|-------|
| **Trades** | 618,021 |
| **Unique paths** | 82,880 |
| **Coverage** | 100.0% (H baseline) |
| **Bias** | +411 (MCPs clear above H on average) |
| **Mean |residual|** | 838 |
| **Median |residual|** | 367 |
| **p90** | 2,024 |
| **p95** | 3,171 |
| **p99** | 7,063 |
| **Direction accuracy** | 66% |
| **Median |H|** | 180 |
| **Legacy bid range** (for median |H|) | 560 |
| **Range adequacy** (vs p95) | **9%** — the legacy bid range covers only 9% of the actual p95 MCP uncertainty range |

**Key finding:** aq1 is a typical R1 cell. The baseline H predicts the right direction 66% of the time, but the magnitude error is enormous (p95 = 3,171). The legacy bid range of 560 for a median-|H| path covers barely any of the actual MCP range.

**Suggestion:** Needs the same R1 treatment as all other quarters — raise cap to ≥4,000, add +411 bias correction, widen parametric bounds.

### 7.2 R1 × aq2 (Sep-Nov, Round 1)

| Metric | Value |
|--------|-------|
| **Trades** | 628,725 |
| **Unique paths** | 83,475 |
| **Coverage** | 100.0% |
| **Bias** | +489 (**highest** bias of any R1 cell) |
| **Mean |residual|** | 943 (**highest** of any R1 cell) |
| **Median |residual|** | 384 |
| **p90** | 2,348 |
| **p95** | 3,712 (**highest** of any R1 cell) |
| **p99** | 8,472 |
| **Direction accuracy** | 67% |
| **Median |H|** | 191 |
| **Legacy bid range** | 582 |
| **Range adequacy** | **8%** |

**Key finding:** aq2 is the **hardest** R1 cell by every magnitude metric. The Sep-Nov delivery quarter has the largest residuals, likely because fall/winter congestion patterns are the most volatile and least well-predicted by prior-year DA congestion.

**Suggestion:** aq2 R1 deserves the widest bands of any cell. Raise price_change_cap to ≥4,500, add +489 bias correction. If per-quarter parameters are implemented, aq2 should have the highest cap.

### 7.3 R1 × aq3 (Dec-Feb, Round 1)

| Metric | Value |
|--------|-------|
| **Trades** | 572,433 |
| **Unique paths** | 75,837 |
| **Coverage** | 100.0% |
| **Bias** | +402 |
| **Mean |residual|** | 816 |
| **p90** | 2,028 |
| **p95** | 3,211 |
| **p99** | 6,782 |
| **Direction accuracy** | 67% |
| **Median |H|** | 179 |
| **Legacy bid range** | 559 |
| **Range adequacy** | **9%** |

**Key finding:** aq3 is middle-of-the-road for R1. Winter delivery quarter — similar difficulty to aq1.

**Suggestion:** Standard R1 treatment. Cap ≥4,000, bias correction +402.

### 7.4 R1 × aq4 (Mar-May, Round 1)

| Metric | Value |
|--------|-------|
| **Trades** | 569,295 |
| **Unique paths** | 74,638 |
| **Coverage** | 100.0% |
| **Bias** | +375 (lowest R1 bias) |
| **Mean |residual|** | 797 (lowest R1 mean |res|) |
| **p90** | 1,957 |
| **p95** | 3,107 (lowest R1 p95) |
| **p99** | 6,828 |
| **Direction accuracy** | 62% (**worst** of any R1 cell) |
| **Median |H|** | 178 |
| **Legacy bid range** | 555 |
| **Range adequacy** | **9%** |

**Key finding:** aq4 is actually the **easiest** R1 cell by residual magnitude, but has the **worst** direction accuracy (62%). The structural reason: aq4 (Mar-May) only gets 1 month of DA congestion data (March) before the April cutoff, whereas other quarters get 3-4 months. Less data → worse direction prediction, but the magnitudes are not larger.

**Suggestion:** Standard R1 treatment. Consider wider symmetric bands to compensate for the 62% direction accuracy (more sign errors → need to cover both directions more evenly).

### 7.5 R2 × aq1 (Jun-Aug, Round 2)

| Metric | Value |
|--------|-------|
| **Trades** | 1,063,809 |
| **Unique paths** | 106,906 |
| **Coverage** | 100.0% (full MISO MCP) |
| **Bias** | +20 (near zero) |
| **Mean |residual|** | 70 |
| **Median |residual|** | 34 |
| **p90** | 162 |
| **p95** | 256 |
| **p99** | 555 |
| **Direction accuracy** | 91% |
| **Median |M|** | 120 |
| **Legacy bid range** | 439 |
| **Range adequacy** | **86%** — the legacy range covers 86% of the p95 MCP range |

**Key finding:** R2 aq1 is well-predicted. The previous round's MCP (M) is a strong baseline with near-zero bias and 91% direction accuracy. The legacy bid range (439 for median |M|) actually covers most of the needed range.

**Suggestion:** The legacy bid range is close to adequate for R2, but the **issue is that the same wide params are used for R1**, where they're catastrophically inadequate. If round-specific params are implemented, R2 params should target p95 ≈ 256.

### 7.6 R2 × aq2 (Sep-Nov, Round 2)

| Metric | Value |
|--------|-------|
| **Trades** | 1,091,559 |
| **Unique paths** | 107,443 |
| **Bias** | +23 |
| **Mean |residual|** | 75 (highest R2 mean |res|) |
| **p95** | 282 (highest R2 p95) |
| **p99** | 590 |
| **Direction accuracy** | 91% |
| **Median |M|** | 123 |
| **Legacy bid range** | 447 |
| **Range adequacy** | **79%** |

**Key finding:** aq2 is the hardest R2 cell (same as for R1), but still very tractable. p95 = 282 is 10% higher than aq1's 256.

**Suggestion:** Same as R2 aq1. If per-quarter R2 params are implemented, aq2 could get slightly wider bands.

### 7.7 R2 × aq3 (Dec-Feb, Round 2)

| Metric | Value |
|--------|-------|
| **Trades** | 1,003,830 |
| **Unique paths** | 98,992 |
| **Bias** | +21 |
| **Mean |residual|** | 67 |
| **p95** | 245 (lowest R2 p95) |
| **p99** | 515 |
| **Direction accuracy** | 91% |
| **Median |M|** | 113 |
| **Legacy bid range** | 427 |
| **Range adequacy** | **87%** |

**Key finding:** aq3 is the easiest R2 cell. Range adequacy is near 90%.

**Suggestion:** Standard R2 treatment.

### 7.8 R2 × aq4 (Mar-May, Round 2)

| Metric | Value |
|--------|-------|
| **Trades** | 1,020,261 |
| **Unique paths** | 97,214 |
| **Bias** | +21 |
| **Mean |residual|** | 68 |
| **p95** | 252 |
| **p99** | 536 |
| **Direction accuracy** | 90% |
| **Median |M|** | 103 |
| **Legacy bid range** | 407 |
| **Range adequacy** | **81%** |

**Key finding:** aq4 in R2 does NOT have the direction accuracy issue seen in R1 (90% vs R1's 62%). This is because R2's baseline (M = R1's MCP) is derived from the actual auction, not from historical DA congestion. The R1 data limitation (1 month of DA for aq4) doesn't propagate to R2.

**Suggestion:** Standard R2 treatment. aq4 is not a special case for R2/R3.

### 7.9 R3 × aq1 (Jun-Aug, Round 3)

| Metric | Value |
|--------|-------|
| **Trades** | 1,255,008 |
| **Unique paths** | 116,368 |
| **Bias** | +13 |
| **Mean |residual|** | 58 |
| **p95** | 214 |
| **p99** | 460 |
| **Direction accuracy** | 93% |
| **Median |M|** | 121 |
| **Legacy bid range** | 443 |
| **Range adequacy** | **103%** — legacy range **exceeds** the p95 range needed |

**Key finding:** For R3, the legacy bid range actually covers MORE than enough. The problem is entirely the opposite of R1: the legacy params are too wide for R3.

**Suggestion:** Tighten R3 params. Target p95 ≈ 214 rather than spreading bids over the full ±2,000 range. This would concentrate bid volume near the likely clearing price, improving fill rates.

### 7.10 R3 × aq2 (Sep-Nov, Round 3)

| Metric | Value |
|--------|-------|
| **Trades** | 1,279,842 |
| **Unique paths** | 113,558 |
| **Bias** | +13 |
| **Mean |residual|** | 60 |
| **p95** | 220 |
| **p99** | 476 |
| **Direction accuracy** | 93% |
| **Median |M|** | 126 |
| **Legacy bid range** | 451 |
| **Range adequacy** | **102%** |

**Key finding:** Same pattern as R3 aq1. Legacy range is adequate but wastefully wide.

### 7.11 R3 × aq3 (Dec-Feb, Round 3)

| Metric | Value |
|--------|-------|
| **Trades** | 1,191,363 |
| **Unique paths** | 106,867 |
| **Bias** | +9 (lowest bias in the entire grid) |
| **Mean |residual|** | 53 |
| **p95** | 190 (lowest p95 in the grid) |
| **p99** | 403 |
| **Direction accuracy** | 93% |
| **Median |M|** | 116 |
| **Legacy bid range** | 431 |
| **Range adequacy** | **113%** |

**Key finding:** The easiest cell in the entire grid. R3 aq3 has the smallest residuals and nearest-to-zero bias. The legacy range overshoots by 13%.

### 7.12 R3 × aq4 (Mar-May, Round 3)

| Metric | Value |
|--------|-------|
| **Trades** | 1,199,634 |
| **Unique paths** | 107,476 |
| **Bias** | +10 |
| **Mean |residual|** | 53 |
| **p95** | 184 |
| **p99** | 415 |
| **Direction accuracy** | 93% |
| **Median |M|** | 111 |
| **Legacy bid range** | 422 |
| **Range adequacy** | **114%** |

**Key finding:** R3 aq4 is essentially the same as R3 aq3. Unlike R1 aq4, there is no direction accuracy penalty — the R2→R3 MCP chain doesn't inherit R1's data sparsity issue.

### 7.13 Grid Summary Table

| Cell | Bias | Mean |Res| | p95 |Res| | Dir Acc | Range Adequacy | Difficulty |
|------|---:|---:|---:|---:|---:|---|
| R1 × aq1 | +411 | 838 | 3,171 | 66% | 9% | Hard |
| R1 × aq2 | +489 | 943 | 3,712 | 67% | 8% | **Hardest** |
| R1 × aq3 | +402 | 816 | 3,211 | 67% | 9% | Hard |
| R1 × aq4 | +375 | 797 | 3,107 | 62% | 9% | Hard (worst direction) |
| R2 × aq1 | +20 | 70 | 256 | 91% | 86% | Tractable |
| R2 × aq2 | +23 | 75 | 282 | 91% | 79% | Tractable |
| R2 × aq3 | +21 | 67 | 245 | 91% | 87% | Easy |
| R2 × aq4 | +21 | 68 | 252 | 90% | 81% | Tractable |
| R3 × aq1 | +13 | 58 | 214 | 93% | 103% | Easy |
| R3 × aq2 | +13 | 60 | 220 | 93% | 102% | Easy |
| R3 × aq3 | +9 | 53 | 190 | 93% | 113% | **Easiest** |
| R3 × aq4 | +10 | 53 | 184 | 93% | 114% | Easy |

**Key patterns across the grid:**

1. **The round dimension dominates everything.** The difference between R1 and R2 (12.7× in p95) dwarfs the difference between quarters (≤20% within a round). Round-specific params are the single most important change.

2. **aq2 is consistently the hardest quarter within each round** — but only by 5-10%. Quarter-specific params are a fine-tuning improvement, not a critical fix.

3. **Range adequacy flips between R1 and R3.** R1 cells are at 8-9% adequacy (catastrophically under-covered). R3 cells are at 102-114% (slightly over-covered). R2 cells are at 79-87% (close to adequate).

4. **R1 aq4's direction accuracy issue (62%) does NOT propagate to R2/R3 aq4** (90-93%). The MCP-based baseline resolves the data sparsity problem that plagues the DA congestion baseline.

### 7.14 Recommendations by Cell

| Priority | Cells | Action |
|----------|-------|--------|
| **Critical** | All R1 (aq1-aq4) | Separate R1 params: cap ≥4,000, bias correction +400-490 per quarter, widen bounds |
| **Important** | All R2 (aq1-aq4) | Separate R2 params: tighten cap to ≈500, minimal bias correction (+20) |
| **Important** | All R3 (aq1-aq4) | Separate R3 params: tighten cap to ≈400, zero bias correction |
| **Nice to have** | Per-quarter within R1 | aq2 slightly wider bounds, aq4 wider symmetric bands for direction uncertainty |
| **Low priority** | Per-quarter within R2/R3 | Minimal difference between quarters (p95 varies ≤15%) |
