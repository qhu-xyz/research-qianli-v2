# Baseline Formula Research — MISO Annual

## 1. Research Scope

**Question:** Using only data from PY 2019 through PY 2022, what formula predicts annual FTR MCP for each path, quarter, and round?

**Focus:** MISO only. 4 quarters (aq1-aq4), 3 rounds per PY.

**Key constraint:** For R1, there is no prior clearing price (no `mtm_1st_mean`) for any quarter. For R2-R3, `mtm_1st_mean` = previous round's MCP for that quarter.

---

## 2. Setup

### 2.1 What we're predicting

For each trade (path × quarter × round), predict **MCP** = market clearing price.
- MCP is what the FTR auction clears the path at.
- This is what the code calls `mcp_pred` → feeds into band generation → optimizer.

### 2.2 MISO annual calendar

| Quarter | Delivery months | PY offset |
|---------|----------------|-----------|
| aq1 | Jun, Jul, Aug | PY year |
| aq2 | Sep, Oct, Nov | PY year |
| aq3 | Dec, Jan, Feb | PY year / PY+1 |
| aq4 | Mar, Apr, May | PY+1 year |

**Round schedule** (approximate, from `pbase/analysis/tools/miso.py:297-303`):
| Round | Submission date | What's available |
|-------|----------------|-----------------|
| R1 | ~April 8 | No prior clearing for any quarter. Historical data only. |
| R2 | ~April 22 | R1 MCP for each quarter (cleared ~2 weeks earlier) |
| R3 | ~May 5 | R2 MCP for each quarter (R1 + R2 results known) |

**Key structural point:** Each round auctions ALL 4 quarters (aq1-aq4) simultaneously. Rounds are sequential time events; quarters are delivery periods. This gives a 3×4 = 12 clearing-event grid per PY.

**The MISO annual grid (what we need a baseline for):**

|       | aq1 (Jun-Aug) | aq2 (Sep-Nov) | aq3 (Dec-Feb) | aq4 (Mar-May) |
|-------|:---:|:---:|:---:|:---:|
| **R1** | no MTM | no MTM | no MTM | no MTM |
| **R2** | R1's MCP | R1's MCP | R1's MCP | R1's MCP |
| **R3** | R2's MCP | R2's MCP | R2's MCP | R2's MCP |

- R1 row: baseline must come entirely from historical signals (H, C).
- R2-R3 rows: baseline is anchored by the previous round's MCP (M) for that quarter.
- Each cell is a separate bid-pricing problem with its own path-level baseline.

### 2.3 Available data at R1 submission (~April)

For PY 2023 R1 aq1 (delivery Jun-Aug 2023), submitted April 2023:

| Signal | What it is | Example months | Staleness |
|--------|-----------|----------------|-----------|
| DA congestion same quarter, Y-1 | DA LMP congestion for Jun-Aug 2022 | Jun-Aug 2022 | 8-10 months old |
| DA congestion same quarter, Y-2 | Same, 2021 | Jun-Aug 2021 | 20-22 months |
| DA congestion same quarter, Y-3 | Same, 2020 | Jun-Aug 2020 | 32-34 months |
| DA congestion recent months | DA congestion Jan-Mar 2023 | Jan-Mar 2023 | 1-3 months, **wrong season** |
| Previous PY FTR clearing | What this path cleared at in PY 2022 aq1 auction | PY 2022 auction (Apr 2022) | 12 months old |
| Recent path revenue | DA revenue from ~April 2023 | Apr 2023 | Days old, **wrong season** |

---

## 3. Clarifying H, C, and R

### 3.1 H — Historical DA Congestion Proxy

**What it is:** The day-ahead congestion price difference (sink − source) for the delivery quarter from prior year(s).

**How `_fill_mtm` computes it** (from `miso_base.py:221-261`):
```
For aq1, PY 2023:
  1. Load DA monthly congestion per node for Jun 2022, Jul 2022, Aug 2022
  2. Apply directional shrinkage:
     - source congestion < 0 → multiply by 0.85 (less negative = conservative)
     - sink congestion > 0 → multiply by 0.85 (less positive = conservative)
  3. Average across months within each planning year
  4. Average across planning years (currently only 1 year: range(1,2))
  5. H = sink_avg - source_avg
  6. H_quarter = H * 3 (scale to 3-month value)
```

**What it is NOT:**
- H is NOT historical FTR clearing price (MCP). It's the realized DA congestion.
- H is NOT total revenue. It's specifically the congestion component of DA LMP.
  (Total DA LMP = energy + congestion + losses. H only uses congestion.)

**Why H ≈ "last year's revenue for those months":**
- For FTR paths, the **FTR settlement value** = DA congestion at sink − DA congestion at source.
- So H is essentially what this FTR path **would have paid out** in the DA market during that quarter.
- In trader terms: H = last year's 同比 (year-over-year same period) realized congestion value.

### 3.2 C — Previous PY FTR Clearing Price

**What it is:** The MCP that this exact path cleared at in last year's annual FTR auction.

**How to get it:**
```python
# Get PY 2022 annual cleared trades
trades = miso_aptools.get_all_cleared_trades(start_date="2022-06-01", end_date="2022-06-01")
# Filter to annual + specific quarter
annual = trades[trades["period_type"] == "aq1"]
# Look up by (source_id, sink_id) → mcp_mean column
```

**How C differs from H:**

| Aspect | H (DA Congestion) | C (FTR Clearing) |
|--------|-------------------|-------------------|
| Source | Real-time DA energy market | FTR auction market |
| What it reflects | Physical congestion that occurred | Market expectations + risk premium |
| Granularity | Monthly per node, 3 months of daily data | One clearing price per path per auction |
| Coverage | All nodes with DA data | Only paths that were bid on and cleared |
| Bias | None (realized outcome) | Contains market risk premium (usually MCP < H) |

**Key relationship:** Historically, `C < H` on average — FTR markets tend to clear below realized DA congestion because participants demand a risk premium for bearing congestion uncertainty. This gap is the "FTR discount."

**Why both are useful:**
- H captures the fundamental value (what congestion actually was)
- C captures what market participants are willing to pay (closer to what MCP will be)
- We're predicting next year's C, not next year's H

### 3.3 R — Recent Revenue / Revenue Trend

**What it is in f0p:** `1(rev)` = current month's DA revenue for the path, scaled to monthly rate.

**For annual R1:**
- Submission is ~April 8. Most recent available revenue data: March 2023 or early April 2023.
- aq1 delivery: Jun-Aug 2023.
- **The seasonal mismatch:** March/April congestion patterns often differ significantly from June/August patterns (summer peaking, different load profiles, different transmission constraints binding).

**Why the design excluded R initially:**
- In f0p, `1(rev)` has 0.23 weight for f0 (near-term delivery). As delivery gets further out, rev weight drops: f1=0.15, f2=0.06.
- For annual aq1, delivery is 2+ months away from submission. For aq4 (Mar-May next year), delivery is 11-13 months away.
- The signal-to-noise ratio of April revenue for predicting June congestion is low.

**When R COULD help:**
- For paths with **persistent, non-seasonal congestion** (e.g., structural bottlenecks that bind year-round), recent revenue IS informative regardless of season.
- As a trend indicator: if a path's congestion has been rising over the past 6 months, that trend may continue.
- **R should be tested in ablation, not assumed useless.**

---

## 4. What Historical Months Should We Consider?

### 4.1 Current approach (what `_fill_mtm` does)

For aq1 PY 2023: loads Jun 2022, Jul 2022, Aug 2022 only.
- 3 months, 1 year back.
- Misses: multi-year patterns, recent trends, adjacent season data.

### 4.2 Expanded options

**Option A: Same quarter, multiple years (recommended first extension)**
```
For aq1 PY 2023:
  Y-1: Jun 2022, Jul 2022, Aug 2022
  Y-2: Jun 2021, Jul 2021, Aug 2021
  Y-3: Jun 2020, Jul 2020, Aug 2020
```
- Pros: Same seasonal pattern, reduces single-year noise
- Cons: Grid changes across years (new lines, generation retirements)
- Recency weighting addresses this: Y-1 gets more weight

**Option B: Recent months (pre-submission, current year)**
```
For aq1 PY 2023 R1 (submitted April 2023):
  Recent: Jan 2023, Feb 2023, Mar 2023
```
- Pros: Captures current grid state, recent binding constraints
- Cons: Wrong season — winter/spring congestion ≠ summer congestion
- Best as: supplementary signal, not primary

**Option C: Trailing 12 months**
```
For aq1 PY 2023:
  Apr 2022, May 2022, Jun 2022, ..., Mar 2023
```
- Pros: Large data volume, captures both seasonal and recent
- Cons: Dilutes seasonal signal with off-season data

**Option D: Same quarter + adjacent months**
```
For aq1 PY 2023:
  May 2022, Jun 2022, Jul 2022, Aug 2022, Sep 2022
```
- Pros: Wider seasonal window, still relevant season
- Cons: May/Sep have different characteristics than Jun-Aug peak

### 4.3 Recommendation

**Primary:** Option A (same quarter, 3 years, recency-weighted)
**Supplementary:** Option B (recent months as a trend adjustment)

Formula:
```
H_primary = 0.60 * H_y1 + 0.30 * H_y2 + 0.10 * H_y3

H_trend = mean(DA_congestion for Jan-Mar of current year)
         (only same nodes, same class_type)

H_adjusted = α * H_primary + β * H_trend
```

Whether `H_trend` adds value is an empirical question (→ hypothesis H5 equivalent).

---

## 5. Recency Bias Analysis

### 5.1 The problem

Using only Y-1 same-quarter data:
- Single year = high variance (one outlier year skews everything)
- Grid changes: new transmission, load growth, generation mix changes
- 2020 COVID year significantly distorted congestion patterns

### 5.2 Multi-year weighted average

```
H = w1 * H_y1 + w2 * H_y2 + w3 * H_y3
```

**Why this helps:**
- 2020 COVID distortion gets 10% weight instead of 100%
- Structural patterns (persistent bottlenecks) reinforce across years
- Temporary distortions (outages, weather events) get averaged out

**The weights `[0.60, 0.30, 0.10]` are a starting point.** Optimal weights should be calibrated by minimizing MAE against actual MCP over PY 2019-2022.

### 5.3 Recency within a year

Another form of recency bias: should months closer to submission date get more weight?

For aq3 (Dec-Feb), PY 2023:
- Y-1 data: Dec 2021, Jan 2022, Feb 2022
- Dec 2021 is 16 months before submission (April 2023)
- Feb 2022 is 14 months before submission

This is all roughly equally stale, so within-year recency weighting is less important than cross-year weighting.

However, for aq4 (Mar-May): only Mar passes the cutoff filter (see learning.md Section 6, aq4 edge case). So aq4 has structural sparsity that needs addressing separately.

---

## 6. Alternative Simple Models

### 6.1 Linear trend extrapolation

```
For each path p, quarter q:
  H_2020, H_2021, H_2022 → fit line → extrapolate to 2023
  MCP_pred = H_2020 + slope * (2023 - 2020)
```
- Pros: Captures trend (e.g., congestion growing due to renewables)
- Cons: Overfits with 3-4 points, amplifies noise
- Verdict: **Too unstable** with ≤4 years of data.

### 6.2 Peer group mean (for sparse paths)

```
For paths without enough history:
  1. Group paths by (region, class_type, approximate distance)
  2. Use group median as proxy
```
- Useful as fallback when path-level data is missing
- Not a primary model

### 6.3 H + C blend (the proposed approach)

```
R1: baseline = 0.80 * H + 0.20 * C
```

Why this is good:
- H captures physical congestion fundamentals
- C captures market pricing behavior (risk premium, liquidity)
- MCP is a market price → putting some weight on C makes sense
- 80/20 split trusts fundamentals more (3 months of daily data vs 1 auction datapoint)

### 6.4 "Simple regression" on available features

```
MCP = β₀ + β₁ * H + β₂ * C + β₃ * H_trend
```

Fit via Ridge regression on PY 2019-2022 data.
- Pros: Learns optimal weights from data
- Cons: With ≤4 years of data, risk of overfitting; need careful cross-validation
- This is essentially what research-bid-price-v5's 2-regime Ridge does for monthly

---

## 7. Concrete Proposal by Round (MISO: 3 rounds × 4 quarters)

Each formula below applies to ALL 4 quarters (aq1-aq4) within that round. The `H` signal is quarter-specific (uses delivery-month congestion for the target quarter).

### R1 (no prior clearing, any quarter)

```
H_y1 = DA_congestion(same_quarter, Y-1, 0.85_shrinkage)
H_y2 = DA_congestion(same_quarter, Y-2, 0.85_shrinkage)
H_y3 = DA_congestion(same_quarter, Y-3, 0.85_shrinkage)

H = 0.60 * H_y1 + 0.30 * H_y2 + 0.10 * H_y3
    (renormalize if fewer years available)

C = previous_PY_FTR_clearing_for_same_path_quarter (if available)

baseline_r1 = 0.80 * H + 0.20 * C    (if C exists)
baseline_r1 = H                        (if C missing)
```

### R2 (has R1 MCP)

```
M = R1 clearing price for this path/quarter (= mtm_1st_mean)
H = same as above

baseline_r2 = 0.90 * M + 0.10 * H
```

Rationale: M is a very recent market signal (2 weeks old). H provides seasonal anchor.

### R3 (final MISO round; has R1 + R2 MCPs)

```
M = R2 clearing price for this path/quarter
H = same as above

baseline_r3 = 0.92 * M + 0.08 * H
```

R3 is MISO's final annual round. More weight on M as we have 2 rounds of clearing data.

---

## 8. Quarter-Specific Considerations

| Quarter | Delivery | Data from `_fill_mtm` | Notes |
|---------|----------|----------------------|-------|
| aq1 | Jun-Aug | 3 months pass cutoff | Summer peak. Best data coverage. Most congestion. |
| aq2 | Sep-Nov | 3 months pass cutoff | Shoulder season. Moderate congestion. |
| aq3 | Dec-Feb | 3 months pass cutoff | Winter. Different constraint patterns from summer. |
| aq4 | Mar-May | **Only 1 month** (Mar) passes cutoff | Shoulder/spring. **Data-sparse.** Need fallback. |

**aq4 problem:** `_fill_mtm` cutoff is April of PY year. For aq4 (Mar-May Y+1), looking back 1 year: Mar Y, Apr Y, May Y. Only Mar Y < April cutoff. Apr Y and May Y are filtered out.

**aq4 fix options:**
1. Relax cutoff for aq4 (allow Apr, May from Y-1 to pass)
2. Use Apr-May from Y-2 as supplement
3. Weight aq4 H toward multi-year average more heavily (less trust in single month)

---

## 9. Gaps Requiring Empirical Validation

All formulas above are **priors**, not calibrated. The next step is:

1. **Collect historical dataset:** For each PY 2019-2025, get:
   - Cleared annual trades with `mcp_mean` (actual clearing price = target variable)
   - Reconstructed `H` values (by running `_fill_mtm` logic on historical DA data)
   - Previous PY clearing prices for C
   - Recent month revenue data for R (if testing ablation)

2. **Grid search baseline weights:**
   - For R1: optimize `(w_H, w_C)` to minimize MAE on holdout PY
   - For R2-R3: optimize `(w_M, w_H)` to minimize MAE

3. **Grid search H year weights:**
   - `[w1, w2, w3]` — optimal recency weighting across 3 lookback years

4. **Calibrate shrinkage per quarter:**
   - Is 0.85 optimal? Or does aq1 need 0.87 and aq4 need 0.80?

5. **Test whether R adds value:**
   - Ablation: with/without recent months trend signal

6. **Compare against current production:**
   - Current: `H = 1yr, no C, no multi-year weighting, uniform 0.85 shrinkage`
   - Proposed: `H = 3yr weighted + C + per-quarter shrinkage`
   - Report MAE improvement per (quarter × class_type)
