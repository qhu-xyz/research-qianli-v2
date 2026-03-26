# PJM Annual Bid Price Research — Plan

**Created:** 2026-03-20
**Updated:** 2026-03-21
**Status:** COMPLETE — champion frozen in `pjm/scripts/run_pjm.py`
**Depends on:** MISO R1/R2/R3 champion (complete)

**Final champion differs from plan:** Research found that `recent_1` (March DA only × 12)
outperforms `trail_12` (Apr N-1 to Mar N) for R1 baseline blending. See
`pjm/docs/pjm-consolidated-report.md` for the full revenue window comparison.

## Goal

Build asymmetric bid prices at specified clearing probabilities (Pxx) for PJM annual FTR
auctions. Same framework as MISO, adapted for PJM-specific structure.

## Terminology

**Pxx = single bid price with xx% clearing probability on training data.**
Not "bands". Not pairs. Each level is one independent bid price.

## PJM Annual Auction Structure

| Property | Value |
|----------|-------|
| Period type | `a` (annual, 12-month settlement Jun–May) |
| Rounds | 4 (R1, R2, R3, R4) — all clear at same event (~April) |
| Classes | onpeak, dailyoffpeak, wkndonpeak |
| Scale | **Annual $** (native). `mtm_1st_period` is already annual. |
| Planning year | Jun N to May N+1 (PY2025 = Jun 2025 – May 2026) |
| Auction timing | ~June 1 of year N (verified: 2024-06-01, 2025-06-01) |
| PY range | 2017–2025 (9 years) |

### Class type history

Pre-2023: only onpeak existed in R1 (historical limitation).
**Teammates have backfilled dailyoffpeak and wkndonpeak for pre-2023.** The canonical loader
should return all 3 classes for all PYs. **MUST verify this in Phase 0.**

### Baseline per round

| Round | Baseline source | Column |
|-------|----------------|--------|
| R1 | LT yr1 R5 MCP (long-term auction, prior year) | `mtm_1st_period` |
| R2 | R1 MCP (same annual auction event) | `mtm_1st_period` |
| R3 | R2 MCP | `mtm_1st_period` |
| R4 | R3 MCP | `mtm_1st_period` |

**All rounds use `mtm_1st_period` directly** — it is already annual scale.
Do NOT multiply by 12. Do NOT use `mtm_1st_mean` (that is monthly = `mtm_1st_period / 12`).

### Key differences from MISO

| Dimension | MISO | PJM |
|-----------|------|-----|
| Period types | aq1-aq4 (4 quarterly) | a (1 annual) |
| Scale | Quarterly (×3) | Annual (native) |
| Rounds | 3 | 4 |
| Classes | 2 (onpeak, offpeak) | 3 (onpeak, dailyoffpeak, wkndonpeak) |
| `break_offpeak` | No-op | Required (splits offpeak → dailyoffpeak + wkndonpeak) |
| R1 baseline | nodal_f0 stitch (no prior annual) | LT yr1 R5 MCP (prior LT auction) |
| R1 1(rev) | Quarter-specific, prior PY same quarter | Trailing 12 months (see below) |
| Quarterly structure | 4 independent quarters per PY | 1 annual period per PY |

## 1(rev) Definition for PJM (CRITICAL)

**Auction timing:** PJM annual auction clears **~June 1** of year N (verified from
`submitting_data` auction_date: 2024-06-01, 2025-06-01). This is later than MISO (~April).

**DA data availability:** `PjmDaLmpMonthlyAgg` has data through 2026-03 as of today.
With a June auction, all months through **May N** are fully available — 5+ months settled.

**Definition:** trailing 12 months of DA congestion revenue ending at the most recent
fully-settled month before auction.

```
1(rev) = sum(sink_DA_congestion - source_DA_congestion) for Apr N-1 through Mar N
```

12 months: Apr N-1, May N-1, Jun N-1, Jul N-1, ..., Feb N, Mar N.

Why this window:
- March N is the most recent fully-settled month before a June auction (3 months prior)
- Verified available: `PjmDaLmpMonthlyAgg` has March 2025 data (57,968 rows)
- 12 months — same scale as the annual target
- No leakage: all months well before June N auction
- April/May N are available but excluded to maintain a consistent 2-month buffer
  (March settles ~45 days later, comfortably before June auction)

**Fallback rule:** If March N monthly aggregate is not yet published for a given year,
fall back to Apr N-2 through Mar N-1 (12 months, fully safe). Log the fallback.

**Implementation:** Load `PjmDaLmpMonthlyAgg` for each month, get per-node congestion,
apply `PjmNodalReplacement` for retired nodes, compute path = sink − source, sum 12 months.

### Node ID format

PJM uses numeric pnode_ids (unlike MISO strings). Source/sink IDs in the canonical data
are also numeric. Direct join should work; replacement mapping handles retired nodes.

## Phases

### Phase 0: Load canonical data (requires Ray)

```python
from pbase.analysis.tools.all_positions import PjmApTools
aptools = PjmApTools()

# Step 1: Load trades (same pattern as MISO load_canonical.py)
trades = aptools.get_trades_of_given_duration(
    participant=None, start_month='2017-06', end_month_in='2026-06'
)

# Step 2: Filter to annual period types
trades_annual = trades[trades['period_type'].isin(aptools.tools.annual_period_types)].copy()

# Step 3: break_offpeak — REQUIRED for PJM (splits offpeak → dailyoffpeak + wkndonpeak)
trades_annual = aptools.tools.break_offpeak(trades_annual)

# Step 4: Filter to valid classes
trades_annual = trades_annual[trades_annual['class_type'].isin(aptools.tools.classtypes)].copy()

# Step 5: Merge cleared volume + MCP + M2M columns
trades_annual = aptools.merge_cleared_volume(trades_annual, merge_mcp=True)
trades_annual = aptools.get_m2m_mcp_for_trades_all(trades_annual)

# Step 6: Filter to buy only, period_type='a' (annual only, not quarterly/monthly forwards)
trades_annual = trades_annual[
    (trades_annual['trade_type'] == 'buy') & (trades_annual['period_type'] == 'a')
].copy()
```

**Note on Step 2:** We filter to `annual_period_types` first (which includes a, aq1-aq4,
af0-af3) because `break_offpeak` and the merge steps may need the full set. After merging,
we filter to `period_type='a'` only in Step 6. This matches the MISO pattern where we filter
to `aq1-aq4` after the merge.

**Verification checks:**
- [ ] All 3 class types present for ALL PYs including pre-2023
- [ ] `mtm_1st_period` is annual scale (NOT monthly)
- [ ] `mcp` is annual scale
- [ ] `mtm_1st_period / mtm_1st_mean` ≈ 12 (confirms scale)
- [ ] Row granularity: 1 row per path, or 12 rows per path (month-level)?
- [ ] Buy-only filter: `trade_type == 'buy'`
- [ ] Path counts per (round, PY, class_type) — expect ~2K-5K per cell

**Output:** `pjm/data/canonical_annual_paths.parquet`

### Phase 1: Compute 1(rev) (requires Ray)

**Definition:** trailing 12 months = Apr N-1 through Mar N for PY N auction.

```python
from pbase.data.dataset.da.pjm import PjmDaLmpMonthlyAgg
from pbase.data.dataset.replacement import PjmNodalReplacement

# Load all needed months (Apr 2016 – Mar 2025 to cover PY2017-2025)
# Per month: get per-node DA congestion, apply replacement mapping
# Sum 12 months per (source, sink, class) → 1(rev) per path
```

**Verification checks:**
- [ ] Coverage: target > 85% of paths
- [ ] Scale: 1(rev) should be annual $ (12-month sum), comparable to `mtm_1st_period`
- [ ] Sign: sink − source (same convention as MCP)
- [ ] Correlation with baseline: expect positive (congested paths have both high MCP and high DA revenue)

**Output:** `pjm/data/pjm_1rev.parquet`

### Phase 2: Baseline diagnostics + blend search

For each round separately:

1. **Raw baseline diagnostics:** bias + MAE per (PY, round, flow, bin, class) at finest grain
2. **Blend search:** `w × mtm_1st_period + (1-w) × 1(rev)`, w per (flow, bin)
   - Report bias, MAE, AND tail risk (P95 of |residual|) per cell
   - Check if 1(rev) helps R1 more than R2-R4 (R1 has weaker baseline from LT auction)
3. **Oracle w analysis:** per-PY oracle w to assess non-stationarity
4. **Decision:** blend or pure baseline, per round

**Key expectations from MISO experience:**
- R2-R4: 1(rev) probably useless (oracle w = 1.0), same as MISO R2/R3
- R1: might help because LT yr1 R5 baseline is older/weaker — but PJM archive showed
  nodal_f0 was 68% worse than mtm for R1, so mtm is already good
- PY2022 will be the problem year (post-COVID volatility, same as MISO)

### Phase 3: Calibration cell experiments

Same A/B/C/D comparison as MISO:
- A: (bin, flow, class)
- B: (bin, class) — drop flow
- C: (bin, flow) — drop class
- D: (bin) — drop both

**PJM-specific considerations:**
- 3 classes (vs MISO's 2): class might matter more here
- dailyoffpeak and wkndonpeak are structurally different from onpeak
- Pre-2023 class backfill: verify the backfilled data is consistent before trusting class split
- PJM has 1 annual period (vs MISO's 4 quarters) → ~4x fewer paths per cell

**Small-cell fallback rule (explicit):**

With 3 classes and 1 period type, PJM calibration cells are ~4x smaller than MISO.
The fallback cascade when a cell has fewer than MIN_CELL (default 200) rows:

1. If config is (bin, flow, class) and cell < MIN_CELL: pool class → (bin, flow)
2. If (bin, flow) still < MIN_CELL: pool flow → (bin) only
3. If (bin) still < MIN_CELL: **raise** — this means the entire bin has too few paths
   for this fold, which should not happen with 5 bins and 1000+ total paths

Each fallback must be logged with cell identity and row count. If fallback rate exceeds
20% of cells for any round, investigate before proceeding (don't silently degrade).

The min cell threshold may need to be 200 (vs MISO's 300) given PJM's smaller cell sizes.
Verify cell sizes in Phase 0 before committing.

**Report at finest grain:** (PY, flow, class, bin) for P10/P20/P50/P90/P95 + bias.
Even when calibrating pooled, always report broken out.

**Expected outcome based on MISO:**
- q1-q3: class and flow don't matter much
- q4-q5: flow pooling likely helps (same non-stationarity)
- Class: unknown for PJM — might matter with 3 types

### Phase 4: Freeze + report

- Frozen script: `pjm/scripts/run_pjm.py`
- Consolidated report: `pjm/docs/pjm-consolidated-report.md`
- Finest-grain holdout tables persisted
- README updated

## Holdout / Dev Split

- **Data available:** PY2017-2025 (9 years)
- **Evaluable folds:** PY2019-2025 (7 years) — expanding window requires min 2 training PYs,
  so PY2017 (0 prior PYs) and PY2018 (1 prior PY) cannot be evaluated. PY2019 is the first
  evaluable year (trained on PY2017-2018).
- **Dev (evaluable):** PY2019-2022 (4 evaluable folds)
- **Holdout:** PY2023-2025 (3 evaluable folds)
- **Training-only:** PY2017-2018 (contribute to training but never appear as test folds)

When reporting "dev" metrics, always state "4 evaluable PYs (2019-2022)" not "6 years".

## Scale Convention (CRITICAL)

**All prices in annual $.** Native scale for PJM annual period type `a`.

- `mcp` = annual clearing price (target)
- `mtm_1st_period` = annual prior-round MCP (baseline)
- `1(rev)` = annual (12-month sum) DA congestion revenue
- Bid prices = annual $
- Do NOT use `mtm_1st_mean` (monthly = annual / 12)
- Do NOT multiply anything by 12

## Risk Items

1. **Class backfill quality:** Pre-2023 dailyoffpeak/wkndonpeak are engineered, not original.
   If they behave differently from PY2023+ classes, calibration cells with class split
   could be unreliable for early PYs.

2. **Fewer paths per cell:** PJM has 1 annual period (vs MISO's 4 quarters).
   Cell sizes are ~4x smaller. Min cell threshold may need adjustment (300 → 200?).

3. **R4 structural weakness:** Archive showed R4 PY2022 q5 at 92.3% P95 — worst across all
   rounds. R4 is last in the sequential clearing and may have different residual dynamics.

4. **Node ID format:** PJM uses numeric IDs. Must verify `PjmNodalReplacement` mapping
   works with the canonical loader's source_id/sink_id format.

## Dependencies

- Ray cluster at `ray://10.8.0.36:10001` for data loading
- `pbase.data.dataset.da.pjm.PjmDaLmpMonthlyAgg` for DA congestion
- `pbase.data.dataset.replacement.PjmNodalReplacement` for retired nodes
- `pbase.analysis.tools.all_positions.PjmApTools` for canonical trade loading
- pmodel venv: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`
