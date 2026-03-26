# Task Plan: Annual Band Research — Port to Canonical Data Sources

**Created:** 2026-03-17
**Last updated:** 2026-03-20
**Status:** MISO R1 frozen. R2/R3 and PJM not started.

**Completed phases:** 0 (archive), 1 (canonical data), 2 (DA revenue), 3 (verify old models), 5a-5e (baseline + banding experiments, ML challengers)
**Current:** Ready for R2/R3 (Phase 4 baseline diagnostics for R2/R3, then repeat banding)
**Frozen results:** `miso/docs/r1-v2-consolidated-report.md`, `miso/scripts/run_r1_v2.py`

**Goal:** Archive old work, port MISO (then PJM) to canonical `get_trades_of_given_duration()` data source, verify consistency, then explore `1(rev)` and flow_type improvements to banding and baseline.
**Source of truth:** `human-input.md`

---

## Background: FTR Annual Auctions

### What is this project?

We build **pricing bands** for annual FTR (Financial Transmission Rights) auctions in PJM and MISO.
A band is a price interval [lower, upper] around a baseline prediction for each path's clearing
price (MCP). Traders use bands to set bid prices at different confidence levels (P10 through P99).

### Auction structure

| | MISO | PJM |
|--|------|-----|
| Period types | aq1, aq2, aq3, aq4 (quarterly, 3-month settlement each) | a (annual, 12-month settlement) |
| Rounds | 3 (R1, R2, R3) | 4 (R1, R2, R3, R4) |
| Planning year (PY) | June–May (PY2025 = Jun 2025 – May 2026) | Same |
| All rounds clear | ~April (R1), ~May (R2), ~Jun (R3) | ~April (all 4 rounds, same event) |
| Classes | onpeak, offpeak | onpeak, dailyoffpeak, wkndonpeak |
| Scale | Quarterly (all prices × 3 relative to monthly) | Annual (all prices × 12) |

### Key terms

- **MCP** = auction clearing price for a path. This is the TARGET we're banding around.
- **Baseline** = our best prediction of MCP before the auction clears. Different per round:
  - R1: nodal_f0 stitch (MISO) or LT yr1 R5 MCP (PJM) — no prior annual round available
  - R2+: mtm_1st_mean = prior round's MCP for the same path
- **Residual** = MCP − baseline. The band captures the distribution of this residual.
- **1(rev)** = DA (day-ahead) congestion revenue for a path over one planning year.
  Measures how much the path actually earned in real-time settlement. Available as historical data.
- **Flow type**: prevail (baseline > 0, you pay to acquire) vs counter (baseline < 0, you get paid)
- **Temporal CV**: expanding window — train on all PYs before the test PY. Never look ahead.

### The banding method

1. **Compute baseline** per path: baseline = f(mtm_1st_mean) or f(nodal_f0)
2. **Assign bins**: 5 quantile bins by |baseline| (q1=smallest, q5=largest)
3. **Calibrate bands per cell**: for each (bin, class_type) cell in training data,
   compute asymmetric quantile pairs of the signed residual:
   - `lower_offset = quantile(residual, (1-level)/2)` (e.g., 2.5th percentile for P95)
   - `upper_offset = quantile(residual, (1+level)/2)` (e.g., 97.5th percentile for P95)
4. **Apply**: `lower = baseline + lower_offset`, `upper = baseline + upper_offset`
5. **Evaluate**: buy clearing rate = P(MCP <= upper). Target for P95 upper = 97.5%.

### Leakage constraint

The annual auction for PY N clears ~April 7 of year N. Any data after March 31 is leak.
At auction time we know: all prior PY settlement data, current year through ~March,
all prior round MCPs.

---

## Context: What We Know So Far

### Previous Research (now archived in `miso/archive_v1/` and `pjm/archive_v1/`)

**MISO V10 bands:**
- Method: asymmetric empirical quantile bands, 5 quantile bins × 2 classes (onpeak/offpeak)
- R1 baseline: nodal_f0 stitch × 3 (quarterly). MAE = 792 quarterly (264 monthly × 3).
- R2/R3 baseline: mtm_1st_mean × 3 (quarterly). MAE = ~210/~168 quarterly.
- Dev P95 two-sided coverage: R1=92.0%, R2=90.2%, R3=91.9%
- P95 half-widths (quarterly $): R1=2,079, R2=532, R3=457
- PY2022 is universally worst year

**PJM V1 bands:**
- Method: same as MISO V10 but 5 bins × 3 classes (onpeak/dailyoffpeak/wkndonpeak)
- All 4 rounds use mtm_1st_mean × 12 (R1 = LT yr1 R5 MCP, R2-R4 = prior round MCP)
- Dev P95 coverage: R1=97.1%, R2=95.7%, R3=95.9%, R4=94.9%
- Holdout PY2025: R1 passes 5pp gate (95.6%), R2-R4 fail (88-90%)
- PY2022 is universally worst year

**Critical finding — mean reversion gap (PJM R1 q5):**
- Bands calibrate per (bin, class) pooling prevail + counter together
- Prevail (baseline > 0): residuals skew negative (MCP regresses toward zero)
- Counter (baseline < 0): residuals skew positive (MCP regresses toward zero)
- At P30 q5 the pooled calibration produces 42pp gap: prevail over-clears at 81%, counter under-clears at 39% (target 65%)
- At P95 the gap is negligible (<2.5pp) — only P30/P50 are affected
- Root cause: mean reversion is symmetric in |MCP|/|baseline| ratio (~0.86) but creates opposite-signed residuals
- Fix: calibrate per (bin, flow_type, class). Smallest cell = 5,892 rows. No data problem.

**Data sources used (OLD — to be replaced):**
- MISO: `all_residuals_v2.parquet`, `aq*_all_baselines.parquet` — built from `MisoApTools.get_all_cleared_trades()`
- PJM: `pjm_annual_with_mcp.parquet` — built from `PjmApTools.get_all_cleared_trades()` + `get_m2m_mcp_for_trades_all()`
- Neither dataset includes DA revenue (`da_award/mw`)

### Production Comparison

| Technique | Prod f0/f1 (conformal) | Prod annual (rule-based) | Ours (research) | Action |
|-----------|:---:|:---:|:---:|--------|
| Per-path ML width (LightGBM + conformal) | Y | N | N | Not for annual — too few PYs |
| Symmetric bands | Y | Y | N | Keep asymmetric |
| Flow type in width calibration | N | N | N | **Add — 42pp gap** |
| Coverage buffer for non-stationarity | Y | N | N | **Consider** |
| Quantile bins (vs fixed) | N | N | Y | Keep |
| Trains on actual annual data | N | N | Y | Keep |

### Reporting Standard (from CLAUDE.md)

- Primary metric: **buy clearing rate** = P(MCP <= upper band price)
- Granularity: **(round, planning_year, bin, flow_type)**
- Flow type: prevail (baseline > 0) or counter (baseline < 0)
- Thresholds: OK (>-3pp), WATCH (-3 to -5), CONCERN (-5 to -10), FLAG (<-10)
- Never report two-sided coverage as primary. Always decompose into one-sided.

---

## Phase 0: Archive [complete]

Moved all existing MISO and PJM work under `archive_v1/`. Added deprecation warnings (DeprecationWarning on import) to all 35 archived Python scripts.

**Result:**
```
miso/archive_v1/   — all old scripts, docs, versions, runbook
pjm/archive_v1/    — all old scripts, docs, versions, knowledge
```

---

## Phase 1: MISO — Load Canonical Data [pending]

### 1a. Load trades via canonical API

**Environment:** All scripts run from the pmodel venv:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```
This provides `pbase`, `pmodel`, `polars`, `pandas`, `numpy`, `lightgbm`, and Ray client.

```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
init_ray()

from pbase.analysis.tools.all_positions import MisoApTools
aptools = MisoApTools()

# Step 0: Print constants to verify expectations
print("annual_period_types:", aptools.tools.annual_period_types)  # expect: ['aq1','aq2','aq3','aq4']
print("classtypes:", aptools.tools.classtypes)  # expect: ['onpeak','offpeak']

# Step 1: Canonical loading — same as production
trades = aptools.get_trades_of_given_duration(
    participant=None, start_month='2018-06', end_month_in='2026-06'
)

# Step 2: Filter to annual
trades_annual = trades[trades['period_type'].isin(aptools.tools.annual_period_types)].copy()

# Step 3: break_offpeak — user says no-op for MISO, VERIFY
before_shape = trades_annual.shape
before_classes = sorted(trades_annual['class_type'].unique())
trades_annual = aptools.tools.break_offpeak(trades_annual)
after_shape = trades_annual.shape
after_classes = sorted(trades_annual['class_type'].unique())
print(f"break_offpeak: {before_shape} -> {after_shape}, classes: {before_classes} -> {after_classes}")

# Step 4: Filter to valid class types
trades_annual = trades_annual[trades_annual['class_type'].isin(aptools.tools.classtypes)].copy()
```

**Key questions to answer in 1a:**
- Does `get_trades_of_given_duration()` return `mtm_1st_mean`, `mcp`, `mcp_mean`, `mtm_now_*`?
- Does it return `split_market_month` / `split_month_mcp`? (i.e., are rows at month level or path level?)
- Does it include all 3 rounds and 4 quarters?
- How does it differ from `get_all_cleared_trades()`?
- If columns differ, document old → new mapping.

### 1b. Understand row granularity

Determine whether the canonical data returns:
- **(A)** One row per path (path-level) — `get_trades_of_given_duration` may already deduplicate
- **(B)** Multiple rows per path (month-level, one per `split_market_month`)

```python
path_key = ['source_id', 'sink_id', 'class_type', 'planning_year', 'round', 'period_type']
n_total = len(trades_annual)
n_unique = len(trades_annual.drop_duplicates(subset=path_key))
print(f"Total rows: {n_total}, unique paths: {n_unique}, ratio: {n_total/n_unique:.1f}")
# Expected ratio for MISO: ~3 (one row per settlement month in a quarter, aq=3 months)
# If ~1.0: already path-level. If ~12: full annual monthly split (unexpected for MISO quarterly).
# If unexpected ratio: STOP and inspect before proceeding.
```

If month-level: keep the full data (needed for revenue joins in Phase 2), but for
baseline/banding work, always group to path level first. **Do NOT silently drop rows** —
verify `mcp` and `mtm_1st_mean` are constant within each path group. If not constant, **raise**.

Cache two files:
- `miso/data/canonical_annual_monthly.parquet` — full month-level rows (needed for Phase 2 revenue joins)
- `miso/data/canonical_annual_paths.parquet` — deduplicated to one row per path (used for baseline/banding)

If data is already path-level (ratio ~1.0), materialize both paths with the same content.

### 1c. Verify consistency with old data

Load old data from `/opt/temp/qianli/annual_research/` (paths in archived scripts:
`miso/archive_v1/scripts/run_v9_bands.py:48`, `pjm/archive_v1/scripts/run_v1_bands.py:59`).

| Check | Old source | New source | Pass if |
|-------|-----------|-----------|---------|
| Row counts per (round, quarter, PY) | `all_residuals_v2.parquet` | new parquet | Exact match ±1% |
| Path universe | old (source_id, sink_id) | new | >98% overlap |
| Class types per PY | old | new | Identical sets |
| mtm_1st_mean values | old | new | Exact match on overlapping paths |
| **MCP scale check** | old `mcp_mean` (monthly) | new | See below |

**MCP scale verification (CRITICAL — catches the ×3 bug that burned us before):**

The archive documents a monthly/quarterly mismatch in `miso/archive_v1/docs/v10-consolidation-report.md:7`.
We use `mcp` (quarterly) as the sole target. `mcp_mean` is referenced below ONLY to verify
old-data consistency — it is never used as a working column.

```python
# 1. New data MUST have 'mcp' (quarterly clearing price)
assert 'mcp' in new_df.columns, "No 'mcp' column — cannot proceed"

# 2. If 'mcp_mean' also exists, verify it's mcp/3 (monthly)
if 'mcp_mean' in new_df.columns:
    ratio = (new_df['mcp'] / new_df['mcp_mean']).median()
    assert 2.9 < ratio < 3.1, f"mcp/mcp_mean ratio = {ratio}, expected ~3.0"

# 3. Cross-check new 'mcp' against old 'mcp' on overlapping paths
# (old all_residuals_v2.parquet has 'mcp' which was already quarterly for R2/R3)
merged = old.merge(new, on=path_key, suffixes=('_old', '_new'))
ratio = (merged['mcp_new'] / merged['mcp_old']).median()
assert 0.95 < ratio < 1.05, f"MCP scale mismatch: ratio = {ratio}"

# 4. After verification, DROP mcp_mean if it exists — we never use it
if 'mcp_mean' in new_df.columns:
    new_df = new_df.drop(columns=['mcp_mean'])
```

If ANY check fails: **STOP and investigate**. Do not proceed with wrong-scale data.

### 1d. Document data availability

Output table: (round, quarter, PY) → {path_count, monthly_row_count, classes, has_mtm_1st, has_mcp}

**Deliverable:** `miso/data/canonical_annual_paths.parquet` + `miso/data/canonical_annual_monthly.parquet` + `miso/docs/data-verification.md`

---

## Phase 2: MISO — Load DA Revenue and Compute 1(rev) [in progress]

### 2a. Revenue loading — DONE

`merge_mcp_revenue_all(trades, mode="rev")` loads each path's own-PY realized DA revenue.
Saved to `miso/data/canonical_annual_with_rev.parquet`. Coverage: 99.97% (1,299 nulls out of 3.97M).

### 2b. 1(rev) definition — Correct calendar logic

**PY N auction clears ~April of year N.** DA settlement through March of year N is available.

**For aq1, aq2, aq3:** `1(rev)` = PY N-1 same-quarter DA revenue. Fully settled by auction time.

| Target | Delivery | 1(rev) source | Example for PY2025 |
|--------|----------|---------------|-------------------|
| aq1 (Jun-Aug) | Jun-Aug PY N | Jun-Aug PY N-1 | Jun-Aug 2024 (settled Oct 2024) |
| aq2 (Sep-Nov) | Sep-Nov PY N | Sep-Nov PY N-1 | Sep-Nov 2024 (settled Jan 2025) |
| aq3 (Dec-Feb) | Dec-Feb PY N | Dec PY N-1 to Feb PY N | Dec 2024 - Feb 2025 (settled Apr 2025) |

**For aq4:** `1(rev)` is a **mixed-vintage** feature because PY N-1's aq4 (Apr-May) is NOT
settled at auction time (~April N):

| Month | Most recent available | Source PY | Age at auction |
|-------|----------------------|-----------|---------------|
| March | March of year N | PY N-1 | **~1 month — CONDITIONAL, must verify availability** |
| April | April of year N-1 | PY N-2 | ~12 months |
| May | May of year N-1 | PY N-2 | ~11 months |

So `1(rev)` for aq4 = `Mar_N_DA + Apr_{N-1}_DA + May_{N-1}_DA`.
April and May come from **PY N-2**, not PY N-1.

**March availability must be empirically verified** before use. If March DA is not available
by auction date, fall back to `Apr_{N-1} + May_{N-1} + Mar_{N-1}` (all from PY N-2, fully safe).

### 2c. 1(rev) via node-level DA (Option B) — PREFERRED

Self-join (Option A) has only **25% coverage** — most paths don't exist in both PY N and PY N-1.

**Option B is preferred:** Load per-node DA congestion directly via `MisoDaLmpMonthlyAgg`,
compute path revenue as `source_node_congestion - sink_node_congestion`, sum over the
relevant months. This gives 1(rev) for ANY path regardless of trading history.
Capping/winsorization of extreme node values is recommended before computing path differences.

### 2d. Leakage rules (CRITICAL)

**The auction cutoff is ~April 7 (user: `human-input.md:39`). Any data after March 31 is leak.**

| Feature | Months used | Leak risk | Status |
|---------|-------------|-----------|--------|
| `1(rev)` aq1/aq2 | Same quarter, PY N-1 | None — fully settled months before auction | **Cleared** |
| `1(rev)` aq3 | Dec N-1 to Feb N | None — Feb settled ~2 months before auction | **Cleared** |
| `1(rev)` aq4 Mar | March of year N | **CONDITIONAL** — must verify DA pipeline availability | **Must verify** |
| `1(rev)` aq4 Apr/May | Apr-May of year N-1 (PY N-2) | None — settled ~11-12 months ago | **Cleared** |
| Apr+ of year N | Apr onward | **LEAK** | **Blocked** |

### 2e. Phase 2 findings so far

- DA revenue loaded: 99.97% coverage, saved to parquet
- Self-join (Option A): only 25% path coverage — NOT viable for production
- DA revenue alone as baseline: 2.6× worse MAE than nodal_f0 — NOT viable standalone
- 70/30 blend (f0 + rev): reduces counter q5 bias from +949 to +109, but overall MAE is worse
- **Option B (per-node DA loading) needed for full coverage** — engineering work required

**Deliverable:** `miso/data/canonical_annual_with_rev.parquet` (own-PY DA revenue).
`miso/docs/revenue-features.md` documenting timing, coverage, and leakage analysis.

---

## Phase 3: MISO — Verify Old Models on New Data [pending]

### 3a. Reproduce old baseline results

Using new canonical data, compute baseline MAE and compare to archived values.

**Scale contract:** Phase 1c drops `mcp_mean` and establishes `mcp` (quarterly) as the sole
target column. All MAEs below are computed as `|mcp - baseline|` where both `mcp` and `baseline`
are in quarterly scale. Baselines: `nodal_f0 * 3` (R1), `mtm_1st_mean * 3` (R2/R3).

**Correct archived benchmarks** (from `miso/archive_v1/docs/v10-consolidation-report.md`):

All MAEs in **quarterly scale** (the only scale we use):

| Round | Baseline | Old MAE (quarterly) | Source |
|-------|----------|--------:|--------|
| R1 | nodal_f0 × 3 | **792** | v10-consolidation-report.md:66 |
| R2 | mtm_1st_mean × 3 | **240** | Computed directly on old all_residuals_v2.parquet (buy-only) |
| R3 | mtm_1st_mean × 3 | **190** | Computed directly on old all_residuals_v2.parquet (buy-only) |

**Note:** NOTES.md claimed ~70/~56 monthly — those were approximate. True old-data values are 80/63 monthly.
532/457 from old context section are R2/R3 quarterly P95 **half-widths**, not MAEs.

| Round | Baseline | Old MAE (quarterly) | New MAE (quarterly) | Diff | Status |
|-------|----------|--------:|--------:|-----:|--------|
| R1 | nodal_f0 × 3 | 792 | **792** | 0.0% | **PASS** |
| R2 | mtm_1st_mean × 3 | 240 | **232** | -3.3% | **PASS** |
| R3 | mtm_1st_mean × 3 | 190 | **186** | -2.1% | **PASS** |

For R1 nodal_f0: this requires `MisoCalculator.get_mcp_df()` via Ray to build the nodal stitch.
Two options:
- **(a)** Rebuild nodal_f0 stitch on new data paths — expensive but clean. Use Ray submit.
- **(b)** Join old `aq*_all_baselines.parquet` nodal_f0 values to new data by path key — fast, verifies path overlap.

Start with (b) for speed. **If join coverage < 95%, do NOT silently drop or fill** — raise and
switch to (a). If (a) also produces > 5% NaN in `nodal_f0`, raise and investigate before proceeding.

### 3b. Reproduce old band results

Run the V10 band calibration logic on new data. **Copy core functions** from
`miso/archive_v1/scripts/run_v9_bands.py` into a new `miso/scripts/band_utils.py` —
strip the deprecation warning, keep only the reusable functions:
- `compute_quantile_boundaries()`
- `assign_bins()`
- `calibrate_asymmetric_per_class()`
- `apply_asymmetric_bands_per_class_fast()`

Compare results. **The archived values are two-sided coverage** — reproduce in the same metric for apples-to-apples comparison, then also compute buy clearing rates for Phase 5.

| Metric | Old value (two-sided) | New value | Pass if |
|--------|-----------|-----------|---------|
| R1 aq1-4 avg P95 two-sided coverage | 92.0% | ? | Within 2pp |
| R2 avg P95 two-sided coverage | 90.2% | ? | Within 2pp |
| R3 avg P95 two-sided coverage | 91.9% | ? | Within 2pp |

Also report buy clearing rates at (round, PY, bin, flow_type) per CLAUDE.md standard — this establishes the Phase 5 baseline.

**Deliverable:** `miso/docs/verification-report.md` — old vs new, pass/fail per metric. + `miso/scripts/band_utils.py` — reusable core functions.

---

## Phase 4: MISO — Explore Revenue and Flow Type for Baseline [pending]

### 4a. Correlation analysis

For each (round, quarter), compute:
```python
residual = mcp - baseline  # quarterly scale
corr_1rev = spearman(residual, 1_rev)
corr_qrev = spearman(residual, q_rev_prior)
corr_recent = spearman(residual, recent_3mo)
corr_mar = spearman(residual, mar_rev)
```

Report as table: (round, quarter) → correlations.

### 4b. Blended baseline search (if correlations are promising)

For R1: `baseline_new = nodal_f0 * 3 + alpha * 1_rev`
- Grid: alpha in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
- Evaluate MAE on temporal CV (expanding window, min_train_pys=2)
- Compare to old baseline MAE

For R2/R3: `baseline_new = mtm_1st_mean * 3 + alpha * 1_rev`
- Same grid and eval

Production uses 0.93×mtm + 0.07×rev for f0p — test analogous weights.

### 4c. March revenue and quarter-specific revenue

Specifically for R1 where baseline is weakest:
- `baseline_new = baseline + beta * mar_rev`
- `baseline_new = baseline + beta * q_rev_prior` (quarter-specific prior year revenue)
- Hypothesis: paths congested recently or in the same season → MCP will be higher
- Grid search beta, evaluate MAE

### 4d. Baseline error diagnostics — flow type and magnitude

Before trying to improve the baseline, first understand WHERE it fails. This is divide-and-conquer.

All diagnostics below are in **quarterly scale**. Groupby always includes quarter (aq1-aq4)
because congestion patterns differ by season.

**Test A: Does flow type matter for baseline MAE?**

```python
for rnd in rounds:
    for qtr in ['aq1', 'aq2', 'aq3', 'aq4']:
        for ft in ['prevail', 'counter']:
            subset = df[(df['round'] == rnd) & (df['period_type'] == qtr) & (df['flow_type'] == ft)]
            mae = (subset['mcp'] - subset['baseline']).abs().mean()
            median_err = (subset['mcp'] - subset['baseline']).median()
            # Report: MAE, median signed error (bias), N
```

Report table: (round, quarter, flow_type) → MAE, median error, N.
- If counter MAE >> prevail MAE: counter paths have more mean-reversion error.
- If median error is far from zero: baseline is biased for that flow type.

**Test B: Does baseline magnitude matter for MAE?**

```python
for rnd in rounds:
    for qtr in ['aq1', 'aq2', 'aq3', 'aq4']:
        for bl in bin_labels:  # q1-q5 by |baseline|
            subset = df[(df['round'] == rnd) & (df['period_type'] == qtr) & (df['bin'] == bl)]
            mae = (subset['mcp'] - subset['baseline']).abs().mean()
            mae_pct = mae / subset['baseline'].abs().mean() * 100  # relative MAE
```

Report table: (round, quarter, bin) → MAE, relative MAE %.
- If q5 MAE >> q1 MAE in absolute terms: large baselines are harder to predict.
- If relative MAE is constant across bins: error scales proportionally (homoscedastic).
- If relative MAE is higher for small bins: small baselines have more noise.

**Test C: Flow type × magnitude interaction**

Cross (quarter, bin, flow_type) → MAE. Is the worst cell prevail-q5 or counter-q5?

Report as: (round, quarter, bin, flow_type) → MAE, N, median signed error.
Start with 4 dimensions. Only add class_type as a 5th if the 4-way analysis shows class-specific
patterns. Check minimum group size at each level — **do not use a groupby if min cell < 200**.

**Test D: Per-flow-type baseline weights (if Tests A-C show differences)**

If counter paths have systematically different bias:
```python
for ft in ['prevail', 'counter']:
    # Grid search: baseline_new = w * mtm + (1-w) * rev_feature
    # Find optimal w per flow type
```

If optimal weights differ by > 0.02: use per-flow-type baselines.
This mirrors prod's v2 f0 approach (separate `prevail_baseline` and `counter_baseline` configs).

**Test E: Regional baseline (if Test B shows magnitude matters)**

If relative MAE differs across bins:
```python
# Fit a simple per-bin scaling: baseline_new = baseline * scale[bin]
# Or: baseline_new = baseline + offset[bin]
# Evaluate MAE improvement per bin
```

**Deliverable:** `miso/docs/baseline-diagnostics.md` — tables for Tests A-E.
`miso/docs/baseline-with-rev.md` — MAE improvement tables per (round, quarter, flow_type) for blending.

---

## Phase 5: MISO — Explore Flow Type + 1(rev) for Banding [pending]

**Note:** Phase 5a (flow_type banding) does NOT depend on Phase 2 (revenue loading).
It can start as soon as Phase 1 + Phase 3 are complete. Phases 2+4 can run in parallel.

### 5a. Flow type split in calibration

**Full banding structure (explicit):**

```
Outer loop (data partition):   round × quarter       ← separate dataset per (round, quarter)
Temporal CV:                   for each test_py, train on PYs < test_py
Calibration cell:              (bin, flow_type, class_type)
                               = 5 bins × 2 flow × 2 classes = 20 cells per (round, quarter, fold)
```

We do NOT pool across quarters or rounds for calibration. Each (round, quarter) is an
independent banding problem.

```python
flow_type = "prevail" if baseline > 0 else "counter"
```

For each cell, compute asymmetric quantile pairs from the signed residual, same as V10 but with the split.

**Minimum group size rule:** If any calibration cell has < 500 rows, **raise** — do not
fall back to pooled. If this happens systematically, reduce bins from 5 to 3 for that
(round, quarter) slice. Report exactly which cells are too small.

**Expected improvement (based on PJM analysis):**
- P95 clearing rate: minimal change (gap was <2.5pp)
- P30/P50 clearing rate: large improvement for counter-flow q4/q5 paths (42pp gap at P30)

Evaluate with full clearing rate tables per CLAUDE.md standard:
(round, quarter, PY, bin, flow_type) × buy@upper_P95 and buy@upper_P50.

**Verify MISO cell sizes first** — compute min cell across all (round, quarter, fold)
combinations before running any banding. MISO has 2 classes (not 3), so cells should be
larger than PJM. But MISO quarterly data may have fewer rows per quarter than PJM annual.

### 5b. 1(rev) as binning dimension

Three options to evaluate:

| Option | Calibration cell | Total cells | Pros | Cons |
|--------|-----------------|------:|------|------|
| A | (rev_quintile, flow, class) | 5×2×2=20 | Direct contention signal | Loses baseline magnitude |
| B | (baseline_bin, rev_tercile, flow, class) | 5×3×2×2=60 | Both signals | Too many cells? |
| C | (baseline_bin, flow, class) + 1_rev as offset to baseline | 5×2×2=20 | No cell explosion | Needs simple model |

For each option:
1. Compute cell sizes. Reject if min < 500.
2. Run bands with temporal CV.
3. Report clearing rates at (round, PY, bin, flow_type).

### 5c. Combined: flow_type + best 1(rev) option

If 5a and 5b both help, combine. Watch cell sizes.

### 5d. Residual capping (after 5a)

Winsorize residuals at P0.5/P99.5 within each (bin, flow_type, class) cell BEFORE computing
quantile pairs. This stabilizes quantile estimates against single extreme paths.

```python
# Within each cell:
lo_cap = residuals.quantile(0.005)
hi_cap = residuals.quantile(0.995)
residuals_capped = residuals.clip(lo_cap, hi_cap)
# Then compute quantile pairs on residuals_capped
```

Evaluate: compare buy clearing rates at all levels (P10-P99) vs uncapped 5a.
If capping improves P50-P80 coverage (where under-coverage is worst, 4-5pp) without
hurting P95/P99, adopt it.

Also test **band width capping**: cap each cell's band width at 2× the pooled-across-bins
width for the same coverage level. Prevents one cell from being absurdly wide.

### 5e. ML banding experiment (after 5a)

LightGBM regression on |residual| per path, then conformal calibration (same method as
production f0/f1 conformal bands).

**Features** (all available at auction time):
- `|baseline|` — residual magnitude scales with baseline
- `sign(baseline)` — prevail vs counter have different residual distributions
- `class_type` — onpeak vs offpeak
- `cleared_volume` — liquidity proxy
- `bid_price - baseline` — trader confidence signal (far bid = uncertain path)

**Method:**
1. Split: train on PYs < test_py (same expanding window as empirical)
2. Train LightGBM to predict |residual| (MAE objective, not quantile regression)
3. Compute conformal nonconformity scores on validation fold: `score = |residual| / predicted_|residual|`
4. For each coverage level L, conformal scalar = `quantile(scores, L)`
5. Apply: `width = scalar × predicted_|residual|`, `upper = baseline + width`, `lower = baseline - width`

**Note:** ML conformal bands are SYMMETRIC (baseline ± width). Compare against our asymmetric
empirical bands (5a) to see if per-path adaptation compensates for losing asymmetry.

**Constraint:** With 2-5 PYs of training (~30-60k rows per quarter), the model must be simple.
Use `max_depth=3`, `n_estimators=100`, `num_threads=4`.

Evaluate: buy clearing rates at all levels vs 5a and 5a+5d.

### 5f. Final comparison and recommendation

Compare all variants on the same clearing rate tables:
- V10 baseline (current, no flow_type)
- 5a: empirical + flow_type
- 5a+5d: empirical + flow_type + capping
- 5e: ML conformal
- 5b: 1(rev) binning (if Phase 2 is done)
- Best combination

**Deliverable:** `miso/docs/banding-v2-report.md` with:
- Clearing rate tables at (round, PY, bin, flow_type) for every variant
- Cell size tables
- Coverage error at all 8 levels (P10-P99)
- Recommendation: which combination to port to production

---

## Phase 6: PJM — Repeat Phases 1-5 [pending]

Same structure as MISO but with PJM-specific differences:

| Aspect | MISO | PJM |
|--------|------|-----|
| ApTools | `MisoApTools` | `PjmApTools` |
| `break_offpeak` | No-op (verify) | Required (splits offpeak → dailyoffpeak + wkndonpeak) |
| Period types | aq1, aq2, aq3, aq4 | a |
| Scale | ×3 (quarterly) | ×12 (annual) |
| Rounds | 3 (R1-R3) | 4 (R1-R4) |
| R1 baseline | nodal_f0 stitch | mtm_1st_mean × 12 (LT yr1 R5) |
| R2+ baseline | mtm_1st_mean × 3 | mtm_1st_mean × 12 |
| DA revenue loader | `MisoDaLmpMonthlyAgg` | `PjmDaLmpMonthlyAgg` |
| Classes | onpeak, offpeak | onpeak, dailyoffpeak, wkndonpeak |
| Quarter-specific rev | Yes (aq1-aq4 have different 3-month windows) | No (single annual period "a") |

**PJM-specific notes:**
- `1(rev)` for PJM = sum of 12 months DA revenue for prior PY (Jun-May). No quarter alignment needed.
- PJM dailyoffpeak/wkndonpeak only exist from PY2023. For PY2017-2022 (R1 onpeak-only),
  the calibration cell (bin, flow_type, class) collapses to (bin, flow_type, onpeak) — 10 cells
  instead of 30. This is NOT a fallback — it's the data reality (those classes didn't exist
  in the auction). The same min-500 rule applies; if any (bin, flow_type, onpeak) cell is < 500,
  reduce bins.
- PJM R1 baseline (LT yr1 R5) already had exhaustive improvement search — revenue blend is the untested angle.

**Deliverable:** PJM results matching MISO structure.

---

## Phase 7: Write Production Porting Plan [pending]

Detailed, independently auditable steps:

1. **Data contract:** input schema (columns, types, nullability) for band generator
2. **Baseline computation:** formula per (RTO, round), with revenue blend weights if Phase 4 succeeds, per-flow-type weights if 4d succeeds
3. **Band calibration:** per (bin, flow_type, class) asymmetric quantile pairs, with exact quantile formulas
4. **CP assignment:** empirical clearing probabilities per (flow_type, band_level), with fallback rules
5. **Integration seam:** exact files and line numbers in pmodel to modify
6. **Verification checksums:** expected P95 buy clearing rate ± tolerance for each (round, quarter/period) on known data

**Production touchpoints (from branch `feature/4496-pjm-first-round-mtm-fix`, line numbers are snapshots):**

| File | Function / symbol (durable locator) | Line (snapshot) | What |
|------|-------------------------------------|-----:|------|
| `pmodel/.../ftr24/v1/band_generator.py` | `generate_bands_for_group()` routing logic | ~1684 | Period type routing: `lgbm_ptypes = {"f0", "f1"}`, else rule-based |
| `pmodel/.../ftr24/v1/band_generator.py` | `generate_f2p_bands()` | ~1711 | Current annual band generation path |
| `pmodel/.../ftr24/v1/autotuning.py` | `run_autotuning()` | ~387 | Autotuning orchestration for annual |
| `pmodel/.../ftr24/v1/calib.py` | `calibrate_conformal_scalars()` | — | Conformal calibration (f0/f1 only, reference) |

**Environment requirements:**
- Python: pmodel venv at `/home/xyz/workspace/pmodel/.venv/`
- `pbase` must be importable (installed in venv or via `extra_modules`)
- Ray cluster at `ray://10.8.0.36:10001` for data loading
- Old parquets at `/opt/temp/qianli/annual_research/` for verification only

**Deliverable:** `miso/docs/porting-plan.md` ready for independent audit.

---

## Dependency Graph

```
Phase 0 (archive) ── complete
Phase 1 (load canonical data) ── complete
Phase 3 (verify old models) ── complete
    │
    ├──→ Phase 5a (flow_type empirical)     ← NEXT, no dependencies
    │        │
    │        ├──→ Phase 5d (residual capping)
    │        └──→ Phase 5e (ML conformal)
    │
    └──→ Phase 2 (load DA revenue)          ← parallel track
             │
             ├──→ Phase 4 (rev for baseline + diagnostics)
             └──→ Phase 5b (rev for banding)

Phase 5f (final comparison) ← waits for 5a, 5d, 5e, and optionally 5b
    │
Phase 6 (PJM) ──→ Phase 7 (porting plan)
```

**Critical path:** 5a → 5d → 5e → 5f. Revenue (Phase 2/4/5b) is parallel and non-blocking.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-17 | Archive all old work, start fresh | New canonical data source (`get_trades_of_given_duration`); old docs confuse |
| 2026-03-17 | MISO first, then PJM | User instruction |
| 2026-03-17 | Flow type in calibration is priority #1 for banding | 42pp gap from mean reversion proven on PJM data |
| 2026-03-17 | 1(rev) exploration is priority #2 | User hypothesis: contended paths → higher MCP; prod blends rev |
| 2026-03-17 | All trades are buy; report one-sided clearing rates | Per CLAUDE.md band reporting standard |
| 2026-03-17 | Self-join approach for 1(rev) over direct DA LMP loading | Simpler, reuses existing merge_mcp_revenue_all; fallback to direct if coverage < 80% |
| 2026-03-17 | Quarter-specific revenue features needed | User explicitly asked about aq1 Jun/Jul/Aug DA and aq4 limited March data |
| 2026-03-17 | Test per-flow-type baseline weights | User said "for baseline consider flow type"; mirrors prod v2 f0 approach |
| 2026-03-19 | Filter to buy trades only | User: "filter to buy-only" — 5% sells dropped |
| 2026-03-19 | PY2023-2025 as holdout (more data = more holdout PYs) | User: "use 23-25 as holdout for example" |
| 2026-03-19 | Canonical pipeline: get_trades_of_given_duration → merge_cleared_volume → get_m2m_mcp_for_trades_all | Discovered during Phase 1a — 3-step process, not single call |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |

## Review Issues Found & Fixed

### Round 1 (self-review, 2026-03-17)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | `merge_mcp_revenue_all` gives current PY revenue, not prior PY | SIGNIFICANT | Changed to self-join approach: load all PYs' own revenue, then join PY N with PY N-1 |
| 2 | Missing quarter-specific revenue features | SIGNIFICANT | Added `q_rev_prior`, `q_rev_2yr`, `recent_3mo` with quarter-to-months mapping |
| 3 | Missing flow type for baseline | SIGNIFICANT | Added Phase 4d: test per-flow-type baseline weights |
| 4 | No deduplication step for split_market_month rows | MODERATE | Added Phase 1b: check row granularity, verify constants within path groups |
| 5 | R1 nodal_f0 stitch dependency unclear | MODERATE | Added two options with strict raise-if-NaN policy |
| 6 | Sequential phases where parallel is possible | MINOR | Added dependency graph showing 5a can start without waiting for Phase 2 |
| 7 | `annual_period_types` and `classtypes` not verified | MINOR | Added explicit print statements in Phase 1a Step 0 |

### Round 2 (user feedback, 2026-03-17)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 8 | Phase 4d too vague — need explicit diagnostic tests | SIGNIFICANT | Expanded into Tests A-E: flow_type MAE, magnitude MAE, interaction, per-flow weights, regional scaling |
| 9 | Unnecessary dedup in Phase 1b | MINOR | Changed to "understand granularity" — check ratio, verify constants, raise if inconsistent |
| 10 | Fallback language too loose — risk of hidden bugs | MODERATE | Tightened: NaN coverage > 5% → raise. No silent defaults. Added to CLAUDE.md. |
| 11 | Ray submission pattern not documented | MINOR | Added to CLAUDE.md: use dual-mode CLI pattern and /parallel-with-ray skill |

### Round 3 (independent review, 2026-03-19)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 12 | Phase 5 says raise on < 500, but Phase 6 says "expect fallback-to-pooled for PJM" — contradiction | SIGNIFICANT | Resolved: PJM onpeak-only PYs have fewer classes (not a fallback, just fewer cells). Same min-500 rule applies. Reduce bins if needed. |
| 13 | Phase 1b caches single file but Phase 1d deliverable lists two files | MODERATE | Fixed: Phase 1b now explicitly caches both `canonical_annual_monthly.parquet` and `canonical_annual_paths.parquet` |
| 14 | Phase 1a code not copy-paste runnable — no venv bootstrap | MODERATE | Fixed: added explicit `cd pmodel && source .venv/bin/activate` before code block |
| 15 | Revenue fallback policy ambiguous across Phase 2a and 2d | MINOR | Clarified: NaN for missing paths (no fill), switch to Option B if >20% NaN. Nodal_f0 DA substitution is explicit documented substitution, not silent fallback. |
| 16 | Row ratio comment says "~3 or ~12" but MISO quarterly should be ~3 | MINOR | Fixed: comment now says ~3 expected for MISO, ~12 would be unexpected and should STOP. |

### Round 4 (independent review, 2026-03-19)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 17 | R2/R3 quarterly MAE ~210/~168 presented as direct but source only has monthly | MINOR | Added "(inferred: ~70 monthly × 3)" and "(monthly only)" source annotation |
| 18 | "symlink" prescription is brittle implementation detail | MINOR | Changed to "materialize both paths with the same content" |
| 19 | Production touchpoint line numbers are brittle for audit | MINOR | Added function names as durable locators, line numbers marked as snapshots |
| 20 | Phase 3 doesn't state scale contract after mcp_mean drop | MINOR | Added explicit scale contract: all MAEs computed as \|mcp - baseline\| in quarterly |
