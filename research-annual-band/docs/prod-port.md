# V10 Annual Band Generator — Production Port Plan

**Date:** 2026-03-14
**Version:** V10 (empirical asymmetric quantile bands, quarterly scale)
**Status:** Research complete. Pre-port validation pending.
**Production Target:** PY 2026 annual auction (R1 ~April 2026, R2 ~May, R3 ~June)

---

## PY 2026 Data Availability

| Data | Available Now? | When Available |
|------|:-:|:---|
| Training data (PY 2019-2025 historical MCPs) | **Yes** | Now — calibrate bands immediately |
| R1 baseline (nodal_f0 for PY 2026 delivery) | **No** | ~April 2026 — needs f0 forward prices for Jun-Aug 2026 from latest monthly auction |
| R2 baseline (R1 MCP for PY 2026) | **No** | After R1 clears (~April 2026) |
| R3 baseline (R2 MCP for PY 2026) | **No** | After R2 clears (~May 2026) |
| PY 2026 actual MCP (for evaluation) | **No** | Never before auction — this is what we're bidding into |

**What we can do NOW:** Steps 1-2 (holdout validation + calibration artifact with empirical CPs).
**What needs ~April:** Step 3 (nodal_f0 lookup for PY 2026 paths).
**What needs the port:** Steps 4-6 + production module.

---

## Why Port

V10 delivers asymmetric pricing bands for MISO annual FTR auctions (aq1-aq4, 3 rounds).
For each path, it produces band edges at 8 coverage levels (P10-P99) that serve as bid prices,
with clearing probabilities derived from the quantile construction.

Current state: research script (`run_v9_bands.py`) produces correct results but is not
integrated into the production pipeline (`pmodel/src/pmodel/base/ftr24/v1/`).

Production needs: `autotuning.py` calls a band generator per (auction_date, period_type, class_type, round)
group. Annual currently uses legacy H-based bands. V10 replaces this with empirical quantile bands
that are 19-49% more accurate (MAE) than H baseline.

---

## Pre-Port Validation Checklist (all in research repo, no production code touched)

### 1. Holdout Validation (PY 2025)

**What:** Run V10 on full PYs (2020-2025) with min_train_pys=3. Save to `versions/bands/v10_holdout/`.
Compare dev (PY 2020-2024) vs holdout (PY 2025) coverage and width.

**Why:** Confirm results generalize. If holdout coverage drops >5pp vs dev, the model may be overfitting.

**How:**
```python
# In run_v9_bands.py, change constants:
VERSION_ID = "v10_holdout"
DEV_R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
DEV_R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MIN_TRAIN_PYS = 3
```

**Success criteria:** PY 2025 fold P95 coverage within 5pp of dev average per round.

### 2. Empirical Clearing Probabilities

**What:** For each band edge, compute actual buy/sell clearing rates from temporal CV test folds.
Store in `calibration_artifact.json` alongside the quantile pairs.

**Why:** Theoretical CPs (quantile levels) are off by 2-5pp at P50-P80 due to temporal non-stationarity.
Empirical CPs from test data are more accurate and what we should report to the optimizer.

**How:** Already analyzed (see CP accuracy analysis output). Need to:
1. Add `empirical_clearing_rates` dict to artifact JSON per quarter
2. Segment by flow_type (prevail/counter) since CPs differ by 2-5pp between them
3. Segment by bin group (q1-q4 vs q5) since q5 CPs are systematically distorted

**Artifact schema addition:**
```json
{
  "calibration": {
    "aq1": {
      "bin_pairs": {...},
      "boundaries": [...],
      "empirical_clearing_rates": {
        "buy_prevail": {"lower_p95": 5.4, "upper_p95": 98.0, ...},
        "buy_counter": {"lower_p95": 2.4, "upper_p95": 96.6, ...},
        "sell_prevail": {"lower_p95": 94.6, "upper_p95": 2.0, ...},
        "sell_counter": {"lower_p95": 97.6, "upper_p95": 3.4, ...},
        "buy_prevail_q5": {"lower_p95": 13.9, "upper_p95": 94.6, ...},
        "buy_counter_q5": {"lower_p95": 2.4, "upper_p95": 94.6, ...}
      }
    }
  }
}
```

### 3. Nodal f0 Lookup Table (R1 Baseline)

**What:** Pre-compute `nodal_f0` for all (source_id, sink_id) pairs in the path pool, per quarter and PY.
Save as `nodal_f0_lookup.parquet`.

**Why:** At auction time, R1 trades don't have `nodal_f0`. R2/R3 have `mtm_1st_mean` in the trades
data, but R1 requires a nodal MCP stitch that takes ~30s and needs Ray. Pre-computing avoids this
at inference time.

**How:** Extract the Phase 2 logic from `archive/phase1_3/run_crossproduct_research.py`:
1. Load f0 nodal MCPs via `MisoCalculator.get_mcp_df()` for each delivery month
2. BFS fallback via `MisoNodalReplacement` for missing nodes
3. Average over 3 delivery months per quarter
4. Save as lookup keyed by (source_id, sink_id, quarter)

**Output:** `/opt/temp/qianli/annual_research/nodal_f0_lookup_PY{YYYY}.parquet`

**Columns:** `source_id, sink_id, quarter, nodal_f0, baseline_q` (where `baseline_q = nodal_f0 * 3`)

### 4. End-to-End Inference Test

**What:** Simulate the production inference path entirely in the research repo:
1. Load a sample of R1 trades (as `autotuning.py` would provide them)
2. Look up `baseline_q` from the nodal_f0 lookup table
3. Load calibration artifact
4. Assign bins, look up quantile pairs, compute band edges
5. Compute clearing probabilities (empirical from artifact)
6. Output `bid_price_1..N, clearing_prob_1..N`

**Why:** Proves the full pipeline works end-to-end before touching production code.

**How:** Write `scripts/test_inference_pipeline.py` that:
- Takes a parquet of trades (source_id, sink_id, class_type, round, period_type)
- Produces a parquet with band edges and CPs
- Validates output schema matches what `autotuning.py` expects

**Success criteria:**
- All paths get bands (no nulls except coverage gap paths with H fallback)
- CPs are between 0-100, monotonic (higher bands → higher buy CP)
- Output schema: `bid_price_1..10, clearing_prob_1..10` (sorted descending for buy)
- Run completes in <10s for ~30K paths (one quarter of R1)

### 5. Fallback for Missing Nodes

**What:** ~1-11% of R1 paths don't have `nodal_f0` (source or sink node missing from f0 data).
Define and test the fallback chain.

**Why:** Production cannot silently drop paths. Every path must get bands.

**Fallback chain:**
1. `nodal_f0 * 3` (primary R1 baseline) — 89-100% coverage
2. `mtm_1st_mean * 3` (H baseline) — 100% coverage, worse accuracy
3. Zero baseline with wide bands (last resort) — should never trigger

### 6. Bug Checks (per CLAUDE.md)

- [ ] `mcp_mean` not used anywhere in inference code
- [ ] All baselines in quarterly scale (`* 3`)
- [ ] class_type validated at entry (`raise ValueError` on unexpected values)
- [ ] No silent fallbacks (explicit warning on every fallback)
- [ ] No `.get(key, {})` on critical calibration data
- [ ] Band widths are quarterly in all outputs
- [ ] CPs validated: 0-100 range, monotonic
- [ ] `num_threads=4` for any LightGBM usage (not applicable to V10 empirical)

---

## Production Port (after pre-port validation passes)

### Architecture: Where Annual Bands Actually Get Generated

The existing production flow for ALL period types (monthly + annual):

```
BaseFtrModel.run() (base.py:566)
  └── generate_bands(df=total_trades, ...) (band_generator.py:1563)
        └── generate_bands_for_group(trades, auction_month, period_type, ...)
              ├── if period_type in {"f0", "f1"}: generate_f0_f1_bands() [LightGBM + conformal]
              └── else (f2+, q*, aq*, a, yr*): generate_f2p_bands() [rule-based binning]
                    └── apply_corrected_bid_prices() → bid_price_1..10, clearing_prob_1..10
                          └── scale_bid_prices_by_duration() → ×3 for aq*, ×12 for a
```

**Annual (aq1-aq4) currently flows through `generate_f2p_bands()`** — symmetric rule-based
binned bands with `mtm_1st_mean` as baseline. This is what V10 replaces.

### Correct Integration Seam: Inside `generate_bands_for_group()`

**NOT** at `autotuning.py` (that's feature prep, not bid generation).
**NOT** as a separate entry point (that bypasses scaling + CP assignment).

The change is inside `generate_bands_for_group()` (band_generator.py:1422-1481):

```python
# BEFORE (line 1422-1455):
lgbm_ptypes = {"f0", "f1"}
if period_type in lgbm_ptypes:
    # ... LightGBM path
else:
    # Rule-based for everything including aq1-aq4
    train_df = load_training_data_for_f2p(...)
    result, val_df_with_bands = generate_f2p_bands(train_df, trades.copy())

# AFTER:
lgbm_ptypes = {"f0", "f1"}
annual_ptypes = {"aq1", "aq2", "aq3", "aq4"}

if period_type in lgbm_ptypes:
    # ... LightGBM path (unchanged)
elif period_type in annual_ptypes:
    # V10: asymmetric quantile bands from pre-calibrated artifact
    from .annual_band_generator import generate_annual_bands_for_group
    result, val_df_with_bands = generate_annual_bands_for_group(
        trades=trades.copy(),
        period_type=period_type,
        class_type=class_type,
        auction_month=auction_month,
        round_num=_get_round_from_trades(trades),  # extract from trades DataFrame
    )
else:
    # Rule-based for f2+, q* (unchanged)
    train_df = load_training_data_for_f2p(...)
    result, val_df_with_bands = generate_f2p_bands(train_df, trades.copy())
```

After this, the existing `apply_corrected_bid_prices()` and `scale_bid_prices_by_duration()`
still run — so the output contract is preserved.

### Scale Contract — RESOLVED

**The existing pipeline works in monthly scale internally:**
- `mtm_1st_mean` is monthly
- `generate_f2p_bands()` produces monthly-scale band edges
- `scale_bid_prices_by_duration()` multiplies by 3 for aq* at the end (line 1335-1343)

**Decision: Match existing contract.** The annual band generator outputs **monthly-scale**
band edges. `scale_bid_prices_by_duration()` stays untouched and handles ×3.

Concretely, inside `generate_annual_bands_for_group()`:
- R1: `baseline = nodal_f0` (monthly, from lookup table). NOT `nodal_f0 * 3`.
- R2/R3: `baseline = mtm_1st_mean` (monthly, already in trades). NOT `mtm_1st_mean * 3`.
- Calibration artifact stores quantile pairs calibrated on monthly residuals.
- Band edges: `lower/upper = baseline + lo/hi` (monthly).
- `scale_bid_prices_by_duration()` converts to quarterly downstream.

**Why monthly internally:** The optimizer and trade finalizer expect bid_price columns to be
pre-scaled by `scale_bid_prices_by_duration()`. If we output quarterly and skip scaling,
we'd need to modify the scaling function and risk breaking f0p. Matching the existing
contract is the zero-risk path.

### New File: `pmodel/src/pmodel/base/ftr24/v1/annual_band_generator.py`

```python
def generate_annual_bands_for_group(
    trades: pd.DataFrame,
    period_type: str,         # aq1-aq4
    class_type: str,          # onpeak/offpeak
    auction_month: str,       # e.g. "2026-06"
    round_num: int,           # 1, 2, or 3
    artifact_path: Path | str | None = None,  # override artifact location
    nodal_f0_lookup: pd.DataFrame | None = None,  # R1 only
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Generate V10 asymmetric bands for one annual (auction_month, period_type) group.

    Matches the return signature of generate_f0_f1_bands() and generate_f2p_bands():
    Returns (result_df_with_bands, val_df_with_bands_or_None).

    Band edges are in MONTHLY scale to match the production contract.
    scale_bid_prices_by_duration() handles the ×3 quarterly conversion downstream.

    Required columns in trades:
      source_id, sink_id, class_type, trade_type, mtm_1st_mean (R2/R3)
    R1 also needs nodal_f0_lookup for baseline computation.
    """
```

### Nodal f0 Lookup Schema (corrected)

Key: `(source_id, sink_id, period_type, planning_year, class_type)`

`nodal_f0` varies by ALL five dimensions:
- **period_type:** different delivery months (aq1=Jun-Aug, aq2=Sep-Nov, etc.)
- **planning_year:** different f0 forward prices each year
- **class_type:** onpeak vs offpeak have very different congestion patterns.
  99.5% of paths have different nodal_f0 for onpeak vs offpeak (mean diff 212 $/MWh monthly = 636 quarterly).
  Previous claim "class-agnostic" was WRONG — the f0 MCPs from `get_mcp_df()` are per-class.

| Column | Type | Notes |
|--------|------|-------|
| source_id | str | Path source node |
| sink_id | str | Path sink node |
| period_type | str | aq1-aq4 |
| planning_year | int | PY of the auction |
| class_type | str | onpeak or offpeak |
| nodal_f0 | float | Monthly avg of 3 delivery months (monthly scale) |

At inference: `baseline = nodal_f0` (monthly, no ×3). The downstream
`scale_bid_prices_by_duration()` handles the ×3 quarterly conversion.

### Clearing Probability: Per-Row Assignment — DETAILED

The annual generator does NOT use the existing `apply_corrected_bid_prices()`.
Instead, it handles CP assignment and bid_price sorting internally, because:
- Annual uses asymmetric bands (lower/upper are different quantiles)
- Annual CPs come from the calibration artifact, not from a val_df_with_bands
- The band column naming differs (lower_p10..upper_p99 vs lower_10..upper_99)

**Per-row CP assignment algorithm:**

```python
def assign_clearing_probs(trades_with_bands, artifact, round_num):
    """Assign per-row clearing probs and produce bid_price_1..10, clearing_prob_1..10.

    Inputs available in trades_with_bands:
      - trade_type: str ('buy' or 'sell') — REQUIRED, already in trades from autotuning
      - baseline: float — computed from nodal_f0 (R1) or mtm_1st_mean (R2/R3)
      - lower_p10..upper_p99: float — band edges (16 columns, 8 levels × lower/upper)

    Steps per row:
      1. flow_type = 'prevail' if baseline > 0, 'counter' if baseline <= 0
      2. bin_group = 'q5' if |baseline| >= artifact['boundaries'][Q80_index] else 'q1-q4'
      3. cp_key = f"{trade_type}_{flow_type}" or f"{trade_type}_{flow_type}_q5"
      4. For each band edge, look up CP from artifact['empirical_clearing_rates'][cp_key][band_name]
      5. Select 10 band edges as bid prices (configurable — e.g., all 16, or top 10 by spread)
      6. Sort: descending for buy (highest price = highest CP first), ascending for sell
      7. Output: bid_price_1..10, clearing_prob_1..10
    """
```

**Why 10 bid prices from 16 band edges?**
We have 8 levels × 2 sides = 16 band edges. The optimizer expects exactly 10 bid points.
Selection options:
- **Option A:** Use all 8 upper + 2 lower (upper covers the buy-relevant range)
- **Option B:** Select the 10 most spread-out points (maximize price diversity)
- **Decision:** Match f0p's BAND_ORDER pattern — use the 10 columns that f0p uses, mapped to our coverage levels.

**What `trade_type` is and where it comes from:**
`trade_type` = 'buy' or 'sell'. It's set by `autotuning.py`'s trade generation logic
BEFORE `generate_bands()` is called. It's already in the `total_trades` DataFrame at
line 566 of `base.py`. The annual generator receives it via the `trades` parameter
of `generate_bands_for_group()` — no changes needed to pass it.

**Q80 boundary for q5 assignment at inference:**
The calibration artifact stores `boundaries` (5 quantile bin edges from training data).
The 80th percentile boundary = `boundaries[4]` (0-indexed, 5 bins means 6 edges: [0, Q20, Q40, Q60, Q80, inf]).
At inference, `|baseline| >= boundaries[4]` → q5. Otherwise q1-q4.
This is the same logic used in calibration — no additional parameter needed.

### Artifacts to Deploy

| Artifact | Location | Refresh Frequency |
|----------|----------|:-:|
| `calibration_artifact.json` per round | `/opt/temp/qianli/annual_research/artifacts/v10/r{N}/` | Annually (after new PY data) |
| `nodal_f0_lookup_PY{YYYY}.parquet` | Same directory | Before each R1 auction |

### What Changes vs What Does NOT Change

**Changes:**
- `generate_bands_for_group()` in `band_generator.py` — add `elif period_type in annual_ptypes` branch
- New file: `annual_band_generator.py` — contains `generate_annual_bands_for_group()` + CP assignment
- New artifacts: `calibration_artifact.json` + `nodal_f0_lookup.parquet`

**Does NOT change:**
- `autotuning.py` — untouched (it still calls `generate_bands()` which dispatches internally)
- `generate_bands()` entry point — untouched (dispatches to `generate_bands_for_group()`)
- `scale_bid_prices_by_duration()` — untouched (×3 for aq* already handled)
- `apply_corrected_bid_prices()` — NOT called for annual (annual handles its own CP+sorting)
- `BaseFtrModel.run()` — untouched (calls `generate_bands()` as before)
- The optimizer, clustering, trade finalization — all downstream unchanged
- All monthly f0p logic — completely untouched

---

## Timeline

| Step | Est. Effort | Depends On |
|------|:-:|:-:|
| 1. Holdout validation | 5 min run | — |
| 2. Empirical CPs in artifact | 30 min code | Step 1 |
| 3. Nodal f0 lookup | 1 hour (Ray + data) | — |
| 4. E2E inference test | 1 hour code | Steps 2, 3 |
| 5. Fallback chain | 30 min | Step 3 |
| 6. Bug checks | 30 min | Step 4 |
| **Total pre-port** | **~4 hours** | |
| 7. Build `annual_band_generator.py` | 2 hours | Steps 1-6 pass |
| 8. Integration test with autotuning | 1 hour | Step 7 |
| **Total including port** | **~7 hours** | |
