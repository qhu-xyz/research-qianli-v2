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

### New File: `pmodel/src/pmodel/base/ftr24/v1/annual_band_generator.py`

```python
def generate_annual_bands(
    trades: pd.DataFrame,
    round_num: int,
    quarter: str,
    class_type: str,
    artifact_path: Path | str,
    nodal_f0_lookup: pd.DataFrame | None = None,  # required for R1
) -> pd.DataFrame:
    """Generate bid prices and clearing probabilities for annual FTR trades.

    Parameters
    ----------
    trades : DataFrame with source_id, sink_id, class_type, mtm_1st_mean
    round_num : 1, 2, or 3
    quarter : aq1, aq2, aq3, aq4
    class_type : onpeak or offpeak
    artifact_path : path to calibration_artifact.json
    nodal_f0_lookup : R1 only — pre-computed nodal f0 lookup table

    Returns
    -------
    DataFrame with added columns:
        baseline_q, lower_p10..upper_p99, bid_price_1..10, clearing_prob_1..10
    """
```

### Integration Point: `autotuning.py`

In the annual R1 processing block (around line 26-33 of `autotuning.py`):

```python
# BEFORE (legacy):
miso_annual_r1_trades = aptools.tools.fill_mtm_1st_period_with_hist_revenue(...)

# AFTER (v10):
from pmodel.base.ftr24.v1.annual_band_generator import generate_annual_bands
trades_with_bands = generate_annual_bands(
    trades=miso_annual_r1_trades,
    round_num=1, quarter=quarter, class_type=class_type,
    artifact_path=ANNUAL_ARTIFACT_DIR / f"r1/{quarter}/calibration_artifact.json",
    nodal_f0_lookup=nodal_f0_lookup,
)
```

### Artifacts to Deploy

| Artifact | Location | Refresh Frequency |
|----------|----------|:-:|
| `calibration_artifact.json` per (round, quarter) | `/opt/temp/qianli/annual_research/artifacts/v10/r{N}/` | Annually (after new PY data) |
| `nodal_f0_lookup_PY{YYYY}.parquet` | Same directory | Annually (before R1 auction) |

### What Does NOT Change

- `band_generator.py` (monthly f0p) — untouched
- R2/R3 flow through `autotuning.py` — only baseline changes (`mtm_1st_mean * 3`)
- The optimizer, bid point generation, trade finalization — all downstream unchanged

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
