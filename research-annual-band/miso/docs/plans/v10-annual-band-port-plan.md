# V10 Annual Band Generator — Port Plan to pmodel ftr24/v1

**Date:** 2026-03-14
**Source:** `research-annual-band/scripts/run_v9_bands.py` (v10 empirical, quarterly scale)
**Target:** `pmodel/src/pmodel/base/ftr24/v1/annual_band_generator.py` (new file)

---

## 1. F0P Pipeline Architecture (for reference)

The monthly f0p band generator (`band_generator.py`) works as follows:

```
Input: trades DataFrame with auction_month, period_type, mtm_1st_mean, trade_type
  │
  ├── Step 1: TOGGLE — Save training data to parquet (per auction_month/period_type/class_type)
  │     cached at /opt/temp/qianli/mcp_pred_training/
  │
  ├── Step 2: BASELINE — weighted blend of mtm_1st_mean + 1(rev)
  │     baseline = mtm_weight * mtm + rev_weight * rev (per period_type)
  │
  ├── Step 3: FEATURES — derived from mtm, rev, baseline
  │     feat_abs_mtm, feat_log_abs_baseline, feat_mtm_volatility, ...
  │
  ├── Step 4: SEGMENT — split into "low" (|mtm| < 300) and "high"
  │
  ├── Step 5: ML MODEL — LightGBM predicts |residual| per segment
  │     conformal calibration on validation set
  │
  ├── Step 6: BANDS — baseline ± width at 15 coverage levels
  │     10 band columns selected (lower_10/30/50/70/99, upper_10/30/50/70/99, upper_50)
  │
  ├── Step 7: CLEARING PROBS — empirical from val/train historical data
  │     hybrid: empirical for some bands, rule-based for others
  │     segmented by flow_type (prevail/counter) × trade_type (buy/sell)
  │
  └── Output: bid_price_1..10, clearing_prob_1..10
```

## 2. Annual Pipeline — What Changes

### 2a. Training Data Toggle — NOT NEEDED

**F0P:** Saves per-month training data to parquet because data loading requires Ray + pbase calls that are expensive. The toggle caches this.

**Annual:** Our data is already pre-cached in:
- R1: `aq*_all_baselines.parquet` (4 files, ~4-5 MB each)
- R2/R3: `all_residuals_v2.parquet` (436 MB, one-time load)

These are small, local, and load in <1 second. **No toggle needed.** The calibration artifact (`calibration_artifact.json`) serves the same purpose — it stores pre-computed band parameters that can be loaded at inference time without re-calibrating.

### 2b. Baseline — Already Corrected

**F0P:** `baseline = mtm_weight * mtm_1st_mean + rev_weight * 1(rev)`

**Annual:**
- R1: `baseline_q = nodal_f0 * 3` (quarterly, no learned weights)
- R2/R3: `baseline_q = mtm_1st_mean * 3` (quarterly)

Both are pure column transforms, no ML. Already implemented in `run_v9_bands.py` data loaders.

### 2c. Band Width — Empirical Quantile (No ML)

**F0P:** LightGBM predicts |residual|, then conformal calibration scales predictions to target coverage. Two segments (low/high |mtm|). 15 coverage levels.

**Annual (v10):** Empirical quantile bands. 5 quantile bins × 2 classes = 10 cells. Signed quantile pairs at 8 coverage levels. No ML, no conformal calibration.

**Key difference:** F0P uses ML → needs train/val/test split. Annual uses empirical quantiles → only needs train/test (the "calibration" IS the quantile computation).

### 2d. Clearing Probabilities — Analysis Required

**F0P:** Empirical clearing rates from validation/training data, segmented by flow_type × trade_type. Hybrid: some bands use empirical, others use rule-based fallback.

**Annual:** For asymmetric bands, the clearing probability at a band edge IS the quantile level by construction:
- `lower_p95` = baseline + quantile(residual, 0.025) → theoretically, 2.5% of MCPs fall below this
- `upper_p95` = baseline + quantile(residual, 0.975) → theoretically, 2.5% of MCPs fall above this

For a **BUY trade** at `upper_p95`: clearing_prob ≈ 97.5% (MCP < upper_p95 in 97.5% of cases)
For a **SELL trade** at `lower_p95`: clearing_prob ≈ 97.5% (MCP > lower_p95 in 97.5% of cases)

**BUT**: the designated coverage may differ from true coverage (temporal non-stationarity, cold-start, q5 under-coverage). We need to analyze this gap.

---

## 3. Clearing Probability Accuracy Analysis

### 3a. What We Need to Check

For each band edge (lower_p10 through upper_p99), compare:
- **Designated CP**: the quantile level used to calibrate (e.g., upper_p95 → 97.5% for buy)
- **Actual CP**: the empirical rate from temporal CV test folds

Break down by:
- Round (R1, R2, R3)
- Quarter (aq1-aq4)
- Bin (q1-q5, especially q5)
- Flow type (prevail/counter)
- Coverage level (all 8)

### 3b. Key Questions

1. **Which band edges are most inaccurate?** Lower bands or upper bands? Low coverage (P10) or high (P95/P99)?
2. **Does q5 distort clearing probs?** We know q5 under-covers at P95 (83-90% vs 95% target). Does this mean buy CPs at upper_p95 are also off?
3. **Prevail vs counter asymmetry?** Do clearing probs differ for positive vs negative baseline paths?
4. **How does the gap vary by PY?** Cold-start folds (PY 2021-2022) likely have worse CP accuracy.

---

## 4. Implementation Plan

### Task 1: Clearing Prob Accuracy Analysis

Run v10 temporal CV, but for each test fold, compute:
- For each band edge × each path: did MCP clear below/above this price?
- Aggregate by (round, quarter, bin, flow_type) to get actual clearing rates
- Compare against designated rates

Output: Table showing designated vs actual clearing probability per band edge, segmented by all dimensions.

### Task 2: Create `annual_band_generator.py` in pmodel

**File:** `pmodel/src/pmodel/base/ftr24/v1/annual_band_generator.py`

Functions needed:

```python
def load_calibration_artifact(artifact_path: Path, quarter: str, round_num: int) -> dict:
    """Load pre-computed band parameters for a specific quarter/round."""

def compute_annual_baseline(trades: pd.DataFrame, round_num: int) -> pd.DataFrame:
    """Compute quarterly baseline.
    R1: nodal_f0 * 3 (requires nodal data lookup)
    R2/R3: mtm_1st_mean * 3 (column already in trades)
    """

def apply_annual_bands(
    trades: pd.DataFrame,
    calibration: dict,
    round_num: int,
) -> pd.DataFrame:
    """Apply pre-calibrated bands to trades.
    1. Assign bins based on |baseline_q| using calibration boundaries
    2. Look up (bin, class) quantile pairs from calibration
    3. Compute lower/upper band edges: baseline_q + lo/hi
    """

def compute_annual_clearing_probs(
    trades_with_bands: pd.DataFrame,
    round_num: int,
) -> pd.DataFrame:
    """Compute clearing probabilities per band edge.
    For asymmetric bands, CP = quantile level (theoretical).
    Optionally: apply empirical correction from accuracy analysis.
    Output: bid_price_1..N, clearing_prob_1..N
    """

def generate_annual_bands(
    trades: pd.DataFrame,
    round_num: int,
    quarter: str,
    class_type: str,
    artifact_path: Path | str | None = None,
) -> pd.DataFrame:
    """Main entry point. Plug-and-play replacement for generate_bands().
    Returns DataFrame with bid_price_1..N, clearing_prob_1..N columns.
    """
```

### Task 3: Integration with autotuning.py

In `autotuning.py`, annual R1 trades already go through `fill_mtm_1st_period_with_hist_revenue()`. After that, we need to:
1. Call `generate_annual_bands(trades, round_num=1, ...)` instead of the current band logic
2. For R2/R3, call with `round_num=2` or `3`

### Task 4: Bug Check (per CLAUDE.md)

- [ ] Verify `mcp_mean` is NOT used anywhere in the new code
- [ ] Verify baseline is quarterly (`* 3`) for all rounds
- [ ] Verify class_type validation (`raise ValueError` on unexpected values)
- [ ] Verify no silent fallbacks (explicit warnings on every bin/class fallback)
- [ ] Verify band widths are quarterly in output
- [ ] Verify clearing probs are between 0-100 and monotonic (higher bands → higher buy CP)
- [ ] Test with edge cases: paths with baseline_q = 0, very large |baseline_q|, single class_type

---

## 5. Difficulty Assessment

| Component | Difficulty | Notes |
|-----------|:---:|-------|
| Baseline calculation | Easy | Already done in run_v9_bands.py loaders |
| Band width calibration | Easy | Load artifact JSON, assign bins, look up quantile pairs |
| Band application | Easy | baseline_q + lo/hi per (bin, class) |
| Clearing prob (theoretical) | Easy | CP = quantile level by construction |
| Clearing prob (empirical correction) | **Medium** | Needs accuracy analysis first (Task 1) |
| Integration with autotuning | **Medium** | Need to understand the groupby flow in autotuning.py |
| R1 nodal_f0 baseline at inference | **Hard** | Requires `MisoCalculator.get_mcp_df()` + nodal replacement. Currently done offline, not at inference time. May need to pre-compute and cache. |

### The Hard Part: R1 Baseline at Inference Time

For R2/R3, `mtm_1st_mean` is already in the trades DataFrame — just multiply by 3.

For R1, `nodal_f0` is NOT in the trades DataFrame. It must be computed from:
1. `MisoCalculator.get_mcp_df()` — monthly f0 nodal MCPs (requires Ray)
2. `MisoNodalReplacement.load_data()` — BFS fallback for missing nodes
3. Average over 3 delivery months
4. Stitch: `sink_f0 - source_f0`

This is the same pipeline that `run_crossproduct_research.py` Phase 2 runs offline. For production, we either:
- Pre-compute and cache `nodal_f0` per (source_id, sink_id, quarter, PY) before the auction
- Or compute it inline (requires Ray, adds ~30s per quarter)

**Recommendation:** Pre-compute and save as a lookup table. The `calibration_artifact.json` already stores band parameters; add a `nodal_f0_lookup.parquet` alongside it.
