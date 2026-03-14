# Design: PJM Band Version v3 — MW Seasonal Adjustment

## Background

PJM's June first-monthly auction covers 12 forward months: f0 (Jun) through f11 (May next year). The banding pipeline predicts each month's MCP from the annual auction result. Prior to v3, the prediction uses `annual_mcp / 12` as `mtm_1st_mean` — a flat, hour-weighted distribution that ignores seasonal patterns.

Research (v1 and v2 in this repo) showed that a market-wide seasonal factor applied to `mtm_1st_mean` yields +7.9% MAE improvement on trades, with 62.0% win rate, statistically significant across every year (PY2020-2025), every month (11/12 improve), and every value tier. The improvement scales with path value: +14.2% on $5K+ paths.

**The v3 band version implements this MW seasonal adjustment natively in `pmodel/base/ftr24/v1/band_generator.py`.**

---

## Research that Led Here

| Phase | Approach | Result | Verdict |
|-------|----------|--------|---------|
| V1 | Path-level historical distributions | +1.8% MAE, 51.1% win | Too noisy, abandoned |
| V2 | Market-wide seasonal factor (expanding median) | **+7.9% MAE, 62.0% win** | **Winner — shipped as v3** |
| V2-alt | Node-level factors | 9x worse | Noise amplification |
| V2-alt | Normalized MW (sum=1.0) | +3.0% only | Jensen's inequality effect |
| V2-alt | Path-level traceback (5+ yr) | +17.9% but 6% coverage | Not viable at scale |

Full results: `v1/REPORT.md`, `v2/REPORT.md`, `SUMMARY.md`

---

## The Factor Table

12 hardcoded constants, computed as `median(fx_path_mcp / annual_path_mcp) × 12` over an expanding window (PY2017-2025). Sum = 0.88 — this is by design, not a bug (Jensen's inequality; normalizing to 1.0 reduces performance).

```
f0  = 0.7846  (Jun)      f6  = 0.9269  (Dec)
f1  = 0.8952  (Jul)      f7  = 0.9709  (Jan)
f2  = 0.8914  (Aug)      f8  = 0.8894  (Feb)
f3  = 0.9424  (Sep)      f9  = 0.8074  (Mar)
f4  = 0.9798  (Oct)      f10 = 0.7853  (Apr)
f5  = 0.8947  (Nov)      f11 = 0.7879  (May)
```

Formula: `adjusted_mtm_1st = mtm_1st_mean × MW_SEASONAL_ADJ[period_type]`

---

## Key Design Decisions

### 1. Adjust only test_df, not train/val

In `generate_f0_f1_bands()` (line 1047-1048):
```python
if band_version == "v3":
    test_df = adjust_mtm_seasonal(test_df, auction_month)
```

The seasonal adjustment is applied **only to test_df** before `prepare_features()` runs. Train and val data are NOT adjusted.

**Rationale**: Training data comes from prior auction months. Each training month's `mtm_1st_mean` already reflects its own auction context. The seasonal distortion is specific to the June first-monthly auction's flat /12 distribution. Adjusting train/val would corrupt their ground truth. The LightGBM model trains on unadjusted baselines and learns residual patterns from them; at inference time, the adjusted test baseline is closer to the actual MCP, so the model's residual corrections are more accurate.

(Note: the research monkey-patch in `v15/seasonal_patch.py` takes a different approach — it wraps `prepare_features()` and adjusts all data. The native v3 is more precise.)

### 2. June-only guard

```python
am_str = get_auction_month_str(auction_month)
if not am_str.endswith("-06"):
    return df
```

The factors were derived from June first-monthly auction data. Non-June months produce different auction dynamics and the factors don't apply. The guard ensures v3 is identical to v2 for all non-June months.

### 3. v3 shares v2's baseline and features

`prepare_features()` treats v2 and v3 identically:
```python
use_v2 = band_version in ("v2", "v3") and ptype == "f0"
```

v3 adds the seasonal adjustment on top of v2's baseline (`0.65 × avg(mtm1,mtm2,mtm3) + 0.35 × avg(rev1,rev2,rev3)`) and v2's extra features. For non-f0 period types, v3 falls back to v1 (same as v2).

### 4. No changes to training data generation

The saved training parquets contain the original, unadjusted `mtm_1st_mean`. The adjustment is applied at band-generation time, not at data save time. This means the same training data works for v1, v2, and v3.

### 5. Auto-select training2 path

```python
def _get_training_base(band_version, training_base):
    if band_version in ("v2", "v3"):
        return training_base or TRAINING_BASE_V2_PJM  # pjm_mcp_pred_training2
    return training_base or TRAINING_BASE              # pjm_mcp_pred_training
```

v3 requires `pjm_mcp_pred_training2` (which has `series_rev_mapped` for v2 features).

---

## Pipeline Flow: v1 vs v2 vs v3

```
base.py → generate_bands(df, band_version="v1|v2|v3")
  │
  ├─ _get_training_base()
  │    v1: pjm_mcp_pred_training
  │    v2/v3: pjm_mcp_pred_training2
  │
  ├─ groupby (auction_month, period_type)
  │
  └─ generate_bands_for_group(auction_month, ptype, ...)
       │
       ├─ [f0/f1] → load train/val from training_base
       │            → generate_f0_f1_bands(train, val, test, ...)
       │                │
       │                ├─ [v3 ONLY, June ONLY] adjust_mtm_seasonal(test_df)
       │                │    mtm_1st_mean *= MW_SEASONAL_ADJ[ptype]
       │                │
       │                ├─ prepare_features(train_df)  ← uses v2 baseline if v2/v3+f0
       │                ├─ prepare_features(val_df)    ← uses v2 baseline if v2/v3+f0
       │                ├─ prepare_features(test_df)   ← uses v2 baseline if v2/v3+f0
       │                │    └─ compute_baseline_v2() or compute_baseline()
       │                │    └─ compute_residual()
       │                │    └─ compute_segment()
       │                │    └─ compute_derived_features()
       │                │    └─ [v2/v3] _compute_v2_extra_features()
       │                │
       │                ├─ train LightGBM per segment (low/high)
       │                ├─ conformal calibration on val_df
       │                └─ apply_corrected_bid_prices(test_df) → HALT
       │                     bid_price_1..10, clearing_prob_1..10, baseline
       │
       └─ [f2+] → rule-based binning (v1/v2/v3 identical)
```

The only difference between v2 and v3 in the entire pipeline is the single `adjust_mtm_seasonal(test_df)` call at the top of `generate_f0_f1_bands()`.

---

## Cascading Effects

| Component | Impact | Risk |
|-----------|--------|------|
| **Baseline** | mtm_1st_mean is scaled 0.78-0.98×. Since mtm_1st is ~22% of v2 baseline weight, baseline shifts by ~0.6-1.6% | LOW |
| **Segment threshold** | `\|mtm_1st_mean\| < SEGMENT_THRESHOLD` — some borderline paths may shift from high→low | LOW (most paths well above/below) |
| **LightGBM features** | `feat_mtm_horizon_vol = \|mtm_1st - mtm_2nd\|` — mtm_1st is scaled, mtm_2nd is not. Changes feature semantics | MEDIUM (model retrains and adapts) |
| **Flow type** | `prevail if mtm_1st_mean > 0 else counter` — scaling preserves sign | NONE |
| **Conformal strata** | Uses `srev_bucket × flow_type` — neither changes | NONE |
| **mcp_pred** | Set in `trade_finalizer.py` before banding — not affected | NONE |

---

## Activation

v3 is activated by setting `mcp_band_version: "v3"` in the PJM params dict. As of now, no PJM params file uses this — v3 is implemented but not yet activated in production.

In `base.py:566`:
```python
total_trades = generate_bands(
    df=total_trades,
    band_version=self.params_dict.get("mcp_band_version", "v1"),
    ...
)
```

---

## Verification

Script: `v2/scripts/10_verify_v1_v2_v3_pipeline.py`

Tests:
1. `save_training_data()` writes valid parquet with required columns
2. `generate_bands()` produces all 21 BAND_COLS for v1, v2, v3
3. Cross-version assertions:
   - v1 != v2 (different baselines)
   - v2 != v3 on June (seasonal adjustment active)
   - v2 == v3 on non-June (seasonal only affects June)

---

## File Locations

| File | Purpose |
|------|---------|
| `pmodel/base/ftr24/v1/band_generator.py` | Native v3 implementation (MW_SEASONAL_ADJ, adjust_mtm_seasonal, band_version="v3" handling) |
| `research-first-monthly-pjm/SUMMARY.md` | Complete research summary |
| `research-first-monthly-pjm/v2/REPORT.md` | V2 MW factor results and statistical tests |
| `research-first-monthly-pjm/v2/scripts/04_refine_market_wide.py` | Factor computation (expanding median, clipping) |
| `research-first-monthly-pjm/v2/scripts/05_mw_trades_deep.py` | Trades deep-dive (per-year, per-month, value tiers) |
| `research-first-monthly-pjm/v2/scripts/10_verify_v1_v2_v3_pipeline.py` | Pipeline verification script |
