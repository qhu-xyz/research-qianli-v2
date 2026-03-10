# Design Document: SPICE_F0P_V6.7B.R1 Signal Generation

## 1. Overview

### 1.1 Objective

Generate the `SPICE_F0P_V6.7B.R1` ConstraintsSignal for **all tradable period types and class types** for both MISO and PJM, using ML predictions from the shadow price prediction pipeline.

### 1.2 Ticket Summary

> Base signal: `TEST.TEST.Signal.{RTO}.SPICE_F0P_V6.2B.R1`. Use the ML pipeline in `research-spice-shadow-price-pred` to generate the final 6.7B signal. Get signals for all period types and class types — monthly, quarterly, and annual.

### 1.3 Current State (on disk)

| RTO | Signal | Auction Months | Period Types | Class Types |
|-----|--------|---------------|--------------|-------------|
| MISO | 6.7B | 2024-06 to 2026-01 (20 mo) | f0 only | onpeak only |
| MISO | 6.2B | 2017-06 to 2025-06 | f0, f1, q4 (varies) | onpeak only |
| PJM | 6.7B | 2025-06 to 2026-01 (8 mo) | f0–f11 (varies) | onpeak, dailyoffpeak, wkndonpeak |
| PJM | 6.2B | (various) | (various) | onpeak |

### 1.4 Target Scope

#### MISO

| Dimension | Current | Target |
|-----------|---------|--------|
| Period types | f0 | f0, f1, f2, f3, q2, q3, q4 (per auction schedule) |
| Class types | onpeak | onpeak, offpeak |
| Auction months | 2024-06 – 2026-01 | Same range, extensible |

**Annual periods (aq1–aq4)**: Auctioned only in June. Density data for June only covers `market_month=YYYY-06` (1 month), so the ML pipeline **cannot produce predictions for aq1–aq4** (which span 3 months each: aq1=Jun-Aug, aq2=Sep-Nov, aq3=Dec-Feb, aq4=Mar-May). **Annual periods are out of scope** unless density data is generated for all market months.

#### PJM

| Dimension | Current | Target |
|-----------|---------|--------|
| Period types | f0–f11 (varies) | Same (PJM is monthly-only, no quarterly) |
| Class types | onpeak, dailyoffpeak, wkndonpeak | Same |
| Auction months | 2025-06 – 2026-01 | Extensible |

**PJM note**: The PJM constraint path template hardcodes `class_type=onpeak`. PJM's existing 6.7B already has dailyoffpeak and wkndonpeak, suggesting there's a separate data path or override for non-onpeak class types.

---

## 2. Architecture

### 2.1 Data Flow

```
Density Data (disk)                    Constraint Info (disk)
/opt/temp/tmp/pw_data/spice6/          /opt/temp/tmp/pw_data/spice6/
  prod_f0p_model_{rto}/density/          prod_f0p_model_{rto}/constraint_info/
        │                                        │
        v                                        v
  ┌───────────────────────────────────────────────────┐
  │              ShadowPricePipeline.run()             │
  │  (per auction_month × market_month × class_type)  │
  └──────────────────────┬────────────────────────────┘
                         │ final_results (per market_month)
                         v
  ┌──────────────────────────────────────┐
  │  aggregate_multi_month_results()      │  ← only for quarterly (3 months → 1)
  └──────────────────────┬───────────────┘
                         v
  ┌──────────────────────────────────────┐
  │  convert_predictions_to_signal()      │  ← ranks, tiers, derived columns
  └──────────────────────┬───────────────┘
                         v
  ┌──────────────────────────────────────┐
  │  save_signal()                        │  ← ConstraintsSignal.save_data()
  └──────────────────────┬───────────────┘
                         v
  /opt/data/xyz-dataset/signal_data/{rto}/constraints/
    TEST.TEST.Signal.{RTO}.SPICE_F0P_V6.7B.R1/
      {auction_month}/{period_type}/{class_type}/*.parquet
```

### 2.2 Input Data Availability

**MISO density data** (`prod_f0p_model_miso/density/`):
- June auction: only `market_month=YYYY-06` (f0 only)
- July auction: `market_month=YYYY-07` through `YYYY+1-05` (11 months — covers f0, f1, q2, q3, q4)
- Other months: varies, generally covers all tradable period market months

**MISO constraint data** (`prod_f0p_model_miso/constraint_info/`):
- Class types available: `onpeak`, `offpeak`
- Period types: match density data coverage

**PJM density data** (`prod_f0p_model_pjm/density/`):
- Available from 2017-06 onward
- Class types: `onpeak` only in constraint template (hardcoded)

---

## 3. Signal Format Specification

### 3.1 Storage Path

```
/opt/data/xyz-dataset/signal_data/{rto}/constraints/
  TEST.TEST.Signal.{RTO}.SPICE_F0P_V6.7B.R1/
    {YYYY-MM}/           # auction_month
      {period_type}/     # f0, f1, q2, etc.
        {class_type}/    # onpeak, offpeak, dailyoffpeak, wkndonpeak
          *.parquet
```

### 3.2 DataFrame Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| *(index)* | `str` | `{constraint_id}\|{-flow_direction}\|spice` |
| `constraint_id` | `object` (str) | Constraint identifier |
| `branch_name` | `object` (str) | Monitored transmission branch |
| `flow_direction` | `int64` | Flow direction (+1 or -1) |
| `hist_da` | `float64` | Historical DA shadow price (log-transformed composite) |
| `predicted_shadow_price` | `float64` | ML-predicted shadow price (always positive) |
| `binding_probability_scaled` | `float64` | Scaled binding probability from classifier |
| `prob_rank` | `float64` | Percentile rank of binding_probability_scaled (desc) |
| `hist_shadow_rank` | `float64` | Percentile rank of hist_da (desc) |
| `pred_shadow_rank` | `float64` | Percentile rank of predicted_shadow_price (desc) |
| `rank` | `float64` | Composite rank in [0, 1] |
| `tier` | `int64` | Tier 0-4 (0 = best, 4 = worst) |
| `shadow_sign` | `int64` | `-flow_direction` |
| `shadow_price` | `float64` | `predicted_shadow_price * shadow_sign` (can be negative) |
| `equipment` | `object` (str) | Same as `branch_name` |

### 3.3 Ranking Formula

```
composite = 0.4 * prob_rank + 0.3 * hist_shadow_rank + 0.3 * pred_shadow_rank
rank      = percentile_rank(composite, method='first')
tier      = pd.qcut(rank, q=5, labels=False)
```

Each sub-rank: `series.rank(ascending=False, pct=True, method='average')`.

Weights reverse-engineered from existing 6.7B f0/onpeak data (max deviation = 0.05).

### 3.4 Key Column Relationships

| Relationship | Formula |
|---|---|
| `shadow_sign` | `= -flow_direction` |
| `shadow_price` | `= predicted_shadow_price * shadow_sign` |
| `equipment` | `= branch_name` |
| Index flow_dir | `= -flow_direction` (negated in index string) |

---

## 4. Period Type Handling

### 4.1 MISO Auction Schedule

From `MisoApTools().tools.get_tradable_period_types(auction_month)`:

| Month | Tradable | Auctioned (includes annual) |
|---|---|---|
| Jun | f0 | **aq1, aq2, aq3, aq4**, f0 |
| Jul | f0, f1, q2, q3, q4 | same |
| Aug | f0, f1, f2, f3 | same |
| Sep | f0, f1, f2 | same |
| Oct | f0, f1, q3, q4 | same |
| Nov | f0, f1, f2, f3 | same |
| Dec | f0, f1, f2 | same |
| Jan | f0, f1, q4 | same |
| Feb | f0, f1, f2, f3 | same |
| Mar | f0, f1, f2 | same |
| Apr | f0, f1 | same |
| May | f0 | same |

**Note**: `get_tradable_period_types()` returns monthly/quarterly only. `get_auctioned_period_types()` includes annual quarters (aq1–aq4) for June.

### 4.2 PJM Auction Schedule

PJM is monthly-only (f0 through f11, depending on month). No quarterly or annual period types.

| Month | Tradable Period Types |
|---|---|
| Jun | f0–f11 (12 periods) |
| Jul | f0–f10 (11 periods) |
| ... | (decreasing by 1 each month) |
| May | f0 (1 period) |

### 4.3 Monthly vs. Quarterly vs. Annual

| Type | Period Types | Market Months | Pipeline Runs | Data Available? |
|---|---|---|---|---|
| Monthly | f0–f3 (MISO), f0–f11 (PJM) | 1 | 1 | Yes |
| Quarterly | q2, q3, q4 (MISO only) | 3 | 3 (aggregated) | Yes |
| Annual | aq1–aq4 (MISO June only) | 3 each | 3 (aggregated) | **No** — density data missing |

### 4.4 Quarterly/Annual Aggregation

For multi-month periods, per-month `final_results` are aggregated:

```python
agg_dict = {
    "branch_name": "first",
    "predicted_shadow_price": "sum",    # Total expected shadow cost
    "binding_probability_scaled": "max", # Bind in any month → flagged
    "hist_da": "max",                    # Conservative historical importance
}
grouped = combined.groupby(["constraint_id", "flow_direction"]).agg(agg_dict)
```

---

## 5. RTO-Specific Differences

| Aspect | MISO | PJM |
|--------|------|-----|
| Signal name | `TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1` | `TEST.TEST.Signal.PJM.SPICE_F0P_V6.7B.R1` |
| Class types | onpeak, offpeak | onpeak, dailyoffpeak, wkndonpeak |
| Period types | Monthly (f0–f3) + Quarterly (q2–q4) | Monthly only (f0–f11) |
| Annual periods | aq1–aq4 (June, but no density data) | N/A |
| Constraint template class_type | Parameterized (`{class_type}`) | Hardcoded (`onpeak`) |
| Density path | `prod_f0p_model_miso` | `prod_f0p_model_pjm` |
| ISO config | `MISO_ISO_CONFIG` | `PJM_ISO_CONFIG` |
| AP Tools | `MisoApTools` | `PjmApTools` |

---

## 6. Implementation Details

### 6.1 Module: `signal_generator.py`

| Function | Purpose |
|---|---|
| `aggregate_multi_month_results(results_list)` | Combine per-month results for quarterly periods |
| `convert_predictions_to_signal(final_results, rank_weights, n_tiers)` | Transform ML output → signal format |
| `save_signal(signal_df, rto, signal_name, ...)` | Persist via `ConstraintsSignal` |
| `generate_and_save_signal(...)` | Convenience: convert + save |

These are RTO-agnostic — same code works for MISO and PJM.

### 6.2 Notebook: `generate_signal_67b.ipynb`

Currently configured for **MISO only**. Needs a PJM variant or parameterization.

Sections:
1. Ray init
2. Configuration (signal name, auction months, class types, model params)
3. Helper functions (`get_market_months_for_period`, `run_pipeline_for_period`)
4. Main loop: `auction_months × class_types × period_types`
5. Summary: coverage matrix
6. Validation: load-back test

### 6.3 Caching

Intermediate `final_results.parquet` saved to `OUTPUT_DIR`:
```
{OUTPUT_DIR}/auction_month={YYYY-MM}/market_month={YYYY-MM}/class_type={type}/final_results.parquet
```

---

## 7. ML Pipeline Summary

Two-stage ensemble:

```
Stage 1: Classification → binding detection
  XGBClassifier (50%) + LogisticRegression (50%)

Stage 2: Regression → shadow price magnitude
  XGBRegressor (50%) + ElasticNet (50%)
```

- Per-branch models (min 10 samples per branch)
- 12-month training lookback
- Anomaly detection enabled
- Threshold: F-beta (beta=2, favoring recall)

**Features**: Density shape (110, 105, 100, 95, 90), probability diffs, risk metrics, probability exceedances, interaction features.

---

## 8. Downstream Consumer

```python
# CIA Multistep trading workflow (pbase/analysis/utils/cia_multistep.py)
signal = ConstraintsSignal(rto, signal_name, period_type, class_type).load_data(auction_month)
# signal['tier'] → constraint prioritization
# signal['shadow_price'] → trade sizing

# Signal backtesting (pbase/analysis/tools/base.py)
evaluate_signal_v2(rto, signal_name, period_type, class_type, auction_month)
```

---

## 9. Gaps and Open Questions

| Issue | Status | Impact |
|---|---|---|
| **MISO annual periods (aq1–aq4)**: June density data only has 1 market month, need 3 per aq period | Blocked — density data not available | Cannot generate annual signals without upstream data |
| **PJM class types**: Constraint template hardcodes `class_type=onpeak`, but existing 6.7B has dailyoffpeak/wkndonpeak | Needs investigation — how was existing PJM 6.7B generated for non-onpeak? | May need different constraint path for non-onpeak |
| **PJM notebook**: Current notebook is MISO-only | Needs PJM parameterization | PJM generation not yet runnable |
| **Quarterly aggregation**: Sum vs. average for `predicted_shadow_price` | Used sum (total expected shadow cost over quarter) | May want to verify against trading desk preference |

---

## 10. Files Created/Modified

| File | Action | Description |
|---|---|---|
| `src/shadow_price_prediction/signal_generator.py` | **New** | Signal conversion and saving (RTO-agnostic) |
| `notebook/generate_signal_67b.ipynb` | **New** | MISO signal generation orchestration |
| `src/shadow_price_prediction/__init__.py` | **Modified** | Added signal_generator exports |
| `document/design_signal_67b.md` | **New** | This design document |
| `document/learning_signal_67b.md` | **New** | Key learnings and findings |
