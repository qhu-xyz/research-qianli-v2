# Learnings: SPICE_F0P_V6.7B.R1 Signal Generation

Key findings from investigating the SPICE signal pipeline, the ML prediction codebase, and the existing 6.7B/6.2B data on disk.

---

## 1. Signal Format (Reverse-Engineered)

### Index convention has a negation trap

The signal index is `{constraint_id}|{fd}|spice` where `fd = -flow_direction`. The `flow_direction` column and the index embed **opposite** signs. This was not documented anywhere — discovered by comparing actual parquet data column values against their index strings.

```
Index:            396264|-1|spice
flow_direction:   1         ← negation of the index fd
shadow_sign:     -1         ← same as the index fd
```

### Rank formula: 40/30/30 weighted composite

Reverse-engineered by testing weighted combinations against existing 6.7B data:
```
rank ≈ percentile_rank(0.4 * prob_rank + 0.3 * hist_shadow_rank + 0.3 * pred_shadow_rank)
```
Maximum deviation from reconstructed values: 0.05. No documentation existed for this formula.

### `equipment` is just `branch_name`

Verified across all 566 rows in a sample month. `equipment == branch_name` for 100% of rows. The column exists because downstream consumers (CIA Multistep) expect it.

### hist_da is log-transformed

`hist_da = log1p(recent_hist_da + sum(season_hist_da_1..3))` — a composite of recent (last 3 months) and seasonal (last 3 years) historical DA shadow prices, with discounting. See `data_loader.py:494`.

---

## 2. Data Availability Constraints

### MISO June = f0 only

June is the annual auction month where aq1–aq4 are auctioned. However, the density model data (`prod_f0p_model_miso/density/`) for June only contains `market_month=YYYY-06`. The annual quarter periods need market months spanning 3 months each (e.g., aq1 = Jun–Aug), so **the ML pipeline cannot generate aq1–aq4 predictions with current data**.

```
2024-06 density data: market_month=2024-06 (only 1 month)
2024-07 density data: market_month=2024-07 through 2025-05 (11 months)
```

This means annual signals are blocked on upstream density data generation.

### MISO constraint data has both onpeak and offpeak

```
constraint_info/auction_month=2025-01/market_round=1/period_type=f0/
  class_type=offpeak/
  class_type=onpeak/
```

So offpeak generation is feasible for MISO.

### PJM constraint template hardcodes onpeak

In `iso_configs.py`:
```python
constraint_path_template = (
    ".../period_type={period_type}/class_type=onpeak"  # ← hardcoded
)
```

Yet the existing PJM 6.7B signal on disk has `dailyoffpeak` and `wkndonpeak`. This means either:
1. The existing PJM 6.7B was generated with a different/overridden path, or
2. There's a separate process for non-onpeak PJM signals

This needs investigation before PJM multi-class generation.

---

## 3. Pipeline Architecture

### One final_results per market_month

`ShadowPricePipeline` produces one `final_results.parquet` per `(auction_month, market_month, class_type)`. For a quarterly period like q2 (Sep–Nov), you get 3 separate result files. The signal generator must aggregate these before conversion.

### Pipeline saves per market_month, not per period_type

The output dir structure is:
```
{output_dir}/auction_month={AM}/market_month={MM}/class_type={CT}/final_results.parquet
```

There is no `period_type` in this path. The same market_month result can be used for different period types (e.g., market_month=2025-09 serves both f2 from auction 2025-07 and q2 from auction 2025-07).

### `get_tradable_period_types` vs `get_auctioned_period_types`

Two different functions on the AP Tools:
- `get_tradable_period_types()` — returns monthly + quarterly only (what the notebook uses)
- `get_auctioned_period_types()` — includes annual quarters (aq1–aq4) for June

For signal generation, `get_tradable_period_types()` is the right one to use, since it represents what can actually be traded in the monthly auction.

---

## 4. pbase Signal Infrastructure

### ConstraintsSignal is a thin wrapper

```python
ConstraintsSignal(rto, signal_name, period_type, class_type)
  └── BaseSignal(signal_type=SignalType.CONSTRAINTS, ...)
        └── save_data(data, auction_month) → update_parquet(df=data, ...)
        └── load_data(auction_month) → load_parquet([full_path])
```

Path template: `{rto}/{signal_type}/{signal_name}/{year_month}/{period_type}/{class_type}`

The save/load is straightforward parquet I/O with hive-style partitioning by path components.

### Signal evaluation expects specific columns

`evaluate_signal_v2()` in `pbase/analysis/tools/base.py:18463` loads the signal and accesses:
- `tier` column for stratification
- Signal index format `{id}|{fd}|spice` for joining with constraint data
- `shadow_price` for value assessment

---

## 5. ML Pipeline Details

### Two-stage ensemble architecture

```
Test data → Classification (binding?) → Regression (how much?) → final_results
```

Stage 1 (classifiers): XGBClassifier + LogisticRegression, weighted 50/50
Stage 2 (regressors): XGBRegressor + ElasticNet, weighted 50/50

Per-branch models trained when a branch has >= 10 training samples. Otherwise falls back to the default (pooled) model.

### Aggregation within predict()

Within a single market month, `predict()` aggregates across outage dates:
- `predicted_shadow_price: sum` across outage dates
- `binding_probability_scaled: aggregate_probabilities` (custom function)
- `binding_probability: max`
- `hist_da: mean` (from features)

This is **different** from the quarterly aggregation we do across market months.

### Feature set is density-based

All features come from the density model output — probability distributions of power flow relative to line ratings. No direct price or load features.

Key features: `110, 105, 100, 95, 90` (raw density at % of rating), `*_diff` (probability mass between intervals), `prob_overload`, `risk_ratio`, `curvature_100`, `prob_exceed_*`, interaction terms.

---

## 6. Existing 6.7B Data Statistics

From reading the actual parquet for `2025-01/f0/onpeak`:
- **566 constraints** with predictions
- **14 columns** (matching schema above)
- **5 tiers**: {0: 110, 1: 117, 2: 112, 3: 112, 4: 115} — roughly equal
- **shadow_price range**: large spread (both positive and negative)
- **constraint_id**: string type (not int)
- **flow_direction**: values are +1 or -1

---

## 7. Key Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Quarterly aggregation for `predicted_shadow_price` | Sum | Represents total expected shadow cost over the quarter |
| Quarterly aggregation for `binding_probability_scaled` | Max | Conservative: if likely to bind in any month, flag it |
| Rank weights | 0.4/0.3/0.3 | Matched existing 6.7B data |
| Filter to `predicted_shadow_price > 0` | Yes | Only include constraints predicted to bind |
| Annual periods | Out of scope | Density data not available for aq1–aq4 market months |
| PJM non-onpeak class types | Needs investigation | Constraint template hardcodes onpeak |

---

## 8. Gotchas and Pitfalls

1. **Don't confuse `get_tradable_period_types` with `get_auctioned_period_types`** — the former excludes annual quarters, which is what we want for signal generation.

2. **Index flow direction is negated** — easy to get wrong. The index string uses `-flow_direction` while the column stores the original.

3. **PredictionConfig.period_type is used for training data loading but the pipeline handles multiple market months** — setting it to 'f0' still works for quarterly predictions because the pipeline derives period types dynamically from test_periods.

4. **June auction months have minimal density data** — only 1 market month. Don't expect to generate anything beyond f0 for June.

5. **The `hist_da` column in final_results is already log-transformed** — don't apply log again during signal generation.

6. **ConstraintsSignal.save_data() overwrites** — calling it twice for the same (auction_month, period_type, class_type) replaces the previous data. This is actually desired for reruns.

7. **The constraint_path_template has `period_type` and `class_type` parameters** — the constraint info is per (period_type, class_type), not shared across them. This means the ML pipeline loads different constraint sets for different period types.
