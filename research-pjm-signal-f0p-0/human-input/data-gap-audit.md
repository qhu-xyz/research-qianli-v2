# PJM V7.0 Data Gap Audit

Date: 2026-03-10

Comprehensive audit of every data source the V7.0 agent needs, with verified coverage and remaining gaps.

---

## 1. V6.2B Signal Data — COMPLETE

**Path**: `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/`

| ptype | months | range | all 3 class types? |
|-------|-------:|-------|:------------------:|
| f0 | 105 | 2017-06 to 2026-03 | yes |
| f1 | 97 | 2017-06 to 2026-03 | yes |
| f2 | 89 | 2017-06 to 2026-03 | yes |
| f3 | 80 | 2017-06 to 2026-01 | yes |
| f4 | 72 | 2017-06 to 2026-01 | yes |
| f5 | 63 | 2017-06 to 2025-12 | yes |
| f6 | 54 | 2017-06 to 2025-11 | yes |
| f7 | 45 | 2017-06 to 2025-10 | yes |
| f8 | 36 | 2017-06 to 2025-09 | yes |
| f9 | 27 | 2017-06 to 2025-08 | yes |
| f10 | 18 | 2017-06 to 2025-07 | yes |
| f11 | 9 | 2017-06 to 2025-06 | yes |

Class types: `onpeak`, `dailyoffpeak`, `wkndonpeak` — all present for every (month, ptype) combo. Zero missing.

Schema (20 columns): `constraint_id`, `flow_direction`, `shadow_price_da`, `da_rank_value`, `density_ori_rank_value`, `density_mix_rank_value`, `ori_mean`, `mix_mean`, `mean_branch_max`, `mean_branch_max_fillna`, `branch_name`, `bus_key`, `bus_key_group`, `shadow_sign`, `shadow_price`, `equipment`, `rank_ori`, `density_mix_rank`, `rank`, `tier`.

**No gap.**

---

## 2. Spice6 ml_pred — COMPLETE (newly filled)

**Path**: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred/auction_month={A}/market_month={M}/class_type={C}/final_results.parquet`

| class_type | files | auction months | range |
|------------|------:|:--------------:|-------|
| onpeak | 614 | 92 | 2018-06 to 2026-01 |
| dailyoffpeak | 614 | 92 | 2018-06 to 2026-01 |
| wkndonpeak | 614 | 92 | 2018-06 to 2026-01 |

`wkndonpeak` was generated 2026-03-10 using `research-spice-shadow-price-pred/scripts/generate_wkndonpeak_mlpred.py`.

Schema identical across all three class types. Safe feature columns: `predicted_shadow_price`, `binding_probability`, `binding_probability_scaled`, `prob_exceed_{80..110}`, `density_skewness`, `hist_da`. Do NOT use: `actual_shadow_price`, `actual_binding`, `error`, `abs_error`.

**ml_pred ↔ V6.2B alignment** (2025-06 f0): 100% of V6.2B constraint_ids found in ml_pred for all three class types. ml_pred has more rows (covers all branches across periods); V6.2B is a filtered subset.

**Temporal gap**: ml_pred starts at 2018-06, V6.2B starts at 2017-06. The 12-month gap (2017-06 to 2018-05) is by design — the pipeline needs training lookback. For months without ml_pred, fill spice6 features with 0.

**End gap**: ml_pred stops at 2026-01 for all class types. V6.2B has 2026-03 (f0/f1/f2). If the V7.0 agent needs 2026-02/03, ALL class types need extension. Density data exists for both months; the generation script can be rerun with `--start 2026-02 --end 2026-03`.

**No gap for V7.0 work as scoped.**

---

## 3. Spice6 Density — COMPLETE

**Path**: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density/`

- 106 auction months (2017-06 to 2026-03)
- Not class-type partitioned — shared across all class types
- Contains: `density.parquet`, `density_multi.parquet`, `limit.parquet`, `limit_multi.parquet`, `score.parquet`

**No gap.**

---

## 4. constraint_info — NO GAP (class-type-invariant)

**Path**: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info/auction_month={A}/market_round=1/period_type={P}/class_type=onpeak/`

- 106 auction months (2017-06 to 2026-03)
- Only stored under `class_type=onpeak` — by design
- Contains physical topology: branch_name, kV ratings, circuit info, device names (41 columns)
- 319 interface entries, ~32K branch_constraint entries per month
- Safe to share across all class types

**No gap.**

---

## 5. SF (Scale Factor) — COMPLETE

**Path**: `/opt/data/xyz-dataset/signal_data/pjm/sf/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/`

- 105 months (2017-06 to 2026-03) — matches V6.2B constraints
- All period types and all 3 class types present
- V7.0 copies SF unchanged (no modification needed)

**No gap.**

---

## 6. Realized DA Shadow Prices (API) — AVAILABLE

**API**: `PjmApTools().tools.get_da_shadow_by_peaktype(st, et_ex, peak_type)`

Import: `from pbase.analysis.tools.all_positions import PjmApTools`

Verified for all three peak types:

| peak_type | 2025-01 rows | unique facilities |
|-----------|:------------:|:-----------------:|
| onpeak | 3,326 | 157 |
| dailyoffpeak | 2,110 | 125 |
| wkndonpeak | 1,163 | 78 |

Goes back to at least 2017-06 (verified: 13,838 rows for 2017-06 onpeak). Requires Ray connection.

**No gap in API availability.**

---

## 7. TARGET JOIN: constraint_id → realized DA — CRITICAL FINDING

**This is the most important item for the V7.0 agent.**

### Naive monitored_facility join (split constraint_id on ":") — LOW COVERAGE

| Month | % of V6.2B MFs with DA match | % of DA value captured |
|-------|:----------------------------:|:----------------------:|
| 2024-06 | ~20% | 46.4% |
| 2025-01 | ~14% | 46.3% |

This approach misses >50% of realized binding value. **Do not use this as the sole join strategy.**

### Branch-level join via constraint_info — HIGH COVERAGE

Using `constraint_info` to map V6.2B `constraint_id → branch_name → monitored_facility`, then joining DA data:

| Month | % of DA rows matched | % of DA value captured |
|-------|:--------------------:|:----------------------:|
| 2022-06 | 93.6% | 96.4% |
| 2023-06 | 98.8% | 99.1% |
| 2024-06 | 97.2% | 99.3% |
| 2025-01 | 94.3% | 97.1% |

The branch-level mapping captures 96-99% of DA binding value.

### How the branch-level join works

The existing PjmDataLoader in `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py` implements this:

```python
# 1. Build spice_map from constraint_info (indexed by constraint_id)
spice_map["monitored_facility"] = constraint_id.split(":")[0]
spice_map["match_str"] = monitored_facility.upper()

# 2. Map DA monitored_facility → branch_name via match_str
# 3. For interfaces: special prefix matching (e.g., "BED-BLA" matches "BED-BLA contingency 24")
# 4. Aggregate DA shadow_price by branch_name
```

### Unmatched residual (~3% of value)

The unmatched DA facilities are mostly interface contingencies:
- `BED-BLA contingency 24` ($3,290)
- `AEP-DOM contingency Bed-Bla` ($2,475)
- `APSOUTH contingency` ($1,725)

These use a naming pattern that the interface matching in `PjmDataLoader.map_constraints_to_branches()` handles. The V7.0 agent should use or adapt that method.

### Recommendation for V7.0 agent

1. **Use the branch-level join**, not the naive monitored_facility join
2. Reference `PjmDataLoader.map_constraints_to_branches()` at `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805`
3. The join is: `constraint_id (split on ":") → match_str (uppercase) → branch_name (via constraint_info) → realized_sp (sum of DA shadow_price per branch)`
4. For the V7.0 ranking target, aggregate realized_sp per branch_name, then join back to all V6.2B rows sharing that branch_name
5. Multiple V6.2B constraint_ids can map to the same branch_name — they'd all get the same realized_sp target. This is intentional.

---

## Summary

| Data source | Status | Gap? |
|-------------|--------|------|
| V6.2B signal | Complete, all ptypes/ctypes | No |
| ml_pred (onpeak) | 614 files, 2018-06 to 2026-01 | No |
| ml_pred (dailyoffpeak) | 614 files, 2018-06 to 2026-01 | No |
| ml_pred (wkndonpeak) | 614 files, 2018-06 to 2026-01 | No (newly filled) |
| Density | 106 months, class-type-agnostic | No |
| constraint_info | 106 months, onpeak-only (by design) | No |
| SF | 105 months, all ptypes/ctypes | No |
| Realized DA API | All 3 peak types, back to 2017-06 | No |
| **Target join strategy** | **Naive join is broken (46% capture)** | **YES — must use branch-level mapping** |

### Remaining risks for V7.0 agent

1. **Target join**: The handoff doc's naive `monitored_facility` join captures only ~46% of DA value. The branch-level join via `constraint_info` + `map_constraints_to_branches()` captures 96-99%. The V7.0 agent MUST use the branch-level approach.

2. **ml_pred for 2026-02/03**: Not generated for any class type. If needed, run `scripts/generate_wkndonpeak_mlpred.py` (adaptable for all class types) with `--start 2026-02 --end 2026-03`.

3. **ml_pred temporal start**: Begins at 2018-06, not 2017-06. For V7.0 training months before 2018-06, spice6 features should be filled with 0.
