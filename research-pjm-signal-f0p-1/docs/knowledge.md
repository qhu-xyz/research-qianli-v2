# PJM Data Pipeline Knowledge Base

Verified facts from f0p-0 exploration and raw data inspection.

## 1. Raw Data Shapes (verified 2023-06)

### score.parquet (per outage_date)
- **Columns**: `score` (f64), `constraint_id` (str), `flow_direction` (i64)
- **Rows**: ~22,552 per outage_date
- **Unique constraint_ids**: ~11,603
- **Unique (cid, flow_dir)**: ~23,206
- 11 outage_dates per month (PJM)
- `score` ranges from ~1e-50 (essentially zero) to ~0.9 (very likely to bind)
- `score` represents probability of exceeding 110% of line rating

### density.parquet (per outage_date)
- **Columns**: 78 columns — 77 probability bins + `constraint_id`
- **Bins**: from -300 to +300 (% of thermal limit), e.g., "-300", "-280", ..., "300"
- **Rows**: ~11,216 per outage_date
- Each row is a probability distribution for one constraint's thermal loading
- No flow_direction column — density is single-directional per constraint_id

### limit.parquet (per outage_date)
- **Columns**: `limit` (f32), `constraint_id` (str)
- **Rows**: ~17,481 per outage_date
- Physical thermal limit in MW

### constraint_info (per period_type, class_type=onpeak only)
- **Columns**: 41 columns including `constraint_id`, `branch_name`, `type`, `contingency`, `limit`, `monitored_facility`, `direction`, etc.
- **Rows**: ~31,741
- **Unique constraint_ids**: ~17,977
- **Unique branch_names**: ~4,781
- **Types**: branch_constraint (31,423), interface (318)
- Many-to-one: multiple constraint_ids map to one branch_name (different contingencies for same physical line)
- Class-type invariant: topology doesn't change with peak type

### ml_pred/final_results.parquet (per class_type)
- **Columns** (24): branch_name, hist_da, prob_exceed_{80,85,90,95,100,105,110}, actual_shadow_price, predicted_shadow_price, binding_probability, binding_probability_scaled, threshold, predicted_binding_count, actual_binding, auction_month, market_month, model_used, predicted_binding, error, abs_error, constraint_id, flow_direction
- **SAFE features**: hist_da, prob_exceed_{80..110}, predicted_shadow_price, binding_probability, binding_probability_scaled
- **LEAKY features** (DO NOT USE): actual_shadow_price, actual_binding, error, abs_error
- **Rows**: ~12,912 per (auction_month, market_month, class_type)
- **Unique constraint_ids**: ~11,543
- **Unique branch_names**: ~3,131
- Has `branch_name` column directly (no need to join through constraint_info)

### V6.2B signal (per month/ptype/ctype)
- **Columns** (21): constraint_id, flow_direction, mean_branch_max, mean_branch_max_fillna, ori_mean, branch_name, bus_key, bus_key_group, mix_mean, shadow_price_da, density_mix_rank_value, density_ori_rank_value, da_rank_value, rank_ori, density_mix_rank, rank, tier, shadow_sign, shadow_price, equipment, __index_level_0__
- **Rows**: ~667 (before dedup) → ~475 unique branches
- `rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value` (verified exact)

## 2. Universe Sizes (verified 2023-06)

| Level | Size |
|-------|------|
| score.parquet unique constraint_ids | 11,603 |
| score.parquet unique (cid, flow_dir) | 23,206 |
| constraint_info unique constraint_ids | 17,977 |
| constraint_info unique branches | 4,781 |
| ml_pred unique constraint_ids | 11,543 |
| ml_pred unique branches | 3,131 |
| V6.2B unique constraint_ids | 667 |
| V6.2B unique branches | 475 |

### Branch Overlap (2023-06)
- ml_pred branches are a **strict subset** of constraint_info branches (3,131 ⊂ 4,781)
- V6.2B branches: 469/475 found in both ml_pred and constraint_info (6 missing — investigate)
- Score constraint_ids closely match ml_pred constraint_ids (11,603 vs 11,543)

## 3. Coverage Gap (verified across 85 months, f0/onpeak)

| Metric | V6.2B | Expanded (ml_pred branches) |
|--------|-------|-----------------------------|
| Avg branches/month | ~450 | ~3,100 |
| Binding constraints captured | 33% (47/142) | 89% |
| Binding value captured | 47% | 89% |
| Top-20 binders captured | ~9/20 | ~18/20 |

Missed binders have similar mean value to captured ones — not low-value noise.

## 4. Feature Quality (verified f0p-0)

Spearman correlation with realized_sp (f0/onpeak, within V6.2B universe):
| Feature | Spearman |
|---------|----------|
| da_rank_value | ~0.27 (NEGATIVE rank, lower = more binding) |
| shadow_price_da | ~0.27 |
| binding_freq_6 | ~0.15-0.25 |
| ori_mean | ~0.09 |
| density_score (raw) | ~0.01-0.09 |
| binding_probability | ~0.01 |
| predicted_shadow_price | ~0.01 |
| prob_exceed_100 | ~0.01 |
| constraint_limit | ~0.01 |

**Key insight**: `da_rank_value` (= hist_da from ml_pred) dominates. Density and ML predictions are near-useless individually. But they may help when combined via LTR, especially for new-binding constraints where hist_da = 0.

## 5. Data Availability Timeline

| Source | Start | End | Months |
|--------|-------|-----|--------|
| Density (score, limit) | 2017-06 | 2026-03 | 106 |
| ml_pred (all class_types) | 2018-06 | 2026-01 | 92 |
| Realized DA cache | 2019-01 | 2026-02 | 86 |
| V6.2B signal | 2017-06 | 2026-03 | 106 |

### Eval Windows
- **Dev**: 2020-06 to 2023-05 (36 months for f0)
- **Holdout**: 2024-01 to 2025-12 (24 months)
- **Training lookback**: 8 months before eval month

## 6. PJM Auction Schedule

```
Month  1: f0, f1, f2, f3, f4
Month  2: f0, f1, f2, f3
Month  3: f0, f1, f2
Month  4: f0, f1
Month  5: f0
Month  6: f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11
Month  7: f0..f10
Month  8: f0..f9
Month  9: f0..f8
Month 10: f0..f7
Month 11: f0..f6
Month 12: f0..f5
```

ML slices: f0 × 3 class_types + f1 × 3 class_types = 6 slices.
f2-f11: passthrough (no ML needed).

## 7. Branch Mapping Details

### How constraint_id → branch_name Works
1. `constraint_info` has direct mapping: `constraint_id` → `branch_name`
2. For DA shadow prices: `DA.monitored_facility.upper()` → match against `constraint_id.split(":")[0].upper()`
3. Interface fallback: prefix-match for interface contingencies (type="interface")
4. Aggregate realized_sp by branch_name (sum of |shadow_price|)

### Mapping Quality
- Direct match captures ~85% of DA binders
- Interface fallback adds ~10-14%
- Total: ~96-99% of DA value mapped
- Remaining ~1-4% are genuinely unmappable (obscure naming conventions)

### Key Gotcha
`constraint_info` is stored under `class_type=onpeak` only — it's physical topology, not class-dependent. Always load with `class_type=onpeak` regardless of target class_type.

## 8. Open Questions for Investigation

### Q1: What is density_mix vs density_ori?
V6.2B has both `ori_mean` and `mix_mean`. We know `ori_mean` ≈ raw density score (Spearman 0.85).
`mix_mean` may blend density with another signal. The V6.2B formula gives it 30% weight.
Need to investigate if there's a `density_multi.parquet` connection or if mix involves cross-element scoring.

### Q2: Flow direction handling
- score.parquet: both directions (flow_dir = 1 and -1)
- density.parquet: no flow_direction column (single per constraint_id)
- ml_pred: both directions
- V6.2B: both directions (keeps both as separate rows)
- Realized DA: direction-agnostic (absolute shadow price by branch)

**Current approach** (f0p-0): V6.2B keeps both directions → dedup by branch_name (keep lowest rank_ori).
**For expanded universe**: unclear whether to keep max-scoring direction or both. Both approaches need testing.

### Q3: Score aggregation across outage_dates
V6.2B may use a specific weighting scheme (not simple mean) across the 11 outage_dates.
Need to test: mean, max, weighted (closer outage_dates weighted more?), or outage-probability-weighted.

### Q4: Constraint universe stability
Do the same ~3,100 branches appear every month, or does the set change significantly?
Important for understanding whether BF features are meaningful across the expanded universe.
