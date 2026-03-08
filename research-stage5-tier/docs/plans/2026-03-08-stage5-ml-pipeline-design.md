# Stage 5: ML-Based Signal Generation Pipeline Design

**Date:** 2026-03-08
**Status:** Approved
**Predecessor:** Stage 4 (abandoned -- circular evaluation)

## Problem

Stage 4 used `shadow_price_da` as both a feature source (via `da_rank_value`) and ground truth. This is circular: `Spearman(da_rank_value, shadow_price_da) = -1.0`. Stage 5 fixes this by using **realized DA shadow prices** as ground truth, fetched independently from MISO market data.

## Approach

Copy reusable modules from stage 4, fix the circularity (ground truth swap), and run versioned experiments (v0-v3) with increasing feature sets.

## Data Verification (2026-03-08)

All data sources verified on disk:

| Source | constraint_id type | Join key | Verified |
|--------|-------------------|----------|----------|
| V6.2B parquet | **String** | constraint_id | Yes |
| spice6 score_df | **String** | (constraint_id, flow_direction) | Yes |
| spice6 limit | **String** | constraint_id | Yes |
| ml_pred final_results | **String** | (constraint_id, flow_direction) | Yes |
| constraint_info | **String** | constraint_id | Yes |
| Realized DA (pandas) | **object (str)** | constraint_id | Yes |

**Correction from experiment-setup.md:** mem.md CHECK 9 claimed constraint_id is i64. It is String everywhere, including non-numeric values like `'SO_MW_Transfer'`.

spice6 score_df columns verified: `110, 105, 100, 95, 90, 85, 80, 70, 60, constraint_id, flow_direction`. Stage 4's spice6_loader.py correctly reads columns `110, 100, 90, 85, 80` and renames them to `prob_exceed_*`.

## Project Structure

```
research-stage5-tier/
  ml/
    __init__.py
    config.py          # Modified (fix comment, add Groups C/D, defaults 8/0)
    data_loader.py     # Rewritten (realized DA ground truth, no engineered feats)
    evaluate.py        # Copy from stage4
    compare.py         # Copy from stage4
    train.py           # Copy from stage4
    features.py        # Copy from stage4
    spice6_loader.py   # Copy from stage4
    mlpred_loader.py   # New: loads ml_pred/final_results.parquet
    realized_da.py     # New: fetches + caches realized DA
    v62b_formula.py    # Copy from stage4
    pipeline.py        # Modified (3-line ground truth swap)
    benchmark.py       # Copy from stage4
  scripts/
    cache_realized_da.py       # New: one-time Ray fetch
    run_v0_formula_baseline.py # Rewritten: eval against realized DA
  data/
    realized_da/               # Cached: {YYYY-MM}.parquet
  registry/                    # Rebuilt from correct v0
```

## Module Changes

### Copied as-is (after line-by-line audit)
- `evaluate.py` -- VC@k, Recall@k, NDCG, Spearman, Tier0-AP
- `compare.py` -- 3-layer gate system
- `train.py` -- LightGBM lambdarank + XGBoost fallback
- `features.py` -- prepare_features, compute_query_groups
- `spice6_loader.py` -- density aggregation
- `v62b_formula.py` -- formula reproduction
- `benchmark.py` -- multi-month runner

### config.py modifications
1. Fix comment: "da_rank_value is a historical 60-month lookback, legitimate as feature"
2. Add `_HIST_DA_FEATURES = ["da_rank_value"]` with monotone `[-1]` (Group C)
3. Add `_MLPRED_FEATURES = ["predicted_shadow_price", "binding_probability", "binding_probability_scaled"]` with monotone `[1, 1, 1]` (Group D)
4. Named feature sets: `FEATURES_V1` (A+B, 11), `FEATURES_V1B` (A+B+C, 12)
5. Change PipelineConfig defaults: `train_months=8, val_months=0`
6. Remove `_ENGINEERED_FEATURES`, `_FULL_FEATURES`

### data_loader.py rewrite
- Remove `_add_engineered_features()` (37 useless derived features)
- Add `load_realized_da(month)` reading from cached parquet
- Join realized DA to V6.2B by `constraint_id` (String), fill missing with 0
- Add optional ml_pred enrichment via `mlpred_loader.py`
- Each training month gets its OWN realized DA labels

### pipeline.py modifications (3 lines)
```
Line 52: y_train = train_df["realized_sp"]      # was shadow_price_da
Line 59: y_val = val_df["realized_sp"]           # was shadow_price_da
Line 84: actual_sp = test_df["realized_sp"]      # was shadow_price_da
```

### New: realized_da.py
- `fetch_realized_da(month) -> pd.DataFrame` -- Ray fetch via get_da_shadow_by_peaktype
- `cache_realized_da(months, output_dir)` -- batch fetch + save to parquet
- `load_realized_da(month, cache_dir) -> pl.DataFrame` -- read cached parquet
- Aggregation: `abs(sum(shadow_price))` per constraint_id

### New: mlpred_loader.py
- Loads `ml_pred/auction_month={m}/market_month={m}/class_type=onpeak/final_results.parquet`
- Extracts: predicted_shadow_price, binding_probability, binding_probability_scaled
- Joins to V6.2B by `(constraint_id, flow_direction)`
- Coverage: 92 months (2018-06 to 2026-01)

## Experiment Plan

| Version | Features | Count | Purpose |
|---------|----------|-------|---------|
| v0 | V6.2B formula | 3 (formula inputs) | Baseline against realized DA |
| v1 | Groups A+B | 11 | Pure forecasts, no historical DA |
| v1b | Groups A+B+C | 12 | Add da_rank_value |
| v2 | Groups A+B+C+E | 16 | Add custom historical DA (deferred) |
| v3 | Groups A+B+C+D+E | 19 | Add ML predictions (deferred) |

Execution: screen (4 months) first, then eval (12 months) if promising.

## v0 Baseline (target numbers, from experiment-setup.md Section 8)

| Metric | Mean |
|--------|------|
| VC@20 | 0.2817 |
| VC@100 | 0.6008 |
| Recall@20 | 0.1833 |
| Spearman | 0.2045 |

## Gate System

Gates recalibrated from correct v0 (not the circular stage 4 values).
Three layers: L1 Mean (>= 0.9 * v0_mean), L2 Tail (<= 1 failure below v0_min), L3 Bot2 (>= v0_bot2 - 0.02).

## Audit Checklist (applied to every copied module)

- [ ] No use of shadow_price_da as y/labels/actual
- [ ] No use of rank, rank_ori, tier as features
- [ ] constraint_id treated as String (no int casts)
- [ ] Score direction: higher = more binding for evaluation
- [ ] Join keys correct: (constraint_id, flow_direction) where applicable
- [ ] Each training month gets its own realized DA labels
- [ ] Monotone directions: flow +1, rank -1, prob_exceed +1
