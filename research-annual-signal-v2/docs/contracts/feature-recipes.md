# Feature Recipes — MISO Annual

**Date**: 2026-03-24

A feature recipe defines the exact columns, source modules, transforms, fill rules, and round behavior for a model. No model may be promoted without a bound feature recipe ID.

---

## `miso_annual_v0c_features_v1`

**Model**: `miso_annual_v0c_formula_v1`
**Universe scope**: `miso_annual_branch_active_v1`
**Valid ctypes**: onpeak, offpeak (separate eval, same recipe)

### Columns

| Column | Source | Transform | Fill rule |
|--------|--------|-----------|-----------|
| `da_rank_value` | `ml/phase6/features.py:_compute_class_shadow_price_da` | Dense rank (descending) of class-specific cumulative SP among positive branches | Sentinel = max_rank + 1 for zero-history branches |
| `rt_max` | Derived | `max(bin_80_cid_max, bin_90_cid_max, bin_100_cid_max, bin_110_cid_max)` | Always available (from density) |
| `bf` | `ml/history_features.py:compute_history_features` | Class-specific BF_12: fraction of last 12 calendar months with binding. `bf_12` for onpeak, `bfo_12` for offpeak | 0.0 for branches with no history |

### Formula

```
score = 0.40 * (1 - minmax(da_rank_value)) + 0.30 * minmax(rt_max) + 0.30 * minmax(bf)
```

`minmax` = linear rescale to [0, 1] within the eval slice. If all values are equal, returns 0.5.

### Round behavior

- **v1**: round-insensitive. All published V7.0/V7.0B artifacts use `market_round=1`.
- **v2** (pending): `da_rank_value` and `bf` will incorporate partial-month data from the daily DA cache up to the round-specific cutoff date. `rt_max` will use the round-specific density partition. Same formula, different feature values.

### Source code

- Scoring: `ml/phase6/scoring.py:score_v0c()`
- Feature table: `ml/phase6/features.py:build_class_model_table()`
- History: `ml/history_features.py:compute_history_features()`

---

## `miso_annual_bucket_features_v1`

**Model**: `miso_annual_bucket_6_20_v1`
**Universe scope**: `miso_annual_branch_active_v1`
**Valid ctypes**: onpeak, offpeak (separate training per ctype)

### Columns

| Column | Source | Transform | Fill rule |
|--------|--------|-----------|-----------|
| `da_rank_value` | `ml/phase6/features.py:_compute_class_shadow_price_da` | Dense rank (descending) of class-specific cumulative SP | Sentinel for zero-history |
| `shadow_price_da` | Same source | Raw class-specific cumulative SP (not ranked) | 0.0 |
| `bf` | `ml/history_features.py` | Class-specific BF_12 alias (`bf_12` or `bfo_12`) | 0.0 |
| `count_active_cids` | `ml/data_loader.py:load_collapsed` | Count of CIDs with `right_tail_max >= threshold` per branch | Always available |
| `bin_80_max` | `ml/data_loader.py` Level 2 collapse | Max across CIDs of bin_80 mean-across-outage-dates | Always available |
| `bin_90_max` | Same | Max across CIDs of bin_90 | Always available |
| `bin_100_max` | Same | Max across CIDs of bin_100 | Always available |
| `bin_110_max` | Same | Max across CIDs of bin_110 | Always available |
| `rt_max` | Derived | `max(bin_80_max, bin_90_max, bin_100_max, bin_110_max)` | Always available |
| `top2_bin_80` | `data/nb_cache/{py}_{aq}_top2.parquet` | Mean of top-2 CID values per branch for bin_80 | 0.0 |
| `top2_bin_90` | Same | Same for bin_90 | 0.0 |
| `top2_bin_100` | Same | Same for bin_100 | 0.0 |
| `top2_bin_110` | Same | Same for bin_110 | 0.0 |

### Label recipe

`miso_annual_bucket_5tier_v1`:

| Tier | Condition | Weight |
|------|-----------|--------|
| 0 | SP = 0 | 1 |
| 1 | 0 < SP <= $200 | 1 |
| 2 | $200 < SP <= $5K | 2 |
| 3 | $5K < SP <= $20K | 6 |
| 4 | SP > $20K | 20 |

### Training

- Objective: LambdaRank (NDCG)
- Groups: (PY, quarter) pairs
- Expanding window: train on all PYs before eval PY
- LGB params: `num_leaves=15, lr=0.05, min_child_samples=5, subsample=0.8, colsample=0.8, num_threads=4, 150 rounds`

### Round behavior

- **v1**: round-insensitive. Current cached tables from `data/nb_cache/` built with `market_round=1`.
- **v2** (pending): `da_rank_value`, `shadow_price_da`, `bf` will use round-specific cutoff. `bin_*_max`, `count_active_cids` will use round-specific density. `top2_bin_*` will need round-aware caching.

### Source code

- Training + eval: `scripts/nb_bucket_model.py`
- 3-way comparison: `scripts/champion_confirmation.py`

---

## `miso_annual_nb_tiered_top2_features_v1`

**Model**: NB dormant scorer (used in R30/R50 reserved-slot deployment)
**Universe scope**: `miso_annual_nb_dormant_v1` (dormant subset of `miso_annual_branch_active_v1`)
**Valid ctypes**: onpeak, offpeak (separate training per ctype)

### Columns

| Column | Source | Transform | Fill rule |
|--------|--------|-----------|-----------|
| `bin_80_max` | `ml/data_loader.py` Level 2 collapse | Max across CIDs | Always available |
| `bin_90_max` | Same | Same | Always available |
| `bin_100_max` | Same | Same | Always available |
| `bin_110_max` | Same | Same | Always available |
| `rt_max` | Derived | `max(bin_80_max, ..., bin_110_max)` | Always available |
| `count_active_cids` | `ml/data_loader.py:load_collapsed` | Active CID count | Always available |
| `shadow_price_da` | `ml/phase6/features.py` | Class-specific cumulative SP | 0.0 |
| `da_rank_value` | Same | Dense rank of cumulative SP | Sentinel |
| `top2_bin_80` | `data/nb_cache/{py}_{aq}_top2.parquet` | Mean of top-2 CID values | 0.0 |
| `top2_bin_90` | Same | Same | 0.0 |
| `top2_bin_100` | Same | Same | 0.0 |
| `top2_bin_110` | Same | Same | 0.0 |

### Label recipe

`miso_annual_nb_tiered_tertile_v1`: per-group tertile tiers [0, 1, 2, 3] with weights [1, 1, 3, 10].

### Training

- Trained on dormant branches only (BF_12 == 0 for the class)
- Same LGB params as Bucket_6_20
- Used in R30/R50 deployment: top-scoring dormant branches fill reserved slots

### Round behavior

Same as `miso_annual_bucket_features_v1` — currently R1-only.

### Source code

- Training: `scripts/nb_v3_ablation.py` (variant `tiered_top2`)
- Deployment: `scripts/nb_native_comparison.py` (`allocate_reserved_slots`)

---

## Relationship between recipes

`miso_annual_bucket_features_v1` is a strict superset of `miso_annual_nb_tiered_top2_features_v1` — same 12 columns plus `bf`. The difference is:
- Bucket trains on ALL branches (full universe)
- NB tiered_top2 trains on DORMANT branches only (NB subset)

`miso_annual_v0c_features_v1` uses only 3 effective inputs (da_rank, rt_max, bf) — a strict subset of both ML recipes.

---

## Rules

1. A feature recipe ID must appear in `spec.json` for any promoted model.
2. Changing any column, transform, or fill rule requires a new recipe version (e.g., `_v2`).
3. Round-sensitive versions must be distinct IDs from round-insensitive versions.
4. `top2_bin_*` features currently depend on `data/nb_cache/` which is R1-only and not round-aware. A round-sensitive recipe must source these from the round-aware pipeline or drop them.
