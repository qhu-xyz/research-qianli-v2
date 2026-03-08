# Annual FTR Constraint Tier Prediction — Full Audit

**Date**: 2026-03-08
**Scope**: Data integrity, leakage, target correctness, evaluation, design
**Verdict**: **PASS — No critical issues found**

---

## 1. Pipeline Architecture

```
V6.1 Signal (pre-auction forecasts)    Spice6 Density (forward simulation)
         \                                    /
          → cache/enriched/ (27 cols, ~325-632 rows/quarter)
                           |
                    [Features extracted]
                           |
                    LightGBM LambdaRank
                           |
                   Realized DA Shadow Prices ← MisoApTools (separate fetch)
                   cache/ground_truth/ (2 cols: branch_name, realized_shadow_price)
                           |
                   Evaluate: VC@K, Recall@K, NDCG, Spearman, Tier-AP
```

**Key design principle**: Features come from V6.1 (planned/historical), target comes from MisoApTools DA (realized/actual) — completely separate sources.

---

## 2. Feature Leakage Audit

### 2.1 Feature-by-Feature Check

| Feature | Source | Time Horizon | Leakage Risk | Verdict |
|---------|--------|-------------|-------------|---------|
| `shadow_price_da` | V6.1 parquet | Historical 60-month lookback | LOW | **CLEAN** — Spearman ~0.81 with hist_shadow, only ~0.36 with actual_shadow_price |
| `mean_branch_max` | V6.1 parquet | SPICE6 forward simulation | NONE | **CLEAN** |
| `ori_mean` | V6.1 parquet | Baseline scenario flow forecast | NONE | **CLEAN** |
| `mix_mean` | V6.1 parquet | Mixed scenario flow forecast | NONE | **CLEAN** |
| `density_mix_rank_value` | V6.1 parquet | Percentile rank of mix_mean | NONE | **CLEAN** |
| `density_ori_rank_value` | V6.1 parquet | Percentile rank of ori_mean | NONE | **CLEAN** |
| `rank_ori` | V6.1 parquet | Formula output (0.60×da_rank + 0.30×mix_rank + 0.10×ori_rank) | LOW | **CLEAN** — Fixed weights on historical features, NOT target-derived |
| `prob_exceed_80..110` | MISO_SPICE_DENSITY_DISTRIBUTION | Forward simulation exceedance probs | NONE | **CLEAN** |
| `constraint_limit` | MISO_SPICE_CONSTRAINT_LIMIT | Static network limit | NONE | **CLEAN** |
| `rate_a` | MISO_SPICE_CONSTRAINT_INFO | Static branch thermal rating | NONE | **CLEAN** |

### 2.2 Leaky Feature Guard

`config.py` defines `_LEAKY_FEATURES` that are **actively filtered** in `LTRConfig.__post_init__()`:

```python
_LEAKY_FEATURES = {"rank", "tier", "shadow_sign", "shadow_price", "density_mix_rank", "mean_branch_max_fillna"}
```

If any leaky feature accidentally enters the feature list, it is removed with a printed warning. This guard runs at config construction time.

### 2.3 Verdict: **NO FEATURE LEAKAGE**

---

## 3. Target Variable Audit

### 3.1 Definition

`realized_shadow_price` = **SUM of absolute onpeak DA shadow prices across 3 market months** per branch

### 3.2 Construction Chain

```
MisoApTools.get_da_shadow_by_peaktype(peak_type="onpeak", dates=3 months)
  → group by constraint_id → SUM(abs(shadow_price))
  → JOIN via MISO_SPICE_CONSTRAINT_INFO → branch_name
  → SUM per branch_name (multiple DA constraint_ids → 1 branch)
  → LEFT JOIN to V6.1 universe
  → Non-matches get 0.0 (non-binding)
```

### 3.3 Key Properties

| Property | Value | Status |
|----------|-------|--------|
| Aggregation | Sum of absolute values | Correct for ranking by impact |
| Peak type | Onpeak only | Matches V6.1 signal (onpeak directory) |
| Time scope | 3 market months per quarter | Correct (aq1=Jun-Aug, aq2=Sep-Nov, etc.) |
| Bridge table | MISO_SPICE_CONSTRAINT_INFO | 17.8M rows, maps DA constraint_id → branch_name |
| Binding rate | ~31-42% per quarter | Reasonable for constraint universe |
| Null handling | 0.0 for non-binding | Correct |
| Duplicate risk | Both flow_directions get same shadow price | Documented, expected |

### 3.4 Is target circular with any feature?

- `shadow_price_da` in V6.1 is **historical** (60-month lookback), NOT realized DA
- Verified: adjacent months share ~65-70% identical `shadow_price_da` values (proves slow evolution)
- Verified: Spearman ~0.36 between `shadow_price_da` and `realized_shadow_price` (low correlation)
- `rank_ori` is formula output on historical features, not target-derived

### 3.5 Verdict: **TARGET IS CORRECT AND INDEPENDENT**

---

## 4. Train/Test Split Audit

### 4.1 Expanding Window Definition

```
split1: train [2019, 2020, 2021] → eval 2022 (4 quarters)
split2: train [2019, 2020, 2021, 2022] → eval 2023 (4 quarters)
split3: train [2019, 2020, 2021, 2022, 2023] → eval 2024 (4 quarters)
holdout: train [2019, 2020, 2021, 2022, 2023, 2024] → eval 2025 (4 quarters)
```

### 4.2 Temporal Leakage Check

| Check | Result |
|-------|--------|
| Train years < eval year? | YES — strict temporal ordering |
| Same year in train AND eval? | NO — never |
| Overlapping market months? | NO — quarters cleanly separated |
| Model reuse within year? | YES — train once per eval_year, evaluate 4 quarters (correct: all share same model) |
| Ground truth added before or after split? | AFTER — features loaded first, ground truth joined second |

### 4.3 Verdict: **NO TEMPORAL LEAKAGE**

---

## 5. Cache Integrity Audit

### 5.1 Cache Structure

| Directory | Content | Columns | Contains Target? |
|-----------|---------|---------|-----------------|
| `cache/enriched/` | V6.1 + spice6 features | 27 (incl. branch_name, constraint_id, all features) | **NO** — no `realized_shadow_price` |
| `cache/ground_truth/` | Realized DA shadow prices | 2 (branch_name, realized_shadow_price) | YES — by design |

### 5.2 Cross-Contamination Check

- Enriched cache built by `load_v61_enriched()` in `data_loader.py` — never calls ground_truth
- Ground truth cache built by `get_ground_truth()` in `ground_truth.py` — never reads enriched
- Different file patterns: `{year}_{aq}.parquet` in both dirs, but separate builders
- Verified: enriched parquet does NOT contain `realized_shadow_price` column

### 5.3 Verdict: **CACHES PROPERLY SEPARATED**

---

## 6. Evaluation Metrics Audit

### 6.1 VC@K (Value Capture)

```python
numerator = sum(realized_shadow_price of model's top-K)
denominator = sum(realized_shadow_price of ALL V6.1 constraints)
```

- Denominator is V6.1 universe sum, NOT total market DA — **correct per spec**
- `k = min(k, len(scores))` handles edge cases

### 6.2 Recall@K

```python
true_top_k = set(argsort(actual)[::-1][:k])
pred_top_k = set(argsort(scores)[::-1][:k])
recall = len(intersection) / k
```

- Standard top-K overlap metric — **correct**

### 6.3 NDCG

- Standard implementation with `log2(position+1)` discounting
- Uses actual shadow prices as relevance (not binary) — **correct for graded relevance**

### 6.4 tier_ap (Average Precision)

- Tier0-AP: top 20% by actual value → binary labels → AP against model scores
- Tier01-AP: top 40% by actual value
- **Bug fix verified**: When binding% < top_frac% (threshold=0), falls back to `actual > 0` as positive class
- Edge case: if no positive values, returns 0.0

### 6.5 Aggregate

- Mean, std (population), min, max, bottom_2_mean across 12 eval groups
- **Correctly implemented**

### 6.6 Verdict: **ALL METRICS CORRECT**

---

## 7. Monotone Constraints Audit

| Feature | Constraint | Direction | Domain Rationale | Correct? |
|---------|-----------|-----------|-----------------|----------|
| shadow_price_da | +1 | Higher → more binding | Higher historical shadow → higher future binding | YES |
| mean_branch_max | +1 | Higher → more binding | Higher max loading → more congestion | YES |
| ori_mean | +1 | Higher → more binding | Higher baseline flow → more congestion | YES |
| mix_mean | +1 | Higher → more binding | Higher mixed flow → more congestion | YES |
| density_mix_rank_value | -1 | Lower → more binding | Lower percentile = tighter constraint | YES |
| density_ori_rank_value | -1 | Lower → more binding | Lower percentile = tighter constraint | YES |
| rank_ori | -1 | Lower → more binding | Lower formula rank = more binding | YES |
| prob_exceed_80..110 | +1 | Higher → more binding | Higher exceedance prob → more binding | YES |
| constraint_limit | 0 | Unconstrained | No clear monotone direction | YES |
| rate_a | 0 | Unconstrained | No clear monotone direction | YES |

### Verdict: **ALL MONOTONE DIRECTIONS CORRECT**

---

## 8. Model Architecture Audit

| Aspect | Choice | Rationale | Sound? |
|--------|--------|-----------|--------|
| Objective | LambdaRank | Ranking task (identify top-K binding constraints) | YES |
| Backend | LightGBM | 22x faster than XGBoost, same results | YES |
| Label mode | Tiered (v5) | Avoids rank-transform noise on 58% non-binding | YES |
| n_estimators | 100 (early stopping at 20) | Enough for small data, ES prevents overfit | YES |
| num_leaves | 31 | Default, reasonable for ~5-9K training rows | YES |
| learning_rate | 0.05 | Standard | YES |
| min_data_in_leaf | 25 | Prevents overfitting on small groups | YES |
| subsample | 0.8 | Standard bagging | YES |
| num_threads | 4 | Prevents 64-core thread contention | YES |
| Query groups | ~300-600 constraints per group | Ample for pairwise ranking | YES |
| Training size | 5,828-8,746 rows | Small but sufficient with regularization | YES |

### Verdict: **MODEL ARCHITECTURE SOUND**

---

## 9. Data Quality Audit

### 9.1 Raw Data

| Source | Files | Rows/File | Nulls | Duplicates | Status |
|--------|-------|-----------|-------|-----------|--------|
| V6.1 Signal | 28 | 276-632 | 0 | 0 | CLEAN |
| Spice6 Density | 1 | 50.2M | 0 | Expected | CLEAN |
| Constraint Info | 1 | 17.8M | 0 | Many-to-1 mapping | CLEAN |
| Constraint Limit | 1 | — | 0 | — | CLEAN |

### 9.2 Cached Data

| Cache | Files | Coverage | Nulls Post-Fill | Status |
|-------|-------|----------|----------------|--------|
| Enriched | 28 | 98.8% spice6 match | 0 (filled with 0.0) | CLEAN |
| Ground Truth | 28 | ~31-42% binding | 0 (non-binding = 0.0) | CLEAN |

### 9.3 Known Data Quirks

| Quirk | Impact | Mitigation |
|-------|--------|-----------|
| 2025/aq4 only 18.2% binding | Low binding rate in holdout | Expected — Mar-May 2026 data incomplete |
| Both flow_directions get same shadow price | Duplicate signal | Both rows contribute equally — OK for ranking |
| ~20-28% of total DA value on V6.1 branches | V6.1 doesn't cover all market | Denominator is V6.1 universe only |
| 65-70% constraint churn year-over-year | Different constraint universe each year | Model learns from features, not IDs |

---

## 10. Results Verification

### 10.1 Registry vs mem.md Cross-Check

| Version | Metric | mem.md | registry JSON | Match? |
|---------|--------|--------|---------------|--------|
| v1 | VC@20 | 0.2934 | 0.29340... | YES |
| v1 | Recall@20 | 0.2708 | 0.27083... | YES |
| v5 | VC@20 | 0.3075 | 0.30752... | YES |
| v5 | Recall@20 | 0.3208 | 0.32083... | YES |
| v5 | NDCG | 0.6098 | 0.60981... | YES |
| v5 | Spearman | 0.3695 | 0.36953... | YES |

### 10.2 Version Config Verification

| Version | Features | Label Mode | Monotone | Config Matches Script? |
|---------|----------|-----------|----------|----------------------|
| v0 | formula (rank_ori only) | — | — | YES |
| v1 | SET_A (6) | rank | correct | YES |
| v2 | SET_B (11) | rank | correct | YES |
| v3 | SET_A (6) | tiered | correct | YES |
| v4 | SET_AF (7) | rank | correct | YES |
| v5 | SET_AF (7) | tiered | correct | YES |

---

## 11. Checklist Summary

| # | Check | Status |
|---|-------|--------|
| 1 | No realized DA data in features | **PASS** |
| 2 | No target column used as feature | **PASS** |
| 3 | No future data in training | **PASS** |
| 4 | Train/eval years strictly ordered | **PASS** |
| 5 | Ground truth fetched independently | **PASS** |
| 6 | Cache directories separated | **PASS** |
| 7 | Leaky feature guard active | **PASS** |
| 8 | Constraint mapping chain correct | **PASS** |
| 9 | VC@K denominator correct (V6.1 universe) | **PASS** |
| 10 | Recall@K correctly computed | **PASS** |
| 11 | NDCG correctly computed | **PASS** |
| 12 | tier_ap bug fixed | **PASS** |
| 13 | Aggregate means correct | **PASS** |
| 14 | Monotone constraints domain-correct | **PASS** |
| 15 | Label transform within-group only | **PASS** |
| 16 | LightGBM num_threads set | **PASS** |
| 17 | Registry results match mem.md | **PASS** |
| 18 | Version configs match scripts | **PASS** |
| 19 | No cross-group label leakage | **PASS** |
| 20 | Market month boundaries correct | **PASS** |

---

## 12. Recommendations (Non-Critical)

1. **Add unit tests** for `tier_ap()` edge cases (all zeros, single constraint, threshold=0 fallback)
2. **Document V6.1 generation process** — parquet files exist but no generation code in this repo
3. **Monitor 2025 data completeness** — aq4 (Mar-May 2026) expected to be incomplete
4. **Consider adding rate_a to V6.1 enrichment** — currently only in Set C which isn't used by any version
