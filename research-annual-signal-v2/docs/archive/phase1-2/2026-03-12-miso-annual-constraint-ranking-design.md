# MISO Annual FTR Constraint Ranking ML Pipeline — Design Spec

**Date**: 2026-03-12
**Status**: Approved (brainstorming complete)
**Source spec**: `docs/implementer-guide.md`

---

## 1. Project Summary

Rank MISO constraints by likelihood of binding in the Day-Ahead (DA) electricity market for the annual FTR auction. The ranking is used for trade selection. Universe size varies by quarter (provisionally estimated at 1,100-1,850 branch_names per quarter after right-tail density filtering and constraint-to-branch collapse — these are inherited planning targets and will be recalibrated once UNIVERSE_THRESHOLD is frozen in Step 1.2).

**Input**: Raw SPICE density distributions + realized DA shadow prices
**Output**: Per-branch_name ranking score for each (planning_year, aq_quarter) group

---

## 2. Architecture: Modular Library + Thin Experiment Scripts

Approach B selected. The `ml/` library handles all shared logic. Experiment scripts are thin orchestrators that import from `ml/`, specify a feature list and params, and write results to `registry/{version}/`. DRY is non-negotiable. No config-driven framework — the experiment scripts ARE the configs.

### 2.1 Module Layout (12 modules)

```
scripts/fetch_realized_da.py          # Run FIRST (requires Ray)

ml/__init__.py
ml/config.py              # Constants, paths, thresholds, feature lists, splits, params, gates
ml/bridge.py               # Annual+monthly bridge loading, convention filter, both-ctype UNION
ml/realized_da.py          # Cache loading, combined/onpeak/offpeak monthly and quarter aggregations
ml/data_loader.py          # Density + limits -> universe filter -> Level 1+2 collapse -> branch features
ml/history_features.py     # BF family (7) + da_rank_value (1) + has_hist_da flag — all branch-level
ml/ground_truth.py         # Continuous SP target + tiered labels + per-ctype split targets + raw coverage diagnostics
ml/nb_detection.py         # NB6/NB12/NB24 + per-ctype NB12 flags — branch-level interface
ml/features.py             # Joins all into ONE model table, cohort assignment, monotone vector, schema contract
ml/train.py                # Expanding-window LambdaRank training + prediction
ml/evaluate.py             # All Tier 1/2/3 metrics, NB metrics, cohort contribution, gates
ml/registry.py             # Write metrics/config/artifacts to registry/{version}/

scripts/calibrate_threshold.py        # Universe threshold calibration (Phase 1, Day 1)
scripts/run_v0a_da_rank.py            # Formula: pure da_rank_value
scripts/run_v0b_blend.py              # Formula: da_rank + right_tail density blend
scripts/run_v0c_full_blend.py         # Formula: da_rank + right_tail density + bf_combined blend
scripts/run_v1_ml_historical.py       # ML: da_rank + BF features only
scripts/run_v2_ml_density.py          # ML: + density bin features (build-up steps 2b-2f)
```

### 2.2 Dependency Graph

```
config.py              <- (no deps)
bridge.py              <- config.py
realized_da.py         <- config.py
data_loader.py         <- config.py, bridge.py
history_features.py    <- config.py, bridge.py, realized_da.py
ground_truth.py        <- config.py, bridge.py, realized_da.py
nb_detection.py        <- config.py (receives monthly_binding_table + gt_df as inputs)
features.py            <- data_loader, history_features, ground_truth, nb_detection
train.py               <- (receives model table from features.py)
evaluate.py            <- (receives model table + scores)
registry.py            <- (receives metrics dict + config dict, independent of evaluate.py)
```

### 2.3 Key Design Rules

**Branch-level boundary**: Everything after Level 2 collapse is branch-level. No downstream module ever sees a constraint_id. The cid-to-branch boundary lives entirely in `data_loader.py` (for density features) and `bridge.py` (for GT/BF/NB mapping).

**Shared bridge mapping contract**: `bridge.py` exposes a single `map_cids_to_branches()` function that handles:
- Both-ctype UNION (onpeak + offpeak)
- Convention < 10 filter
- Ambiguous cid detection (cids mapping to multiple branch_names after union)
- Logging of ambiguous cid count and affected SP
- Dropping ambiguous cids

All three consumers — `data_loader.py`, `ground_truth.py`, `history_features.py` — use this single function. One ambiguity rule, one place. No module implements its own bridge-join logic.

```python
def map_cids_to_branches(
    cid_df: pl.DataFrame,           # must have constraint_id column
    auction_type: str,              # 'annual' or 'monthly'
    auction_month: str,             # e.g., '2025-06'
    period_type: str,               # 'aq1'..'aq4' for annual, 'f0' for monthly
) -> tuple[pl.DataFrame, dict]:
    """Returns (mapped_df with branch_name, diagnostics dict with ambiguous_cids, ambiguous_sp)."""
```

---

## 3. Data Loading & Collapse Pipeline (`ml/data_loader.py`)

**Input**: Raw density parquet + constraint limits parquet + bridge table
**Output**: Branch-level DataFrame with forward-looking features for all (PY, quarter) slices

### 3.1 Pipeline per (PY, quarter)

```
Step 1: Derive market_months from (PY, quarter)
  e.g., 2025-06/aq1 -> ['2025-06', '2025-07', '2025-08']

Step 2: Load raw density via partition-specific paths (NOT hive scan, NOT pbase loaders)
  For each market_month:
    pl.read_parquet(f'{DENSITY_PATH}/spice_version=v6/auction_type=annual/
      auction_month={PY}/market_month={mm}/market_round=1/')
  Select: constraint_id + 10 selected bins
  Concat all months

Step 3: Compute right_tail_max and is_active flag for ALL cids (before any filtering)
  a. Per raw row: right_tail = max(bin_80, bin_90, bin_100, bin_110)
  b. Per cid: right_tail_max = max(right_tail) across all outage_dates and months
  c. Flag: is_active = (right_tail_max >= UNIVERSE_THRESHOLD)
  Note: UNIVERSE_THRESHOLD was calibrated at branch level (see DR#1) but is
  applied here at cid level. Consistent because branch_rtm = max(cid_rtm).

Step 4: Map cids -> branch_name via bridge.map_cids_to_branches()
  Use bridge.map_cids_to_branches(auction_type='annual', auction_month=PY, period_type=quarter)
  Shared function handles both-ctype UNION, convention < 10, ambiguity detect+log+drop
  Compute per-branch:
    - count_cids = total mapped cids on the branch (active + inactive)
    - count_active_cids = mapped cids with is_active=True
  Filter: keep only branches with at least one active cid

Step 5: Level 1 collapse (on active cids only)
  Mean across outage_dates AND quarter months per cid per bin
  -> 1 row per active cid, 10 bin columns

Step 6: Load constraint limits for the same quarter months
  Level 1 limits: mean(limit) across dates/months per cid
  Join onto cid-level density rows

Step 7: Level 2 collapse: group_by branch_name
  Density: max + min per bin (20 features) -- scaffolded for std variant
  Limits: min, mean, max, std across cids per branch (4 features)
  Metadata: count_cids, count_active_cids from Step 4 (2 features)
  -> 1 row per branch, ~26 forward-looking features

Step 8: Cache to data/collapsed/{PY}_{quarter}_t{threshold_version}_f{feature_version}.parquet
```

### 3.2 Selected Bins (10)

`-100, -50, 60, 70, 80, 90, 100, 110, 120, 150`

Evidence-based selection from empirical Spearman analysis (implementer-guide SS7.2). Dropped: -10, 0, 20, 40 (negligible/inverted), 85, 95, 105 (redundant with adjacent), 200, 300 (too sparse).

### 3.3 Assertions

- Raw density row sum approximately 20.0 (validates bin semantics — Trap 22)
- No null branch_names after convention filter
- Ambiguous cids (multiple branch_names after union) logged and dropped
- Branch count within expected range: monitoring log, not hard assert until threshold frozen (current 1,100-1,850 estimates are provisional)

---

## 4. Ground Truth Pipeline (`ml/ground_truth.py`)

**Input**: Realized DA cache + bridge table
**Output**: Per-branch DataFrame with continuous SP target, tiered labels, per-ctype split targets, and raw coverage diagnostics

### 4.1 Pipeline per (PY, quarter)

```
Step 1: Load realized DA cache for the 3 market months in the quarter
  - Load BOTH onpeak AND offpeak per month via realized_da.py
  - Each cache row is (constraint_id, realized_sp) where
    realized_sp = abs(sum(shadow_price)) already netted within that month+ctype
  - Quarter aggregation: sum(realized_sp) across 3 months x 2 ctypes per constraint_id
  - These are nonnegative values being summed — no abs() or netting at this stage

Step 2: Map DA constraint_id -> branch_name via ANNUAL bridge (primary)
  - Use bridge.map_cids_to_branches(auction_type='annual', auction_month=PY, period_type=quarter)
  - Shared function handles both-ctype UNION, convention < 10, ambiguity detection+drop
  - Map DA cids via bridge (unmapped cids tracked in diagnostics for monthly fallback)
  - Track unmapped DA cids

Step 3: Monthly bridge fallback for unmapped DA cids
  - For each market_month, use bridge.map_cids_to_branches(auction_type='monthly', period_type='f0')
  - Same shared ambiguity rule applied to monthly bridge
  - Match remaining unmapped DA cids against monthly bridges
  - Only use monthly mapping for cids NOT already mapped by annual bridge
  - Log: recovered cid count, recovered SP, still unmapped count/SP

Step 4: Aggregate to branch_name
  - Combine annual-mapped + monthly-recovered
  - group_by(branch_name).agg(sum(realized_sp))
  - Multiple DA cids -> same branch -> SUM (not mean)

Step 5: Compute tiered labels per (PY, quarter) group
  - 0 = non-binding (realized_sp == 0)
  - Tertile boundaries on positive-SP branches only within this group
  - 1 = bottom tertile, 2 = middle, 3 = top

Step 6: Compute per-ctype split targets (monitoring only)
  - onpeak_sp: sum of realized_sp from onpeak DA only
  - offpeak_sp: sum of realized_sp from offpeak DA only
  - NOT used for training — only for evaluation split reporting
```

### 4.2 Return DataFrame

```
branch_name | planning_year | aq_quarter |
realized_shadow_price |          # continuous target (combined ctype)
label_tier |                     # 0/1/2/3
onpeak_sp | offpeak_sp |         # class-type split (monitoring)
```

### 4.3 Raw Coverage Diagnostics

Returned alongside the DataFrame (raw counts + raw SP, not just percentages):

```python
{
    "total_da_cids": int,
    "annual_mapped_cids": int,
    "monthly_recovered_cids": int,
    "still_unmapped_cids": int,
    "total_da_sp": float,
    "annual_mapped_sp": float,
    "monthly_recovered_sp": float,
    "still_unmapped_sp": float,
}
```

Validate where published reference exists (SS8.7 table); otherwise log raw diagnostics.

**Important**: `total_da_sp` from this diagnostics dict is propagated by `features.py` into the model table as `total_da_sp_quarter` (group-level constant). This is the denominator for `Abs_SP@K` metrics in `evaluate.py`.

---

## 5. History Features (`ml/history_features.py`)

**Input**: Realized DA cache + bridge table
**Output**: Branch-level DataFrame with 8 features + has_hist_da flag

### 5.1 Internal Structure

Build a monthly branch-binding table first (single DA scan), then derive both BF windows and da_rank_value from it.

```
Step 1: Build monthly branch-binding table
  For each month in [2017-04, cutoff_month]:
    - Load onpeak + offpeak cache via realized_da.py
    - Map DA cid -> branch via bridge.py using the EVAL PY's annual bridge:
      bridge.map_cids_to_branches(auction_type='annual', auction_month=eval_PY, period_type=quarter)
      For unmapped cids, monthly fallback:
      bridge.map_cids_to_branches(auction_type='monthly', auction_month=M, period_type='f0')
    - Bridge partition rule: always use the eval PY (the PY you're building features for),
      NOT the PY that contains historical month M. This keeps the bridge consistent with
      how GT and data_loader map for the same PY.
    - Same shared ambiguity rule (detect, log, drop) applied via the shared function.
    - Per branch per month: onpeak_bound, offpeak_bound, combined_bound,
      onpeak_sp, offpeak_sp, combined_sp

Step 2: BF features (from binding flags, fixed denominator)
  - bf_N = count(onpeak_bound=True in last N months) / N   (Float64)
  - bfo_N = count(offpeak_bound=True in last N months) / N
  - bf_combined_N = count(combined_bound=True in last N months) / N
  - Denominator is ALWAYS fixed N, even if fewer months of history exist

Step 3: da_rank_value (from cumulative SP, ranked within universe)
  - cumulative_sp = sum(combined_sp) across all months up to cutoff per branch
  - Dense rank descending within current (PY, quarter) universe branches
  - Zero-history branches: rank = n_positive_history + 1
  - has_hist_da = cumulative_sp > 0  (exported for cohort assignment)
  - Float64
```

### 5.2 Features

| Feature | Definition | Type | Monotone |
|---------|-----------|------|:---:|
| `bf_6` | Fraction of last 6 months with onpeak binding | Float64 | +1 |
| `bf_12` | Fraction of last 12 months onpeak | Float64 | +1 |
| `bf_15` | Fraction of last 15 months onpeak | Float64 | +1 |
| `bfo_6` | Fraction of last 6 months offpeak | Float64 | +1 |
| `bfo_12` | Fraction of last 12 months offpeak | Float64 | +1 |
| `bf_combined_6` | Fraction of last 6 months either ctype | Float64 | +1 |
| `bf_combined_12` | Fraction of last 12 months either ctype | Float64 | +1 |
| `da_rank_value` | Rank of cumulative historical SP (lower = more binding) | Float64 | -1 |

### 5.3 Leakage Prevention

- Lookback cutoff = March of submission year (Trap 1)
- Backfill floor = 2017-04 (v16 champion insight)
- For PY 2025-06 (submitted ~April 10, 2025): months 2017-04 through 2025-03
- Features returned only for branches in the modeling universe for that (PY, quarter)

---

## 6. NB Detection & Cohorts

### 6.1 NB Detection (`ml/nb_detection.py`)

Branch-level interface. Reuses monthly binding table from history_features (no duplicate DA scans).

```python
def compute_nb_flags(
    universe_branches: list[str],
    planning_year: str,
    aq_quarter: str,
    gt_df: pl.DataFrame,              # has realized_shadow_price, onpeak_sp, offpeak_sp
    monthly_binding_table: pl.DataFrame,
) -> pl.DataFrame:
    """Returns branch-level NB flags."""
```

**Combined-ctype NB (gate + monitoring):**
- `is_nb_N`: NO cid on branch had binding in EITHER ctype for last N months, AND branch binds in target quarter (realized_shadow_price > 0)
- Windows: N = 6, 12, 24

**Per-ctype NB12 (monitoring only):**
- `nb_onpeak_12`: no onpeak binding for 12 months AND onpeak_sp > 0 in target quarter
- `nb_offpeak_12`: no offpeak binding for 12 months AND offpeak_sp > 0 in target quarter

**Rules:**
- Branches with < N months of history: treat as NB-eligible
- NB requires actual binding in target quarter (not just universe membership)

### 6.2 Cohort Assignment (`ml/features.py`)

Mutually exclusive, based on branch-level features. Uses `has_hist_da` from history_features (not da_rank_value sentinel).

| Cohort | Rule | Signal available |
|--------|------|-----------------|
| `established` | `bf_combined_12 > 0` | All features |
| `history_dormant` | `bf_combined_12 == 0 AND has_hist_da` | DA history but no recent binding |
| `history_zero` | `bf_combined_12 == 0 AND NOT has_hist_da` | Density features only |

Priority: established > history_dormant > history_zero (non-overlapping).

---

## 7. Model Table Contract (`ml/features.py`)

`features.py` owns the final model table schema. All downstream modules (train, evaluate) receive this single DataFrame.

### 7.1 Schema

```
# Keys
branch_name        | Utf8
planning_year      | Utf8
aq_quarter         | Utf8

# Forward-looking features (from data_loader)
bin_{X}_cid_max    | Float64    # 10 bins x max = 10 features
bin_{X}_cid_min    | Float64    # 10 bins x min = 10 features (scaffolded for std)
limit_min          | Float64
limit_mean         | Float64
limit_max          | Float64
limit_std          | Float64
count_cids         | UInt32
count_active_cids  | UInt32

# History features (from history_features)
da_rank_value      | Float64
bf_6               | Float64
bf_12              | Float64
bf_15              | Float64
bfo_6              | Float64
bfo_12             | Float64
bf_combined_6      | Float64
bf_combined_12     | Float64

# Target (from ground_truth)
realized_shadow_price | Float64    # continuous target (combined ctype)
label_tier            | Int32      # 0/1/2/3
onpeak_sp             | Float64    # monitoring only
offpeak_sp            | Float64    # monitoring only

# NB flags (from nb_detection)
is_nb_6            | Boolean
is_nb_12           | Boolean
is_nb_24           | Boolean
nb_onpeak_12       | Boolean
nb_offpeak_12      | Boolean

# Cohort (from features.py)
cohort             | Utf8        # 'established', 'history_dormant', 'history_zero'

# Metadata (from features.py)
has_hist_da        | Boolean

# Group-level GT diagnostics (from ground_truth via features.py)
total_da_sp_quarter | Float64    # ALL DA binding SP for the quarter (including outside-universe)
                                 # Used as Abs_SP@K denominator — NOT in-universe SP
```

**Note on `total_da_sp_quarter`**: This is the sum of ALL realized DA shadow prices for the quarter across ALL constraints (not just those in the model universe). It comes from `ground_truth.py` coverage diagnostics (`total_da_sp`) and is attached as a group-level constant by `features.py`. This is required for `Abs_SP@K` metrics in `evaluate.py` — using in-universe SP as the denominator would be wrong.
```

### 7.2 Monotone Constraints

Order must exactly match feature_cols (enforced by assertion).

| Feature | Direction | Reasoning |
|---------|:-:|---|
| bf_6, bf_12, bf_15 | +1 | Higher freq = more binding |
| bfo_6, bfo_12 | +1 | Higher freq = more binding |
| bf_combined_6, bf_combined_12 | +1 | Higher freq = more binding |
| da_rank_value | -1 | Lower rank = more binding |
| density bins (cid_max/cid_min) | 0 | NOT monotone — density weights, not probabilities |
| limit_* | 0 | Relationship unclear |
| count_cids, count_active_cids | 0 | Direction unclear |

---

## 8. Training (`ml/train.py`)

### 8.1 Expanding Window Split

| Eval Year | Training PYs | Train groups | Eval groups |
|:-:|---|:-:|:-:|
| 2022-06 | 2019, 2020, 2021 | 12 | 4 (dev) |
| 2023-06 | 2019-2022 | 16 | 4 (dev) |
| 2024-06 | 2019-2023 | 20 | 4 (dev) |
| 2025-06 | 2019-2024 | 24 | 3 (holdout: aq1-aq3) |

- Dev = 12 groups (2022-2024 x 4 quarters)
- Holdout = 3 groups (2025-06/aq1-aq3). aq4 monitored when available, excluded from gates.

### 8.2 LambdaRank Parameters

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "num_threads": 4,           # Trap 3: 64-CPU container causes contention
    "verbose": -1,
}
```

### 8.3 Key Training Rules

- **Query groups**: Each (planning_year, aq_quarter) is one group. Sort by (planning_year, aq_quarter, branch_name) before building group sizes — stable ordering prevents silent misalignment.
- **Eval-only scoring**: Predict only eval year groups, never in-sample.
- **train_and_predict() returns**: model table with added `score` (Float64), `eval_year` (Utf8), `split` (Utf8: 'dev' or 'holdout').
- **Feature importance**: LightGBM gain importance, normalized to percentages.
- **Walltime**: Record train walltime and full experiment walltime separately.

---

## 9. Evaluation & Gates (`ml/evaluate.py`)

### 9.1 Tier 1 — Blocking Gates

Must pass 2/3 holdout groups + mean >= baseline for each metric.

| # | Metric | Definition |
|---|--------|-----------|
| 1 | VC@50 | sum(branch SP in top 50) / sum(branch SP in universe) |
| 2 | VC@100 | sum(branch SP in top 100) / sum(branch SP in universe) |
| 3 | Recall@50 | count(binding branches in top 50) / count(binding branches in universe) |
| 4 | Recall@100 | count(binding branches in top 100) / count(binding branches in universe) |
| 5 | NDCG | Standard ranking NDCG using tiered labels 0/1/2/3 |
| 6 | Abs_SP@50 | sum(branch SP in top 50) / total DA binding SP for quarter (including outside-universe) |
| 7 | NB12_Recall@50 | count(NB12 binding branches in top 50) / count(NB12 binding branches in universe) |

### 9.2 Tier 2 — Monitoring

VC@20, VC@200, Recall@20, Spearman, Tier0-AP, Tier01-AP, Abs_Binders@50, NB6_Recall@50, NB24_Recall@50, NB_Median_Rank, NB_SP_Capture@50, onpeak-only VC@50, offpeak-only VC@50

### 9.3 Tier 3 — Cohort Contribution

Per cohort (established, history_dormant, history_zero):
- Count of cohort branches in global top-50
- SP from cohort branches in global top-50
- Cohort recall@50 (cohort SP in top-50 / cohort total SP)
- Cohort miss rate (cohort binder branches ranked >100)

All cohort metrics are branch-level.

### 9.4 Gate Checking

```python
def check_gates(
    candidate_metrics: dict,       # per-group metrics
    baseline_metrics: dict,        # per-group metrics for named baseline
    baseline_name: str,            # e.g., 'v0c'
    holdout_groups: list[str],     # ['2025-06/aq1', '2025-06/aq2', '2025-06/aq3']
) -> dict:
    """Returns {metric: {passed: bool, wins: int, mean_delta: float}}"""
```

Compares candidate against one named baseline (usually v0c, the strongest formula baseline) on the same holdout groups.

### 9.5 Reporting Contract

Every experiment reports:
1. Per-group breakdown for all holdout quarters (not just mean)
2. Dev AND holdout numbers
3. NB breakdown: NB6/NB12/NB24 recall and median rank
4. Cohort breakdown
5. Class-type split (combined primary + onpeak/offpeak monitoring)
6. Feature importance (LightGBM gain, normalized %)
7. Train walltime + full experiment walltime

---

## 10. Decisions Resolved

### DR#1: Universe Threshold Calibration (SS5.4)

Run elbow analysis from scratch on Day 1. The SS5.4 tables are inherited planning targets from the old density_signal_score filter, not validated raw-only numbers.

**Calibration is BRANCH-LEVEL**: right_tail_max is computed per cid, then aggregated to max per branch. The 95% SP capture target uses annual-bridge-mapped branch SP as the denominator (not total quarter DA SP). The elbow is chosen by branch rank.

**Application is CID-LEVEL**: `is_active = (right_tail_max >= UNIVERSE_THRESHOLD)` is evaluated per cid in `load_collapsed()`. Branches with ≥1 active cid are kept. This is consistent because `branch_rtm = max(cid_rtm) >= threshold` implies at least one active cid.

Calibration produces:
- Chosen UNIVERSE_THRESHOLD value
- Calibration script output (branch-level threshold sweep)
- Rationale note

Saved to `registry/threshold_calibration/`.

### DR#2: Level-2 Density Second Stat (SS6.2)

Deferred to Phase 2 empirics. Implementation scaffolds all three variants (max+min, max+std, max-only). Phase 2 build-up (Steps 2b -> 2d) resolves it. Keep/drop requires improvement on main blocking metrics (VC@50, Abs_SP@50, NB12_Recall@50) and lift must beat expected noise across dev groups. The >2% heuristic is a guide, not the final decision rule.

### DR#3: Per-ctype NB Monitoring (SS11.1)

Compute per-ctype NB at 12-month window only (nb_onpeak_12, nb_offpeak_12). Monitoring only, not gated. Per-ctype NB uses per-ctype target binding (onpeak_sp > 0 / offpeak_sp > 0).

---

## 11. Phased Implementation Plan

### Phase 1: Foundation (formula baselines + diagnostics)

```
Step 1.0: scripts/fetch_realized_da.py
  - Build realized DA cache: 2017-04 through 2026-02, both ctypes
  - Requires Ray
  - Output: data/realized_da/{YYYY-MM}.parquet, {YYYY-MM}_offpeak.parquet

Step 1.1: ml/config.py + ml/bridge.py + ml/realized_da.py
  - Shared infrastructure: paths, constants, bridge loading, DA cache access
  - Validate bridge loading on one (PY, quarter) slice

Step 1.2: Universe threshold calibration (Day 1 task)
  - scripts/calibrate_threshold.py
  - Calibration at BRANCH level: cid right_tail_max aggregated to max per branch
  - SP denominator: annual-bridge-mapped branch SP (not total quarter DA SP)
  - Sort branches by right_tail_max descending, compute cumulative SP capture
  - Find elbow at 95% SP capture, cross-check against 2023-06/aq1 (within +/-20%)
  - Freeze UNIVERSE_THRESHOLD in config.py (applied at cid level in data_loader)
  - Produce artifact: threshold value, branch-level sweep table, rationale note
  - Save to registry/threshold_calibration/

Step 1.3: ml/data_loader.py
  - Full pipeline: density -> universe filter -> Level 1 -> bridge -> Level 2
  - count_cids / count_active_cids with pre-filter is_active flag
  - Constraint limits with proper Level 1 aggregation
  - Ambiguous bridge cids: detect, log, drop
  - Cache collapsed slices
  - Validate where published reference exists; otherwise log raw diagnostics

Step 1.4: ml/ground_truth.py
  - Combined ctype GT with annual + monthly bridge fallback
  - Continuous SP + tiered labels + per-ctype split targets
  - Raw coverage diagnostics (counts + SP, not just percentages)
  - Validate where published reference exists (SS8.7 table)

Step 1.5: ml/history_features.py
  - Monthly branch-binding table (single DA scan, shared mapping helper)
  - BF family (7 features) with fixed denominators
  - da_rank_value ranked within universe, has_hist_da flag
  - Leakage prevention: cutoff = March of submission year

Step 1.6: ml/nb_detection.py
  - NB6/NB12/NB24 (combined ctype, branch-level)
  - nb_onpeak_12 / nb_offpeak_12 (per-ctype target binding)
  - Reuses monthly binding table from history_features

Step 1.7: ml/features.py
  - Joins all sources into ONE model table
  - Cohort assignment (established / history_dormant / history_zero using has_hist_da)
  - Monotone constraint vector (order matches feature_cols, assertion enforced)
  - Explicit schema contract with assertions

Step 1.8: ml/evaluate.py + ml/registry.py
  - All Tier 1/2/3 metrics with exact denominators
  - NB metrics (NB6/NB12/NB24, per-ctype NB12)
  - Cohort contribution (branch-level)
  - Gate checking against named baseline
  - Registry: write config.json + metrics.json to registry/{version}/

Step 1.9: Formula baselines + baseline contract freeze
  - v0a: pure da_rank_value
  - v0b: 0.60 * da_rank_norm + 0.40 * right_tail_rank_norm
  - v0c: 0.40 * da_rank_norm + 0.30 * right_tail_rank_norm + 0.30 * bf_combined_12_rank_norm
  - Run on dev (12 groups) + holdout (3 groups)
  - Full Tier 1/2/3 + NB + cohort reporting
  - BASELINE CONTRACT FREEZE after formula baselines run:
    - Freeze which baseline is authoritative for promotion (v0c unless results say otherwise)
    - Freeze exact metric names and denominators
    - No drift in gate comparisons after this point
```

### Phase 2: ML Build-Up (incremental feature groups)

```
Step 2a: ML baseline — historical features only (8 features)
  da_rank_value + BF family (bf_6, bf_12, bf_15, bfo_6, bfo_12, bf_combined_6, bf_combined_12)
  Must beat v0c to justify ML

Step 2b: Add core density bins — max only (5 features -> 13)
  bin_80, bin_100, bin_110, bin_120, bin_150 (cid_max)
  Key metric: NB12_Recall@50

Step 2c: Add counter-flow + mid-range (3 features -> 16)
  bin_-100, bin_-50, bin_60 (cid_max)
  Drop if delta < noise (+/-1% VC@50 across dev groups)

Step 2d: Add second Level-2 stat (10 features -> 26)
  Test BOTH 2d-min (cid_min) and 2d-std (cid_std) variants
  Keep/drop: requires improvement on main blocking metrics (VC@50, Abs_SP@50, NB12_Recall@50)
  and lift must beat expected noise across dev groups

Step 2e: Add structural features (6 features -> up to 32)
  limit_min/mean/max/std, count_cids, count_active_cids
  Prune features with < 2% importance

Step 2f: Final pruning
  Drop < 2% importance, deduplicate rho > 0.95 pairs
  Re-train, confirm no degradation -> candidate champion

Step 2g: Density-only model (diagnostic, NOT candidate)
  If density-only NB12_Recall@50 >> full model -> consider dual-model blend
```

### Phase 3: Exploration (evidence-driven, only if Phase 2 shows density signal)

```
- Try remaining bins (70, 90 — adjacent, likely redundant)
- Try std instead of min at Level 2
- Universe expansion (lower right_tail_max thresholds)
- Dual model blend if NB detection warrants it
```

Each step produces a registry entry with full metrics, feature list, params, and walltime. Every step compares against v0c and the previous step.

---

## 12. Traps & Pitfalls (inherited from implementer-guide)

Critical traps that affect implementation (see implementer-guide SS15 for full list):

1. **Trap 1**: Temporal leakage — BF/DA features must use only months <= March of submission year
2. **Trap 3**: LightGBM thread contention — always set num_threads=4
3. **Trap 4/5**: Bridge table partition sensitivity and schema mismatch — use partition-specific paths
4. **Trap 7**: Monotone constraint signs — wrong signs silently degrade, no error
5. **Trap 12**: Memory budget 128 GiB — use polars, lazy scans, gc.collect() between stages
6. **Trap 19**: Ray required for pbase data loaders
7. **Trap 21**: pbase spice loaders broken for annual data — use pl.scan_parquet
8. **Trap 22**: Density bins are NOT probabilities — sum to 20.0, can exceed 1.0, NOT monotone
9. **Trap 23**: GT must be combined onpeak + offpeak
10. **Trap 24**: DA and density cid systems only ~60% overlap — bridge required
