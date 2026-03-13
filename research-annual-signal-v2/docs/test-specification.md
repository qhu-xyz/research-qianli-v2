# Test Specification: Annual Signal v2

**Purpose**: Verify correctness of the branch-level MISO annual FTR constraint ranking pipeline. A fresh implementer should run these tests at each build step to confirm the implementation matches the spec.

**How to use**: Each test has a name, what it checks, setup, action, expected result, and which requirement it validates. Tests are ordered by pipeline stage. Run them sequentially — later tests depend on earlier ones passing.

**Anchor data**: Tests use PY `2024-06/aq1` as the primary verification slice (good bridge coverage, complete GT data). Some tests also reference `2025-06/aq1` (holdout, known outlier for bridge coverage) and `2023-06/aq1`.

**Data availability (verified)**: Annual density distribution exists for all 7 PYs: 2019-06 through 2025-06. Bridge table and constraint limits exist for all 7 PYs. Realized DA cache covers 2017-04 through 2026-02.

---

## A. Data Loading

### A1: Density distribution shape and columns

**Tests**: Raw density distribution loads correctly for a specific (PY, quarter, market_month).
**Setup**: Load density for `auction_type=annual, auction_month=2024-06, market_month=2024-06, market_round=1` using partition-specific path.
**Action**: `pl.read_parquet(f'{DENSITY_PATH}/spice_version=v6/auction_type=annual/auction_month=2024-06/market_month=2024-06/market_round=1/')`
**Expected**:
- DataFrame has column `constraint_id` (String)
- DataFrame has 77 bin columns named: `-300, -280, ..., 280, 300` (exact list in §4.2)
- Row count > 100,000 (multiple outage_dates × ~12,800 cids)
- There is an `outage_date` column (Date type)
- There is NO `flow_direction` column
- There is NO `class_type` column
**Validates**: §4.2 density schema, Trap 22 (bins are not probabilities)

### A2: Density bin values are density weights, not probabilities

**Tests**: Bin values have the correct semantics — sum to 20.0, can exceed 1.0.
**Setup**: Load one partition as in A1.
**Action**: Sum all 77 bin columns per row.
**Expected**:
- Every row sums to exactly `20.0` (within floating point tolerance ±0.01)
- At least 10% of rows have at least one bin value > 1.0
- Values are NOT monotonically decreasing across bins for most rows
**Validates**: §4.2 semantics, Trap 22

### A3: Density distribution — partition-specific path required

**Tests**: Using `pl.scan_parquet()` with hive on the FULL density parquet does NOT produce a `SchemaError` (unlike the bridge table), but partition-specific paths are still preferred for performance.
**Setup**: Attempt both loading methods.
**Action**:
1. `pl.scan_parquet(DENSITY_PATH, hive_partitioning=True).filter(auction_type='annual', auction_month='2024-06', market_month='2024-06').collect()`
2. `pl.read_parquet(f'{DENSITY_PATH}/spice_version=v6/auction_type=annual/auction_month=2024-06/market_month=2024-06/market_round=1/')`
**Expected**: Both produce the same row count and columns. Partition path is faster.
**Validates**: §4.2, Trap 21

### A4: Bridge table loads with convention filter (single partition)

**Tests**: Bridge table loads correctly and convention < 10 filter works.
**Setup**: Load bridge for `auction_type=annual, auction_month=2024-06, period_type=aq1, class_type=onpeak` using partition-specific path. (This tests a SINGLE class_type partition; see A8 for the UNION test.)
**Action**:
```python
bridge = pl.read_parquet(
    f'{BRIDGE_PATH}/spice_version=v6/auction_type=annual/auction_month=2024-06'
    f'/market_round=1/period_type=aq1/class_type=onpeak/'
).filter(
    (pl.col('convention') < 10) & pl.col('branch_name').is_not_null()
).select(['constraint_id', 'branch_name']).unique()
```
**Expected**:
- `constraint_id` column exists (String type)
- `branch_name` column exists (String type)
- Row count: ~14,000 (before convention filter), ~14,000 after (convention filter does NOT change cid count — §8.4)
- `convention` values before filter include: -1, 1, 999
- After filter: only -1 and 1 remain
- No null branch_names after filter
- `bridge['branch_name'].n_unique()` is between 4,000 and 6,000
**Validates**: §8.4, §8.5, Trap 4, Trap 5

### A5: Bridge table schema mismatch — hive scan fails

**Tests**: Using hive scan on the full bridge parquet raises SchemaError.
**Setup**: Attempt `pl.scan_parquet(BRIDGE_PATH, hive_partitioning=True).collect()`
**Expected**: Raises `polars.exceptions.SchemaError` or similar (due to `device_type` column mismatch across partitions).
**Validates**: Trap 5

### A6: Constraint limits load

**Tests**: Constraint limit data loads and has expected schema.
**Setup**: Load limits for `auction_type=annual, auction_month=2024-06` from `MISO_SPICE_CONSTRAINT_LIMIT.parquet`.
**Action**: Scan with hive partitioning and filter, or use partition-specific path.
**Expected**:
- Has columns: `constraint_id`, `limit`
- `limit` values are positive floats (MW)
- Row count: ~14,000 unique constraint_ids
**Validates**: §4.2

### A7: Realized DA shadow prices load — BOTH class types

**Tests**: Realized DA data loads for both onpeak and offpeak.
**Setup**: Build local DA cache via `scripts/fetch_realized_da.py` or load from `data/realized_da/` for July 2024 (aq1 month).
**Action**: Load onpeak (`2024-07.parquet`) and offpeak (`2024-07_offpeak.parquet`) files.
**Expected**:
- Both files exist and are non-empty
- Columns: `constraint_id` (String), `realized_sp` (Float64)
- `realized_sp` values are non-negative (already abs-aggregated per constraint_id)
- There are constraint_ids that appear in offpeak but NOT in onpeak (and vice versa)
**Validates**: §8.3 (combined GT), Trap 23

### A8: Bridge table class_type UNION

**Tests**: Bridge onpeak and offpeak partitions are loaded and UNIONed correctly.
**Setup**: Load bridge for both `class_type=onpeak` and `class_type=offpeak` for `2024-06/aq1`.
**Action**:
```python
on = load_bridge_partition(bridge_path, 'annual', '2024-06', '1', 'aq1')
# Manually load each class_type separately for comparison
on_only = pl.read_parquet(f'{BRIDGE_PATH}/.../class_type=onpeak/')
off_only = pl.read_parquet(f'{BRIDGE_PATH}/.../class_type=offpeak/')
```
**Expected**:
- Union row count >= max(onpeak rows, offpeak rows)
- For 2024-06/aq1: onpeak and offpeak have IDENTICAL constraint_id sets (no exclusive cids)
- For 2024-06/aq1: zero cids with divergent branch_name mappings
- For 2021-06/aq4 (outlier): 2 divergent cids (227076, 331594)
- `load_bridge_partition()` raises `FileNotFoundError` if NEITHER class_type partition exists
- `load_bridge_partition()` logs a warning if only one class_type is found
**Validates**: §8.5 (bridge UNION rule), verified 2026-03-12

---

## B. Universe Filter

### B1: right_tail_max computation

**Tests**: Universe filter is computed correctly from raw bins.
**Setup**: Load Level 1 collapsed density (mean across outage_dates per cid) for 2024-06/aq1.
**Action**: `right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` per cid, taking max across outage_dates.
**Expected**:
- `right_tail_max` is a single float per cid
- For cids with all four bins = 0 across all outage_dates: `right_tail_max = 0`
- For cids with bin_110 = 0.5 on one date and 0 on others: `right_tail_max >= 0.5` (max across dates)
- `right_tail_max` can exceed 1.0 (bins are density weights, not probabilities)
**Validates**: §5.4 universe filter

### B2: Universe filter produces expected sizes

**Tests**: Applying the threshold produces universe sizes consistent with §5.4 tables.
**Setup**: Compute right_tail_max for all cids in 2024-06/aq1.
**Action**: Apply threshold, count filtered cids and resulting branches.
**Expected**:
- Filtered cid count should be ~4,343 (from §5.4 table)
- After bridge join + collapse to branch_name: ~1,712 branches
- cid:branch ratio ~2.54
- If counts deviate by more than ±10%, the threshold needs recalibration
**Validates**: §5.4

### B3: Universe filter uses max across outage_dates, not mean

**Tests**: A cid with intermittent signal (high right_tail_max on 1 date, zero on 10 dates) passes the filter.
**Setup**: Find a cid where right_tail_max is > threshold for 1 outage_date but mean < threshold.
**Action**: Apply filter with max-across-dates rule vs mean-across-dates rule.
**Expected**:
- The cid passes with max rule
- The cid FAILS with mean rule
- This confirms the filter is using max, not mean (§5.4 filter scope note)
**Validates**: §5.4 filter scope

### B4: Raw cid counts per PY

**Tests**: Total raw density cids match expected numbers.
**Setup**: Load density for each quarter of 2025-06.
**Action**: Count unique constraint_ids per quarter.
**Expected**:
- aq1: ~12,876
- aq2: ~12,833
- aq3: ~12,931
- aq4: ~12,936
- (±100 tolerance)
**Validates**: §5.4 table

---

## C. Two-Level Collapse

### C1: Level 1 — mean across outage_dates per cid

**Tests**: Level 1 collapse produces one row per cid with mean values.
**Setup**: Load density for 2024-06/aq1 (3 market_months × ~11 outage_dates each).
**Action**: Group by `constraint_id`, compute `mean` of each bin column across all rows.
**Expected**:
- Output has exactly 1 row per constraint_id
- Number of unique cids matches raw count (~12,876 for 2025-06/aq1)
- For a cid with constant bin values across dates: mean = that constant value
- For a cid with varying values: mean is between min and max of the date-level values
**Validates**: §6.2 Level 1

### C2: Bridge join does NOT fan out

**Tests**: Joining cid-level data with bridge (convention < 10) produces at most ~1.01× the input rows.
**Setup**: Level 1 collapsed density for 2024-06/aq1 + bridge with convention < 10.
**Action**: Inner join on `constraint_id`.
**Expected**:
- Output row count ≤ 1.05 × input row count (near 1:1 after convention filter)
- If row count > 1.1× input: convention filter is missing or wrong
- Each constraint_id maps to exactly 1 branch_name (after convention < 10 + unique)
**Validates**: §8.4, §6.2 prerequisite

### C3: Level 2 — max/min across cids per branch

**Tests**: Level 2 collapse produces one row per branch with correct aggregation stats.
**Setup**: Joined cid-level data from C2.
**Action**: Group by `branch_name`, compute max and min per bin column.
**Expected**:
- Output has exactly 1 row per branch_name
- For single-cid branches: max = min = the cid's value
- For multi-cid branches: max >= min for every bin column
- `count_cids` = number of cids mapped to that branch (should be ≥ 1)
- Branch count for 2024-06/aq1: ~1,712
**Validates**: §6.2 Level 2

### C4: count_cids and count_active_cids

**Tests**: Branch metadata features are computed correctly.
**Setup**: Level 2 collapse output for 2024-06/aq1.
**Action**: Check `count_cids` and `count_active_cids` columns.
**Expected**:
- `count_cids` ≥ 1 for every branch (no branch with 0 cids)
- `count_active_cids` ≤ `count_cids` for every branch
- `count_active_cids` ≥ 1 for every branch (all branches in universe passed the filter, so at least 1 cid was active)
- `count_cids.mean()` ≈ 2.5 (matches cid:branch ratio from §5.4)
- Max `count_cids` ≈ 51 (the branch "EQIN-HAM-4 A" per §5.4b)
**Validates**: §6.2, §7.1

### C5: Constraint limit Level 2 collapse

**Tests**: Limit features produce 4 stats per branch.
**Setup**: Limit data joined with bridge, grouped by branch.
**Action**: Compute min, mean, max, std of `limit` per branch.
**Expected**:
- `limit_min` ≤ `limit_mean` ≤ `limit_max` for every branch
- For single-cid branches: `limit_min = limit_mean = limit_max`, `limit_std` = NaN or 0
- `limit_std` ≥ 0 for all multi-cid branches
- All limit values are positive
**Validates**: §6.2, §7.1

### C6: Quarter-to-market-month mapping

**Tests**: Market months are correctly derived from (PY, quarter).
**Action**: Verify mapping function.
**Expected**:
- `2025-06/aq1` → `['2025-06', '2025-07', '2025-08']`
- `2025-06/aq2` → `['2025-09', '2025-10', '2025-11']`
- `2025-06/aq3` → `['2025-12', '2026-01', '2026-02']` (year rollover!)
- `2025-06/aq4` → `['2026-03', '2026-04', '2026-05']` (year rollover!)
- `2024-06/aq3` → `['2024-12', '2025-01', '2025-02']`
**Validates**: §6.4

---

## D. Feature Construction

### D1: All 34 features are present

**Tests**: Final feature DataFrame has exactly the expected columns.
**Setup**: Fully collapsed branch-level DataFrame for one (PY, quarter).
**Action**: Check column names.
**Expected 34 features**:
```
# Density bins (20): 10 bins × 2 stats
bin_-100_cid_max, bin_-100_cid_min,
bin_-50_cid_max, bin_-50_cid_min,
bin_60_cid_max, bin_60_cid_min,
bin_70_cid_max, bin_70_cid_min,
bin_80_cid_max, bin_80_cid_min,
bin_90_cid_max, bin_90_cid_min,
bin_100_cid_max, bin_100_cid_min,
bin_110_cid_max, bin_110_cid_min,
bin_120_cid_max, bin_120_cid_min,
bin_150_cid_max, bin_150_cid_min,

# Constraint limit (4)
limit_min, limit_mean, limit_max, limit_std,

# Branch metadata (2)
count_cids, count_active_cids,

# Historical DA (1)
da_rank_value,

# Onpeak BF (3)
bf_6, bf_12, bf_15,

# Offpeak BF (2)
bfo_6, bfo_12,

# Combined BF (2)
bf_combined_6, bf_combined_12,
```
**Validates**: §7.1

### D2: Density bin selection — exactly 10 bins

**Tests**: Only the 10 selected bins are used, not all 77.
**Setup**: Level 2 collapse output.
**Action**: Count density feature columns.
**Expected bins**: `-100, -50, 60, 70, 80, 90, 100, 110, 120, 150`
**NOT included**: `-10, 0, 20, 40` (negligible/inverted), `85, 95, 105` (redundant), `200, 300` (too sparse)
**Validates**: §7.2 bin selection

### D3: BF features — correct windows and backfill

**Tests**: BF features use correct lookback windows and backfill from 2017-04.
**Setup**: Compute BF for 2024-06 (annual submission ~April 2024).
**Action**: Check bf_12 for a branch known to have bound in last 12 months.
**Expected**:
- `bf_12` = (months with onpeak binding in last 12) / 12
- BF uses only months through **March** of submission year (April 2024 submission → months ≤ 2024-03)
- `bf_12` for 2024-06 uses months 2023-04 through 2024-03
- `bf_15` for 2024-06 uses months 2023-01 through 2024-03
- With backfill to 2017-04: for 2019-06 eval, bf_15 uses months 2018-01 through 2019-03 (all available)
- All BF values are in [0, 1]
**Validates**: §7.4, Trap 1 (temporal leakage)

### D4: BF temporal leakage check

**Tests**: BF does NOT use April data for April-submission signals.
**Setup**: For PY 2024-06 (submitted April 2024).
**Action**: Verify the maximum month used in BF computation.
**Expected**:
- Last month used: `2024-03` (March)
- `2024-04` is NOT used (even though partial data may exist)
- If any BF value changes when April data is included, this test FAILS
**Validates**: Trap 1

### D5: Combined BF — either ctype logic

**Tests**: `bf_combined_N` counts months where the branch bound in EITHER onpeak OR offpeak.
**Setup**: Find a branch with bfo_12 > 0 but bf_12 = 0 (binds offpeak only).
**Action**: Check bf_combined_12 for this branch.
**Expected**:
- `bf_combined_12 >= bfo_12` (strictly, since combined includes both)
- `bf_combined_12 > bf_12` (combined picks up offpeak months)
- For a branch with bf_12 = 0, bfo_12 = 0.5: bf_combined_12 >= 0.5
**Validates**: §7.5b

### D6: da_rank_value computation

**Tests**: da_rank_value is the rank of historical DA SP per branch.
**Setup**: Compute da_rank_value for 2024-06.
**Action**: Rank branches by cumulative `sum(abs(SP))` from realized DA history.
**Expected**:
- `da_rank_value` = 1 for the branch with the highest total historical SP
- Lower rank value = more binding (rank 1 is most binding)
- Uses combined onpeak + offpeak DA
- Lookback window ends at March of submission year (same as BF)
- All values are positive integers, no ties expected for most branches
**Validates**: §7.3, Trap 1

### D7: Monotone constraint signs

**Tests**: Monotone constraints are correctly specified for LightGBM.
**Setup**: Build the monotone_constraints vector.
**Action**: Verify sign per feature.
**Expected**:
- BF features (bf_6, bf_12, bf_15, bfo_6, bfo_12, bf_combined_6, bf_combined_12): **+1**
- da_rank_value: **-1** (lower rank = more binding)
- Density bin features: **0** (unconstrained)
- Limit features: **0** (unconstrained)
- count_cids, count_active_cids: **0** (unconstrained)
**Validates**: §7.6, Trap 7

---

## E. Ground Truth Pipeline

### E1: Combined GT — both ctypes loaded

**Tests**: Ground truth loads both onpeak and offpeak DA data.
**Setup**: Load GT for 2024-06/aq1 (months: Jun, Jul, Aug 2024).
**Action**: Load onpeak and offpeak separately, then combine.
**Expected**:
- `onpeak_cids = set(onpeak['constraint_id'])`
- `offpeak_cids = set(offpeak['constraint_id'])`
- `offpeak_only = offpeak_cids - onpeak_cids` — this set is non-empty
- Combined `sum(abs(SP))` > onpeak-only `sum(abs(SP))`
- If combined == onpeak-only: offpeak was not loaded (BUG)
**Validates**: §8.3, Trap 23

### E2: Bridge mapping with convention < 10

**Tests**: DA cids map to branch_names via annual bridge with correct filter.
**Setup**: DA constraint_ids for 2024-06/aq1 + annual bridge.
**Action**: LEFT JOIN DA cids to bridge.
**Expected**:
- Mapped percentage: ~95-99% of DA cids (for 2022-2024 training years)
- For 2025-06/aq1: only ~62% mapped (known outlier — §8.2)
- No fan-out: each DA cid maps to at most 1 branch_name
- Track unmapped count and SP
**Validates**: §8.2, §8.4

### E3: Monthly bridge fallback

**Tests**: Unmapped DA cids are recovered via monthly bridge tables.
**Setup**: DA cids unmapped after annual bridge for 2025-06/aq1.
**Action**: Try monthly bridges for 2025-06, 2025-07, 2025-08 via `load_bridge_partition(bridge_path, 'monthly', mm, '1', 'f0')` — loads BOTH class types and UNIONs them.
**Expected**:
- For 2025-06/aq1: recovers ~123 of 208 unmapped cids (+13.4% SP)
- For 2024-06/aq1: recovers very few (annual bridge already covers ~98.7%)
- Monthly-mapped cids are NOT in the annual bridge (no double-counting)
- Combined coverage for 2025-06/aq1: ~84% of DA cids, ~89.2% of SP
**Validates**: §8.3 Steps 2-3, §8.7

### E4: Target aggregation — sum(abs(SP)) per branch

**Tests**: Target is sum of absolute shadow prices per branch.
**Setup**: Mapped DA data for 2024-06/aq1.
**Action**: Group by branch_name, sum `abs(shadow_price)`.
**Expected**:
- All target values >= 0 (absolute values summed)
- Multiple DA cids can map to same branch → their SPs are summed (NOT averaged)
- A branch with 3 DA cids (SP = 100, -50, 200) gets target = 100 + 50 + 200 = 350
- Branches not in DA get target = 0.0
**Validates**: §6.5, §8.3 Step 4

### E5: Tiered labels (0/1/2/3)

**Tests**: Tiered label assignment uses per-group tertile boundaries.
**Setup**: Target values for one (PY, quarter) group.
**Action**: Assign tier labels.
**Expected**:
- Label 0 = target == 0 (non-binding)
- Labels 1/2/3 = tertiles of the POSITIVE targets only
- Tertile boundaries computed per (PY, quarter) — NOT globally
- Count of label 0 > count of labels 1+2+3 (majority are non-binding)
- Count of label 1 ≈ count of label 2 ≈ count of label 3 (by tertile definition)
- For 2024-06/aq1: binding rate ~14-16% → ~85% label 0
**Validates**: §8.6

### E6: NB detection — combined ctype check

**Tests**: NB12 checks binding in BOTH onpeak and offpeak for the lookback window.
**Setup**: For 2024-06/aq1, check NB12 for a branch that binds offpeak but not onpeak in lookback.
**Action**: Compute is_nb_12 for this branch.
**Expected**:
- A branch with bfo_12 > 0 (offpeak binding in lookback) is NOT NB12
- A branch with bf_12 = 0 AND bfo_12 = 0 (no binding in either ctype) IS NB12 eligible
- NB12 is confirmed only if it ALSO binds in the target quarter's combined GT
**Validates**: §11.1, Trap 23

---

## F. Training Pipeline

### F1: Query group construction

**Tests**: Each (PY, quarter) forms one query group for LambdaRank.
**Setup**: Training DataFrame with `planning_year` and `aq_quarter` columns.
**Action**: Compute group sizes.
**Expected**:
- Number of groups = number of unique (PY, quarter) combinations in training set
- For 2025-06 holdout training (2019-2024): 24 groups (6 PYs × 4 quarters)
- Each group has ~1,100-1,850 rows
- `sum(group_sizes) == len(training_df)`
**Validates**: §9.4

### F2: Expanding window split

**Tests**: Train/eval year assignment is correct — no future data leakage.
**Setup**: Build splits per §9.3.
**Action**: Verify for eval year 2025-06.
**Expected**:
- Training PYs: 2019-06, 2020-06, 2021-06, 2022-06, 2023-06, 2024-06
- Eval PYs: 2025-06 only
- No 2025-06 data in training set
- Training has 24 groups (6 years × 4 quarters)
- Eval has 3 groups (aq1, aq2, aq3 only — aq4 incomplete)
**Validates**: §9.3

### F3: LightGBM parameters

**Tests**: Model uses correct parameters.
**Setup**: Build LightGBM model.
**Action**: Check params dict.
**Expected**:
- `objective` = `"lambdarank"`
- `metric` = `"ndcg"`
- `num_threads` = 4 (NOT auto-detected 64)
- `n_estimators` = 200
- `learning_rate` = 0.03
- `num_leaves` = 31
- NOT regression, NOT XGBoost
**Validates**: §9.2, Trap 3

### F4: Training completes in reasonable time

**Tests**: Training does not hang due to thread contention.
**Setup**: Train model on dev data (2019-2023 training, 2024 eval).
**Action**: Time the training.
**Expected**:
- Training completes in < 10 seconds per fold (not 57+ seconds)
- If > 30 seconds: num_threads is probably wrong (Trap 3)
**Validates**: Trap 3

---

## G. Evaluation

### G1: VC@K computation

**Tests**: Value Captured at K is computed correctly.
**Setup**: Predicted scores and actual targets for one (PY, quarter) group.
**Action**: Compute VC@50.
**Expected**:
- `VC@50 = sum(target[top_50_by_score]) / sum(target[all_branches_in_universe])`
- Denominator is total binding SP **within universe** (not all DA SP)
- VC@50 ∈ [0, 1]
- If model is perfect: VC@50 approaches the maximum possible given 50 picks
- If model is random: VC@50 ≈ 50 / n_branches (roughly)
**Validates**: §10.1

### G2: Recall@K computation

**Tests**: Recall at K counts binding branches correctly.
**Setup**: Same as G1.
**Action**: Compute Recall@50.
**Expected**:
- `Recall@50 = count(top_50 with target > 0) / count(all branches with target > 0)`
- Recall@50 ∈ [0, 1]
- Denominator = total binding branches in universe
**Validates**: §10.1

### G3: Abs_SP@K computation — cross-universe denominator

**Tests**: Abs_SP uses ALL DA binding SP, not just within-universe SP.
**Setup**: Same as G1, plus total DA binding SP for the quarter.
**Action**: Compute Abs_SP@50.
**Expected**:
- `Abs_SP@50 = sum(target[top_50_by_score]) / total_da_binding_sp_all_constraints`
- Denominator includes DA binding SP from constraints NOT in our universe
- `Abs_SP@50 <= VC@50` (because denominator is larger)
- `Abs_SP@50` enables comparison across different universe sizes
**Validates**: §10.1, §10.6

### G4: NB12_Recall@K computation

**Tests**: NB recall measures detection of new binding constraints.
**Setup**: Same as G1, plus NB12 labels.
**Action**: Compute NB12_Recall@50.
**Expected**:
- `NB12_Recall@50 = count(NB12 binders in top_50) / count(all NB12 binders)`
- Only counts branches that are BOTH NB12 AND actually bound in target quarter
- NB12 uses combined ctype check (no binding in EITHER onpeak or offpeak for 12 months)
**Validates**: §10.1, §11.1

### G5: Cohort assignment

**Tests**: Every branch is assigned to exactly one cohort.
**Setup**: Feature DataFrame with has_hist_da and bf_combined_12.
**Action**: Assign cohorts.
**Expected**:
- **history-zero**: `has_hist_da = False AND bf_combined_12 = 0`
- **history-dormant**: `has_hist_da = True AND bf_combined_12 = 0`
- **established**: `bf_combined_12 > 0`
- Priority: established > history-dormant > history-zero (non-overlapping)
- Every branch gets exactly 1 cohort label
- `count(established) + count(history-dormant) + count(history-zero) == total_branches`
**Validates**: §11.2

### G6: Per-group reporting

**Tests**: Metrics are reported per (PY, quarter) group, not just mean.
**Setup**: Full evaluation run.
**Action**: Check output format.
**Expected**:
- Each holdout group (aq1, aq2, aq3) has its own row of metrics
- Mean across groups is also reported
- Min/max/spread across groups is reported
- NB metrics are reported per group
- Class-type split (onpeak-only, offpeak-only) is reported as monitoring
**Validates**: §10.7

---

## H. Formula Baselines

### H1: v0a — pure da_rank_value

**Tests**: v0a formula ranks by historical DA shadow price rank alone.
**Setup**: da_rank_value computed for dev + holdout.
**Action**: Score = -da_rank_value (lower rank = higher score), evaluate.
**Expected**:
- Produces valid VC@50, Recall@50, NDCG for all groups
- All scores are distinct (no ties for most branches)
- VC@50 > 0 (model is not random)
- This is the floor baseline — all other versions must beat it
**Validates**: §12 Phase 1

### H2: v0b — da_rank + density blend

**Tests**: v0b formula combines da_rank with right-tail density.
**Setup**: Compute `right_tail_rank_norm = rank(mean(bin_80 + bin_90 + bin_100 + bin_110))` per branch, normalize to [0,1].
**Action**: `score = 0.60 * da_rank_norm + 0.40 * right_tail_rank_norm`, evaluate.
**Expected**:
- Both terms are normalized to [0, 1] within each (PY, quarter) group
- Score is in [0, 1]
- v0b should beat v0a on at least some dev groups (density adds signal for NB constraints)
**Validates**: §12 Phase 1

### H3: v0c — da_rank + density + bf blend

**Tests**: v0c formula adds BF on top of v0b.
**Action**: `score = 0.40 * da_rank_norm + 0.30 * density_term + 0.30 * bf_combined_12_rank_norm`
**Expected**:
- v0c should beat v0a and v0b on most dev groups (BF is the #1 feature family)
- This is the primary formula baseline ML must beat
**Validates**: §12 Phase 1

---

## I. Integration / End-to-End

### I1: Full pipeline on one slice

**Tests**: End-to-end pipeline produces valid predictions for one (PY, quarter).
**Setup**: Run full pipeline for 2024-06/aq1 (dev slice).
**Action**: Load density → collapse → features → GT → formula baselines → evaluate.
**Expected**:
- Branch count: ~1,712
- Feature count: 34
- All features are non-null (no missing values except limit_std for single-cid branches)
- Target has > 0 binding branches
- VC@50 > 0 for v0a, v0b, v0c
- NDCG > 0.5 (better than random)
- Walltime < 5 minutes (no runaway memory or thread contention)
**Validates**: Full pipeline

### I2: ML Step 2a — historical features beat formula

**Tests**: ML with historical features outperforms v0c formula.
**Setup**: Train LightGBM with 8 features (da_rank_value + 7 BF) on dev data.
**Action**: Evaluate on dev groups.
**Expected**:
- ML VC@50 > v0c VC@50 on majority of dev groups
- If ML does NOT beat formula: the ML framework has a problem
**Validates**: §12 Phase 2, Step 2a

### I3: ML Step 2b — density adds NB signal

**Tests**: Adding density features improves NB detection.
**Setup**: Train with 13 features (8 historical + 5 core density max).
**Action**: Compare NB12_Recall@50 vs Step 2a.
**Expected**:
- NB12_Recall@50 improves (density helps find NB constraints)
- If no improvement: density is not earning its place
**Validates**: §12 Phase 2, Step 2b

---

## J. Edge Cases

### J1: Empty density data for a market_month

**Tests**: Pipeline handles missing monthly density gracefully.
**Setup**: Simulate by passing a market_month that doesn't exist.
**Action**: Load density for non-existent month.
**Expected**:
- Returns empty DataFrame (not an error)
- The quarter still processes with remaining months
- If ALL months are missing: returns empty, skips this (PY, quarter) — logged as warning
**Validates**: Robustness

### J2: Single-cid branches

**Tests**: Level 2 collapse handles branches with only 1 constraint_id.
**Setup**: Filter to branches where count_cids = 1.
**Action**: Check Level 2 stats.
**Expected**:
- `cid_max = cid_min` for every bin (only 1 value to aggregate)
- `limit_min = limit_max = limit_mean` (single value)
- `limit_std` is NaN or 0 (single value has no spread)
- `count_cids = 1`, `count_active_cids = 1`
**Validates**: §6.2 edge case

### J3: Branch with all density zeros

**Tests**: A branch where all density bins are zero for all cids.
**Setup**: This shouldn't happen after universe filter (right_tail_max > 0 implies at least one bin > 0), but test defensively.
**Action**: Check that universe filter excludes such branches.
**Expected**:
- After filter: no branch has all-zero density features
- right_tail_max = 0 → cid is excluded → branch is excluded if ALL cids have right_tail_max = 0
**Validates**: §5.4 universe filter

### J4: New planning year — all BF = 0

**Tests**: For PY 2019-06 with backfill to 2017-04, some branches may have zero history.
**Setup**: Load BF for a branch that first appears in 2019 (no DA history before).
**Action**: Check BF values.
**Expected**:
- bf_6 = bf_12 = bf_15 = bfo_6 = bfo_12 = bf_combined_6 = bf_combined_12 = 0
- da_rank_value = n_positive + 1 (sentinel rank for zero-history branches — §7.3)
- has_hist_da = False
- The branch falls into "history-zero" cohort (has_hist_da = False AND bf_combined_12 = 0)
- Model still makes a prediction (density features provide signal)
**Validates**: §11.2, NB detection value proposition

### J5: Zero binding in a quarter

**Tests**: Pipeline handles a quarter where no constraints bind in DA.
**Setup**: Unlikely but possible for some historical quarter.
**Action**: Compute GT and labels.
**Expected**:
- All targets = 0
- All labels = 0
- Tiered labeling handles this (no positive targets → no tertile computation needed)
- Evaluation metrics degenerate gracefully (VC@50 = 0/0 → handle as 0 or skip)
**Validates**: Robustness

---

## K. Trap Verification Tests

One test per critical documented trap from §15.

### K1: Trap 1 — Temporal leakage

**Tests**: BF and da_rank_value do NOT use data beyond March of submission year.
**Action**: For PY 2025-06 (submitted April 2025), verify:
- BF uses months ≤ 2025-03
- da_rank_value lookback ends at 2025-03
- No April 2025+ data is used anywhere
**If violated**: Metrics are inflated by 6-20%

### K2: Trap 3 — LightGBM thread contention

**Tests**: `num_threads=4` is set in all LightGBM params.
**Action**: Grep all LightGBM param dicts in code.
**Expected**: Every dict has `"num_threads": 4`
**If violated**: Training takes 57s instead of 0.1s

### K3: Trap 4 — Bridge partition sensitivity

**Tests**: Bridge table is filtered on all 4 partition levels.
**Action**: Verify partition-specific path includes: `auction_type`, `auction_month`, `period_type`, `class_type`.
**If violated**: Incorrect many-to-many mappings

### K3b: Bridge class_type UNION

**Tests**: Bridge loading UNIONs BOTH class types (not onpeak-only).
**Action**: Grep bridge loading code. Must load BOTH `class_type=onpeak` and `class_type=offpeak` and UNION.
**Expected**: `load_bridge_partition()` iterates over both class types. No hardcoded `class_type='onpeak'`.
**If violated**: Loses 0-21 class-type-exclusive cids per slice and up to 0.87% of binding SP.

### K4: Trap 5 — Bridge schema mismatch

**Tests**: Bridge table is loaded via partition-specific paths, NOT hive scan.
**Action**: Grep code for `pl.scan_parquet.*CONSTRAINT_INFO` — should not exist.
**If violated**: SchemaError from device_type column mismatch

### K5: Trap 22 — Density bins NOT probabilities

**Tests**: No code assumes bin values are in [0, 1] or monotonically decreasing.
**Action**: Verify no `assert val <= 1.0` or monotone assumptions on raw bins.
**Expected**: Bins can be 5.83, can be non-monotonic

### K6: Trap 23 — Combined GT

**Tests**: Ground truth loads BOTH onpeak and offpeak DA.
**Action**: Grep GT loading code for both `peak_type='onpeak'` AND `peak_type='offpeak'`.
**Expected**: Both loads present. If only onpeak: BUG.

### K7: Trap 24 — Partial cid overlap

**Tests**: Pipeline uses bridge mapping (not direct cid matching) for GT.
**Action**: Verify GT pipeline joins DA cids to bridge table, not directly to density cids.
**If violated**: ~40% of binding DA cids are missed

### K8: Trap 26 — Two-level collapse

**Tests**: Output is branch-level (not cid-level).
**Action**: Check that training DataFrame has unique `branch_name` per (PY, quarter) — no duplicates.
**Expected**: `df.group_by(['branch_name', 'planning_year', 'aq_quarter']).len().filter(pl.col('len') > 1)` returns 0 rows.
**If violated**: Training overweight, inflated metrics

---

## L. Gate Verification

### L1: Blocking gate rule — 2 of 3 holdout groups

**Tests**: Gate logic correctly requires 2/3 holdout groups to pass.
**Setup**: Mock results where candidate beats baseline on 2 of 3 groups.
**Action**: Apply gate rule.
**Expected**:
- 2 of 3 groups pass + mean >= baseline → PASS
- 1 of 3 groups pass → FAIL (even if mean is higher)
- All 3 groups pass but mean < baseline → FAIL
**Validates**: §10.2

### L2: Seven blocking metrics checked

**Tests**: All 7 Tier 1 metrics are checked in gates.
**Action**: Verify gate code checks: VC@50, VC@100, Recall@50, Recall@100, NDCG, Abs_SP@50, NB12_Recall@50.
**Expected**: All 7 are present. Missing any one is a BUG.
**Validates**: §10.2

---

## Appendix: Verification Anchors

These are concrete numbers from the spec that serve as ground truth for verification.

| Check | Expected Value | Source |
|-------|:---:|---|
| 2025-06/aq1 raw density cids | ~12,876 | §5.4 |
| 2025-06/aq1 branches after collapse | ~1,527 | §5.4 |
| 2024-06/aq1 branches after collapse | ~1,712 | §5.4 |
| 2023-06/aq1 branches after collapse | ~1,852 | §5.4 |
| cid:branch ratio | ~2.5 | §5.4 |
| Max cids per branch | ~51 | §5.4b |
| Bridge convention values | -1, 1, 999 | §8.4 |
| Convention < 10 keeps: | -1 and 1 | §8.4 |
| 2025-06/aq1 annual bridge SP coverage | 75.7% | §8.2 |
| 2025-06/aq1 + monthly fallback SP coverage | 89.2% | §8.2 |
| 2024-06/aq1 annual bridge SP coverage | 98.7% | §8.7 |
| Density row sum across 77 bins | 20.0 | §4.2 |
| Feature count | 34 | §7.1 |
| LightGBM num_threads | 4 | §9.2, Trap 3 |
| Dev groups | 12 (2022-2024 × 4q) | §9.3 |
| Holdout groups | 3 (2025-06 aq1-3) | §9.3 |
| Blocking gate metrics | 7 | §10.2 |
| Gate rule | 2/3 groups + mean >= baseline | §10.2 |
| Tiered labels | 0, 1, 2, 3 | §8.6 |
| BF backfill start | 2017-04 | §7.4 |
| Bins selected | 10 | §7.2 |
| Level 2 density stats | 2 (max + min) | §6.2 |
