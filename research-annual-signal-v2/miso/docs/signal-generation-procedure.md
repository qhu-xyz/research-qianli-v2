# SPICE Signal Generation Procedure — Reference for Annual Signal Writer

> **Purpose:** Document the complete SPICE V6.2B signal generation pipeline so that the annual signal writer can be reviewed, designed, and built independently. Covers: what the production code does, what pmodel expects, and what our annual signal must produce.

---

## 1. Overview

The SPICE signal is a **constraint ranking** consumed by pmodel's optimizer. It tells the optimizer:
- **Which constraints** to include in the optimization (the constraint universe)
- **How important** each constraint is (the tier: 0=most important, 4=least)
- **What direction** each constraint binds (flow_direction / shadow_sign)
- **How each constraint affects each pricing node** (shift factors)

The signal is a pair of parquet files:
1. **Constraint signal** — one row per constraint, with ranking, tier, metadata
2. **Shift factor signal** — matrix of pnode × constraint, with SF values

Production code: `psignal/src/psignal/spice/signal/base.py` → `BaseSpiceSignal.get_signal()`

---

## 2. Production Pipeline (V6.2B `get_signal()`)

### Step 1: Load & Aggregate Density Scores

```
Input:  density score parquets per (auction_month, market_month, outage_date)
Output: DataFrame indexed by (constraint_id, flow_direction), columns = outage dates
```

- For each market_month in the delivery period, for each outage_date (every 3 days):
  - Load density score parquet
  - Extract `score` column indexed by `(constraint_id, flow_direction)`
- Concatenate into a wide DataFrame (rows = constraints, cols = outage dates)
- **Filter:** keep only rows where any score ≥ `score_threshold` (0.03)

### Step 2: Load Network Branches & Constraint Info

```
Input:  branch network data, constraint_info table
Output: constraint_id → branch_name mapping, branch_name → bus_key mapping
```

- Load `branches` table for this auction_month/period_type (fallback to f0 if missing)
- Compute `bus_key = "{from_number},{to_number}"` for each branch
- Load `constraint_info` for this auction_month/period_type/class_type="onpeak"
- Filter out convention=999 and null branch_name
- Group by (constraint_id, type) → deduplicate branch_names (comma-join if multiple)
- Map each constraint → branch_name → bus_key

### Step 3: Compute bus_key_group (Connected Components)

```
Input:  bus_key strings (e.g. "12345,67890")
Output: bus_key_group label per constraint
```

- Extract all unique bus numbers from bus_key strings
- Run `group_pair_components()` — a **union-find** algorithm that:
  - Creates a node per bus number
  - For each bus_key pair (from, to), unions the two nodes
  - Returns connected components: component_root → list of bus_key strings
- Invert the mapping: bus_key → component_root
- Each constraint gets `bus_key_group = component_root` (or branch_name if bus_key is null)

**Purpose:** Constraints sharing any bus belong to the same equipment group. The SF diversity filter (Step 7) operates per group to avoid redundant constraints.

### Step 4: Compute Ranking Metrics

```
Input:  score DataFrame with branch_name, bus_key, bus_key_group
Output: ori_mean, mix_mean per constraint
```

- Per constraint:
  - `ori_mean` = mean of raw scores across outage dates
  - `mean_branch_max` = mean of per-branch max scores (NaN-aware)
  - `mean_branch_max_fillna` = same but treating NaN as 0
  - `mix_mean = 0.6 * mean_branch_max + 0.4 * mean_branch_max_fillna`
- Sort by (mix_mean, ori_mean) descending
- **MISO only:** Inject `SO_MW_Transfer` constraint at rank #1 (hard-coded)

### Step 5: Load & Aggregate Shift Factors

```
Input:  SF parquets per (auction_month, market_month, outage_date)
Output: pnode × constraint matrix (averaged across outage dates)
```

- For each market_month in delivery period, for each outage_date (every 3 days):
  - Load SF parquet (columns = constraint_ids, rows = pnodes)
  - Sum into running total
- Divide by count to get average SF
- **Filter constraints:** keep only those present in both SF columns AND the scored constraint list
- **Index alignment:** `test.index = constraint_id + "|" + str(-flow_direction)`, SF columns match

### Step 6: Load Historical DA Shadow Prices

```
Input:  realized DA binding data from past months
Output: shadow_price_da per branch (cumulative absolute SP)
```

- Cutoff = auction_month - 1 month (with day cutoff at run_at_day)
- Build month list: last 3 calendar months + seasonal months (same season, 4 years back)
- For each month:
  - Load DA shadow prices filtered by class_type (onpeak/offpeak)
  - If cutoff month: truncate to days ≤ run_at_day
- Map DA constraint_id → branch_name via constraint_info
- Aggregate: `shadow_price_da = sum(abs(shadow_price)) / n_months` per branch
- Map back to constraint level via branch_name
- **MISO only:** `SO_MW_Transfer` gets shadow_price_da = 100,000

### Step 7: V6.2B Formula Ranking (Pre-Filter)

```python
density_mix_rank_value  = 1 - MinMaxScaler(0,1).fit_transform(log1p(mix_mean))
density_ori_rank_value  = 1 - MinMaxScaler(0,1).fit_transform(log1p(ori_mean))
da_rank_value           = 1 - MinMaxScaler(0,1).fit_transform(log1p(shadow_price_da))

rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

- Lower rank_ori = more binding (all three sub-values: lower = more binding)
- Sort ascending by rank_ori

### Step 8: SF Diversity Filter

```
Input:  pre-ranked constraints, SF matrix, bus_key_group
Output: filtered constraint list (deduplicated)
```

For each `bus_key_group`, iterate through constraints in rank order:
1. Always keep the top-ranked constraint
2. For each subsequent constraint `idx`:
   - Count how many constraints with the **same branch_name** are already selected
   - **If ≥ 3 with same branch_name:** skip (max 3 per branch per group)
   - Compute **Chebyshev distance** between `idx`'s SF vector and all selected constraints' SF vectors
   - Compute **Pearson correlation** between `idx`'s SF vector and all selected constraints' SF vectors
   - **Keep only if ALL of:**
     - Chebyshev distance ≥ 0.05 to every selected constraint
     - Correlation ≥ -0.21 to every selected constraint
     - Correlation ≥ 0.1 to every selected same-branch constraint
3. Collect surviving constraint indices across all groups

### Step 9: Post-Filter Ranking & Tier Assignment

After the diversity filter, **re-compute** ranking values on the surviving set:

```python
# Re-fit MinMaxScaler on filtered set only
density_mix_rank_value = 1 - MinMaxScaler(0,1).fit_transform(log1p(mix_mean))
density_ori_rank_value = 1 - MinMaxScaler(0,1).fit_transform(log1p(ori_mean))
da_rank_value          = 1 - MinMaxScaler(0,1).fit_transform(log1p(shadow_price_da))

# Additional filter: drop constraints with zero DA history AND high density rank
density_mix_rank = density_mix_rank_value.rank(method="dense", ascending=True, pct=True)
keep = (shadow_price_da > 0) | (density_mix_rank < 0.6)

# Re-apply V6.2B formula
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
rank = rank_ori.rank(method="dense", ascending=True, pct=True)
```

**Tier assignment** (V6.2B, factor=1.0):

| Percentile rank | Tier |
|:---:|:---:|
| ≤ 0.2 | 0 (most binding) |
| ≤ 0.4 | 1 |
| ≤ 0.6 | 2 |
| ≤ 0.8 | 3 |
| ≤ 1.0 | 4 (least binding) |

- **MISO only:** `SO_MW_Transfer` forced to tier 1
- Drop any tier > 4 (there shouldn't be any with factor=1.0)

### Step 10: Build Output DataFrames

**Constraint signal columns:**

| Column | Type | Description |
|---|---|---|
| index | String | `{constraint_id}\|{-flow_direction}\|spice` |
| `constraint_id` | String | Original constraint ID |
| `flow_direction` | Int64 | Original flow direction (+1 or -1) |
| `shadow_sign` | Int64 | `-flow_direction` |
| `shadow_price` | Float64 | `shadow_sign * shadow_price_da` (or 1 if no DA history) |
| `shadow_price_da` | Float64 | Historical DA SP (unsigned) |
| `branch_name` | String | Network branch name |
| `equipment` | String | Same as branch_name |
| `bus_key` | String | `"{from_number},{to_number}"` |
| `bus_key_group` | String | Connected component label |
| `ori_mean` | Float64 | Mean density score |
| `mix_mean` | Float64 | Blended density score |
| `da_rank_value` | Float64 | Normalized DA rank (0–1, lower=more binding) |
| `density_mix_rank_value` | Float64 | Normalized mix rank |
| `density_ori_rank_value` | Float64 | Normalized ori rank |
| `density_mix_rank` | Float64 | Percentile rank of mix |
| `rank_ori` | Float64 | Composite ranking score |
| `rank` | Float64 | Percentile rank (0–1) |
| `tier` | Int64 | 0–4 |

**Shift factor signal:**

| Dimension | Description |
|---|---|
| Index (rows) | pnode_id (pricing node IDs) |
| Columns | Same as constraint signal index: `{constraint_id}\|{-flow_direction}\|spice` |
| Values | Float64 shift factor values |

---

## 3. How pmodel Consumes the Signal

Source: `pmodel/src/pmodel/base/ftr24/v1/constraint_loader.py`

### Loading

```python
tmp_cstrs = ConstraintsSignal(
    rto=rto,
    signal_name=signal_name,        # e.g. "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
    period_type=period_type,         # e.g. "f0"
    class_type=class_type,           # e.g. "onpeak"
).load_data(auction_month=pd.Timestamp(auction_month))

tmp_sf = ShiftFactorSignal(
    rto=rto,
    signal_name=signal_name,
    period_type=period_type,
    class_type=class_type,
).load_data(auction_month=pd.Timestamp(auction_month))
```

### Storage path

```
{data_root}/signal_data/{rto}/constraints/{signal_name}/{year_month}/{period_type}/{class_type}/
{data_root}/signal_data/{rto}/sf/{signal_name}/{year_month}/{period_type}/{class_type}/
```

### Index deduplication

When multiple signals are loaded, each signal's constraint indices get a suffix:
```python
tmp_cstrs.index += f"__{signal_name}__{signal_id}"
tmp_sf.columns += f"__{signal_name}__{signal_id}"
```

This prevents index collisions when combining constraints from different signals.

### Tier set construction

```python
tier_set_dict = build_tier_set_dict(constraints=cstrs_df)
```

- Creates `{tier: set(constraint_names)}` for tiers 0–4
- **Cumulative:** tier 0 ⊂ tier 1 ⊂ tier 2 ⊂ tier 3 ⊂ tier 4
- The optimizer uses different tiers for different optimization passes

### Critical columns consumed by pmodel

| Column | Required | Used For |
|---|---|---|
| `tier` | Yes | Tier set construction, optimization passes |
| `shadow_price` | Yes | Objective function direction, exposure calculation |
| `shadow_sign` | Yes | Derived from index if missing, direction disambiguation |
| `equipment` / `branch_name` | Yes | Equipment grouping, position limits |
| `rating` / `limit` | Optional | Exposure rating multiplier |

### SF matrix requirements

- Rows = pnode_ids, Columns = constraint indices (matching constraint signal index)
- No NaN allowed in final SF (rows with >97% zeros are dropped, remaining NaN → blocked nodes)
- Column order must match constraint signal index

---

## 4. What Our Annual Signal Must Produce

### Key differences from V6.2B

| Aspect | V6.2B (Monthly) | Annual Signal (Ours) |
|---|---|---|
| Constraint universe | Built from density model scores | Built from **our own ranking** (v0c formula or ML) |
| Ranking method | V6.2B formula (0.6/0.3/0.1) | Our champion formula (v0c: 0.40*da_rank_norm + 0.30*right_tail_norm + 0.30*bf_combined_12_norm) |
| Period types | f0, f1, f2, ... | Annual: aq1, aq2, aq3, aq4 |
| Class types | onpeak, offpeak | onpeak, offpeak |
| SF source | SPICE density model SF parquets | Must load from network model or bridge mapping |
| Density scores | Available (score_threshold filter) | Not available for annual — we use BF + DA history |
| Delivery period | Single month (or multi-month for fN) | Quarter (3 months per aq) |

### Requirements checklist

1. **Constraint signal parquet** with correct index format: `{constraint_id}|{flow_direction}|spice`
2. **SF signal parquet** with pnode rows × constraint columns
3. **Tier column** (Int64, 0–4) based on percentile rank of our scoring
4. **shadow_price** column (Float64, signed)
5. **shadow_sign** column (Int64)
6. **equipment** / **branch_name** column (String)
7. **SF diversity filter** to avoid redundant constraints in same bus_key_group
8. **Saved via** `ConstraintsSignal.save_data()` and `ShiftFactorSignal.save_data()` for pmodel compatibility
9. **Period type** and **class type** must be passed through correctly (never hardcoded, never assumed)

### Open questions for implementation

1. **SF source for annual:** V6.2B loads SF from SPICE density model. Our annual signal builds its own universe — where do we get shift factors? Options:
   - Load from the same SPICE SF parquets (if available for annual auction_type)
   - Compute from network model data
   - Use the bridge mapping to map constraint_ids to branches, then load branch-level SFs

2. **bus_key_group for annual:** We need branch → (from_number, to_number) mapping to compute bus_key and run the union-find. This requires:
   - Loading the branch network table for the annual auction
   - Or reusing V6.2B's branch data (period_type sensitivity!)

3. **Signal naming convention:** Our signal needs a unique name (e.g., `TEST.TEST.Signal.MISO.ANNUAL_V2.R1`) registered in pmodel's config so it can be loaded alongside other signals.

---

## 5. Pipeline Steps for Annual Signal Writer

```
┌─────────────────────────────────────────────────────────┐
│ 1. Build constraint universe                            │
│    - Load branches for annual auction via bridge        │
│    - Get all constraint_ids in our universe             │
│    - Compute bus_key and bus_key_group                  │
├─────────────────────────────────────────────────────────┤
│ 2. Score constraints                                    │
│    - Compute history features (BF, da_rank_value)       │
│    - Compute density features (if available)            │
│    - Apply v0c formula (or ML model)                    │
│    - Produce rank_ori, rank, tier                       │
├─────────────────────────────────────────────────────────┤
│ 3. Load shift factors                                   │
│    - Load SF matrix for constraints in universe         │
│    - Average across outage dates in delivery quarter    │
│    - Align columns with constraint index                │
├─────────────────────────────────────────────────────────┤
│ 4. Apply SF diversity filter                            │
│    - Per bus_key_group: max 3 per branch                │
│    - Chebyshev ≥ 0.05, correlation ≥ -0.21             │
│    - Same-branch correlation ≥ 0.1                      │
├─────────────────────────────────────────────────────────┤
│ 5. Re-rank and assign tiers                             │
│    - Re-compute rank on surviving constraints           │
│    - Assign tier 0–4 by percentile                      │
├─────────────────────────────────────────────────────────┤
│ 6. Build output DataFrames                              │
│    - Constraint signal: proper index, all required cols │
│    - SF signal: pnode × constraint matrix               │
├─────────────────────────────────────────────────────────┤
│ 7. Save via pbase signal loaders                        │
│    - ConstraintsSignal.save_data()                      │
│    - ShiftFactorSignal.save_data()                      │
│    - One file per (auction_month, period_type, class_type) │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Reference: Key Source Files

| File | Purpose |
|---|---|
| `psignal/src/psignal/spice/signal/base.py` | Production V6.2B signal generator (`get_signal()`) |
| `psignal/src/psignal/spice/utils.py:387` | `group_pair_components()` — union-find for bus_key_group |
| `pmodel/src/pmodel/base/ftr24/v1/constraint_loader.py` | How pmodel loads and consumes signals |
| `pbase/src/pbase/data/dataset/signal/base.py` | `BaseSignal` — path format, save/load |
| `pbase/src/pbase/data/dataset/signal/general.py` | `ConstraintsSignal`, `ShiftFactorSignal` classes |
| `research-annual-signal-v2/ml/history_features.py` | Our BF + DA history feature computation |
| `research-annual-signal-v2/ml/config.py` | Feature lists, eval splits, LGBM params |
| `research-miso-signal7/scripts/generate_v70_signal.py` | V7.0 pattern (re-ranks V6.2B — NOT our approach) |
