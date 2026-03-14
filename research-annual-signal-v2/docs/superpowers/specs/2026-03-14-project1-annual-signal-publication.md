# Project 1: Annual Signal Publication

**Date**: 2026-03-14
**Status**: Draft

---

## 1. Goal

Publish the annual constraint ranking signal (v0c + NB blend) as a production-ready
signal artifact consumable by pmodel's trade generation pipeline. The output must be
constraint-level, matching the schema and contract of existing signals like
`Signal.MISO.SPICE_ANNUAL_V6.1`.

## 2. Output Contract

### 2.1 Constraints Parquet

**Path**: `{data_root}/signal_data/miso/constraints/{signal_name}/{YYYY-06}/aq{1-4}/{onpeak|offpeak}/*.parquet`

**Index**: `"{constraint_id}|{shadow_sign}|{scenario}"` — this is the join key into SF
and tier sets. Must be unique and stable. `scenario` is `"spice"` for our signal.

**Required columns** (matching V6.1 schema):

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `constraint_id` | str | density data | MISO constraint ID |
| `flow_direction` | int | density data | 1 or -1 |
| `branch_name` | str | bridge mapping | grouping key for dedup |
| `bus_key` | str | density data | bus identifier |
| `bus_key_group` | str | density data | equipment grouping |
| `equipment` | str | density data | physical equipment name |
| `shadow_price_da` | float | V6.2B signal / history | historical DA congestion. **No NaN.** |
| `shadow_price` | float | derived | our signal's price estimate. **No NaN.** |
| `shadow_sign` | int | derived from flow_direction | 1 or -1. Must be explicit, not inferred. |
| `da_rank_value` | float | V6.2B signal | rank of shadow_price_da |
| `ori_mean` | float | density data | density score |
| `mix_mean` | float | density data | mixed density score |
| `density_mix_rank_value` | float | derived | rank of mix_mean |
| `density_ori_rank_value` | float | derived | rank of ori_mean |
| `rank_ori` | float | V6.1 formula | 0.60*da_rank + 0.30*mix_rank + 0.10*ori_rank |
| `rank` | float | our blend score | v0c + α×NB blend score |
| `tier` | int | derived from rank | 0-4, integer. Cumulative semantics. |
| `mean_branch_max` | float | density data | max outage density per branch |
| `mean_branch_max_fillna` | float | density data | same, NaN-filled |
| `limit` | float | density data | constraint MW limit. Near-required. |

### 2.2 SF Parquet

**Path**: `{data_root}/signal_data/miso/sf/{signal_name}/{YYYY-06}/aq{1-4}/{onpeak|offpeak}/*.parquet`

**Index (rows)**: pnode_id strings
**Columns**: same constraint index strings as constraints parquet index

**Validation requirements**:
- Columns must exactly match constraints parquet index (assert on mismatch)
- Rows with >97% zero values may be pruned by downstream (document but don't prune ourselves)
- No NaN in SF values (fill with 0.0 if missing)
- dtype: float32 or float64

### 2.3 Tier Contract

Tiers are integers 0-4 assigned by score rank:
- Tier 0: top-ranked constraints (most likely to bind with high SP)
- Tier 4: lowest-ranked

pmodel builds cumulative tier sets: tier_set[0] = {tier 0}, tier_set[1] = {tier 0, 1}, etc.
Our tier assignment must be compatible with this cumulative semantics.

**Tier assignment**: rank all constraints by our blend score descending, assign tiers
by quantile breaks (e.g., top 10% = tier 0, next 20% = tier 1, etc.). Exact breaks TBD
during implementation — should match V6.1's tier distribution.

## 3. Pipeline

```
Step 1: Score branches (v0c + NB blend)
  └─ Input: model table per (PY, aq)
  └─ Output: per-branch scores for all branches

Step 2: Expand branch → constraints
  └─ Input: branch scores + CID mapping from data_loader
  └─ Output: per-constraint scores (each constraint inherits its branch's score)

Step 3: Attach metadata
  └─ Input: per-constraint scores + V6.2B signal columns + density data
  └─ Output: full 21-column constraints DataFrame

Step 4: Assign tiers
  └─ Input: scored constraints
  └─ Output: constraints with integer tier 0-4

Step 5: Build SF matrix
  └─ Input: constraint IDs + SPICE outage data
  └─ Output: pnode × constraint SF matrix
  └─ Validate: columns = constraints index, no NaN

Step 6: Constraint selection / dedup
  └─ Input: constraints + SF
  └─ Output: final constraint set (cap per branch, SF-distinct)
  └─ This is the SF dedup logic (Chebyshev + correlation)

Step 7: Format and publish
  └─ Build index: "{constraint_id}|{shadow_sign}|{scenario}"
  └─ Validate: no NaN in shadow_price, shadow_sign explicit
  └─ Save via ConstraintsSignal.save_data() and ShiftFactorSignal.save_data()
```

### Step 2 Detail: Branch → Constraint Expansion

Our model scores at branch level. Each branch has 1-N constraint IDs (from the
bridge/CID mapping cached by `load_cid_mapping()`). Expansion rule:

```python
# Each constraint inherits its branch's blend score
for constraint_id in branch_cids:
    constraint_score = branch_blend_score
```

This is a 1-to-many fanout. After expansion, constraints from the same branch share
the same score, and the dedup step (Step 6) selects among them by SF distinctiveness.

### Step 6 Detail: Constraint Selection

Before publication, reduce the constraint set:
1. **Cap**: max 3 constraints per `branch_name` within each `bus_key_group`
2. **SF distinctiveness**: Chebyshev distance ≥ 0.05 between selected constraints
3. **Correlation bounds**: no strong counter-correlation (corr ≥ -0.21)
4. **Same-branch correlation**: corr ≥ 0.1 for constraints on the same branch

This matches the existing dedup logic. Whether we run this at publication time or leave
it to the consumer is a design decision — V6.1 publishes the full set and lets the
consumer dedup. We should match that behavior.

## 4. Verification Strategy

### 4.1 Schema Verification
- Assert output schema matches V6.1 exactly (same columns, same dtypes)
- Assert constraint index format matches `"{cid}|{sign}|{scenario}"`
- Assert SF columns match constraints index

### 4.2 Content Verification
- Load V6.1 for same PY/aq/class and compare:
  - Same constraint universe? (expect ~90% overlap)
  - Tier distribution similar?
  - shadow_price_da values identical? (they should be — same source)
- Load with `ConstraintsSignal.load_data()` and verify round-trip

### 4.3 Integration Verification
- Load our signal in pmodel's `load_constraints_and_tier_set()` and verify:
  - No crashes
  - Tier sets build correctly
  - SF intersection produces reasonable constraint count
  - shadow_price clipping works

### Reference data:
- V6.1 signal at `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/`
- V6.2B signal at `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/`

## 5. Implementation

| File | Description |
|------|-------------|
| `scripts/publish_annual_signal.py` | NEW — main signal generation + publication script |
| `ml/signal_publisher.py` | NEW — constraint expansion, metadata, tier assignment, SF build, validation |
| `tests/test_signal_publisher.py` | NEW — schema + content tests |

### Dependencies
- `pbase.data.dataset.signal.general.ConstraintsSignal` (save_data)
- `pbase.data.dataset.signal.general.ShiftFactorSignal` (save_data)
- `ml/data_loader.py:load_cid_mapping()` (branch→constraint expansion)
- V6.2B signal (metadata columns: shadow_price_da, da_rank_value, etc.)
- SPICE density data (SF matrix source)

## 6. Signal Name

Proposed: `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1`

Follows naming convention: `{prefix}.Signal.MISO.{source}_ANNUAL_{version}.R{round}`
