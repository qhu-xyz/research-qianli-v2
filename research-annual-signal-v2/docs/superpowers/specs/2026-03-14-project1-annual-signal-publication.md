# Project 1: Annual Signal Publication

**Date**: 2026-03-14
**Status**: Draft (rev 2 — review fixes applied)

---

## 1. Goal

Publish the annual constraint ranking signal (v0c + NB blend) as a production-ready,
post-dedup, constraint-level signal artifact consumable by pmodel's trade generation
pipeline. The output schema must exactly match `Signal.MISO.SPICE_ANNUAL_V6.1`.

## 2. Output Contract

### 2.1 Published Artifact = Post-Dedup

pmodel loads whatever constraints/SF we publish and uses that set directly — it does NOT
apply our dedup logic on load (`base.py:968`). The `pdist("chebyshev")` code in pmodel
is for node clustering, not signal dedup (`base.py:1540`).

Therefore:
- **Published signal = final post-dedup constraint set**
- Pre-dedup expanded universe saved as debug artifact only (not consumed by pmodel)
- This ensures production and Project 2 use the same constraint universe

### 2.2 Constraints Parquet

**Path**: `{data_root}/signal_data/miso/constraints/{signal_name}/{YYYY-06}/aq{1-4}/{onpeak|offpeak}/*.parquet`

**Index**: `"{constraint_id}|{shadow_sign}|{scenario}"` — the join key into SF and tier
sets. Must be unique and stable. `scenario` = `"spice"` for our signal.

**20 data columns + parquet index** (matching V6.1 and V6.2B schema exactly).
Note: `__index_level_0__` appears when loaded via polars but is the parquet index,
not a data column. The actual parquet has 20 data columns:

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| `constraint_id` | str | SPICE annual density | MISO constraint ID |
| `flow_direction` | int | SPICE annual density | 1 or -1 |
| `branch_name` | str | annual bridge mapping | grouping key for position logic |
| `bus_key` | str | SPICE annual density | bus identifier |
| `bus_key_group` | str | SPICE annual density | equipment grouping |
| `equipment` | str | SPICE annual density | physical equipment name |
| `shadow_price_da` | float | **V6.1 annual signal** | historical DA congestion. **No NaN.** |
| `shadow_price` | float | derived from our model | our signal's price estimate. **No NaN.** Fill 0 if missing. |
| `shadow_sign` | int | V6.1 annual signal | 1 or -1. Publish explicitly, not inferred. |
| `da_rank_value` | float | **V6.1 annual signal** | rank of shadow_price_da |
| `ori_mean` | float | SPICE annual density | density score |
| `mix_mean` | float | SPICE annual density | mixed density score |
| `density_mix_rank_value` | float | V6.1 annual signal | rank of mix_mean |
| `density_ori_rank_value` | float | V6.1 annual signal | rank of ori_mean |
| `rank_ori` | float | V6.1 formula | 0.60×da_rank + 0.30×mix_rank + 0.10×ori_rank |
| `rank` | float | **our blend score** | v0c + α×NB blend score |
| `tier` | int | derived from `rank` | 0-4, integer. Cumulative semantics. |
| `mean_branch_max` | float | SPICE annual density | max outage density per branch |
| `mean_branch_max_fillna` | float | SPICE annual density | same, NaN-filled |
| `density_mix_rank` | float | V6.1 annual signal | rank of mix_mean (numeric rank, not rank_value) |
| `__index_level_0__` | str | constraint index string | pandas artifact: same as DataFrame index. V6.1 stores this as a column when serialized from pandas with index. |

**Verified**: V6.1 parquet has exactly 21 columns (20 data + `__index_level_0__`).
Our publication must match this exactly.

**CRITICAL: Metadata source is V6.1 annual, NOT V6.2B monthly.** Verified: V6.1 and
V6.2B have different values for overlapping constraints (`shadow_price_da` max diff =
$30k, `ori_mean` max diff = 0.76). V6.1 is annual-specific SPICE; V6.2B is f0p-specific.
Our signal must inherit metadata from V6.1 for the annual pipeline.

`limit` column: V6.1 does not contain `limit` — it is loaded separately by pmodel from
the density data as `rating` (renamed from `limit`). Our signal matches V6.1 which does
not include `limit`. pmodel will load it separately if needed.

### 2.3 SF Parquet

**Path**: `{data_root}/signal_data/miso/sf/{signal_name}/{YYYY-06}/aq{1-4}/{onpeak|offpeak}/*.parquet`

**Index (rows)**: pnode_id strings
**Columns**: same constraint index strings as constraints parquet index

**Validation**:
- Columns must exactly match constraints parquet index (`assert set(sf.columns) == set(cstrs.index)`)
- No NaN in SF values (fill with 0.0 if missing)
- dtype: float64

### 2.4 Tier Contract

Tiers are integers 0-4 assigned by our blend score ranking:
- Tier 0: top-ranked (most likely to bind with high SP)
- Tier 4: lowest-ranked

pmodel builds cumulative tier sets (`base.py:359`): `tier_set[0] = {tier 0}`,
`tier_set[1] = {tier 0, 1}`, etc. Missing tiers are allowed (empty set).

Tier breaks match V6.1 distribution per PY/aq. Computed as quantile breaks of the
blend score within the post-dedup constraint set.

## 3. Pipeline

```
Step 1: Score branches (v0c + NB blend)
  └─ Input: model table per (PY, aq)
  └─ Output: per-branch blend scores

Step 2: Expand branch → constraints
  └─ Input: branch scores + CID mapping (load_cid_mapping)
  └─ Output: per-constraint scores (constraint inherits branch score)

Step 3: Attach metadata from V6.1
  └─ Input: per-constraint scores + V6.1 annual signal for same PY
  └─ Join on constraint_id to get: shadow_price_da, da_rank_value,
     ori_mean, mix_mean, rank_ori, shadow_sign, bus_key, bus_key_group,
     equipment, branch_name, flow_direction, mean_branch_max, etc.
  └─ Constraints not in V6.1: fill shadow_price_da=0, shadow_sign from
     flow_direction

Step 4: Assign tiers (0-4) based on blend score rank

Step 5: Build SF matrix from MISO_SPICE_SF.parquet
  └─ Source: /opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_SF.parquet
  └─ NOT pw_data (that only has 1 market_month for annual)
  └─ 12 market months per PY: 3 per quarter (aq1=Jun/Jul/Aug, etc.)
  └─ 13,207 constraints × 2,225 pnodes per outage date
  └─ Aggregate: mean across all outage dates within the 3 quarter months
  └─ Subset columns to our post-dedup constraint set
  └─ SF is class-type-agnostic (same values, different constraint subsets)
  └─ Validate: columns = constraints index, no NaN

Step 6: Constraint selection / dedup (PUBLISH POST-DEDUP)
  └─ Cap: max 3 constraints per branch_name within bus_key_group
  └─ SF distinctiveness: Chebyshev ≥ 0.05
  └─ Correlation: ≥ -0.21 cross, ≥ 0.1 same-branch
  └─ Save pre-dedup as debug artifact only

Step 7: Format index + validate + publish
  └─ Index: "{constraint_id}|{shadow_sign}|spice"
  └─ shadow_price = blend-derived estimate, no NaN
  └─ shadow_sign explicit
  └─ Save via ConstraintsSignal.save_data() + ShiftFactorSignal.save_data()
```

## 4. Verification Strategy

### 4.1 Schema Verification
- Assert 21 columns match V6.1 exactly (names, dtypes)
- Assert constraint index format: `"{cid}|{sign}|spice"`
- Assert `set(sf.columns) == set(cstrs.index)`

### 4.2 Content Verification
- Load V6.1 for same PY/aq/class and compare:
  - Constraint universe overlap (expect ~80-90%)
  - Metadata columns identical where joined from V6.1
  - Tier distribution reasonable (not degenerate)
  - No NaN in shadow_price or shadow_sign

### 4.3 Round-Trip Verification
- Load with `ConstraintsSignal.load_data()` and verify parseable
- Load in pmodel's `load_constraints_and_tier_set()`:
  - No crashes
  - Tier sets build correctly
  - SF intersection produces ≥80% of published constraints

### Reference data
- V6.1: `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/`
- V6.2B: NOT used for metadata (values differ from V6.1 for overlapping constraints)

## 5. Implementation

| File | Description |
|------|-------------|
| `scripts/publish_annual_signal.py` | NEW — signal generation + publication |
| `ml/signal_publisher.py` | NEW — expand, metadata, tier, SF, dedup, validate |
| `tests/test_signal_publisher.py` | NEW — schema + content + round-trip tests |

### Dependencies
- `pbase.data.dataset.signal.general.ConstraintsSignal` (save_data, load_data)
- `pbase.data.dataset.signal.general.ShiftFactorSignal` (save_data, load_data)
- `ml/data_loader.py:load_cid_mapping()` (branch→constraint mapping)
- V6.1 annual signal (metadata source: class-specific shadow_price_da, da_rank_value)
- `MISO_SPICE_SF.parquet` at `/opt/data/xyz-dataset/spice_data/miso/` (13,207 constraints × 2,225 pnodes, 12 market months per PY)
- Aggregation: mean across outage dates within 3 market months per quarter

## 6. Signal Name

Proposed: `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1`
