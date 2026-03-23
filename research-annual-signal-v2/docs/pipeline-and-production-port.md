# Annual Constraint Signal: Pipeline, Artifacts & Production Port

**Date**: 2026-03-15
**Signal**: `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1`
**Champion**: v0c + NB blend (α=0.05)

---

## 1. What V6.2B / V6.1 Produce (Reference)

### 1.1 V6.2B (Monthly f0p Signal) Artifacts

V6.2B is the monthly constraint signal for f0p period types. For each
`(auction_month, period_type, class_type)`, it publishes:

**Constraints parquet** (20 data columns + parquet index):
```
Index: "{constraint_id}|{shadow_sign}|{scenario}" (parquet index, not a data column)
Columns (20): constraint_id, flow_direction, branch_name, bus_key, bus_key_group,
  equipment, shadow_price_da, shadow_price, shadow_sign, da_rank_value, ori_mean,
  mix_mean, density_mix_rank_value, density_ori_rank_value, rank_ori,
  density_mix_rank, rank, tier, mean_branch_max, mean_branch_max_fillna
```
Note: `__index_level_0__` appears as a column when loaded via polars (which reads
the parquet index as a column). The actual parquet has 20 data columns.

**SF parquet**: pnode × constraint matrix (rows = pnode_ids, columns = constraint index)

**Index format**: `"{constraint_id}|{shadow_sign}|{scenario}"` (e.g., `"296436|-1|spice"`)

### 1.2 V6.1 (Annual Signal) Artifacts

V6.1 is the existing annual constraint signal. **Identical 20-column schema** as V6.2B.

| Difference | V6.2B | V6.1 |
|---|---|---|
| Period types | f0, f1, f2, ... | aq1, aq2, aq3, aq4 |
| Auction month | Calendar month (2024-01, etc.) | Planning year (2024-06) |
| Constraint universe | ~550 per (month, ptype, ctype) | **~280-480 per (PY, aq, ctype)** |
| shadow_price_da values | f0p-specific historical DA | Annual-specific historical DA |
| Values overlap? | **NO** — $30k max diff on same constraint_id | |

### 1.3 Both Produce

| Artifact | Format | Consumer |
|----------|--------|----------|
| Constraints parquet | 20 cols, indexed by `{cid}\|{sign}\|{scenario}` | pmodel `load_constraints_and_tier_set()` |
| SF parquet | pnode × constraint matrix | pmodel exposure computation |
| Tier assignment | Integer 0-4, cumulative semantics | pmodel tier_set construction |

---

## 2. What We Can Produce — Exact Counterparts

### 2.1 Our Annual Signal (V7.0) vs V6.1

| Artifact | V6.1 (existing) | V7.0 (ours) | Match? |
|----------|-----------------|-------------|--------|
| **Constraints parquet** | 20 cols, ~280-480 rows per (aq, ctype) | 20 cols, **~1,000** per (aq, ctype) post-dedup | **Schema: YES. Count: LARGER** |
| **SF parquet** | pnode × constraint | pnode × constraint | **YES (exact parity verified)** |
| **Index format** | `{cid}\|{sign}\|spice` | `{cid}\|{sign}\|spice` | **YES** |
| **Tier 0-4** | Assigned by V6.1 rank_ori | Assigned by our blend score | **Format: YES. Values: DIFFERENT** |
| **shadow_price_da** | V6.1 annual-specific | For V6.1 overlap: inherited. For new: computed from DA history (branch-level) | **Overlap: IDENTICAL. New: computed** |
| **da_rank_value** | V6.1 annual-specific | Recomputed from shadow_price_da | **DIFFERENT (our universe may differ)** |
| **ori_mean, mix_mean** | V6.1 annual-specific | From SPICE density (same source) | **IDENTICAL for overlap** |
| **rank_ori** | 0.60*da_rank + 0.30*mix + 0.10*ori | Same formula, different inputs if universe differs | **IDENTICAL for overlap** |
| **rank** | = rank_ori in V6.1 | **Our blend score** (v0c + NB) | **DIFFERENT** |
| **tier** | Derived from rank_ori | Derived from our blend rank | **DIFFERENT** |
| **shadow_price** | SPICE-computed | `shadow_sign` (direction only, per business decision) | **DIFFERENT** |

**For overlapping constraints** (our universe ∩ V6.1): metadata inherited from V6.1,
values identical.

**For new constraints** (in our universe but not V6.1): metadata computed from raw
sources — `shadow_price_da` from DA history at branch level, density features from
SPICE density, SF from `MISO_SPICE_SF.parquet`. These constraints need explicit
source contracts, not V6.1 inheritance.

### 2.2 Column-Level Source Map

| Column | Source (V6.1 overlap) | Source (new constraints) |
|--------|----------------------|------------------------|
| constraint_id | Bridge mapping | Bridge mapping |
| flow_direction | V6.1 | **RESOLVED** — from `MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet`: pick direction (+1/-1) with max score per CID. Confirmed 2026-03-23. |
| branch_name | V6.1 | Bridge mapping |
| bus_key | V6.1 | **OPEN GAP** — not in density; need MISO_SPICE_CONSTRAINT_INFO |
| bus_key_group | V6.1 | **OPEN GAP** — not in density; need MISO_SPICE_CONSTRAINT_INFO |
| equipment | V6.1 | **OPEN GAP** — not in density; need MISO_SPICE_CONSTRAINT_INFO |
| shadow_price_da | V6.1 (inherited) | DA history (branch-level, class-specific) |
| da_rank_value | V6.1 (inherited) | Rank of computed shadow_price_da |
| ori_mean | V6.1 (inherited) | SPICE density |
| mix_mean | V6.1 (inherited) | SPICE density |
| density_mix_rank_value | V6.1 (inherited) | Rank of mix_mean |
| density_ori_rank_value | V6.1 (inherited) | Rank of ori_mean |
| rank_ori | V6.1 (inherited) | Computed from ranks |
| density_mix_rank | V6.1 (inherited) | Rank of mix_mean |
| mean_branch_max | V6.1 (inherited) | SPICE density |
| mean_branch_max_fillna | V6.1 (inherited) | SPICE density |
| shadow_sign | V6.1 | **OPEN GAP** — derived from flow_direction; same gap as flow_direction |
| **rank** | Our blend score | **NO — our ranking** |
| **tier** | Our blend ranking | **NO — our tiers** |
| **shadow_price** | Our model estimate | **NO — our estimate** |
| __index_level_0__ | Index string | Yes |

---

## 3. Remaining Gaps

> **TOP BLOCKER for M3 (publication)**: New-constraint metadata source is unresolved.
> `flow_direction`, `shadow_sign`, `bus_key`, `bus_key_group`, `equipment` have no
> known raw source outside V6.1. For V6.1-overlapping constraints these are inherited;
> for new constraints they are OPEN GAPS. Need `MISO_SPICE_CONSTRAINT_INFO` or
> equivalent. **M1-M2 (modeling) can proceed without this; M3 cannot.**

### 3.1 shadow_price — FROZEN: use shadow_sign

**Decision**: `shadow_price = shadow_sign` (just +1 or -1).

pmodel uses `shadow_price` in `get_node_exposure()` as: `exposure = SF × shadow_price × tier_weight`.
With `shadow_price = shadow_sign`, this becomes `exposure = SF × direction × tier_weight`,
which is tier-driven ranking. This matches the business decision to use tiers only.

This is simpler, avoids the SPICE shadow_price computation gap, and does not break
pmodel (verified: the optimizer accepts any non-NaN float in shadow_price).

### 3.2 Constraint Universe Size

**Pre-dedup density universe**: ~12,800-13,000 constraints per (PY, aq). Class-agnostic.

**V6.1 published**: ~280-480 per (PY, aq, class_type).

**V7.0 target: 1,000 per (PY, aq, ctype)** = 5 tiers × 200 each.

This is larger than V6.1 (367 avg) but within the range of existing annual signals
(SPICE_ANNUAL_V4.5 peaks at 889, DA_ANNUAL_V1.4 at 549). The larger count gives
more coverage for the optimizer's constraint universe.

**Published signal is always post-dedup.** pmodel does not apply our branch-cap +
SF correlation dedup on load. If V7.0 is loaded alongside other signals, pmodel
unions the constraint sets and uses all of them.

### 3.3 SF Matrix Construction — RESOLVED

**Source**: `MISO_SPICE_SF.parquet` in the canonical spice_data path:
`/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_SF.parquet/spice_version=v6/auction_type=annual/auction_month={YYYY-06}/market_month={YYYY-MM}/market_round=1/outage_date={date}/`

NOT the `pw_data` path (`prod_f0p_model_miso/sf/`) — that only has 1 market_month
per annual PY because it's the f0p pipeline output.

**Key facts** (verified):
- Raw SPICE has **13,207 constraints × 2,225 pnodes** per outage date
- **12 market months** per annual PY (one per delivery month Jun-May)
- 10-11 outage dates per market month
- Quarter mapping: aq1 = Jun/Jul/Aug, aq2 = Sep/Oct/Nov, aq3 = Dec/Jan/Feb, aq4 = Mar/Apr/May
- V6.1 selected ~300-480 from these 13,207. Our post-dedup set is a different subset.
- **Any constraint in our universe has SF in the raw data** — no gaps.
- Pnodes are **identical across quarters within a PY** (100% overlap aq1-aq4).
  Slowly grows across PYs (91-95% overlap year-to-year).
- SF values are **class-type-agnostic** (onpeak SF == offpeak SF for same constraint).
  Only the constraint selection differs per class_type.

**Aggregation method**: Mean across outage dates within the 3 market months of each
quarter. **Verified: exact parity** — max_diff = 0.000000, correlation = 1.000000
for all 50 tested constraints (2024-06/aq1). V6.1 SF is exactly the mean of raw
SPICE SF from `MISO_SPICE_SF.parquet`.

**Our approach**: For each quarter's constraint set:
1. Load raw SF from `MISO_SPICE_SF.parquet` for the 3 market months of that quarter
2. Mean across all outage dates (30-33 per quarter)
3. Subset columns to our post-dedup constraint set for that (aq, class_type)
4. Publish as the SF parquet

This is NOT inherited from V6.1 — we build our own SF from the same upstream source.

### 3.4 Dedup Parameters

The SF dedup (Chebyshev ≥ 0.05, correlation ≥ -0.21, max 3/branch) was derived from
the code snippet shared by the user. These parameters match the existing dedup logic.

**Gap**: We haven't verified that these exact thresholds are used in production for
annual signals. The pbase `reduce_constraints_by_similar_sf()` uses only Chebyshev
threshold, not the correlation or branch cap. The branch cap + correlation is
custom logic from the team.

**Mitigation**: Confirm with team which dedup logic to use — pbase's or the custom one.

### 3.5 Per-Class-Type Signal — MUST BE CLASS-SPECIFIC

V6.1 publishes **genuinely different signals** for onpeak and offpeak (verified):
- Different constraint sets: 425 onpeak vs 413 offpeak (78% overlap)
- Different `shadow_price_da`: max diff $7,318 for same constraint
- Different rankings and tiers: up to 3 tier difference for same constraint

**Current limitation**: Phase 5 models were trained on combined SP. Class-specific
pipeline is required for production.

**See**: `docs/superpowers/specs/2026-03-15-class-specific-pipeline-design.md` for the
full class-separation design — what changes, what stays, experiment plan, and
implementation details.

---

## 4. Research Pipeline (Detailed)

### 4.1 Data Sources

| Source | Path | Content |
|--------|------|---------|
| SPICE Density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` | Simulated SP distribution per constraint × outage |
| V6.1 Annual Signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` | Constraint-level metadata |
| Realized DA | Via `pbase` loaders | Monthly DA shadow prices per constraint |
| MISO Bridge | Via `pbase` loaders | Constraint-to-branch mapping per auction |

### 4.2 Feature Extraction

**Density Features** (`ml/data_loader.py`):
- Raw → Level 1 (mean/CID) → bridge join → Level 2 (max+min/branch)
- CID mapping cached for Project 1 expansion
- Output: `bin_{X}_cid_max/min`, `count_cids`, `count_active_cids`, limits

**History Features** (`ml/history_features.py`):
- Monthly binding table from [BF_FLOOR_MONTH, cutoff]
- BF windows: `bf_6`, `bf_12`, `bf_15`, `bfo_6`, `bfo_12`, `bf_combined_12`
- `da_rank_value`: rank of historical DA shadow price

**NB Track B Features** (`ml/features_trackb.py`):
- `historical_max_sp`: peak single-month SP from older history
- `months_since_last_bind`: recency of last binding event
- Density shape features (entropy, skewness — marginal impact, included for completeness)

### 4.3 Scoring (v0c + NB blend)

```python
# v0c formula (all branches):
v0c = 0.40 × norm(1 - da_rank_value) + 0.30 × norm(rt_max) + 0.30 × norm(bf_combined_12)

# NB model (dormant only): LightGBM binary, 14 features, sqrt(SP) sample weights
nb_prob = nb_model.predict(dormant_features)

# Blend:
final[established] = v0c_score
final[dormant] = v0c_score + 0.05 × (nb_prob - nb_prob.min()) / (nb_prob.max() - nb_prob.min()) × v0c_range
final[zero_history] = v0c_score
```

### 4.4 Evaluation

**Metrics**: VC@K, Recall@K, Abs_SP@K, NB12_SP@K, Dang_Recall@K, Dang_SP_Ratio@K
**K levels**: 150, 200, 300, 400
**Paired scorecard**: composite of VC, Recall, DangR, NB12_SP weighted 0.4/0.2/0.2/0.2

---

## 5. Production Port Pipeline

```
Step 1: Score branches (v0c + NB blend)
  └─ Same as research scoring, applied to target PY

Step 2: Expand branch → constraints (FROZEN: pure branch inheritance)
  └─ CID mapping from load_cid_mapping()
  └─ Each constraint inherits its branch's blend score
  └─ No within-branch constraint tie-breaking in V7.0

Step 3: Join V6.1 metadata
  └─ shadow_price_da, da_rank_value, ori_mean, mix_mean, rank_ori, etc.
  └─ shadow_price = shadow_sign (direction only, frozen contract)

Step 4: Assign tiers (0-4) by blend score rank

Step 5: Build SF matrix from SPICE outage data

Step 6: Selection + dedup (walk-and-fill, publish post-dedup)
  └─ Target: 5 tiers × 200 = 1,000 constraints
  └─ Algorithm: walk ranking top-to-bottom, apply dedup while selecting:
     1. Rank all ~13k candidates by blend score (descending)
     2. For each candidate:
        - Skip if branch already has 3 in this bus_key_group
        - Skip if SF Chebyshev < 0.05 vs any selected in same group
        - Otherwise: accept, assign to current tier
     3. When tier reaches 200: advance to next tier
     4. Stop when tier 4 filled (or candidates exhausted)
  └─ Do NOT "select 1000 then dedup" — dedup during selection

Step 7: Validate (schema, no NaN, SF alignment) + publish
  └─ ConstraintsSignal.save_data()
  └─ ShiftFactorSignal.save_data()
```

### Consumer interface:
```python
from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal

cstrs = ConstraintsSignal(rto="miso", signal_name="TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1",
                           period_type="aq1", class_type="onpeak").load_data(pd.Timestamp("2025-06"))
sf = ShiftFactorSignal(rto="miso", signal_name="TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1",
                        period_type="aq1", class_type="onpeak").load_data(pd.Timestamp("2025-06"))
```

---

## 6. Data Paths

| Data | Path |
|------|------|
| SPICE density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` |
| **SPICE SF (annual, canonical)** | **`/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_SF.parquet/`** |
| SPICE SF (f0p only, 1 month) | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/sf/` |
| V6.1 annual signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` |
| V6.1 annual SF | `/opt/data/xyz-dataset/signal_data/miso/sf/Signal.MISO.SPICE_ANNUAL_V6.1/` |
| V6.2B monthly signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/` |
| Our signal (V7.0) | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/` |
| Phase 5 registry | `registry/archive/phase5_final_150_300/`, `registry/archive/phase5_final_200_400/` |
| Annual band port | `research-annual-band/docs/prod-port.md` |

**NOTE**: The `pw_data` SF path (`prod_f0p_model_miso/sf/`) only has 1 market_month
per annual PY. The canonical annual SF source with all 12 delivery months is
`MISO_SPICE_SF.parquet` in `spice_data`. Always use the `spice_data` path for annual.
