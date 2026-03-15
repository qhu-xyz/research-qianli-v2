# Annual Constraint Signal: Pipeline, Artifacts & Production Port

**Date**: 2026-03-15
**Signal**: `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1`
**Champion**: v0c + NB blend (α=0.05)

---

## 1. What V6.2B / V6.1 Produce (Reference)

### 1.1 V6.2B (Monthly f0p Signal) Artifacts

V6.2B is the monthly constraint signal for f0p period types. For each
`(auction_month, period_type, class_type)`, it publishes:

**Constraints parquet** (21 columns):
```
constraint_id, flow_direction, branch_name, bus_key, bus_key_group, equipment,
shadow_price_da, shadow_price, shadow_sign, da_rank_value, ori_mean, mix_mean,
density_mix_rank_value, density_ori_rank_value, rank_ori, density_mix_rank,
rank, tier, mean_branch_max, mean_branch_max_fillna, __index_level_0__
```

**SF parquet**: pnode × constraint matrix (rows = pnode_ids, columns = constraint index)

**Index format**: `"{constraint_id}|{shadow_sign}|{scenario}"` (e.g., `"296436|-1|spice"`)

### 1.2 V6.1 (Annual Signal) Artifacts

V6.1 is the existing annual constraint signal. **Identical 21-column schema** as V6.2B.

| Difference | V6.2B | V6.1 |
|---|---|---|
| Period types | f0, f1, f2, ... | aq1, aq2, aq3, aq4 |
| Auction month | Calendar month (2024-01, etc.) | Planning year (2024-06) |
| Constraint universe | ~1,111 per (month, ptype) | ~2,900-3,400 per (PY, aq) |
| shadow_price_da values | f0p-specific historical DA | Annual-specific historical DA |
| Values overlap? | **NO** — $30k max diff on same constraint_id | |

### 1.3 Both Produce

| Artifact | Format | Consumer |
|----------|--------|----------|
| Constraints parquet | 21 cols, indexed by `{cid}\|{sign}\|{scenario}` | pmodel `load_constraints_and_tier_set()` |
| SF parquet | pnode × constraint matrix | pmodel exposure computation |
| Tier assignment | Integer 0-4, cumulative semantics | pmodel tier_set construction |

---

## 2. What We Can Produce — Exact Counterparts

### 2.1 Our Annual Signal (V7.0) vs V6.1

| Artifact | V6.1 (existing) | V7.0 (ours) | Match? |
|----------|-----------------|-------------|--------|
| **Constraints parquet** | 21 cols, ~3,000 rows/aq | 21 cols, ~300-1,500 rows/aq (post-dedup) | **Schema: YES. Row count: SMALLER** |
| **SF parquet** | pnode × constraint | pnode × constraint | **YES** |
| **Index format** | `{cid}\|{sign}\|spice` | `{cid}\|{sign}\|spice` | **YES** |
| **Tier 0-4** | Assigned by V6.1 rank_ori | Assigned by our blend score | **Format: YES. Values: DIFFERENT** |
| **shadow_price_da** | V6.1 annual-specific | Inherited from V6.1 | **IDENTICAL** |
| **da_rank_value** | V6.1 annual-specific | Inherited from V6.1 | **IDENTICAL** |
| **ori_mean, mix_mean** | V6.1 annual-specific | Inherited from V6.1 | **IDENTICAL** |
| **rank_ori** | 0.60*da_rank + 0.30*mix + 0.10*ori | Same formula | **IDENTICAL** |
| **rank** | = rank_ori in V6.1 | **Our blend score** (v0c + NB) | **DIFFERENT** |
| **tier** | Derived from rank_ori | Derived from our blend rank | **DIFFERENT** |
| **shadow_price** | SPICE-computed | **Our model-derived estimate** | **DIFFERENT** |

**What's identical**: All metadata columns inherited from V6.1 (shadow_price_da, da_rank_value, ori_mean, mix_mean, density ranks, rank_ori, branch_name, bus_key_group, equipment, flow_direction, shadow_sign, mean_branch_max).

**What's different**: `rank` (our blend score vs rank_ori), `tier` (our ranking vs V6.1 ranking), `shadow_price` (our estimate vs SPICE estimate), and the constraint universe (smaller after our dedup).

### 2.2 Column-Level Source Map

| Column | Source for V7.0 | Same as V6.1? |
|--------|----------------|---------------|
| constraint_id | SPICE density → bridge | Yes |
| flow_direction | V6.1 | Yes |
| branch_name | V6.1 | Yes |
| bus_key | V6.1 | Yes |
| bus_key_group | V6.1 | Yes |
| equipment | V6.1 | Yes |
| shadow_price_da | V6.1 | **Yes (inherited)** |
| da_rank_value | V6.1 | **Yes (inherited)** |
| ori_mean | V6.1 | **Yes (inherited)** |
| mix_mean | V6.1 | **Yes (inherited)** |
| density_mix_rank_value | V6.1 | **Yes (inherited)** |
| density_ori_rank_value | V6.1 | **Yes (inherited)** |
| rank_ori | V6.1 formula | **Yes (inherited)** |
| density_mix_rank | V6.1 | **Yes (inherited)** |
| mean_branch_max | V6.1 | **Yes (inherited)** |
| mean_branch_max_fillna | V6.1 | **Yes (inherited)** |
| shadow_sign | V6.1 | **Yes (inherited)** |
| **rank** | Our blend score | **NO — our ranking** |
| **tier** | Our blend ranking | **NO — our tiers** |
| **shadow_price** | Our model estimate | **NO — our estimate** |
| __index_level_0__ | Index string | Yes |

---

## 3. Remaining Gaps

### 3.1 shadow_price Derivation

V6.1's `shadow_price` is computed by SPICE from forward-looking simulation. We don't
have access to the SPICE shadow price computation for our custom constraint ranking.

**Options**:
1. **Inherit V6.1's shadow_price** — for constraints in V6.1, use their shadow_price.
   For new constraints not in V6.1 (rare), use shadow_price_da as fallback.
2. **Derive from density** — compute an expected shadow price from the density
   distribution. Less accurate than SPICE but purely from our data.
3. **Use shadow_price_da** — historical DA shadow price. Available for all constraints
   but backward-looking, not forward.

**Recommendation**: Option 1 (inherit from V6.1 where available, shadow_price_da fallback).
This preserves the SPICE-quality forward estimate for the vast majority of constraints.

### 3.2 Constraint Universe Size

V6.1 publishes ~3,000 constraints per aq. After our dedup (max 3/branch, SF Chebyshev,
correlation bounds), we'll have ~300-1,500. This is intentional — we publish the
post-dedup set that pmodel will actually use.

**Gap**: If pmodel expects to load multiple signals and union their constraint sets
(as it does for f0p: DZ + SPICE + DA), our smaller set may not cover all constraints
the optimizer needs. This depends on whether V7.0 is the sole annual signal or one
of several.

**Mitigation**: Check with the team whether V7.0 replaces V6.1 entirely or supplements
it. If supplement, publish the full pre-dedup set and let pmodel handle dedup.

### 3.3 SF Matrix Availability

V6.1's SF matrix comes from SPICE outage simulation data. We need the same SF source.

**Current state**: The SF data exists in SPICE density partitions, but extracting a
clean pnode × constraint SF matrix requires processing the raw outage data differently
than our current density pipeline (which collapses to branch level).

**Gap**: `ml/data_loader.py` collapses SF information during Level 2 (max/min per branch).
The SF matrix needs to be extracted BEFORE collapse, at the constraint level.

**Mitigation**: The CID mapping cache (`load_cid_mapping()`) preserves the
constraint→branch mapping. The SF matrix can be built by:
1. Loading the same SPICE outage data
2. Keeping per-constraint SF values (not collapsing to branch)
3. Pivoting to pnode × constraint format

This requires new code but uses the same source data.

### 3.4 Dedup Parameters

The SF dedup (Chebyshev ≥ 0.05, correlation ≥ -0.21, max 3/branch) was derived from
the code snippet shared by the user. These parameters match the existing dedup logic.

**Gap**: We haven't verified that these exact thresholds are used in production for
annual signals. The pbase `reduce_constraints_by_similar_sf()` uses only Chebyshev
threshold, not the correlation or branch cap. The branch cap + correlation is
custom logic from the team.

**Mitigation**: Confirm with team which dedup logic to use — pbase's or the custom one.

### 3.5 Per-Class-Type Signal

V6.1 publishes separate signals for onpeak and offpeak. Our research pipeline evaluates
combined (onpeak + offpeak SP as `realized_shadow_price`). We need to decide:

**Options**:
1. **Publish identical signal for both class_types** — same ranking for onpeak and offpeak
2. **Publish class-specific signals** — train separate NB models per class_type

**Recommendation**: Option 1 for V7.0 (same ranking). The v0c formula and NB model were
both trained on combined SP. Class-specific models can be a future enhancement.

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

Step 2: Expand branch → constraints
  └─ CID mapping from load_cid_mapping()
  └─ Each constraint inherits branch score

Step 3: Join V6.1 metadata
  └─ shadow_price_da, da_rank_value, ori_mean, mix_mean, rank_ori, etc.
  └─ shadow_price: inherit V6.1 where available, shadow_price_da fallback

Step 4: Assign tiers (0-4) by blend score rank

Step 5: Build SF matrix from SPICE outage data

Step 6: Dedup (publish post-dedup)
  └─ Max 3/branch/bus_key_group
  └─ Chebyshev ≥ 0.05, correlation bounds

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
| V6.1 annual | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` |
| V6.2B monthly | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/` |
| Our signal (V7.0) | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/` |
| Phase 5 registry | `registry/phase5_final_150_300/`, `registry/phase5_final_200_400/` |
| Annual band port | `research-annual-band/docs/prod-port.md` |
