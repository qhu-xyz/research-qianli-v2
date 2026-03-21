# V7.0 Signal Verification Plan

**Date**: 2026-03-19
**Goal**: Verify the published V7.0 signal parquets are correct by merging realized DA ground truth into the published constraints and inspecting the mapping quality.

---

## 1. What to Build

For selected slices, produce a merged file: **published V7.0 constraints + realized DA SP per constraint + branch-level aggregated SP**. Save as parquet for human inspection.

Columns:
- From V7.0: `constraint_id`, `branch_name`, `tier`, `rank`, `v0c_score` (rank column), `flow_direction`, `shadow_sign`, `bus_key`, `da_rank_value`, `shadow_price_da`
- From DA GT: `realized_sp_onpeak`, `realized_sp_offpeak`, `realized_sp_total`
- Derived: `branch_sp_total` (sum of all CIDs on that branch), `n_cids_on_branch` (sibling count), `is_binding` (realized_sp_total > 0)

## 2. Slices to Produce

| Slice | PY | Quarter | Class | Why |
|-------|-----|---------|-------|-----|
| A | 2021-06 | aq1 | onpeak | Early PY, good bridge coverage (~1%), training set |
| B | 2021-06 | aq1 | offpeak | Same group, other class for comparison |
| C | 2025-06 | aq2 | onpeak | Holdout, worst bridge coverage (~30%), most unmapped |
| D | 2025-06 | aq2 | offpeak | Same group, other class, worst Abs/VC (0.619) |

Two PYs (one early, one late) x two classes each = 4 files.

## 3. Pre-Run Inspection Checklist

Before generating the merged files, verify these potential issues:

### 3.1 Schema checks
- [ ] V7.0 constraint parquet has all 20 SCHEMA_COLUMNS + `__index_level_0__` index
- [ ] SF parquet has `pnode_id` + N constraint columns matching the constraint file
- [ ] Constraint count = 1000 per file (or fewer if insufficient candidates)
- [ ] Index format matches V6.1: `"{cid}|{shadow_sign}|spice"`

### 3.2 Branch-CID mapping sanity
- [ ] How many unique branches in 1000 constraints? (expect ~700-800 due to branch_cap=3)
- [ ] Max CIDs per branch = branch_cap = 3
- [ ] No branch appears with >3 CIDs
- [ ] All constraint_ids are in the annual bridge (by construction — they come from the density universe)

### 3.3 DA merge coverage
- [ ] How many published CIDs have nonzero DA SP? (binding rate)
- [ ] How many published CIDs have zero DA SP? (non-binding)
- [ ] Any published CIDs missing from DA entirely? (shouldn't happen if DA cache covers the quarter)
- [ ] Per-tier binding rate: does tier 0 have higher binding rate than tier 4?

### 3.4 Potential anomalies to check
- [ ] **Sibling inflation**: Do sibling CIDs on the same branch all get the same branch SP credited? (They should NOT — each CID has its own DA SP. Branch SP is the sum.)
- [ ] **Score monotonicity**: Are higher-scored constraints more likely to be binding?
- [ ] **Flow direction consistency**: Does `flow_direction` match between V7.0 and V6.1 for overlapping CIDs?
- [ ] **Shadow sign vs DA sign**: Does the constraint bind in the direction `shadow_sign` predicts?
- [ ] **NaN/null columns**: Any nulls in `branch_name`, `constraint_id`, `tier`, `flow_direction`?
- [ ] **da_rank_value distribution**: Should be 0-1 range after normalization. Any sentinels (values > 1)?
- [ ] **Zero-density branches in published set**: Any branches where all density bins are zero but still published? (Shouldn't happen — these would be filtered by `is_active` threshold)

### 3.5 Data drift between 2021-06 and 2025-06
- [ ] Total DA SP: how much bigger/smaller?
- [ ] Binding rate among published CIDs: stable or changing?
- [ ] Score distribution shift: are v0c scores higher/lower in 2025?
- [ ] Tier 0 concentration: does tier 0 capture more or less SP in 2025?
- [ ] Branch overlap: how many branches appear in both PYs' published sets?

### 3.6 SF matrix checks
- [ ] SF columns match published constraint_ids (same count, same CIDs)
- [ ] SF values are in reasonable range (typically -1 to +1 for shift factors)
- [ ] No all-zero SF columns (would mean the constraint has no impact on any pnode)
- [ ] No all-NaN SF columns
- [ ] pnode_id count is reasonable (~2000 for MISO)

## 4. Output Files

Save to `data/v7_verification/`:
```
data/v7_verification/
├── 2021-06_aq1_onpeak_merged.parquet
├── 2021-06_aq1_offpeak_merged.parquet
├── 2025-06_aq2_onpeak_merged.parquet
├── 2025-06_aq2_offpeak_merged.parquet
└── verification_summary.json
```

`verification_summary.json` should contain per-slice:
- n_constraints, n_unique_branches, max_cids_per_branch
- n_binding, n_nonbinding, binding_rate
- per_tier_binding_rate (tier 0 through 4)
- total_da_sp, captured_da_sp, branch_vc (collapsed)
- n_nulls per column
- score_p25, score_p50, score_p75
- sf_shape, sf_nan_count, sf_zero_columns

## 5. Execution Order

1. Run schema + structural checks (no DA needed)
2. Merge DA and produce 4 parquet files
3. Compute verification_summary.json
4. Human reviews the parquets (sort by tier, sort by realized_sp, spot-check branches)
5. Flag any anomalies found
