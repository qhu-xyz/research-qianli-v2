# Universe Catalog — MISO Annual

**Date**: 2026-03-24

Every model, benchmark, and comparison must name its universe explicitly. This catalog defines the named universes used in the MISO annual signal pipeline.

---

## Universes

### `miso_annual_branch_active_v1`

**Description**: All annual branches with at least one active SPICE CID.

**Inclusion rule**:
1. Load SPICE density distribution for `(planning_year, aq_quarter, market_round)`
2. Compute `right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` per CID across all outage dates
3. A CID is "active" if `right_tail_max >= 0.0003467728739657263` (calibrated on 2024-06/aq1)
4. Map CIDs to branches via bridge for `(planning_year, aq_quarter, market_round)`
5. A branch is included if it has at least one active CID

**Key properties**:
- Size: ~2,500-2,800 branches per (PY, quarter)
- Round-sensitive: different `market_round` may produce different CID sets and therefore different branch sets
- Ctype-independent: same branch universe for onpeak and offpeak (class type affects features and targets, not universe)
- Source: `ml/data_loader.py:load_collapsed()`

**Used by**:
- `miso_annual_v0c_formula_v1`
- `miso_annual_bucket_6_20_v1`
- All internal model comparisons

**Calibration artifact**: `registry/threshold_calibration/threshold.json`

---

### `miso_annual_v44_published_v1`

**Description**: Branches published in the V4.4 external signal.

**Inclusion rule**:
1. Load V4.4 signal parquet for `(planning_year, aq_quarter, class_type, market_round)`
2. Filter to rows where `equipment != ""`
3. Each unique `equipment` value is one branch

**Key properties**:
- Size: ~1,200 branches per (PY, quarter, ctype)
- External: not reproducible from our pipeline
- Round-specific: V4.4 has separate R1/R2/R3 directories
- Ctype-specific: onpeak and offpeak may differ

**Signal path**: `TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R{round}/{planning_year}/{aq_quarter}/{class_type}/`

**Rank direction**: lower rank = better (ascending sort)

**Used by**:
- External benchmark comparisons only
- Not used as a training or scoring universe for internal models

---

### `miso_annual_overlap_v1`

**Description**: Intersection of `miso_annual_branch_active_v1` and `miso_annual_v44_published_v1`.

**Inclusion rule**:
- Branch must appear in both universes for the same `(planning_year, aq_quarter, class_type, market_round)`

**Key properties**:
- Size: ~800-1,100 branches per cell (depends on V4.4 coverage)
- Used only for ranking quality diagnostics (`rank_overlap`)
- Never used for production top-K selection

**Used by**:
- Overlap-only comparison tables in metric contract
- Not used for deployment decisions

---

### `miso_annual_nb_dormant_v1`

**Description**: Subset of `miso_annual_branch_active_v1` where the branch has no class-specific binding history in the last 12 months.

**Inclusion rule**:
1. Start from `miso_annual_branch_active_v1`
2. Compute class-specific BF_12 (onpeak: `bf_12`, offpeak: `bfo_12`)
3. A branch is dormant if its class-specific BF_12 == 0

**Key properties**:
- Size: varies by ctype and PY (~60-70% of full universe)
- Ctype-specific: a branch may be dormant for onpeak but not offpeak
- Round-sensitive: partial-month binding near cutoff can change dormancy status

**Used by**:
- NB reserved-slot allocation in R30/R50 deployment policies
- NB-specialist model training (when applicable)
- NB metric computation (NB_SP, NB_Recall, etc.)

---

## Universe rules

1. **Every model spec must name one universe**. If the universe is `miso_annual_branch_active_v1`, say so. If it's a filtered subset, define it.

2. **Cross-universe comparisons must use absolute metrics** (`Abs_SP@K`, `Abs_Binders@K`) not within-universe metrics (`VC@K`). VC@K is only valid when comparing models on the same universe.

3. **Overlap comparisons must be labeled as such**. `rank_overlap` numbers cannot be presented as `rank_native`.

4. **V4.4 is a benchmark, not an internal universe**. Do not use V4.4's branch list as a training or scoring universe for internal models.

5. **Universe definitions are round-sensitive**. A universe ID without `market_round` in context is ambiguous. Specs must always declare the round alongside the universe.

---

## Future universes (not yet defined)

- `pjm_annual_branch_active_v1` — PJM equivalent, pending PJM pipeline
- `miso_annual_branch_active_v2` — if threshold or bridge mapping changes
- `miso_annual_combined_ctype_v1` — if combined-ctype universe is needed (currently all models use class-independent universe)
