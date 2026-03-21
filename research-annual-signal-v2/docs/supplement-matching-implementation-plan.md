# Supplement Key Matching — Implementation Plan (v2)

**Date**: 2026-03-20
**Goal**: Integrate `MisoDaShadowPriceSupplement` key matching into the shared mapping layer so all consumers (GT, history features, evaluation) use consistent branch resolution.

**Scope**: #2 — full canonical mapping correction, not GT-only.

---

## Why

The current mapping layer (`bridge.py`) resolves DA CID → branch using only the SPICE bridge (CID-level matching). DA CIDs with new constraint_ids on branches already in the SPICE universe are dropped. For 2025-06, this loses 86 recoverable CIDs (22% of DA SP).

This affects:
- **GT labels**: `realized_shadow_price` is undercounted for some branches
- **History features**: `shadow_price_da` and `da_rank_value` miss DA binding on known branches
- **Evaluation**: Abs_SP@K is worse than it should be
- **v0c ranking**: da_rank_value (40% of v0c) is wrong for affected branches

All of these must be fixed together. A GT-only fix would create two inconsistent mapping truths.

## Publication versioning

**Do NOT overwrite V7.0.R1.** Publish the corrected signal as a new version:

```
OLD: TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1  (54+54 files, current)
NEW: TEST.Signal.MISO.SPICE_ANNUAL_V7.0B.R1 (54+54 files, with supplement matching)
```

This allows side-by-side comparison of the old and new signals.

---

## What Changes

### Layer 1: Shared mapping (`ml/bridge.py`)

Add a new function that wraps the existing CID-level mapping with supplement key fallback:

```python
def map_cids_to_branches_with_supplement(
    cid_df: pl.DataFrame,
    auction_type: str,
    auction_month: str,
    period_type: str,
    market_months: list[str],
) -> tuple[pl.DataFrame, dict]:
    """Map CIDs to branches: bridge first, then supplement key fallback.

    1. Try annual bridge (existing map_cids_to_branches)
    2. For still-unmapped CIDs, load supplement keys
    3. Construct branch: XF → key1+key3, LN → key2+key3
    4. Match against SPICE branches from the same bridge
    5. Return combined mapping with provenance
    """
```

The existing `map_cids_to_branches()` stays unchanged for backward compatibility.

### Layer 2: Ground truth (`ml/ground_truth.py`)

Replace `map_cids_to_branches()` call at line 82 with `map_cids_to_branches_with_supplement()`.

The monthly fallback (step 4) stays — it runs BEFORE supplement matching. Order:
1. Annual bridge
2. Monthly fallback
3. Supplement key matching ← NEW

### Layer 3: History features (`ml/history_features.py`)

The monthly binding table construction also maps DA CIDs to branches. It must use the same supplement-aware mapping.

Check: `compute_history_features()` calls `map_cids_to_branches()` for each historical month. Replace with `map_cids_to_branches_with_supplement()` where applicable.

### Layer 4: Nothing else changes

- `load_collapsed()` — density universe stays CID-based (no supplement needed, these are SPICE CIDs)
- `signal_publisher.py` — consumes scored model table, no mapping logic
- `evaluate.py` — consumes GT from model table
- `features.py` — consumes GT + history, no direct mapping

---

## Supplement data source

```
Path: /opt/data/xyz-dataset/modeling_data/miso/MISO_DA_SHADOW_PRICE_SUPPLEMENT.parquet
Columns used: constraint_id, key1, key2, key3, device_type, year, month
Coverage: 2014-2026, 128/129 CID-unmapped constraints have entries (2025-06)
```

### Matching rules

| device_type | Rule | Example |
|:-----------:|------|---------|
| XF | `key1 + " " + key3` | MNTCELO + TR6__2 → `MNTCELO TR6__2` |
| LN | `key2 + " " + key3` | MAPLEWINGE23_1 + 1 → `MAPLEWINGE23_1 1` |

Normalize whitespace (collapse multiple spaces to one) before matching against SPICE bridge branch_names.

---

## Diagnostics

Add to mapping diagnostics:
- `supplement_recovered_cids`: int
- `supplement_recovered_sp`: float
- `supplement_no_entry_cids`: int (CIDs with no supplement row)

---

## Execution order

1. **Implement** `map_cids_to_branches_with_supplement()` in `bridge.py`
2. **Wire into** `ground_truth.py` (replaces step 3 mapping call)
3. **Wire into** `history_features.py` (replaces monthly DA mapping calls)
4. **Write tests**:
   - Unit: known CID recoveries (511847 → MNTCELO TR6__2, 513025 → MAPLEWINGE23_1 1)
   - Unit: unrecoverable CID stays unmapped (513621)
   - Unit: CID-unmapped = Recovered + Residual
   - Integration: `build_ground_truth("2025-06", "aq2")` has more binding branches
   - Regression: 2021-06 GT changes minimally
   - Round-trip: total DA SP unchanged
5. **Run all 99 existing tests** — must still pass
6. **Re-run model ladder** (4 jobs: onpeak/offpeak × dev/holdout) — compare metrics V7.0 vs V7.0B
7. **Publish V7.0B.R1** for all PYs (do NOT overwrite V7.0.R1)
8. **Run verification** on V7.0B: same 2-slice inspection (2021-06/aq1, 2025-06/aq2)
9. **Compare V7.0 vs V7.0B**:
   - Per-group VC@K, Abs_SP@K, Recall@K differences
   - Per-group binding branch count changes
   - Tier 0 binding rate changes
   - Constraint overlap between old and new published sets

---

## Expected impact

### Small for 2019-2024 (0-2% SP recovered)
- da_rank_value changes slightly for a few branches
- v0c ranking barely moves
- Published signal nearly identical

### Significant for 2025-06 (22% SP recovered)
- Many branches get higher da_rank_value
- v0c ranking reshuffles for 2025-06 groups
- Published signal may change 50-100 constraints (out of 1000)
- Abs_SP@400 improves because more DA SP is credited to modeled branches

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Supplement data missing for some months | Check coverage across all PYs before implementing |
| device_type values beyond XF/LN | Fall through to LN rule (key2+key3). Log unknown types. |
| Normalization mismatch | Already verified whitespace collapsing matches bridge |
| History features path differs from GT path | Both will call the same `map_cids_to_branches_with_supplement()` |
| Breaking existing tests | Run all 99 tests. The new function is additive — existing mapping unchanged, supplement is fallback only. |
