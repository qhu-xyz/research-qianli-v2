# Supplement Key Matching — Implementation Plan (v3)

**Date**: 2026-03-21
**Goal**: Integrate `MisoDaShadowPriceSupplement` key matching into the shared mapping layer so all consumers (GT, history features) use consistent DA CID → branch resolution. Publication path unchanged — always publishes SPICE CIDs.

**Scope**: #2 — full canonical mapping correction. Supplement matching affects the DA→branch path only. The branch→SPICE CID publication path is NOT changed.

**Status**: Partially implemented. Code changes done, 2019-2024 published as V7.0B.R1, 2025-06 blocked by stale CID mapping cache. See "Known issues" section.

---

## How the pipeline works

Two separate mapping paths exist. Only one is being changed.

### Path 1: DA CID → branch (GT + history features) ← CHANGED

```
DA CID
  ├─ Annual bridge (CID-level match)
  ├─ Monthly f0 fallback (CID-level match)
  └─ Supplement key fallback ← NEW
       XF: key1 + key3 → branch_name
       LN: key2 + key3 → branch_name
  → branch_name → realized_shadow_price, da_rank_value, bf
```

This is used by `ground_truth.py` and `history_features.py` to compute:
- GT labels (realized_shadow_price per branch)
- History features (shadow_price_da, da_rank_value, bf_12, bfo_12)

Supplement matching recovers DA CIDs that have new constraint_ids but monitor branches already in the SPICE universe.

### Path 2: branch → SPICE CID (publication) ← NOT CHANGED

```
branch_name (scored by v0c)
  → load_cid_mapping() → SPICE CIDs from density universe
  → expand, dedup by SF, publish 1000 constraints
```

This is used by `signal_publisher.py`. It always publishes SPICE CIDs (from the density universe), never DA CIDs.

### How supplement matching affects publication

Supplement matching changes Path 1 → changes da_rank_value and bf → changes v0c scores → changes which branches rank in top 750 → changes which SPICE CIDs get published via Path 2.

The published CIDs are always SPICE CIDs. Only the RANKING changes.

---

## What Changes

### Layer 1: Shared mapping (`ml/bridge.py`) — DONE

Added:
- `load_supplement_keys()` — loads structured keys from `MisoDaShadowPriceSupplement`, cached
- `supplement_match_unmapped()` — XF: key1+key3, LN: key2+key3, match against SPICE branches
- `map_cids_to_branches_with_supplement()` — wraps existing bridge mapping with supplement fallback

Existing `map_cids_to_branches()` unchanged for backward compatibility.

### Layer 2: Ground truth (`ml/ground_truth.py`) — DONE

Added step 4b after monthly fallback:
1. Annual bridge
2. Monthly fallback
3. Supplement key matching ← NEW

Diagnostics include `supplement_recovered_cids`, `supplement_recovered_sp`, `supplement_no_entry`.

### Layer 3: History features (`ml/history_features.py`) — DONE

Pre-loads supplement keys once for all months (avoids per-month parquet scan). Applies supplement fallback after annual bridge + monthly fallback in `build_monthly_binding_table()`.

### Layer 4: Nothing else changes

- `load_collapsed()` — density universe stays SPICE-CID-only
- `load_cid_mapping()` — maps branches back to SPICE CIDs for publication. NOT changed.
- `signal_publisher.py` — consumes scored model table, expands branches to SPICE CIDs via `load_cid_mapping()`
- `evaluate.py`, `features.py` — consume GT/history from model table

---

## Known Issues (discovered during V7.0B publication)

### Stale CID mapping cache

`load_cid_mapping()` caches the branch↔CID mapping at `data/collapsed/{py}_{aq}_cid_map_*.parquet`. These caches were written by earlier runs and may contain stale entries.

**Symptom**: DA CID `131822` appeared in the CID mapping cache for 2025-06/aq1 with branch `SMITHOMU MITH`. This CID is not in the SPICE density distribution or bridge — it should never be in the cache.

**Impact**: The publisher tried to look up `flow_direction` for CID `131822` in the density signal score → not found → publish failed for all 2025-06 slices.

**Root cause**: The cache was written before the supplement matching changes. It is NOT caused by supplement matching — `load_collapsed()` does not use supplement matching.

**Fix**: Delete all CID mapping caches and let them rebuild from `load_collapsed()`:
```bash
rm data/collapsed/*cid_map*.parquet
```

Then republish 2025-06.

### 2025-06 not yet published as V7.0B

2019-06 through 2024-06: 48 files published as V7.0B.R1 (6 PYs × 4 quarters × 2 classes).
2025-06: blocked by stale cache. After cache deletion and rebuild, should publish cleanly.

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
| XF | `key1 + " " + key3 | MNTCELO + TR6__2 → `MNTCELO TR6__2` |
| LN | `key2 + " " + key3` | MAPLEWINGE23_1 + 1 → `MAPLEWINGE23_1 1` |

### Flow direction lookup

The publisher gets `flow_direction` from `MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet`, keyed by SPICE CID. Since the publisher only publishes SPICE CIDs (from `load_cid_mapping()`), every published CID should have a density signal score entry. DA CIDs should never appear in the publication path.

---

## Execution order

### Already done
1. Implemented `bridge.py` supplement functions ✓
2. Wired into `ground_truth.py` step 4b ✓
3. Wired into `history_features.py` with pre-loaded keys ✓
4. Tests: 98/99 pass (1 pre-existing) ✓
5. Model ladder: 4 jobs complete, results in `registry/{class}/m2b_{split}/` ✓
6. Published V7.0B.R1 for 2019-06 through 2024-06 (48 files) ✓

### Remaining
7. **Delete stale CID mapping caches** for 2025-06
8. **Republish 2025-06** as V7.0B.R1
9. **Run verification** on V7.0B: loss waterfall for 2025-06/aq2/offpeak
10. **Compare V7.0 vs V7.0B**:
    - Per-group VC@K, Abs_SP@K, Recall@K differences
    - Binding branch count changes
    - Constraint overlap between old and new published sets

---

## Results so far

### V7.0 vs V7.0B model ladder comparison (v0c)

| Slice | VC@400 delta | Abs_SP@400 delta | Recall@400 delta |
|-------|:---:|:---:|:---:|
| onpeak/dev | -0.004 | **+0.001** | -0.003 |
| onpeak/holdout | -0.008 | **+0.026** | -0.018 |
| offpeak/dev | +0.001 | **+0.003** | -0.002 |
| offpeak/holdout | -0.006 | **+0.027** | -0.018 |

- Abs_SP@400 improves on holdout (+2.6-2.7pp) — more DA SP credited to modeled branches
- VC@400 drops slightly (-0.4 to -0.8pp) — expected ranking reshuffle
- The ranking changes because da_rank_value and bf shift for branches with recovered DA SP

---

## Risks

| Risk | Status | Mitigation |
|------|--------|-----------|
| Stale CID mapping cache | **FOUND** | Delete caches, rebuild. See Known Issues. |
| DA CIDs leaking into publication | **FOUND** | Root cause: stale cache, not supplement code. Cache rebuild fixes. |
| Supplement data missing for some months | OK | 128/129 CIDs covered for 2025-06 |
| device_type values beyond XF/LN | OK | Falls through to LN rule. Only XF and LN observed. |
| Normalization mismatch | OK | Whitespace collapsing verified against bridge |
| Breaking existing tests | OK | 98/99 pass. 1 pre-existing failure unrelated. |
