# Project 1: Test Specification — Annual Signal Publication

**Date**: 2026-03-14
**Coverage target**: 90 test cases across 6 categories

---

## Verification Sources

| Source | What it proves | How we access it |
|--------|---------------|-----------------|
| V6.1 annual signal | Schema correctness, metadata values | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` |
| pmodel loader | Consumer compatibility | `pmodel.base.ftr24.v1.constraint_loader.load_constraints_and_tier_set()` |
| Phase 5 results | Ranking correctness | `registry/phase5_final_*/metrics.json` |

---

## Category 1: Schema Compliance (15 cases)

Tests that the published constraints parquet matches V6.1's exact schema.

| # | Test | Assert |
|---|------|--------|
| 1 | Column count | `len(df.columns) == 21` |
| 2 | Column names match V6.1 | `set(df.columns) == set(v61.columns)` |
| 3 | Column dtypes match V6.1 | Each column has same dtype as V6.1 |
| 4 | Index format | Every index value matches regex `r"^\d+\|[+-]?1\|spice$"` |
| 5 | Index uniqueness | `df.index.is_unique` |
| 6 | constraint_id is string, non-empty | `df["constraint_id"].str.len().min() > 0` |
| 7 | branch_name is string, non-empty | `df["branch_name"].str.len().min() > 0` |
| 8 | flow_direction values | `set(df["flow_direction"].unique()) ⊆ {-1, 1}` |
| 9 | shadow_sign values | `set(df["shadow_sign"].unique()) ⊆ {-1, 1}` |
| 10 | tier values | `set(df["tier"].unique()) ⊆ {0, 1, 2, 3, 4}` |
| 11 | tier dtype is integer | `df["tier"].dtype` in integer types |
| 12 | shadow_price no NaN | `df["shadow_price"].isna().sum() == 0` |
| 13 | shadow_sign no NaN | `df["shadow_sign"].isna().sum() == 0` |
| 14 | shadow_price_da no NaN | `df["shadow_price_da"].isna().sum() == 0` |
| 15 | __index_level_0__ matches index | `(df["__index_level_0__"] == df.index).all()` |

## Category 2: SF Matrix Compliance (12 cases)

Tests that the published SF parquet is valid and aligned with constraints.

| # | Test | Assert |
|---|------|--------|
| 16 | SF columns match constraints index | `set(sf.columns) == set(cstrs.index)` |
| 17 | SF index (pnode_ids) are strings | `sf.index.dtype == object` (string) |
| 18 | SF index non-empty | `sf.index.str.len().min() > 0` |
| 19 | SF no NaN | `sf.isna().sum().sum() == 0` |
| 20 | SF dtype is float | `sf.dtypes.unique()` all float |
| 21 | SF has reasonable row count | `500 ≤ len(sf) ≤ 5000` (pnodes) |
| 22 | SF has reasonable column count | `100 ≤ len(sf.columns) ≤ 5000` (constraints) |
| 23 | SF is not all zeros | `(sf != 0).any().any()` |
| 24 | SF sparsity reasonable | `(sf == 0).mean().mean()` between 0.5 and 0.99 |
| 25 | SF values reasonable range | `sf.abs().max().max() < 10.0` (SFs are fractions) |
| 26 | No duplicate pnode_ids in SF index | `sf.index.is_unique` |
| 27 | No duplicate constraint keys in SF columns | `sf.columns.is_unique` |

## Category 3: Metadata Content Correctness (18 cases)

Tests that inherited metadata matches V6.1 source and derived columns are correct.

| # | Test | Assert |
|---|------|--------|
| 28 | shadow_price_da matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 29 | da_rank_value matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 30 | ori_mean matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 31 | mix_mean matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 32 | density_mix_rank_value matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 33 | density_ori_rank_value matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 34 | rank_ori matches V6.1 | For overlapping constraints: `max_abs_diff < 1e-6` |
| 35 | shadow_sign matches V6.1 | For overlapping constraints: exact match |
| 36 | flow_direction matches V6.1 | For overlapping constraints: exact match |
| 37 | branch_name matches V6.1 | For overlapping constraints: exact match |
| 38 | bus_key matches V6.1 | For overlapping constraints: exact match |
| 39 | bus_key_group matches V6.1 | For overlapping constraints: exact match |
| 40 | equipment matches V6.1 | For overlapping constraints: exact match |
| 41 | Overlap with V6.1 ≥ 80% | `len(overlap) / len(v61) ≥ 0.80` |
| 42 | shadow_price not all zero | `(df["shadow_price"] != 0).sum() > 0` |
| 43 | shadow_price not all identical | `df["shadow_price"].nunique() > 1` |
| 44 | rank column is finite | `df["rank"].isna().sum() == 0` and no inf |
| 45 | rank is monotonic with tier | within each tier, `rank` values don't overlap with adjacent tiers |

## Category 4: Tier and Dedup Correctness (15 cases)

Tests that tier assignment and constraint dedup follow the contract.

| # | Test | Assert |
|---|------|--------|
| 46 | All 5 tiers present | `set(df["tier"].unique()) == {0, 1, 2, 3, 4}` |
| 47 | Tier 0 is smallest | `(df["tier"] == 0).sum() ≤ (df["tier"] == 4).sum()` |
| 48 | Tier distribution not degenerate | Each tier has ≥ 5% of constraints |
| 49 | Lower tier = higher rank score | `df[df["tier"]==0]["rank"].min() > df[df["tier"]==4]["rank"].max()` |
| 50 | Max 3 constraints per branch per bus_key_group | Group by (bus_key_group, branch_name): max count ≤ 3 |
| 51 | SF Chebyshev ≥ 0.05 within bus_key_group | For selected constraints in same group: pairwise Chebyshev ≥ 0.05 |
| 52 | No duplicate (constraint_id, shadow_sign) | `df.groupby(["constraint_id", "shadow_sign"]).size().max() == 1` |
| 53 | Constraint count reasonable | `200 ≤ len(df) ≤ 4000` per (aq, class_type) |
| 54 | Tier 0 count reasonable | `20 ≤ (df["tier"]==0).sum() ≤ 500` |
| 55 | Every branch_name appears in CID mapping | All published branch_names exist in `load_cid_mapping()` |
| 56 | Dedup reduces constraint count | `len(post_dedup) < len(pre_dedup)` |
| 57 | Dedup preserves top-ranked constraints | Tier-0 constraints are all in post-dedup set |
| 58 | Same-branch correlation ≥ 0.1 | For selected same-branch constraints: SF correlation ≥ 0.1 |
| 59 | Cross-constraint correlation ≥ -0.21 | For selected constraints in same group: SF correlation ≥ -0.21 |
| 60 | Constraint count similar to V6.1 | `0.5 ≤ len(ours) / len(v61) ≤ 2.0` per (aq, class_type) |

## Category 5: Round-Trip and Consumer Compatibility (15 cases)

Tests that the published signal is consumable by pmodel.

| # | Test | Assert |
|---|------|--------|
| 61 | Save + load constraints — no crash | `ConstraintsSignal.load_data()` succeeds |
| 62 | Save + load SF — no crash | `ShiftFactorSignal.load_data()` succeeds |
| 63 | Loaded constraints shape matches saved | Same row count and column count |
| 64 | Loaded SF shape matches saved | Same row and column count |
| 65 | Loaded constraints index matches loaded SF columns | `set(cstrs.index) == set(sf.columns)` |
| 66 | Loaded constraints values match saved | Max abs diff < 1e-10 for floats |
| 67 | pmodel load_constraints_and_tier_set — no crash | Loads without error |
| 68 | pmodel tier sets build correctly | tier_set[0] ⊆ tier_set[1] ⊆ ... (cumulative) |
| 69 | pmodel SF intersection ≥ 80% | `len(common_constraints) / len(cstrs) ≥ 0.80` |
| 70 | pmodel shadow_price clipping works | After clip: `abs(shadow_price).max() ≤ clip_value` |
| 71 | pmodel rename limit→rating — no crash | If limit column present, rename succeeds |
| 72 | pmodel fillna shadow_price — no change | Already no NaN, so fillna is no-op |
| 73 | Signal loads for all 4 aq | aq1, aq2, aq3, aq4 all load successfully |
| 74 | Signal loads for both class_types | onpeak, offpeak both load successfully |
| 75 | Signal loads for multiple PYs | At least 2025-06 and 2024-06 load |

## Category 6: Ranking Consistency and Cross-Validation (15 cases)

Tests that the published ranking reproduces Phase 5 evaluation results.

| # | Test | Assert |
|---|------|--------|
| 76 | Blend scores match Phase 5 | v0c + 0.05×NB produces same ranking as Phase 5 |
| 77 | Dormant constraints have rank reflecting NB boost | Dormant with high NB score → low tier |
| 78 | Established constraints ranked by v0c | Established constraint rank ∝ v0c score |
| 79 | top-K from published rank reproduces Phase 5 VC | At K=300: VC within 0.001 of Phase 5 result |
| 80 | top-K from published rank reproduces Phase 5 NB12_SP | At K=300: NB12_SP within 0.01 |
| 81 | Constraint universe stable across aq for same PY | ≥ 50% overlap between aq1 and aq2 |
| 82 | shadow_price_da identical across aq for same constraint | Historical value doesn't change by quarter |
| 83 | Tier breaks computed per (aq, class_type) | Tier boundaries differ between aq1 and aq2 |
| 84 | SF matrices share pnode universe within PY | ≥ 90% pnode overlap across aq |
| 85 | Published signal for 2025-06 holdout | Can load and evaluate; results match Phase 5 holdout |
| 86 | Published signal for 2024-06 dev | Can load and evaluate; results match Phase 5 dev |
| 87 | Dormant branches with 0 blend score → high tier | bf_combined_12=0 + low NB score → tier 3-4 |
| 88 | History_zero branches excluded from constraints | No constraint whose branch_name has has_hist_da=False |
| 89 | V6.1 rank_ori and our rank column differ | Our ranking ≠ V6.1 ranking (we have NB blend) |
| 90 | Top-tier constraint set is production-ready | Pass all schema + content + round-trip tests simultaneously |

---

## Test Organization

```
tests/
  test_signal_publisher.py       # Categories 1-4 (schema, SF, metadata, dedup)
  test_signal_roundtrip.py       # Category 5 (consumer compatibility)
  test_signal_ranking.py         # Category 6 (ranking consistency)
```

Each test file is independent. Categories 1-4 can run without pmodel.
Category 5 requires pmodel imports. Category 6 requires Phase 5 registry artifacts.

## Test Data

- **V6.1 reference**: load from production path for 2024-06/aq1/onpeak
- **Our signal**: generate for same PY/aq/class, save to temp dir
- **Phase 5 results**: load from `registry/phase5_final_*/metrics.json`
- **SF reference**: load V6.1 SF for same PY/aq/class
