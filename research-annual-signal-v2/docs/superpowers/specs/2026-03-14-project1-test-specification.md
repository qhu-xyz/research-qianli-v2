# Comprehensive Test Specification: 90 Cases Across 8 Layers

**Date**: 2026-03-15 (rev 3 — class-specific pipeline + verified checkpoints)
**Scope**: Class-specific pipeline, production publication, path rating, ground truth checkpoints

---

## Test Architecture

```
Layer 1: Config / Schema Contract (10)        — can the system load?
Layer 2: Data Loading / Bridge / GT (15)       — is class-specific data correct?
Layer 3: Scoring / Merge / Evaluation (15)     — do class-specific models work?
Layer 4: Registry / Reproducibility (10)       — can we reproduce saved results?
Layer 5: Ground Truth Checkpoints (10)         — do our computations match verified references?
Layer 6: Project 1 Publication (15)            — is the published signal correct?
Layer 7: Project 2 Path Rating (10)            — does path rating work correctly?
Layer 8: Annual-Band Join / Segmentation (5)   — does the validation pipeline work?
```

---

## Layer 1: Config / Schema Contract (10 cases)

| # | Test | Assert |
|---|------|--------|
| 1 | PHASE5_K_LEVELS exists | `[150, 200, 300, 400]` |
| 2 | DANGEROUS_THRESHOLD exists | `50000.0` |
| 3 | CLASS_TYPES defined | `["onpeak", "offpeak"]` |
| 4 | Class-specific BF mapping | onpeak → `bf_12`, offpeak → `bfo_12` |
| 5 | Class-specific target mapping | onpeak → `onpeak_sp`, offpeak → `offpeak_sp` |
| 6 | Class-specific NB flag mapping | onpeak → `nb_onpeak_12`, offpeak → `nb_offpeak_12` |
| 7 | evaluate_group computes @150/@300/@400 | Extra K metrics include 150, 200, 300, 400 |
| 8 | evaluate_group computes dangerous metrics | `Dang_Recall@K`, `Dang_SP_Ratio@K` present |
| 9 | merge_tracks accepts tau param | Backward compatible |
| 10 | Old TIER1_GATE_METRICS unchanged | @50/@100 gates still work |

## Layer 2: Data Loading / Class-Specific GT (15 cases)

| # | Test | Assert |
|---|------|--------|
| 11 | load_cid_mapping returns constraint-branch pairs | Has constraint_id, branch_name, is_active |
| 12 | CID mapping is 1-to-1 | Each constraint_id → exactly 1 branch_name |
| 13 | Bridge handles ambiguous CIDs | Ambiguous dropped, not duplicated |
| 14 | Ground truth onpeak: uses `onpeak_sp` as target | `realized_shadow_price = onpeak_sp` when class_type="onpeak" |
| 15 | Ground truth offpeak: uses `offpeak_sp` as target | `realized_shadow_price = offpeak_sp` when class_type="offpeak" |
| 16 | GT tiered labels recomputed per class | Tier tertiles based on class-specific SP, not combined |
| 17 | `onpeak_sp + offpeak_sp ≈ combined realized_shadow_price` | Verify sum relationship |
| 18 | Cohort onpeak: dormant = `bf_12 == 0` | NOT `bf_combined_12` |
| 19 | Cohort offpeak: dormant = `bfo_12 == 0` | NOT `bf_combined_12` |
| 20 | Cross-class dormancy: some branches dormant in one class, active in other | Count > 0 |
| 21 | NB flags per class: `nb_onpeak_12` ≠ `nb_offpeak_12` for some branches | Flags differ |
| 22 | History features expose cross-class columns | Onpeak model has access to `bfo_12` |
| 23 | shadow_price_da is branch-level | All constraints on same branch have identical shadow_price_da |
| 24 | shadow_price_da differs by class_type | onpeak spda ≠ offpeak spda for some branches |
| 25 | BF temporal leakage: uses only months < cutoff | Same rule per class |

## Layer 3: Scoring / Merge / Evaluation (15 cases)

| # | Test | Assert |
|---|------|--------|
| 26 | v0c onpeak uses `bf_12` not `bf_combined_12` | Formula input is class-specific BF |
| 27 | v0c offpeak uses `bfo_12` | Same |
| 28 | v0c uses class-specific `da_rank_value` | Different scores for same branch across class types |
| 29 | NB model onpeak trains on `bf_12 == 0` population | NOT `bf_combined_12 == 0` |
| 30 | NB model includes cross-class BF as feature | `bfo_12` is a feature in onpeak NB model |
| 31 | Blend score = v0c + α × normalized NB for dormant | Established unchanged |
| 32 | Blend with α=0 equals v0c solo | Identical scores |
| 33 | evaluate_group VC@K correct with class-specific target | Manual computation matches |
| 34 | evaluate_group Dang_Recall@K uses class-specific threshold | Correct threshold per class |
| 35 | evaluate_group NB12_SP@K uses class-specific NB12 SP | Ratio of class-specific captured/total |
| 36 | Paired scorecard computation correct | 0.5 × score_lo + 0.5 × score_hi |
| 37 | Gate check uses class-specific solo baseline | Not combined baseline |
| 38 | Onpeak and offpeak champions can differ | Different winning configs allowed |
| 39 | Cross-class feature has signal | Spearman(offpeak_sp, bf_12) > 0.3 |
| 40 | Dangerous thresholds per class | $20k (low) and $40k (high) per class |

## Layer 4: Registry / Reproducibility (10 cases)

| # | Test | Assert |
|---|------|--------|
| 41 | Phase 5 combined-ctype registry exists | `phase5_final_150_300/`, `phase5_final_200_400/` |
| 42 | Phase 5 configs are valid JSON | Has "champion", "paired_score" |
| 43 | Phase 6 class-specific registry exists (when built) | Separate onpeak/offpeak entries |
| 44 | v0c baseline registry exists | Has holdout metrics |
| 45 | save_experiment creates all expected files | config.json, metrics.json |
| 46 | load_metrics round-trip | save then load = identical |
| 47 | Phase 5 champion is combined-ctype baseline | Correctly labeled |
| 48 | No registry version claims class-specific until Phase 6 | Consistency check |
| 49 | run_phase5_reeval.py executes without crash | Exit code 0 |
| 50 | Script output matches saved registry | Regression check |

## Layer 5: Ground Truth Checkpoints (10 cases)

Verified reference values that must pass before implementation proceeds.

| # | Test | Assert | Reference |
|---|------|--------|-----------|
| 51 | **SF exact parity** | V6.1 SF = mean(raw SPICE SF across outage dates) for aq1 | max_diff = 0.000000 |
| 52 | **SF quarter construction** | 3 market months per quarter, 10-11 outage dates each | 32 total SF matrices for aq1 |
| 53 | **Constraint key mapping** | V6.1 ∩ density = 100%, V6.1 ∩ raw SF = 100% | No normalization needed |
| 54 | **shadow_price_da branch invariant** | All constraints on same branch have identical spda in V6.1 | 0 branches with multiple values |
| 55 | **V6.1 class-type difference** | shadow_price_da max diff > $1,000 across onpeak/offpeak | Verified: $7,318 |
| 56 | **V6.1 constraint set differs by class** | onpeak ≠ offpeak constraint sets | Verified: 78% overlap |
| 57 | **Pre-dedup universe size** | ~12,800-13,000 constraints per (PY, aq) | Class-agnostic |
| 58 | **V6.1 published size** | ~280-480 per (PY, aq, ctype) | NOT ~3,000 |
| 59 | **Raw SF source** | `MISO_SPICE_SF.parquet` has 12 market months per annual PY | NOT pw_data (1 month) |
| 60 | **Pnodes stable within PY** | 100% pnode overlap across aq1-aq4 | Verified |

## Layer 6: Project 1 Publication (15 cases)

| # | Test | Assert |
|---|------|--------|
| 61 | Published constraints has exactly 20 data columns | Matches V6.1 (NOT 21 — __index_level_0__ is parquet index) |
| 62 | Column dtypes match V6.1 | Per-column comparison |
| 63 | Constraint index format correct | Regex `r"^.+\|[+-]?1\|spice$"` |
| 64 | Constraint index unique | No duplicates |
| 65 | SF columns exactly match constraints index | `set(sf.columns) == set(cstrs.index)` |
| 66 | SF no NaN, values < 10.0 | Physically reasonable |
| 67 | shadow_price = shadow_sign (frozen contract) | All values ∈ {-1, 1} or {-1.0, 1.0} |
| 68 | Metadata matches V6.1 for overlapping constraints | shadow_price_da, da_rank_value exact match |
| 69 | Overlap with V6.1 ≥ 70% | Sufficient coverage |
| 70 | Tier distribution: all 5 tiers present, not degenerate | Each tier ≥ 5% |
| 71 | Rank monotonic with tier | Higher rank → lower tier number |
| 72 | Post-dedup: max 3 per branch per bus_key_group | Dedup enforced |
| 73 | Separate onpeak and offpeak published | Different constraint sets per class_type |
| 74 | ConstraintsSignal.save_data then load_data round-trip | Same DataFrame |
| 75 | ShiftFactorSignal.save_data then load_data round-trip | Same DataFrame |

## Layer 7: Project 2 Path Rating (10 cases)

| # | Test | Assert |
|---|------|--------|
| 76 | MisoNodalReplacement.load_data() returns DataFrame | No crash |
| 77 | Nodal replacement resolves chains | A→B→C becomes A→C |
| 78 | Replaced nodes exist in SF index | Valid pnode_ids |
| 79 | Path exposure = sf[sink] - sf[source] | Manual computation matches |
| 80 | Aligned exposure = shadow_sign × path_exposure | Sign adjustment correct |
| 81 | Path rating score non-negative | `max(0, aligned_exposure)` ensures this |
| 82 | Tier weight: tier0=1.0, tier1=0.5, others=0 | Matches spec |
| 83 | high_rated: any tier-0 aligned_exposure > 0.1 | Definition matches |
| 84 | Segments exhaustive | high + medium + low + unrated = all paths |
| 85 | Segments disjoint | No path in multiple segments |

## Layer 8: Annual-Band Join / Segmentation (5 cases)

| # | Test | Assert |
|---|------|--------|
| 86 | Join uses all 6 keys | planning_year, round, period_type, class_type, source_id, sink_id |
| 87 | Join produces no duplicates | One row per unique key combination |
| 88 | Per-segment metrics decompose to full-set | Weighted average ≈ full-set |
| 89 | Row-level data used, not metrics.json | Correct data source |
| 90 | Segment sizes reasonable | High ≥ 5%, ≤ 50% of paths |

---

## Test File Organization

```
tests/
  test_config.py                — Layer 1 (extend existing)
  test_data_loader.py           — Layer 2 (extend existing)
  test_ground_truth_class.py    — Layer 2 (NEW: class-specific GT)
  test_evaluate.py              — Layer 3 (extend existing)
  test_merge.py                 — Layer 3 (extend existing)
  test_registry.py              — Layer 4 (extend existing)
  test_checkpoints.py           — Layer 5 (NEW: verified ground truth references)
  test_signal_publisher.py      — Layer 6 (NEW: requires Project 1)
  test_path_rating.py           — Layer 7 (NEW: requires Project 2)
  test_band_validation.py       — Layer 8 (NEW: requires Project 2)
```

## Execution Order

1. **Now**: Layer 5 (ground truth checkpoints) — codify verified results as tests
2. **With class-specific pipeline**: Layers 1-4 extensions
3. **With Project 1**: Layer 6
4. **With Project 2**: Layers 7-8
