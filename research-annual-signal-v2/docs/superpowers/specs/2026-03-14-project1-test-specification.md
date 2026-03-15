# Comprehensive Test Specification: 90-Case Layered Suite

**Date**: 2026-03-14
**Scope**: Full pipeline — existing ML library through Project 1 publication and Project 2 path rating

> **NOTE**: Layers 1-5 test the current combined-ctype pipeline. Once the class-specific
> pipeline is built (see `2026-03-15-class-specific-pipeline-design.md`), additional
> class-specific test cases will be needed for: class-specific targets, BF, cohorts,
> and per-class champion validation.

---

## Test Architecture

```
Layer 1: Config / Schema Contract (10)     — can the system even load?
Layer 2: Data Loading / Bridge / GT (15)   — is the data pipeline correct?
Layer 3: Scoring / Merge / Evaluation (15) — do models produce correct outputs?
Layer 4: Registry / Reproducibility (10)   — can we reproduce saved results?
Layer 5: Phase 5 Champion Regression (10)  — do champion results still hold?
Layer 6: Project 1 Publication (15)        — is the published signal correct?
Layer 7: Project 2 Path Rating (10)        — does path rating work correctly?
Layer 8: Annual-Band Join / Segmentation (5) — does the validation pipeline work?
```

---

## Layer 1: Config / Schema Contract (10 cases)

Unit tests for `ml/config.py` and `ml/evaluate.py` contracts.

| # | Test | Assert |
|---|------|--------|
| 1 | PHASE5_K_LEVELS exists | `[150, 200, 300, 400]` |
| 2 | DANGEROUS_THRESHOLD exists | `50000.0` |
| 3 | EVAL_SPLITS has dev + holdout | Both split types present, train/eval PYs correct |
| 4 | DEV_GROUPS count | 12 groups (3 PYs × 4 quarters) |
| 5 | HOLDOUT_GROUPS count | 3 groups (2025-06 × aq1/aq2/aq3) |
| 6 | evaluate_group computes @150/@300 | Extra K metrics include 150, 200, 300, 400 |
| 7 | evaluate_group computes dangerous metrics | `Dang_Recall@K`, `Dang_SP_Ratio@K`, `Dang_Count@K` present |
| 8 | check_nb_threshold accepts k param | `check_nb_threshold(per_group, groups, k=300)` works |
| 9 | merge_tracks accepts tau param | `merge_tracks(a, b, k=300, r=30, tau=0.5)` works |
| 10 | TIER1_GATE_METRICS unchanged | Old @50/@100 gates still work (backward compat) |

## Layer 2: Data Loading / Bridge / History / GT (15 cases)

Unit + integration tests for data pipeline modules.

| # | Test | Assert |
|---|------|--------|
| 11 | load_raw_density returns correct columns | Has all bin columns + constraint_id + outage_date |
| 12 | load_collapsed returns branch-level features | Unique branch_name, has bin_*_cid_max columns |
| 13 | load_cid_mapping returns constraint-branch pairs | Has constraint_id, branch_name, is_active |
| 14 | CID mapping is 1-to-1 | Each constraint_id maps to exactly 1 branch_name |
| 15 | Bridge mapping handles ambiguous CIDs | Ambiguous CIDs (>1 branch) are dropped, not duplicated |
| 16 | Ground truth has tiered labels | label_tier ∈ {0, 1, 2, 3}, tier 0 = non-binding |
| 17 | GT combined ctype | onpeak_sp + offpeak_sp = realized_shadow_price |
| 18 | History features: BF temporal leakage | bf_combined_12 uses only months < cutoff |
| 19 | History features: BF values in [0, 1] | All bf_* columns between 0 and 1 |
| 20 | Model table unique branches | No duplicate branch_names per (PY, aq) |
| 21 | Cohort assignment correct | established ↔ bf_combined_12 > 0, dormant ↔ bf=0 + has_hist_da |
| 22 | NB detection aligned with cohort | All is_nb_12=True branches are dormant or zero-history |
| 23 | compute_recency_features returns correct shape | One row per dormant branch |
| 24 | months_since_last_bind ≥ 12 for dormant | By definition, dormant branches haven't bound in 12 months |
| 25 | compute_density_shape returns correct columns | tail_sum_ge_100, density_entropy, etc. |

## Layer 3: Scoring / Merge / Evaluation (15 cases)

Tests for model scoring, merge logic, and metric computation.

| # | Test | Assert |
|---|------|--------|
| 26 | v0c formula gives dormant branches nonzero scores | da_rank + density contribute even when bf=0 |
| 27 | v0c scores are in [0, 1] | min ≥ 0, max ≤ 1 |
| 28 | NB model (tiered) trains without crash | LightGBM binary on dormant population |
| 29 | NB model (sqrt) trains without crash | LightGBM binary with sqrt(SP) weights |
| 30 | NB model predictions are probabilities | All in [0, 1] for logistic, ≥0 for lgbm |
| 31 | Blend score = v0c + α × normalized NB | For dormant: score > v0c_score; for established: score = v0c_score |
| 32 | Blend with α=0 equals v0c solo | Identical scores for all branches |
| 33 | merge_tracks with tau filters correctly | Only Track B branches with score ≥ tau get slots |
| 34 | merge_tracks R=0 gives all Track A | No Track B branches in top-K |
| 35 | evaluate_group VC@K is correct | Manual computation matches |
| 36 | evaluate_group Dang_Recall@K is correct | Manual computation matches |
| 37 | evaluate_group NB12_SP@K is correct | Ratio of captured NB12 SP / total NB12 SP |
| 38 | _append_zero_history keeps universe intact | history_zero in evaluation but not in top-K |
| 39 | Paired scorecard computation correct | 0.5 × score_lo + 0.5 × score_hi with correct weights |
| 40 | Gate check: VC regression ≤ 2% | Candidates within 0.02 of solo baseline pass |

## Layer 4: Registry / Reproducibility (10 cases)

Tests that saved artifacts are correct and reproducible.

| # | Test | Assert |
|---|------|--------|
| 41 | phase5_final_150_300 config exists and is valid JSON | Loads, has "champion", "paired_score" |
| 42 | phase5_final_150_300 metrics exists and is valid JSON | Loads, has "per_group_lo", "per_group_hi" |
| 43 | phase5_final_200_400 config exists and is valid JSON | Same structure |
| 44 | phase5_final_200_400 metrics exists and is valid JSON | Same structure |
| 45 | v0c registry has holdout metrics | per_group has 2025-06/aq1, aq2, aq3 |
| 46 | v3a registry has holdout metrics | Same |
| 47 | save_experiment creates all expected files | config.json, metrics.json, gate_results.json |
| 48 | load_metrics round-trip | save then load produces identical dict |
| 49 | Registry versions don't conflict | No two versions claim to be champion for same K pair |
| 50 | All registry configs have "version" field | Structural consistency |

## Layer 5: Phase 5 Champion Regression (10 cases)

Re-run Phase 5 script and verify results match saved artifacts.

| # | Test | Assert |
|---|------|--------|
| 51 | run_phase5_reeval.py executes without crash | Exit code 0 |
| 52 | (150,300) champion is S1_sqrt_a0.05 | Matches registry/phase5_final_150_300/config.json |
| 53 | (200,400) champion is C1_a0.05 | Matches registry/phase5_final_200_400/config.json |
| 54 | (150,300) champion score within tolerance | `|score - 0.5023| < 0.001` |
| 55 | (200,400) champion score within tolerance | `|score - 0.5716| < 0.001` |
| 56 | Holdout VC@300 for blend matches saved | `|VC - 0.7237| < 0.005` |
| 57 | Holdout NB12_SP@300 for blend matches saved | `|NB12_SP - 0.2159| < 0.01` |
| 58 | Holdout DangR@300 for blend matches saved | `|DangR - 0.8758| < 0.005` |
| 59 | v0c solo holdout VC@300 matches saved | `|VC - 0.7195| < 0.005` |
| 60 | v3a fails DangR gate at (200,400) | v3a solo DangR@200 < v0c DangR@200 - 0.05 |

## Layer 6: Project 1 Publication (15 cases)

Tests for the signal publication pipeline.

| # | Test | Assert |
|---|------|--------|
| 61 | Published constraints has exactly 21 columns | Same names as V6.1 |
| 62 | Published constraints column dtypes match V6.1 | Per-column dtype comparison |
| 63 | Constraint index format correct | Regex `r"^.+\|[+-]?1\|spice$"` for all rows |
| 64 | Constraint index unique | No duplicates |
| 65 | SF columns exactly match constraints index | `set(sf.columns) == set(cstrs.index)` |
| 66 | SF no NaN | `sf.isna().sum().sum() == 0` |
| 67 | SF values in reasonable range | `sf.abs().max().max() < 10.0` |
| 68 | Metadata matches V6.1 for overlapping constraints | shadow_price_da, da_rank_value, ori_mean, etc. exact match |
| 69 | Overlap with V6.1 ≥ 80% | Sufficient constraint universe coverage |
| 70 | Tier distribution not degenerate | Each tier has ≥ 5% of constraints |
| 71 | Rank monotonic with tier | Higher rank → lower tier number |
| 72 | Post-dedup: max 3 per branch per bus_key_group | Dedup cap enforced |
| 73 | Post-dedup: Chebyshev ≥ 0.05 within group | SF distinctiveness enforced |
| 74 | ConstraintsSignal.save_data then load_data round-trip | Same DataFrame (within float tolerance) |
| 75 | ShiftFactorSignal.save_data then load_data round-trip | Same DataFrame |

## Layer 7: Project 2 Path Rating (10 cases)

Tests for path rating computation and nodal replacement.

| # | Test | Assert |
|---|------|--------|
| 76 | MisoNodalReplacement.load_data() returns DataFrame | No crash, has from_node / to_node columns |
| 77 | Nodal replacement resolves chains | A→B→C becomes A→C |
| 78 | Replaced nodes exist in SF index | After replacement, source/sink are valid pnode_ids |
| 79 | Path exposure = sf[sink] - sf[source] | Manual computation matches |
| 80 | Aligned exposure = shadow_sign × path_exposure | Sign adjustment correct |
| 81 | Path rating score is non-negative | `max(0, aligned_exposure)` ensures this |
| 82 | Tier weight applied correctly | tier0 × 1.0, tier1 × 0.5, others × 0 |
| 83 | high_rated segment: any tier-0 aligned_exposure > 0.1 | Definition matches spec |
| 84 | Segments are exhaustive | high + medium + low + unrated = all paths |
| 85 | Segments are disjoint | No path in multiple segments |

## Layer 8: Annual-Band Join / Segmentation (5 cases)

Tests for the validation pipeline joining path ratings to band data.

| # | Test | Assert |
|---|------|--------|
| 86 | Join uses all 6 keys | planning_year, round, period_type, class_type, source_id, sink_id |
| 87 | Join produces no duplicates | One row per (path, PY, round, period_type, class_type) |
| 88 | Per-segment metrics decompose to full-set metrics | Weighted average of segments ≈ full-set metric |
| 89 | Row-level data used, not metrics.json | Data source is per-path DataFrame, not aggregate |
| 90 | Segment sizes are reasonable | High segment has ≥ 5% and ≤ 50% of paths |

---

## Test File Organization

```
tests/
  test_config.py              — Layer 1 (extend existing)
  test_data_loader.py          — Layer 2 (extend existing)
  test_bridge.py               — Layer 2 (extend existing)
  test_ground_truth.py         — Layer 2 (extend existing)
  test_features.py             — Layer 2 (extend existing)
  test_evaluate.py             — Layer 3 (extend existing)
  test_merge.py                — Layer 3 (extend existing)
  test_registry.py             — Layer 4 (extend existing)
  test_phase5_regression.py    — Layer 5 (NEW)
  test_signal_publisher.py     — Layer 6 (NEW, requires Project 1 implementation)
  test_path_rating.py          — Layer 7 (NEW, requires Project 2 implementation)
  test_band_validation.py      — Layer 8 (NEW, requires Project 2 implementation)
```

Layers 1-4 extend existing test files (104 tests already passing).
Layers 5-8 are new test files created alongside implementation.
Layer 5 can be written immediately (regression tests against saved artifacts).
Layers 6-8 are written alongside their respective implementations.

## Execution Order

1. **Now**: Write Layer 5 (regression tests) — verifies Phase 5 results are stable
2. **With Project 1**: Write Layers 1-4 extensions + Layer 6
3. **With Project 2**: Write Layers 7-8
