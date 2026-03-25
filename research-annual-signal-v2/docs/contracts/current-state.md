# Current State Freeze — MISO Annual Signal

**Date**: 2026-03-24
**Branch**: `feature/pjm-v70b-ml-pipeline`

---

## 1. Production baseline

**Model**: `miso_annual_v0c_formula_v1`
**Feature recipe**: `miso_annual_v0c_features_v1`
**Universe**: `miso_annual_branch_active_v1` (density right-tail threshold, ~2,500-2,800 branches per quarter)

Formula: `0.40 * (1 - minmax(da_rank_value)) + 0.30 * minmax(rt_max) + 0.30 * minmax(bf)`

Published signals:
- `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1` — 7 PYs (2019-06 through 2025-06)
- `TEST.Signal.MISO.SPICE_ANNUAL_V7.0B.R1` — 9 PYs (2017-06 through 2025-06), adds supplement key matching + zero-SF filter

Round status: **R1-only**. V7.0B.R2 and V7.0B.R3 directories exist on NFS; their provenance is unverified (may have been published by another pipeline or manually).

## 2. Candidate model

**Model**: `miso_annual_bucket_6_20_v1`
**Feature recipe**: `miso_annual_bucket_features_v1`
**Universe**: `miso_annual_branch_active_v1` (same as v0c)
**Label recipe**: `miso_annual_bucket_5tier_v1` — 5-tier SP buckets [0, 200, 5K, 20K] with weights [1, 1, 2, 6, 20]

3-way comparison completed (`scripts/champion_confirmation.py`, `registry/champion_confirmation/`):
- v0c wins SP in 10/16 cells (dominates K=200)
- Bucket_6_20 wins SP in 6/16 cells (dominates K=400)
- V4.4 wins 0/16 on SP, 10/16 on NB_SP
- R30 deployment (170 v0c + 30 Bucket dormant) is the best observed tradeoff

**Scope of confirmation**: R1-only. Uses `data/nb_cache/` built from `market_round=1`. Not yet re-evaluated with round-aware features.

## 3. External benchmark

**Benchmark**: V4.4 published signal
**Signal path**: `TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1` (R2/R3 also exist)
**Universe**: ~1,200 branches (external, not reproducible)
**Rank direction**: lower rank = better (ascending sort)

Authoritative champion-confirmation loading is still R1-only (`scripts/champion_confirmation.py:28`). A separate round-aware comparison runner now loads matched `V4.4.R1/R2/R3`, but that result has not yet been normalized into the standard registry/release surface.

## 4. Round-sensitive core status

| Component | Status | Details |
|-----------|--------|---------|
| Cutoff calendar | Done | `ml/config.py` — MISO R1/R2/R3, PJM R1-R4 |
| Loader interfaces | Done | All 9 core APIs require explicit `market_round` |
| Cache keys | Done | Include `_r{round}_` — no stale-hit risk |
| Daily DA cache | Done | `/opt/tmp/qianli/realized_da_daily/` — 6,150 files, signed sums |
| History cutoff | Done | Partial-month data included in all BF/NB/recency features |
| Bridge round-awareness | Done | `build_monthly_binding_table` passes `market_round` to bridge |
| No-default enforcement | Done | All core APIs raise TypeError without `market_round` |
| Tests | Done | 53 tests passing |
| Publication CLI | Done | `--market-round` required in `publish_annual_signal.py` |
| Benchmark loading R2/R3 | Partial | `champion_confirmation.py` is still R1-only, but `scripts/round_comparison.py` loads matched `V4.4.R1/R2/R3` |
| Round-aware scoring comparison | Partial | `scripts/round_comparison.py` compares `v0c` vs `V4.4` across `R1/R2/R3`; `Bucket_6_20` is not yet re-evaluated round-aware |
| Publish smoke test | Done | R1/R2/R3 × onpeak/offpeak dry-run, zero missing-SF errors (`releases/miso/annual/7.1b/smoke_test.json`) |
| Release manifest | Done | `releases/miso/annual/7.1b/manifest.json` — `aq4` supported for prior PYs; `2025-06/aq4` excluded pending `2026-03` DA |
| Round comparison registry | Done | `registry/miso/annual/comparisons/round_comparison_v1/` normalized to `spec.json` + `metrics.json` (504 raw cells, 84 aggregated head-to-head cells) |
| Published output contract | Done | `docs/contracts/output-schema.md` + `ml/products/annual/output_schema.py` |

## 5. Registry entries

| Path | Model | Status |
|------|-------|--------|
| `registry/onpeak/bucket_6_20/` | Bucket_6_20 onpeak | config.json + metrics.json (R1-only) |
| `registry/offpeak/bucket_6_20/` | Bucket_6_20 offpeak | config.json + metrics.json (R1-only) |
| `registry/champion_confirmation/` | 3-way comparison | all_results.json + config.json (R1-only) |
| `registry/miso/annual/comparisons/round_comparison_v1/` | `v0c` vs `V4.4` round-aware comparison | **spec.json + metrics.json** (504 cells at base grain) + analysis.json |
| `registry/archive/` | Phase 3-5 combined-ctype | 31 entries (legacy, not production) |

Some legacy entries still do not have:
- explicit `universe_id`
- explicit `feature_recipe_id`
- explicit `market_round` in spec
- standardized `spec.json` shape

## 6. Cache dependencies

| Cache | Location | Round-aware? | Manifest? |
|-------|----------|-------------|-----------|
| Monthly DA | `data/realized_da/` | No (round-independent) | No |
| Daily DA | `/opt/tmp/qianli/realized_da_daily/` | Yes (date-level cutoff) | Yes |
| Collapsed density | `data/collapsed/` | Yes (`_r{round}_` in key) | No |
| CID mapping | `data/collapsed/` | Yes (`_r{round}_` in key) | No |
| NB model cache | `data/nb_cache/` | **No** (legacy, no round in key) | No |

`data/nb_cache/` is the stale artifact. It was built by the old pipeline with `market_round=1` implicitly. The round-aware on-disk caches are written by `load_collapsed()` and `_cid_mapping_cache_path()` in `ml/data_loader.py`, which include `_r{round}_` in the filename. `build_class_model_table` consumes these round-aware loaders but does not write its own cache. Scripts that use `data/nb_cache/` (champion_confirmation, nb_bucket_model) read from the old cache directly and are therefore R1-only.

## 7. What is NOT frozen

- `scripts/archive/` — legacy research scripts, will raise TypeError if run without updating
- `data/nb_cache/` — stale R1-only model tables, not authoritative
- Round-aware evaluation results for `Bucket_6_20` — only `v0c` vs `V4.4` has been run across `R1/R2/R3`
- `7.1b` smoke test after the `constraint_limit` output-schema fix — should be rerun once before final publish freeze
- `7.1b` on NFS is currently only published for `2025-06`
- `7.1b` `2025-06/aq4` remains blocked by missing `2026-03` DA cache; older `aq4` is available in prior releases/benchmark artifacts
- PJM annual — no code, no data, no models

## 8. Promotion rules (not yet enforced)

No model may be promoted to champion or attached to a release without:
- bound `universe_id`
- bound `feature_recipe_id`
- bound `label_recipe_id`
- eval grain `(planning_year, aq_quarter, ctype, market_round)` in spec
- standardized `spec.json` + `metrics.json`

These rules will be enforced after the universe catalog and feature recipe contracts are defined.
