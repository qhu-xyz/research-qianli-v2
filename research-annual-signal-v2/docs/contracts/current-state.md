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
- `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1` — 9 PYs (2017-06 through 2025-06)
- `TEST.Signal.MISO.SPICE_ANNUAL_V7.0B.R1` — same + supplement key matching + zero-SF filter

Round status: **R1-only**. V7.0B.R2 and V7.0B.R3 directories exist on NFS but were published by a different pipeline (not this repo).

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

Current benchmark loading is R1-only (`scripts/champion_confirmation.py:28`). V4.4 R2/R3 exist on NFS but are not loaded.

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
| Benchmark loading R2/R3 | Not done | `champion_confirmation.py` still hardcodes V4.4.R1 |
| Round-aware scoring comparison | Not done | No R1 vs R2 vs R3 feature delta evaluation yet |

## 5. Registry entries

| Path | Model | Status |
|------|-------|--------|
| `registry/onpeak/bucket_6_20/` | Bucket_6_20 onpeak | config.json + metrics.json (R1-only) |
| `registry/offpeak/bucket_6_20/` | Bucket_6_20 offpeak | config.json + metrics.json (R1-only) |
| `registry/champion_confirmation/` | 3-way comparison | all_results.json + config.json (R1-only) |
| `registry/archive/` | Phase 3-5 combined-ctype | 31 entries (legacy, not production) |

No entry currently has:
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

`data/nb_cache/` is the stale artifact. It was built by the old pipeline with `market_round=1` implicitly. The new `build_class_model_table` writes to `data/collapsed/` with round-aware keys. Scripts that use `data/nb_cache/` (champion_confirmation, nb_bucket_model) read from the old cache directly and are therefore R1-only.

## 7. What is NOT frozen

- `scripts/archive/` — legacy research scripts, will raise TypeError if run without updating
- `data/nb_cache/` — stale R1-only model tables, not authoritative
- Round-aware evaluation results — no R1/R2/R3 comparison has been run yet
- PJM annual — no code, no data, no models

## 8. Promotion rules (not yet enforced)

No model may be promoted to champion or attached to a release without:
- bound `universe_id`
- bound `feature_recipe_id`
- bound `label_recipe_id`
- eval grain `(planning_year, aq_quarter, ctype, market_round)` in spec
- standardized `spec.json` + `metrics.json`

These rules will be enforced after the universe catalog and feature recipe contracts are defined.
