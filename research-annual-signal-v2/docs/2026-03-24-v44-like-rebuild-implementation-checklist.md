# 2026-03-24 V4.4-Like Rebuild Implementation Checklist

## Objective

The target is:

- be about as predictive as V4.4 on NB / dormant-branch prediction
- while staying roughly as good as `Bucket_6_20` on overall value capture

This is the constraint for all implementation choices.

We are not trying to reproduce V4.4 for its own sake.
We are trying to recover the useful signal family without giving up the strengths of our current best internal model.

## Phase 0: Contracts

### 0.1 Define feature-recipe ids

- `miso_annual_dev_profile_features_v1`
- `miso_annual_bucket_plus_dev_features_v1`

### 0.2 Define experiment ids

- `v44_like_formula_v1`
- `bucket_plus_dev_v1`
- `nb_dev_specialist_v1`

### 0.3 Define benchmark roles

- NB / dormant benchmark: V4.4
- overall SP benchmark: `Bucket_6_20`
- deployment benchmark: `R30`

## Phase 1: CID-Level Deviation Features

### 1.1 Add CID-level deviation builder

Suggested module:

- `ml/features/deviation_profile.py`

Responsibilities:

- load annual density rows at CID × outage-date granularity
- support round-aware loading
- compute positive-tail exceedance thresholds:
  - `60, 65, 70, 75, 80, 85, 90, 95, 100`
- optionally compute negative-tail equivalents

### 1.2 Add the two profile styles

For each CID compute:

- `profile_max`
- `profile_sum`

Definitions:

- `profile_max[t] = fraction of outage_dates where exceedance_t > 0`
- `profile_sum[t] = average exceedance_t across outage_dates`

### 1.3 Add histogram-mass conversion

Convert cumulative exceedance to mass:

- `mass_0 = 1 - exc_60`
- `mass_60 = exc_60 - exc_65`
- ...
- `mass_95 = exc_95 - exc_100`
- `mass_100 = exc_100`

### 1.4 Add weighted deviation transform

Implement weighted Wasserstein-style distance from baseline-at-0.

Support at least:

- `7 ** i`
- `5 ** i`
- `3 ** i`

Primary CID outputs:

- `cid_dev_max`
- `cid_dev_sum`

### 1.5 Add branch aggregation

Outputs:

- `best_cid_dev_max`
- `best_cid_dev_sum`
- `top2_cid_dev_max`
- `top2_cid_dev_sum`
- `dev_gap_max`
- `dev_gap_sum`
- `count_cids_dev_gt_threshold`

## Phase 2: Formula Prototype

### 2.1 Add a direct formula benchmark

Suggested script:

- `scripts/v44_like_formula.py`

Initial score:

- `0.4 * dev_sum_rank + 0.4 * dev_max_rank + 0.2 * shadow_rank`

### 2.2 Save registry artifacts

Write to:

- `registry/v44_like_formula_v1/`

Required files:

- `config.json`
- `metrics.json`
- `all_results.json`

### 2.3 First evaluation slices

Run first on:

- `2024-06`
- `2025-06`
- `aq1-3`
- both ctypes

## Phase 3: Known-Miss Validation

### 3.1 Build a named-branch case-study table

Check whether the new features lift:

- `MNTCELO`
- `13866`
- `AUST_TAYS`
- `GOOSEMNPIP`
- `ULRICMAHNO`
- `BURNHMUNST`

### 3.2 For each branch report

- current `Bucket_6_20` rank
- V4.4 rank
- new formula rank
- new deviation features
- key history features

Goal:

- verify that the new signal changes the known dormant misses for the right reason

## Phase 4: ML Integration

### 4.1 Extend Bucket_6_20 feature set

New recipe:

- `miso_annual_bucket_plus_dev_features_v1`

Base:

- current `Bucket_6_20` features

Added:

- `best_cid_dev_max`
- `best_cid_dev_sum`
- `top2_cid_dev_max`
- `top2_cid_dev_sum`
- `dev_gap_max`
- `dev_gap_sum`

### 4.2 Keep current bucket labels first

Do not change label/objective on the first pass.

Reason:

- isolate feature value before changing multiple variables

### 4.3 Train side-by-side

Compare:

- `Bucket_6_20`
- `bucket_plus_dev_v1`
- V4.4

## Phase 5: NB Specialist Variant

### 5.1 Add a deviation-enhanced dormant specialist

Suggested script:

- `scripts/nb_dev_specialist.py`

Goal:

- test whether deviation features improve dormant ranking more strongly in a specialist path

### 5.2 Evaluate in deployment

Compare:

- `R30` with current Bucket
- `R30` with deviation-enhanced model
- `R50` only if needed

## Phase 6: Round-Aware Upgrade

Only after the feature family is validated.

### 6.1 Recompute features per round

Use:

- round-aware density partitions
- round-aware bridge
- round-aware DA cutoff

### 6.2 Re-evaluate `R1`, `R2`, `R3`

Check whether:

- dormant ranking improves further
- round-specific changes are material enough to matter in production

## Mandatory Metrics

Every experiment must report:

- `SP@200`
- `SP@400`
- `NB_SP@200`
- `NB_SP@400`
- `Binders@200`
- `Binders@400`
- top true dormant binder hit rates
- overlap-only dormant absolute-rank tables
- deployment metrics for `R30`

## Decision Rules

### Pass for formula prototype

- clear gain on known dormant misses
- some improvement versus current Bucket on dormant metrics

### Pass for final candidate

- NB / dormant prediction roughly comparable to V4.4
- overall SP roughly comparable to `Bucket_6_20`
- no major deployment regression under `R30`

### Fail conditions

- NB gain but large overall SP collapse
- overall SP preserved but no meaningful dormant improvement
- overlap-only gain with no useful native/deployment gain

## Deliverables

At the end of this workstream, we should have:

1. one reusable deviation-feature module
2. one formula benchmark
3. one Bucket-plus-deviation benchmark
4. one NB-specialist benchmark
5. one case-study report for known dormant misses
6. one final recommendation:
   - promote
   - keep as research only
   - or stop because density parquet cannot recover enough of the missing signal
