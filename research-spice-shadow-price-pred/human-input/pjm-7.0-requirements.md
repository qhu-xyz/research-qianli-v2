# PJM 7.0 Requirements And Data Availability

Date: 2026-03-10

## Purpose

This note lists exactly what PJM 7.0 needs in order to work, what data/code already exists, what is optional, and where PJM differs from MISO.

The intended product is:

- PJM 7.0 constraint-tier signal
- ML replacement only for `f0` and `f1`
- passthrough for `f2` through `f11`
- separate handling for each class type:
  - `onpeak`
  - `dailyoffpeak`
  - `wkndonpeak`

## Short Answer To The Availability Question

No, `onpeak` and `dailyoffpeak` do **not** have every enhancement dataset for all possible PJM `ptype`s.

More precisely:

- The baseline signal data exists across the expected PJM monthly schedule.
- SPICE6 density also exists across the expected PJM monthly schedule.
- SPICE6 `ml_pred` exists for `onpeak` and `dailyoffpeak`, but only from `2018-06` onward and only through `2026-01`.
- `wkndonpeak` has **no verified `ml_pred` coverage at all**.
- PJM `constraint_info` appears stored only under `class_type=onpeak`.

So:

- `f0/onpeak` and `f1/onpeak` can likely use the full intended feature stack.
- `f0/dailyoffpeak` and `f1/dailyoffpeak` can likely use most of it, but not perfectly.
- `wkndonpeak` cannot use the `ml_pred` feature group with the currently verified data.

## What PJM 7.0 Must Produce

For each target auction month:

1. ML-score these slices:
   - `f0/onpeak`
   - `f0/dailyoffpeak`
   - `f0/wkndonpeak`
   - `f1/onpeak`
   - `f1/dailyoffpeak`
   - `f1/wkndonpeak`

2. Pass through these slices unchanged from PJM V6.2B:
   - `f2` through `f11`
   - all three class types for those period types

3. Write output in PJM V6.2B signal format:
   - same constraint row universe
   - same parquet schema
   - same SF shape and location conventions

4. For ML slices, only replace:
   - `rank_ori`
   - `rank`
   - `tier`

5. Keep the rest of the V6.2B columns intact unless a downstream consumer explicitly requires more.

## Hard Requirements

These are required for PJM 7.0 to work at all.

### A. Baseline source signal and SF

Required because 7.0 is a V6.2B-row-universe replacement, not a new universe.

Canonical predecessor:

- Constraints:
  - `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`
- Shift factors:
  - `/opt/data/xyz-dataset/signal_data/pjm/sf/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`

What must be proven/assumed:

- this is the correct predecessor signal for deployment
- downstream code accepts the exact same schema/index conventions

### B. Correct target definition

Required because this is where leakage and false wins happen.

The next agent must define:

- what the true ranking target is for PJM 7.0
- whether it lives at:
  - raw `constraint_id` level
  - `branch_name` level
  - interface-normalized branch/facility level

This is not optional.

### C. Realized DA loader with PJM-specific mapping

Required because PJM DA mapping is more complex than MISO.

Relevant source code:

- `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`
- `/home/xyz/workspace/pbase/src/pbase/data/const/period/pjm.py`

Important existing behavior:

- PJM uses `aptools.tools.get_da_shadow_by_peaktype(...)`
- PJM branch/interface mapping logic already exists in `PjmDataLoader.map_constraints_to_branches()`

Implication:

- do not assume MISO-style direct `constraint_id` joins are correct for PJM

### D. Walk-forward training logic with PJM schedule

Required because PJM has `f0` through `f11`, not MISO's shorter schedule.

Needed behavior:

- train only on eligible prior auction months
- respect PJM period availability by auction month
- use strict cutoff logic for any realized-DA-derived features

### E. Output writer in PJM V6.2B format

Required because downstream consumers expect the PJM signal shape.

Verified PJM V6.2B columns:

- `constraint_id`
- `flow_direction`
- `mean_branch_max`
- `mean_branch_max_fillna`
- `ori_mean`
- `branch_name`
- `bus_key`
- `bus_key_group`
- `mix_mean`
- `shadow_price_da`
- `density_mix_rank_value`
- `density_ori_rank_value`
- `da_rank_value`
- `rank_ori`
- `density_mix_rank`
- `rank`
- `tier`
- `shadow_sign`
- `shadow_price`
- `equipment`

## Strongly Recommended Inputs

These are not optional in practice for a credible PJM 7.0.

### 1. SPICE6 density

Path:

- `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density`

Why needed:

- this is one of the most important feature groups inherited from V6.2B / stage5 logic
- it appears available across all class types because the density tree is partitioned by auction month and market month, not class type

### 2. Historical-DA-derived features

Why needed:

- these were important in MISO 7.0
- PJM V6.2B already contains `da_rank_value` and `shadow_price_da`
- additional lagged realized-DA features may help

But:

- all such features must use a provably correct decision-time cutoff

### 3. Deterministic ML rank-to-tier conversion

Recommended reference:

- MISO 7.0 row-percentile ranking with deterministic tie-breaks

Why needed:

- raw ML scores can have many ties
- dense-rank-based tiering may create unstable tier sizes

## Optional / Enhancement Inputs

These can improve the model but are not strictly required for the system to function.

### 1. SPICE6 `ml_pred`

Path:

- `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred`

Candidate features:

- `predicted_shadow_price`
- `binding_probability`
- `binding_probability_scaled`
- possibly `prob_exceed_95`
- possibly `prob_exceed_105`

Do **not** use:

- `actual_shadow_price`
- `actual_binding`
- `error`
- `abs_error`

### 2. `constraint_info`

Path:

- `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info`

Use only if:

- the next agent confirms the onpeak-only storage is valid to share across class types

### 3. raw SPICE6 SF-derived features

Path:

- `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/sf`

Likely enhancement only, not a minimum requirement.

## PJM Data Availability

## A. PJM baseline signal availability

These counts are the number of auction months where the slice exists in PJM V6.2B.

Same counts apply to:

- constraint signal parquet
- saved signal SF parquet

And they are the same across all three PJM class types.

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 105 | 2017-06 | 2026-03 |
| `f1` | 97 | 2017-06 | 2026-03 |
| `f2` | 89 | 2017-06 | 2026-03 |
| `f3` | 80 | 2017-06 | 2026-01 |
| `f4` | 72 | 2017-06 | 2026-01 |
| `f5` | 63 | 2017-06 | 2025-12 |
| `f6` | 54 | 2017-06 | 2025-11 |
| `f7` | 45 | 2017-06 | 2025-10 |
| `f8` | 36 | 2017-06 | 2025-09 |
| `f9` | 27 | 2017-06 | 2025-08 |
| `f10` | 18 | 2017-06 | 2025-07 |
| `f11` | 9 | 2017-06 | 2025-06 |

Applies to:

- `onpeak`
- `dailyoffpeak`
- `wkndonpeak`

Interpretation:

- passthrough coverage is available exactly where expected
- front slices have plenty of history
- later slices naturally have fewer months because PJM only exposes them early in the planning year

## B. PJM SPICE6 density availability

Counts by implied `(ptype, class_type)` coverage:

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 106 | 2017-06 | 2026-03 |
| `f1` | 98 | 2017-06 | 2026-03 |
| `f2` | 90 | 2017-06 | 2026-03 |
| `f3` | 81 | 2017-06 | 2026-02 |
| `f4` | 72 | 2017-06 | 2026-01 |
| `f5` | 63 | 2017-06 | 2025-12 |
| `f6` | 54 | 2017-06 | 2025-11 |
| `f7` | 45 | 2017-06 | 2025-10 |
| `f8` | 36 | 2017-06 | 2025-09 |
| `f9` | 27 | 2017-06 | 2025-08 |
| `f10` | 18 | 2017-06 | 2025-07 |
| `f11` | 9 | 2017-06 | 2025-06 |

Treat as available to all three class types:

- `onpeak`
- `dailyoffpeak`
- `wkndonpeak`

because the density tree is not class-type partitioned.

## C. PJM SPICE6 `ml_pred` availability

This is the most important enhancement-coverage table.

### `onpeak`

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 92 | 2018-06 | 2026-01 |
| `f1` | 85 | 2018-06 | 2026-01 |
| `f2` | 78 | 2018-06 | 2026-01 |
| `f3` | 71 | 2018-06 | 2026-01 |
| `f4` | 64 | 2018-06 | 2026-01 |
| `f5` | 56 | 2018-06 | 2025-12 |
| `f6` | 48 | 2018-06 | 2025-11 |
| `f7` | 40 | 2018-06 | 2025-10 |
| `f8` | 32 | 2018-06 | 2025-09 |
| `f9` | 24 | 2018-06 | 2025-08 |
| `f10` | 16 | 2018-06 | 2025-07 |
| `f11` | 8 | 2018-06 | 2025-06 |

### `dailyoffpeak`

Counts are the same as `onpeak`:

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 92 | 2018-06 | 2026-01 |
| `f1` | 85 | 2018-06 | 2026-01 |
| `f2` | 78 | 2018-06 | 2026-01 |
| `f3` | 71 | 2018-06 | 2026-01 |
| `f4` | 64 | 2018-06 | 2026-01 |
| `f5` | 56 | 2018-06 | 2025-12 |
| `f6` | 48 | 2018-06 | 2025-11 |
| `f7` | 40 | 2018-06 | 2025-10 |
| `f8` | 32 | 2018-06 | 2025-09 |
| `f9` | 24 | 2018-06 | 2025-08 |
| `f10` | 16 | 2018-06 | 2025-07 |
| `f11` | 8 | 2018-06 | 2025-06 |

### `wkndonpeak`

No verified coverage:

| ptype | months |
|---|---:|
| `f0` | 0 |
| `f1` | 0 |
| `f2` | 0 |
| `f3` | 0 |
| `f4` | 0 |
| `f5` | 0 |
| `f6` | 0 |
| `f7` | 0 |
| `f8` | 0 |
| `f9` | 0 |
| `f10` | 0 |
| `f11` | 0 |

Meaning:

- `wkndonpeak` cannot use the `ml_pred` feature group unless another source is found

## D. PJM `constraint_info` availability

Verified only under `class_type=onpeak`.

### `onpeak`

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 106 | 2017-06 | 2026-03 |
| `f1` | 98 | 2017-06 | 2026-03 |
| `f2` | 90 | 2017-06 | 2026-03 |
| `f3` | 81 | 2017-06 | 2026-02 |
| `f4` | 72 | 2017-06 | 2026-01 |
| `f5` | 63 | 2017-06 | 2025-12 |
| `f6` | 54 | 2017-06 | 2025-11 |
| `f7` | 45 | 2017-06 | 2025-10 |
| `f8` | 36 | 2017-06 | 2025-09 |
| `f9` | 27 | 2017-06 | 2025-08 |
| `f10` | 18 | 2017-06 | 2025-07 |
| `f11` | 9 | 2017-06 | 2025-06 |

### `dailyoffpeak`

- no verified class-specific tree

### `wkndonpeak`

- no verified class-specific tree

Interpretation:

- this may still be usable as a shared structural feature source
- but it is not a class-specific dataset for off-hours slices

## E. PJM raw SPICE6 SF availability

Counts match density-style availability:

| ptype | months | first | last |
|---|---:|---|---|
| `f0` | 106 | 2017-06 | 2026-03 |
| `f1` | 98 | 2017-06 | 2026-03 |
| `f2` | 90 | 2017-06 | 2026-03 |
| `f3` | 81 | 2017-06 | 2026-02 |
| `f4` | 72 | 2017-06 | 2026-01 |
| `f5` | 63 | 2017-06 | 2025-12 |
| `f6` | 54 | 2017-06 | 2025-11 |
| `f7` | 45 | 2017-06 | 2025-10 |
| `f8` | 36 | 2017-06 | 2025-09 |
| `f9` | 27 | 2017-06 | 2025-08 |
| `f10` | 18 | 2017-06 | 2025-07 |
| `f11` | 9 | 2017-06 | 2025-06 |

Treat as enhancement only unless the next agent proves they materially help.

## MISO Comparison

This is only here to calibrate expectations, not to justify copying MISO blindly.

## A. MISO baseline signal coverage

| slice | months | first | last |
|---|---:|---|---|
| `f0/onpeak` | 106 | 2017-06 | 2026-03 |
| `f0/offpeak` | 106 | 2017-06 | 2026-03 |
| `f1/onpeak` | 89 | 2017-07 | 2026-03 |
| `f1/offpeak` | 89 | 2017-07 | 2026-03 |
| `f2/onpeak` | 54 | 2017-08 | 2026-03 |
| `f2/offpeak` | 54 | 2017-08 | 2026-03 |
| `f3/onpeak` | 27 | 2017-08 | 2026-02 |
| `f3/offpeak` | 27 | 2017-08 | 2026-02 |

## B. MISO density coverage

| slice | months | first | last |
|---|---:|---|---|
| `f0/onpeak` | 106 | 2017-06 | 2026-03 |
| `f0/offpeak` | 106 | 2017-06 | 2026-03 |
| `f1/onpeak` | 89 | 2017-07 | 2026-03 |
| `f1/offpeak` | 89 | 2017-07 | 2026-03 |
| `f2/onpeak` | 81 | 2017-07 | 2026-03 |
| `f2/offpeak` | 81 | 2017-07 | 2026-03 |
| `f3/onpeak` | 54 | 2017-07 | 2026-02 |
| `f3/offpeak` | 54 | 2017-07 | 2026-02 |

## C. MISO `ml_pred` coverage

| slice | months | first | last |
|---|---:|---|---|
| `f0/onpeak` | 92 | 2018-06 | 2026-01 |
| `f0/offpeak` | 92 | 2018-06 | 2026-01 |
| `f1/onpeak` | 77 | 2018-07 | 2026-01 |
| `f1/offpeak` | 77 | 2018-07 | 2026-01 |
| `f2/onpeak` | 70 | 2018-07 | 2026-01 |
| `f2/offpeak` | 70 | 2018-07 | 2026-01 |
| `f3/onpeak` | 47 | 2018-07 | 2026-01 |
| `f3/offpeak` | 47 | 2018-07 | 2026-01 |

## D. MISO `constraint_info` coverage

| slice | months | first | last |
|---|---:|---|---|
| `f0/onpeak` | 106 | 2017-06 | 2026-03 |
| `f0/offpeak` | 106 | 2017-06 | 2026-03 |
| `f1/onpeak` | 89 | 2017-07 | 2026-03 |
| `f1/offpeak` | 89 | 2017-07 | 2026-03 |
| `f2/onpeak` | 54 | 2017-08 | 2026-03 |
| `f2/offpeak` | 54 | 2017-08 | 2026-03 |
| `f3/onpeak` | 27 | 2017-08 | 2026-02 |
| `f3/offpeak` | 27 | 2017-08 | 2026-02 |

## What PJM 7.0 Needs To Decide Immediately

These are the real decisions, not bookkeeping:

1. True target definition
   - This is the biggest unresolved item.

2. Join key for target construction
   - likely not a trivial raw `constraint_id` join for every PJM row

3. Whether `wkndonpeak` gets:
   - a reduced ML feature set
   - or V6.2B passthrough in the first version

4. Whether `constraint_info` is valid to share across class types

5. Whether to include SPICE6 `ml_pred` at launch
   - it is available for `onpeak` and `dailyoffpeak`
   - it is not available for `wkndonpeak`
   - so one global feature contract across all 6 ML slices may be a bad idea

## Recommended Minimum Launch Scope

If the goal is a practical first PJM 7.0:

### Launchable first-wave ML slices

- `f0/onpeak`
- `f1/onpeak`

### Second-wave ML slices

- `f0/dailyoffpeak`
- `f1/dailyoffpeak`

### Highest-risk slice

- `wkndonpeak`

Reason:

- no verified `ml_pred` support
- likely more awkward DA-history behavior
- easiest place to accidentally force a fake symmetry with the other class types

## Final Bottom Line

PJM 7.0 can work with the currently verified data, but not every slice has the same feature availability.

The exact practical situation is:

- Baseline V6.2B signal and SF:
  - good enough for all required PJM slices
- SPICE6 density:
  - good enough for all required PJM slices
- SPICE6 `ml_pred`:
  - available for `onpeak` and `dailyoffpeak`
  - unavailable for `wkndonpeak`
- `constraint_info`:
  - clearly present for `onpeak`
  - not clearly class-specific for `dailyoffpeak` or `wkndonpeak`

So yes: PJM `wkndonpeak` likely cannot use some ML features that `onpeak` and `dailyoffpeak` can use.

And yes: even for `onpeak` and `dailyoffpeak`, the enhancement datasets are not universal across all historical months/ptypes in the same way the baseline signal is.
