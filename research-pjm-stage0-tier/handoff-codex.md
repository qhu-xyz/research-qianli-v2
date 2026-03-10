# PJM 7.0 Codex Handoff

Date: 2026-03-10

## Goal

Build a PJM "7.0b" constraint-tier system modeled on MISO 7.0, but adapted to PJM's actual market structure.

Intended scope from the human note:

- Replace only `f0` and `f1` with ML.
- Carry over `f2p` unchanged.
- For PJM, `f2p` means `f2` through `f11`, not just `f2+` in the MISO sense.
- Treat each `(period_type, class_type)` slice independently.

The original note is:

- `/home/xyz/workspace/research-qianli-v2/research-pjm-stage0-tier/human-input/memory.md`

This handoff expands that note with the facts that were verified from code and data, plus the traps the next agent must avoid.

## Source Repos To Read First

These are the four repos that matter most:

1. `/home/xyz/workspace/research-qianli-v2/research-pjm-stage0-tier`
   - This repo is currently just a handoff shell.

2. `/home/xyz/workspace/research-qianli-v2/research-stage5-tier`
   - This is the ranking-pipeline template.
   - Read:
     - `experiment-setup.md`
     - `stage5-handoff.md`
     - `ml/config.py`
     - `ml/data_loader.py`

3. `/home/xyz/workspace/research-qianli-v2/research-miso-signal7`
   - This is the reference 7.0 signal implementation.
   - Read:
     - `README.md`
     - `docs/v70-deployment-handoff.md`
     - `docs/v70-design-choices.md`
     - `v70/inference.py`
     - `v70/signal_writer.py`

4. `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli`
   - This is the source of the SPICE6 shadow-price ML outputs that MISO stage5 reused as new features.
   - For PJM, this repo is also important because it already contains PJM-specific:
     - period schedule
     - DA shadow fetch API usage
     - branch/interface mapping logic
     - SPICE6 path templates

## What Is Already Proven

### 1. The PJM predecessor signal is real and on disk

The exact signal exists here:

- Constraints:
  - `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`
- Shift factors:
  - `/opt/data/xyz-dataset/signal_data/pjm/sf/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`

There are also other similarly named variants:

- `Signal.PJM.SPICE_F0P_V6.2B`
- `MANUAL.TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`
- `TEST.TEST.Signal.PJM.SPICE_F0P_V6.2BCIA1.R1`

Do not assume the canonical predecessor is whichever one looks nicest. The safest working baseline is `TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`, because it has the expected constraint and SF trees and matches the MISO naming style. But the next agent should still confirm which signal name downstream PJM code actually consumes.

### 2. PJM period structure is not MISO-like

The canonical PJM period schedule is in:

- `/home/xyz/workspace/pbase/src/pbase/data/const/period/pjm.py`
- `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/iso_configs.py`

Verified facts:

- PJM monthly periods are `f0` through `f11`.
- June auctions expose all 12 monthly periods.
- Then the valid set shrinks as the planning year progresses.
- March has `f0`, `f1`, `f2`.
- May has only `f0`.

This matters for both:

- training window construction
- passthrough logic for `f2` through `f11`

### 3. PJM class types are three separate slices

Verified from the real V6.2B tree:

- `onpeak`
- `dailyoffpeak`
- `wkndonpeak`

Example:

- `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/2025-03/f0/{onpeak,dailyoffpeak,wkndonpeak}`

Do not reuse MISO's `offpeak` assumptions. In PJM:

- production constraint signals are stored as `onpeak`, `dailyoffpeak`, `wkndonpeak`
- some `pbase` utilities synthesize `offpeak = dailyoffpeak + wkndonpeak`
- for this project, the signal slices must stay separate unless there is an explicit reason not to

### 4. PJM V6.2B schema matches the MISO-style 20-column constraint parquet

Verified on:

- `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/2025-03/f0/onpeak`

Columns:

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

This is a very important result because it means the stage5/MISO 7.0 replacement pattern is structurally reusable:

- keep the V6.2B row set
- keep the same output schema
- replace only `rank_ori`, `rank`, `tier` on ML slices
- copy SF unchanged

### 5. PJM V6.2B formula is exactly the same 0.60 / 0.30 / 0.10 blend

This was verified directly on PJM parquet samples.

Exact formula:

```text
rank_ori = 0.60 * da_rank_value
         + 0.30 * density_mix_rank_value
         + 0.10 * density_ori_rank_value

rank = dense_rank(rank_ori) / max_dense_rank
tier = floor(rank * 5), clamped to 0..4
```

Checks on sampled slices gave `max_abs_diff = 0.0` for both `rank_ori` and `rank`.

So the answer to the human note's blend question is:

- yes, PJM V6.2B uses the same exact `(.6, .3, .1)` blend as MISO V6.2B

### 6. V6.2B data coverage is long enough

Verified counts:

- PJM V6.2B constraints:
  - `105` auction months
  - `2017-06` through `2026-03`

That is enough history to support:

- 8-month walk-forward training
- `f0` and `f1` ML slices
- later extension to more periods if desired

### 7. PJM SPICE6 data exists and its paths are known

Verified base:

- `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm`

Verified subtrees:

- `density`
- `ml_pred`
- `constraint_info`
- `sf`

Canonical templates are in:

- `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/iso_configs.py`

Important paths:

- Density:
  - `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density/auction_month={A}/market_month={M}/market_round=1/outage_date={D}`
- ML predictions:
  - `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred/auction_month={A}/market_month={M}/class_type={C}/final_results.parquet`
- Constraint info:
  - `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info/auction_month={A}/market_round=1/period_type={P}/class_type=onpeak`
- Shift factors:
  - `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/sf/auction_month={A}/market_month={M}/market_round=1/outage_date={D}`

### 8. SPICE6 `ml_pred` is only partially usable for PJM

Verified coverage:

- `92` auction months
- `2018-06` through `2026-01`

Verified class-type coverage:

- `onpeak`: present
- `dailyoffpeak`: present
- `wkndonpeak`: not found anywhere in the `ml_pred` tree

This is a major PJM-specific gap. It means:

- `onpeak` can likely use SPICE6 ML features
- `dailyoffpeak` can likely use SPICE6 ML features
- `wkndonpeak` cannot simply copy the same feature group unless another source exists

Do not bury this as a minor edge case. It directly affects the feature plan and the registry layout.

### 9. `ml_pred` aligns well with V6.2B rows for usable class types

Sanity checks on `2025-03/f0`:

- `onpeak`: V6.2B overlap with `ml_pred` was `100%`
- `dailyoffpeak`: overlap was about `98.0%` (`549/560`)
- `wkndonpeak`: no `ml_pred` slice exists

So `ml_pred` is a valid candidate feature group for:

- `onpeak`
- `dailyoffpeak`

But you still need a null-fill or fallback policy for missing rows, especially in `dailyoffpeak`.

### 10. The shadow-price repo's exported columns are easy to misuse

A sampled PJM `final_results.parquet` contains columns like:

- `predicted_shadow_price`
- `binding_probability`
- `binding_probability_scaled`
- `hist_da`
- `prob_exceed_95`
- `prob_exceed_105`
- `actual_shadow_price`
- `actual_binding`
- `error`
- `abs_error`

Important warning:

- Do not use `actual_shadow_price`, `actual_binding`, `error`, or `abs_error` as features.
- In production-like future outputs these can be zero, stale, or unavailable.
- Treat only the forward-looking prediction columns as candidate features.

### 11. PJM constraint-info appears to be onpeak-only

Verified directory shape under `constraint_info` only shows:

- `class_type=onpeak`

This strongly suggests PJM structural constraint info is shared across class types, or at least stored once under `onpeak`.

Implication:

- if you use `constraint_info`, treat it as structural metadata, not class-type-specific signal
- verify that joining it into `dailyoffpeak` and `wkndonpeak` is valid before relying on it

### 12. PJM DA mapping logic is harder than MISO

Read:

- `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`

Important PJM-specific behavior:

- `PjmDataLoader.fetch_da_shadow_wrapper()` uses:
  - `aptools.tools.get_da_shadow_by_peaktype(st=..., et_ex=..., peak_type=...)`
- But labels/history are not joined directly on raw `constraint_id`.
- `PjmDataLoader.map_constraints_to_branches()` maps DA data to `branch_name` with special handling for interfaces.

This is a huge warning for the stage5-style target builder:

- do not assume PJM realized DA can be joined to V6.2B rows the way MISO stage5 joins on `constraint_id`
- you may need a `constraint_id -> branch_name` mapping step, especially for interface rows

### 13. The shadow-price repo's training label is not automatically the right 7.0 ranking target

In the shadow-price repo:

- per-outage data gets a `label`
- then `prediction.py` renames it to `actual_shadow_price`
- then monthly `final_results` aggregate it by summing across outage dates

That label is an outage-window weighted shadow-price label, not automatically the same as the stage5-style monthly ranking target.

For PJM 7.0, you must explicitly decide which target is correct:

- full delivery-month realized DA absolute shadow-price sum?
- binary binds-anytime target?
- weighted outage-window label from the shadow-price repo?

Do not conflate:

- "SPICE6 shadow-price model training label"
- "V6.2B / 7.0 ranking ground truth"

That distinction is central.

## What Should Be Borrowed From MISO 7.0

These are the MISO 7.0 ideas that are likely correct to inherit:

- ML only on front slices (`f0`, `f1`)
- passthrough on later slices
- keep V6.2B row universe
- copy SF unchanged
- train one model per `(period_type, class_type)`
- use walk-forward training with strict temporal lag
- use row-percentile ranking with deterministic tie-breaks
- use V6.2B formula score as a feature, not as the output
- keep hierarchical registry layout:
  - `registry/{ptype}/{ctype}/{version}/...`

Key MISO reference details:

- MISO 7.0 inference features:
  - `binding_freq_1`
  - `binding_freq_3`
  - `binding_freq_6`
  - `binding_freq_12`
  - `binding_freq_15`
  - `v7_formula_score`
  - `prob_exceed_110`
  - `constraint_limit`
  - `da_rank_value`
- MISO 7.0 blend weights used for `v7_formula_score` are optimized by slice, not fixed at 0.60/0.30/0.10
- MISO 7.0 ranking uses:
  - primary key: ML score descending
  - tie-break: V6.2B `rank_ori` ascending
  - then original row order

Read:

- `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/inference.py`
- `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/signal_writer.py`
- `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-design-choices.md`

## What Must Not Be Cargo-Culted From MISO

- Do not reuse MISO's `offpeak`; PJM uses `dailyoffpeak` and `wkndonpeak`.
- Do not reuse MISO's schedule; PJM goes out to `f11`.
- Do not assume PJM joins realized DA on raw `constraint_id`.
- Do not assume PJM has `ml_pred` for `wkndonpeak`.
- Do not assume MISO's optimized formula-blend weights transfer to PJM.
- Do not assume the stage5 target definition is already correct for PJM.

## Non-Negotiable Leakage Rules

1. Never use information from the target auction month or later to construct features.
2. Treat any realized-DA-derived feature as cutoff-sensitive.
3. Separate two concepts:
   - training-month eligibility
   - per-row historical feature cutoff
4. Never use derived V6.2B outputs as features:
   - `rank`
   - `rank_ori` as output target
   - `tier`
5. Do not use `actual_*` columns from `ml_pred`.
6. Do not let `wkndonpeak` silently inherit `dailyoffpeak` feature availability assumptions.
7. Do not compare models on circular targets.

## The First Questions The Next Agent Must Prove

These are still open and need evidence, not guesses:

1. What is the correct stage5-style PJM target?
   - monthly realized DA absolute shadow-price sum by delivery month?
   - monthly bind indicator?
   - branch-level aggregate?
   - something else?

2. What is the correct join key for that target?
   - raw `constraint_id`
   - `branch_name`
   - interface-normalized monitored facility string

3. What exact cutoff should be used for PJM historical DA features?
   - The shadow-price repo uses `run_at_day = 13` for PJM.
   - Verify that this matches the actual signal production timing.

4. Can `wkndonpeak` support the same ML feature set?
   - If not, should it get:
     - a reduced feature set
     - a separate model family
     - or passthrough for 7.0b

5. Does PJM 7.0 replace only `rank_ori/rank/tier`, or is there any downstream consumer that also reads raw score semantics?

## Recommended Build Plan

### Phase 1: Reproduce PJM V6.2B exactly

Before any ML work:

- build a reproducibility script for PJM V6.2B
- prove exact formula parity across:
  - `f0/onpeak`
  - `f0/dailyoffpeak`
  - `f0/wkndonpeak`
  - `f1/onpeak`
  - `f1/dailyoffpeak`
  - `f1/wkndonpeak`

This should be the PJM `v0`.

### Phase 2: Build the correct target

Implement a PJM target loader with these properties:

- uses realized DA, not `shadow_price_da`
- applies strict cutoff rules for historical features
- keeps target generation independent from forward-looking signal features
- explicitly documents whether the join happens on:
  - `constraint_id`
  - or `branch_name`

This is the highest-risk part of the whole project.

### Phase 3: Build a minimal stage5-style PJM dataset

Start with the cleanest possible slices:

- `f0/onpeak`
- `f1/onpeak`

Then extend to:

- `dailyoffpeak`
- `wkndonpeak`

Use groups similar to stage5:

- Group A: V6.2B forecast features
- Group B: SPICE6 density
- Group C: historical DA features
- Group D: SPICE6 `ml_pred` features where available

For `wkndonpeak`, expect Group D to be unavailable unless a second source is found.

### Phase 4: Train slice-by-slice models

At minimum, keep separate registries for:

- `f0/onpeak`
- `f0/dailyoffpeak`
- `f0/wkndonpeak`
- `f1/onpeak`
- `f1/dailyoffpeak`
- `f1/wkndonpeak`

Do not share gates/champions across class types.

### Phase 5: Assemble a PJM 7.0 signal

Mirror MISO's deployment shape:

- load V6.2B parquet for the target slice
- compute ML scores for ML-enabled slices
- convert scores to `rank` and `tier`
- replace only:
  - `rank_ori`
  - `rank`
  - `tier`
- write constraints parquet in V6.2B schema
- copy SF parquet unchanged
- pass through `f2` to `f11`

## Strong Recommendation On Scope

Do not try to solve all 6 PJM ML slices at once.

Recommended order:

1. `f0/onpeak`
2. `f1/onpeak`
3. `f0/dailyoffpeak`
4. `f1/dailyoffpeak`
5. `wkndonpeak`

Reason:

- `onpeak` has the cleanest overlap and the least ambiguity.
- `dailyoffpeak` has some missing `ml_pred` rows but still exists.
- `wkndonpeak` has no verified `ml_pred` source and likely needs bespoke handling.

## Suggested Repo Structure

If this repo becomes the real implementation repo, use a stage5-style hierarchy from day one:

```text
research-pjm-stage0-tier/
  handoff-codex.md
  docs/
    pjm-v0-audit.md
    pjm-target-definition.md
    pjm-feature-audit.md
    pjm-v70-design.md
  ml/
    config.py
    data_loader.py
    realized_da.py
    spice6_loader.py
    features.py
    train.py
    evaluate.py
    registry_paths.py
  registry/
    f0/
      onpeak/
      dailyoffpeak/
      wkndonpeak/
    f1/
      onpeak/
      dailyoffpeak/
      wkndonpeak/
```

## Validation Checklist

Before calling anything "7.0", the next agent should produce evidence for all of these:

1. V6.2B reproduction is exact.
2. Target definition is documented and justified.
3. No forward-looking columns are used as features.
4. Training windows respect PJM schedule and lag.
5. Feature availability is audited per `(ptype, ctype)`.
6. `wkndonpeak` feature gap is explicitly handled.
7. Holdout metrics are computed against the true target.
8. Passthrough slices are bit-identical to V6.2B.
9. Output parquet schema matches PJM V6.2B exactly.
10. SF output is copied unchanged.

## Concrete Facts Worth Remembering

- PJM V6.2B exists and is usable.
- PJM V6.2B uses the same exact `0.60/0.30/0.10` formula as MISO.
- PJM has `f0` through `f11`.
- PJM has `onpeak`, `dailyoffpeak`, `wkndonpeak`.
- PJM `ml_pred` exists for `onpeak` and `dailyoffpeak`, not `wkndonpeak`.
- PJM `constraint_info` appears stored only under `class_type=onpeak`.
- PJM DA mapping may need branch/interface normalization.
- The shadow-price repo's `label` is not automatically the 7.0 ranking target.

## The Most Important Failure Mode

The easiest way to get a fake win here is to silently evaluate on the wrong target or to join the target incorrectly.

For PJM, the highest-risk mistakes are:

- using V6.2B historical DA columns as if they were realized labels
- treating shadow-price-pipeline labels as equivalent to monthly ranking labels
- joining realized DA directly on raw `constraint_id` when branch/interface mapping is actually required
- pretending `wkndonpeak` has the same feature coverage as `onpeak`

If the next agent is uncertain on any of those, the right action is to stop and prove the data lineage first.
