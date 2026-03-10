# Review: F1 Implementation Plan

Reviewed plan:

- [2026-03-09-f1-implementation.md](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/docs/plans/2026-03-09-f1-implementation.md)

## Verdict

The plan is directionally correct on the main `f1` timing rules:

- ground truth should use `delivery_month`
- Spice6 should use `market_month = delivery_month`
- `binding_freq` cutoff should follow decision timing, not delivery offset
- training rows should be collected as previous **usable** rows, not previous calendar months

But I would **not** implement it exactly as written. There are several gaps and one important architectural mistake.

## Findings

### 1. Do not push lagged/as-of semantics into the shared generic loader

This is the biggest issue in the plan.

Task 5 proposes changing [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py) so `load_train_val_test()` starts enforcing:

- usable-row filtering
- auction-schedule filtering
- a leakage guard based on `target_auction_month - 2`

The problem is that [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py) is a shared generic loader used by:

- [ml/pipeline.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/pipeline.py)
- [ml/benchmark.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/benchmark.py)
- lagged and non-lagged experiment paths

That means the plan would silently redefine the baseline loader contract from:

- “load previous calendar months”

to:

- “load previous usable months with an as-of leakage guard”

Those are not the same contract.

Recommendation:

- keep the existing generic loader behavior intact
- add a separate explicit loader path such as:
  - `load_train_val_test_asof(...)`
  - or `collect_usable_auction_months(...)` used only by lagged / as-of scripts

This keeps the temporal semantics explicit and avoids breaking unrelated pipelines.

### 2. The plan does not fully close feature leakage

The plan fixes:

- GT month selection
- Spice6 market month selection
- `binding_freq` cutoff
- row inclusion / missing-month backfill

Those are all real fixes.

But the plan does **not** prove point-in-time safety for the V6.2B snapshot features that still drive the model:

- `da_rank_value`
- `density_mix_rank_value`
- `density_ori_rank_value`
- and therefore `v7_formula_score`

The plan assumes the `f1` V6.2B snapshot is the correct pre-auction artifact, but there is no explicit verification step for that.

So even after the proposed fixes, this statement would still be too strong:

- “f1 is fully non-leaky”

The accurate statement would be:

- “row leakage, GT month leakage, and binding_freq leakage are addressed, assuming the V6.2B / Spice6 monthly snapshots are point-in-time correct”

Recommendation:

- add an explicit verification task for V6.2B snapshot provenance
- at minimum, document this assumption clearly in the plan and final results

### 3. Task 5 has two competing sources of truth for training-month selection

Task 5 first updates [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py) to use `collect_usable_months()`.

Then the same task says [scripts/run_v10e_lagged.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py) should also replace its own `shifted_eval` / manual month generation with `collect_usable_months()`.

That creates an ambiguous design:

- is the loader responsible for row selection?
- or is the script responsible for row selection?

Unless this is cleaned up, the implementation is likely to drift or double-apply logic.

Recommendation:

- choose one owner of training-month selection
- best option: the lagged/as-of loader owns it
- the script should not separately reconstruct month lists except for logging / assertions

### 4. The expected f1 dev-month count appears wrong

The plan says:

- “f1 V6.2B data: 34 dev months”

But the repo’s full eval window is:

- `2020-06` through `2023-05`

from [ml/config.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/config.py#L106).

Within that range:

- months with no `f1` are May and June
- across `2020-06` to `2023-05`, that removes:
  - `2020-06`
  - `2021-05`
  - `2021-06`
  - `2022-05`
  - `2022-06`
  - `2023-05`

So the expected dev count looks like **30**, not 34.

The holdout count of 20 for 2024-2025 appears reasonable.

Recommendation:

- correct the expected month counts before implementation
- add a quick verification snippet in the plan so this is checked programmatically

### 5. Version naming becomes misleading after Task 4

Task 4 changes the logic so:

- row inclusion for `f1` uses `lag=2`
- `binding_freq` cutoff always uses `lag=1`

Then Task 11 still saves the version as:

- `v10e-lag2`

That is understandable, but it is now only describing the row-inclusion lag, not the feature cutoff behavior.

Recommendation:

- either document this explicitly in the saved config
- or rename more explicitly, for example:
  - `v10e-rowlag2-bflag1`

At minimum, the saved `config.json` or `metrics.json` should record both concepts separately.

## What I Agree With

These parts of the plan are correct:

- `delivery_month(auction_month, period_type)` helper
- GT join should use `delivery_month`
- Spice6 should use `market_month = delivery_month`
- `binding_freq` cutoff should be based on decision timing, not period offset
- for `f1+`, collect previous 8 usable rows instead of previous 8 calendar months
- eval months should be filtered by auction schedule

## Recommended Revision

Before implementation, I would rewrite the architecture summary as:

1. Add period helpers:
   - `period_offset`
   - `delivery_month`
   - `has_period_type`

2. Fix source-of-truth joins:
   - GT uses `delivery_month`
   - Spice6 uses `market_month = delivery_month`

3. Add a new explicit lagged/as-of row-selection helper:
   - do **not** repurpose the generic loader contract

4. Keep row inclusion and feature cutoff as separate concepts:
   - row inclusion depends on label settlement
   - `binding_freq` cutoff depends on decision-time realized-data availability

5. Add a verification task for V6.2B snapshot provenance:
   - otherwise the final claim should remain conditional

## Bottom Line

I agree with the intended `f1` logic, but I would not approve the plan unchanged.

The main fix needed is architectural:

- **do not bake lagged/as-of semantics into the shared generic loader**

And the main missing verification is:

- **prove or explicitly assume that the V6.2B / monthly snapshot inputs are point-in-time safe**
