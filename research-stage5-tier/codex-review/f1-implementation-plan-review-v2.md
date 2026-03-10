# Review: Updated F1 Implementation Plan

Reviewed plan:

- [2026-03-09-f1-implementation.md](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/docs/plans/2026-03-09-f1-implementation.md)

## Verdict

This revision is materially better than the earlier version.

The biggest architectural problem from the first draft was fixed:

- the plan no longer pushes lagged/as-of semantics into the shared generic loader

It also now explicitly separates:

- row inclusion
- per-row `binding_freq` cutoff
- point-in-time snapshot verification

Those are the right corrections.

I still would not call the plan fully ready to implement without adjustment. There are a few remaining issues.

## Findings

### 1. The plan still over-claims full leakage closure from timestamp checks

Task 6 adds a provenance check for V6.2B and Spice6 snapshots using file timestamps. That is better than nothing, but it is not strong enough to support a claim like:

- “feature leakage is fully resolved”

Why:

- matching file timestamps only shows files were written around the same time
- it does **not** prove the snapshots were generated before the actual auction cutoff
- it does **not** prove the upstream feature computation itself avoided using post-cutoff information

This matters because the plan’s goal line says “correct temporal leakage guards,” and the V6.2B / Spice6 snapshot assumption is still the biggest remaining feature-leakage uncertainty.

Recommendation:

- keep Task 6, but downgrade the conclusion
- final results should say:
  - “row leakage, GT month selection, and binding_freq leakage fixed”
  - “snapshot provenance checked for consistency, but still an upstream assumption”

### 2. `collect_usable_months()` is now scoped correctly, but its contract should be renamed to make the target explicit

The new Task 5 is much better because it keeps the generic loader untouched and makes [scripts/run_v10e_lagged.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py) the sole owner of training-month selection.

But the helper name and arguments are still a bit too vague:

- `collect_usable_months(target_auction_month, period_type, ...)`

This helper is actually returning:

- usable **training auction months**
- relative to a specific **target auction month**

That distinction matters because the repo already uses “month” ambiguously for eval month, query month, auction month, and delivery month.

Recommendation:

- rename to something like:
  - `collect_usable_training_auction_months(...)`
- document explicitly that the returned values are auction months, not delivery months

This is not a correctness blocker, but it will prevent future confusion.

### 3. The version naming is now more misleading than before

The updated plan now defines:

- `v1` = f1 blend-search result
- `v2` = full ML model

while the script still conceptually derives from `v10e-lagged`.

At the same time, Task 4 says:

- row inclusion for f1 behaves like “lag 2”
- `binding_freq` still uses lag 1

So the saved version identity will be under-described unless the config records:

- period type
- row-selection rule
- binding_freq cutoff rule
- blend weights used for the formula feature

Recommendation:

- do not rely on the version ID alone to convey timing semantics
- make sure the saved config/metrics writes all of:
  - `period_type`
  - `last_full_known rule`
  - `row_selection = usable_auction_rows`
  - `binding_freq_lag = 1`
  - `blend_weights = (...)`

### 4. The new `v1` blend-search step is valid, but the plan should state more explicitly that it is not a production formula replacement

Task 12 introduces:

- `v1` = f1-specific blend search over `(da, dmix, dori)`

This is reasonable as an experimental baseline and a feature-engineering input for `v2`.

But the plan could easily be read as:

- “replace the production formula for f1 with the blend-search result”

That is not the same thing as reproducing V6.2B’s native `rank_ori`.

Recommendation:

- explicitly separate:
  - `v0`: exact V6.2B production formula reproduction
  - `v1`: empirical f1 blend-search baseline
  - `v2`: ML model using the chosen blend as one feature

This is mostly a documentation clarity issue, but it matters because the plan is now comparing three conceptually different baselines.

### 5. The plan still assumes the same 8-row training depth is sufficient for sparse future period types

For `f1`, 8 usable rows is a reasonable continuation of the `f0` setup.

But the plan is already written as if the same pattern will naturally extend to `f2/f3`, which may not be stable:

- `f1`: sparse but workable
- `f2`: much sparser
- `f3`: sparse enough that “8 usable rows” may reach too far back or become unstable

This is not a problem for the stated `f1` goal, but the plan should avoid implying the same recipe automatically generalizes.

Recommendation:

- keep the implementation scoped to `f1`
- document that `f2/f3` need separate minimum-data checks before reuse

## What Improved

Compared with the earlier draft, these changes are solid:

- generic loader contract is preserved
- per-row `binding_freq` cutoff is now explicitly distinguished from target-row cutoff
- the data-availability count was corrected to 30 dev months
- provenance verification is now at least acknowledged instead of ignored
- the plan now treats `v0`, blend search, and ML as separate phases

## Recommended Final Tweaks

Before implementation, I would make these small revisions:

1. In Task 6, weaken the claim from “verified non-leaky” to “upstream snapshot assumption checked for consistency.”
2. Rename `collect_usable_months()` to make it explicit that it returns training auction months.
3. Ensure saved configs record row-selection and feature-cutoff semantics explicitly.
4. Clarify in Task 12 that `v1` is an empirical experimental baseline, not the production formula.
5. Keep all future-period generalization text clearly out of scope for this `f1` plan.

## Bottom Line

This version is close.

I would approve it as an implementation plan **after** tightening the provenance language and the naming/metadata pieces. The core temporal logic is now mostly right.
