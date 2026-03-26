# Handoff Note: Round-Based Specialist Modeling Branch

**Date**: 2026-03-24  
**Audience**: branch working on round-based specialist / reranker release candidate  
**Purpose**: clarify what must be made round-aware and ctype-aware before the specialist branch can be compared fairly to `7.1b` or treated as a publish candidate.

---

## Executive summary

Yes: if the specialist branch is moving toward a real annual release candidate, it should adopt the same release standards as the round-sensitive `v0c` path.

That means the branch should not just:

- add a new selector
- show improved dormant / NB metrics at `K=400`

It also needs to make the **inputs**, **features**, **evaluation grain**, and **publish path** consistent with round-sensitive annual production.

The key requirements are:

1. Use explicit `(planning_year, aq_quarter, class_type, market_round)` as the base grain.
2. Use round-specific annual auction-side inputs, not implicit `market_round=1`.
3. Use day-level DA cutoff logic for history-derived features.
4. Re-audit every feature used by the specialist model for round/ctype correctness.
5. Recompute all comparison metrics at the new base grain, even though GT itself is unchanged.

The GT of realized branches does **not** change by round, but the candidate set, features, ranking, and therefore top-K portfolio metrics absolutely do.

---

## What this means in practice

### 1. Inputs must be class- and round-aware

If the specialist branch is intended to compete with or extend `7.1b`, the following inputs should be explicit in the feature pipeline:

- `class_type`
  - `onpeak`
  - `offpeak`
- `market_round`
  - `R1`
  - `R2`
  - `R3`

This applies to:

- density / flow inputs
- constraint limits
- bridge / CID mapping
- branch metadata
- flow-direction source
- SF if the model is evaluated in a publish-like path

If a feature currently hardcodes `market_round=1`, then that feature is still R1-only.

If a feature is built from mixed `onpeak/offpeak` data without an explicit class contract, then it is not yet suitable for a ctype-sensitive release.

### 2. DA history should use daily cache + cutoff function

If the goal is a true round-sensitive annual model, history-derived features should use:

- day-level DA cache
- round-specific cutoff date
- explicit rule for excluding post-close days

This is needed because:

- `R1`, `R2`, and `R3` see different pre-close history
- a branch can be dormant in `R1` but not in `R2`
- partial-month history can change BF / recency / cumulative-SP-style features materially

For release parity with `7.1b`, the specialist branch should not remain on month-only DA history if it claims round-sensitive behavior.

### 3. Each feature should be inspected carefully

Yes. Each feature should be inspected individually.

Do not assume that because the model is “round-based” overall, every feature is correct.

Each feature should be classified as one of:

- fully round- and ctype-aware
- ctype-aware only
- round-aware only
- legacy R1-only
- mixed / ambiguous

For the specialist branch, that audit should be explicit for at least:

- deviation / flow profile features
- density-bin features
- top-k / top2 / tail-shape features
- BF / NB / recency features
- cumulative DA / shadow features
- any branch aggregation over CID-level signals

If one critical feature family is still R1-only, then the model is still operationally R1-only even if the runner accepts a round argument.

### 4. Metrics must be recomputed under the new base grain

Yes. Every comparison metric should be updated.

Even though the realized GT for a given `(planning_year, aq_quarter, class_type)` is the same, the following change by round:

- branch universe
- CID mapping
- feature values
- ranking
- top-K membership
- deployment portfolio composition

So all model-vs-model comparisons need to be recomputed at:

- `(planning_year, aq_quarter, class_type, market_round)`

This applies to comparisons against:

- `v0c`
- `7.1b`
- `V4.4`
- base / reranker / reserved-slot policies

---

## Minimum technical requirements for the specialist branch

### A. Feature pipeline requirements

The specialist branch should explicitly confirm whether each of these is round-aware:

- density / flow source path
- limits source path
- bridge source path
- branch metadata source path
- flow-direction source path
- SF source path if used in publish-like validation
- DA history source
- cutoff function
- class-specific BF / NB logic

Release standard:

- no silent default `market_round=1`
- no mixed-round loading
- no hidden use of legacy `data/nb_cache/`-style artifacts unless clearly declared R1-only

### B. Evaluation requirements

The specialist branch should rerun evaluation at the base grain:

- `planning_year`
- `aq_quarter`
- `class_type`
- `market_round`

At minimum, it should report:

- native `SP@200`
- native `SP@400`
- native `VC@200`
- native `VC@400`
- binder counts / precision style metrics
- heavy-binder metrics such as `D20` / `D40`
- NB metrics such as `NB12_Count`, `NB12_SP`
- coverage / unlabeled accounting for `V4.4`

And it should keep metric views separate:

- `native_topk`
- `overlap_topk`
- `deployment_policy`

Do not mix overlap-style conclusions into native-release claims.

### C. Policy validation requirements

If the intended shipped behavior is:

- `150 base + 50 specialist` at `K=200`
- `300 base + 100 specialist` at `K=400`

then those exact policies must be run and saved.

It is not enough to show:

- `R30`
- `R50`
- `R100`

at `K=400` only and then infer what `150+50` or `300+100` would do.

### D. Publish-path requirements

If the specialist selector is part of the release claim, it must be wired into a publish-like path.

At minimum:

- the production selector logic must exist in the publish path or a publish-equivalent runner
- output should be tested for both ctypes
- round scope must be explicit:
  - `R1-only`
  - or `R1/R2/R3`

If the branch remains R1-only, that is acceptable only if it is stated explicitly in the release scope.

---

## Specific implications for the specialist branch summary

Based on the summary provided:

- The current saved eval appears meaningful, but it is still `K=400`-centric.
- It is not enough for a release candidate if the intended shipped policy includes `K=200`.
- If `deviation_profile.py` is still hardcoded to `market_round=1`, then the specialist model remains R1-only.
- If the publish path still uses `score_v0c` only, then the specialist is not yet productionized.

So the right interpretation is:

- promising specialist candidate
- not yet a release candidate
- must be upgraded to the same round-sensitive standards as `7.1b` before fair release comparison

---

## Recommended handoff tasks for the specialist branch

### 1. Input audit

Produce a table with one row per input family:

- source
- current grain
- ctype-aware?
- round-aware?
- release-safe?

### 2. Feature audit

Produce a table with one row per feature:

- feature name
- depends on class type?
- depends on round?
- depends on DA cutoff?
- currently correct?
- action needed

### 3. Metric rerun

Rerun all model comparisons at:

- `(planning_year, aq_quarter, class_type, market_round)`

against:

- `v0c`
- matched-round `V4.4`
- the intended final policy

### 4. Policy rerun

Run the exact target policies:

- `150 + 50` at `K=200`
- `300 + 100` at `K=400`

### 5. Release-scope decision

Decide explicitly:

- R1-only release
- or full R1/R2/R3 release

Do not leave this implicit.

---

## Bottom line

Yes:

- the specialist branch should use class-based flow/density files
- it should use daily DA cache plus round cutoff logic if it wants to claim round sensitivity
- each feature should be inspected carefully
- and each comparison metric should be updated under the new base grain

The GT itself does not change by round, but the evaluation still must, because the ranking problem changes by round.

Treat round-sensitive annual production as a full contract:

- inputs
- features
- metrics
- policy
- publish path

not just a model improvement on top of legacy R1-only infrastructure.
