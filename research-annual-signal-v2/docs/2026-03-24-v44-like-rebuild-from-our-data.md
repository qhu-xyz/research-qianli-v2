# 2026-03-24 V4.4-Like Rebuild From Our Own Data

## Goal

Rebuild the core idea behind V4.4 using only inputs we control in this repo.

This is not an exact reproduction of the original V4.4 notebook.
It is an internal replacement design intended to recover the same signal family:

- forward-looking flow stress
- extreme-tail emphasis
- weaker dependence on BF/history
- better dormant-branch ranking

The target is not to copy V4.4 end to end.

The actual target is:

- be about as predictive as V4.4 on NB / dormant-branch prediction
- while staying roughly as good as `Bucket_6_20` on overall value capture

So success requires both:

- strong dormant/NB behavior
- strong general top-K value capture

## Motivation

From the recovered source notebook:

- [se-signal-annual.ipynb](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/human-inputs/se-signal-annual.ipynb)

the likely reason V4.4 beats us on top dormant branches is:

- it uses richer exceedance/deviation features
- it ranks at constraint/CID resolution before dedup
- it uses DA history only as a minority component

We do not currently have the raw `flow_memo.parquet` source used by that notebook on this machine, so we should rebuild the signal family from data we do have.

## Available Internal Inputs

We already have:

- raw annual density distribution parquet
  - [MISO_SPICE_DENSITY_DISTRIBUTION.parquet](/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet)
- annual bridge
  - [MISO_SPICE_CONSTRAINT_INFO.parquet](/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet)
- annual limits
  - [MISO_SPICE_CONSTRAINT_LIMIT.parquet](/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet)
- round-aware annual partitions
- DA history / realized DA cache
  - `data/realized_da/`

This is sufficient to build an approximate V4.4-like feature family.

## Success Criteria

This rebuild should be judged against two baselines at the same time:

1. NB / dormant benchmark:
- published V4.4

2. Overall value-capture benchmark:
- `Bucket_6_20`

The intended outcome is:

- close the dormant prediction gap versus V4.4
- without losing the overall SP capture that makes `Bucket_6_20` strong

That means:

- beating V4.4 on NB while collapsing on total SP is not acceptable
- matching Bucket on total SP while staying weak on dormant branches is also not acceptable

## Core Idea

Replace our current narrow branch-level tail summary:

- `bin_80/90/100/110`
- `rt_max`
- `top2_bin_*`

with a richer CID-level deviation feature built from the full tail shape.

The main shift is:

- from `max of a few bins`
- to `distance over an exceedance profile`

## Proposed Signal Family

### Step 1: Build CID-level exceedance profiles

From the raw density distribution rows, for each CID and outage-date:

compute cumulative exceedance probabilities for thresholds:

- positive tail: `60, 65, 70, 75, 80, 85, 90, 95, 100`
- negative tail: `-60, -65, -70, -75, -80, -85, -90, -95, -100`

For the positive side:

- `exc_60 = P(flow >= 60)`
- `exc_65 = P(flow >= 65)`
- ...
- `exc_100 = P(flow >= 100)`

For the negative side:

- use absolute negative-tail exceedance in the same way

Then pick direction per CID:

- `chosen_tail = argmax(dev_positive, dev_negative)`

or keep both and let the model use them.

### Step 2: Build two profile styles per CID

Approximate the notebook’s `max` and `sum` styles.

For each CID and threshold:

1. `profile_max`
- fraction of outage dates where `exc_t > 0`

2. `profile_sum`
- average `exc_t` across outage dates

These are not exact notebook equivalents, but they preserve the same idea:

- one feature family measures whether a threshold is ever stressed
- the other measures how broadly/often it is stressed

### Step 3: Convert cumulative exceedance into histogram mass

For each profile, convert cumulative exceedance to histogram-like mass:

- `mass_0 = 1 - exc_60`
- `mass_60 = exc_60 - exc_65`
- `mass_65 = exc_65 - exc_70`
- ...
- `mass_95 = exc_95 - exc_100`
- `mass_100 = exc_100`

This creates a probability vector over stress levels.

### Step 4: Compute deviation distance

Use the same weighted Wasserstein-style idea as the original implementation:

- baseline distribution concentrated at bin `0`
- compare each CID profile against that baseline

Distance:

- cumulative difference from baseline
- weighted by threshold severity

Initial weight schedule:

- `w_i = 7 ** i`

Also test lighter variants:

- `3 ** i`
- `5 ** i`

Primary CID-level outputs:

- `cid_dev_max`
- `cid_dev_sum`

### Step 5: Aggregate from CID to branch

Do not collapse too early.

Compute CID-level deviation first, then aggregate to branch with:

- `best_cid_dev_max`
- `best_cid_dev_sum`
- `top2_cid_dev_max`
- `top2_cid_dev_sum`
- `dev_gap_max = best - second`
- `dev_gap_sum = best - second`
- `count_cids_dev_max_gt_x`
- `count_cids_dev_sum_gt_x`

This keeps more within-branch structure than our current branch-first collapse.

## Round Awareness

This rebuild should be round-aware from day 1.

Use:

- round-specific density partitions
- round-specific bridge partitions
- round-specific limits if needed
- round-aware DA cutoff logic

This aligns with the round-sensitive annual design docs:

- [2026-03-24-repo-reformulation-and-round-sensitive-annual-design.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-repo-reformulation-and-round-sensitive-annual-design.md)
- [2026-03-24-round-sensitive-annual-test-design.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-round-sensitive-annual-test-design.md)

## History Component

History should remain a minority component.

Recommended history inputs:

- `shadow_price_da`
- `shadow_rank`
- maybe `bf_12` / `bfo_12`

But history should not dominate the rank for dormant branches.

Initial blend to test:

- `0.4 * dev_sum_rank`
- `0.4 * dev_max_rank`
- `0.2 * shadow_rank`

This is intentionally close to the original V4.4 logic.

## Candidate Models

### Model A: Formula prototype

No ML initially.

Use:

- `rank_score = 0.4 * dev_sum_rank + 0.4 * dev_max_rank + 0.2 * shadow_rank`

This tests whether the feature family itself is valuable.

### Model B: Add deviation features to Bucket_6_20

Take the current feature recipe:

- `miso_annual_bucket_features_v1`

and extend with:

- `best_cid_dev_max`
- `best_cid_dev_sum`
- `top2_cid_dev_max`
- `top2_cid_dev_sum`
- `dev_gap_max`
- `dev_gap_sum`

This tests whether the new signal helps a learned ranker.

This is the most important path, because it directly targets the desired final product:

- preserve the general value-capture strengths of `Bucket_6_20`
- import the dormant-branch signal family that makes V4.4 useful

### Model C: Dormant specialist only

Use the same deviation features inside a dormant-only NB model.

This isolates whether the gain is specifically on dormant branches.

## Evaluation Plan

Evaluate against:

- `v0c`
- `Bucket_6_20`
- published V4.4

using the existing metric contract views:

1. native top-K
2. overlap-only
3. deployment R30/R50

Priority metrics:

- `SP@200`, `SP@400`
- `NB_SP@200`, `NB_SP@400`
- top true dormant binder hit rate
- overlap-only top dormant absolute rank

Primary decision rule:

- NB metrics should approach or beat V4.4
- overall SP metrics should remain roughly in the `Bucket_6_20` range

## Implementation Plan

### Phase 1: Feature builder

Create a new builder for CID-level deviation features from annual density parquet.

Suggested module:

- `ml/features/deviation_profile.py`

Outputs:

- CID-level deviation table
- branch-level aggregated deviation table

### Phase 2: Formula benchmark

Implement a direct formula ranker using only:

- `dev_sum_rank`
- `dev_max_rank`
- `shadow_rank`

Suggested script:

- `scripts/v44_like_formula.py`

### Phase 3: ML integration

Add branch-level deviation features to the bucket model and compare.

Suggested feature recipe id:

- `miso_annual_bucket_plus_dev_features_v1`

### Phase 4: Round-aware version

Once round-sensitive plumbing is complete:

- recompute deviation features per round
- compare `R1`, `R2`, `R3`

## Risks

1. Density distribution is still a compressed representation.
It may not capture everything `flow_memo.parquet` captured.

2. Branch collapse can still lose signal.
Even with better CID-level scoring, some loss remains if the final model is branch-level only.

3. Weight schedule may matter a lot.
The original notebook’s `7 ** i` weighting may be too aggressive or too noisy for our data.

4. Direction handling needs care.
Negative-tail and positive-tail stress should not be mixed incorrectly.

## Recommended First Experiment

Before any large refactor, run the smallest useful test:

1. Build CID-level deviation features from density parquet for one holdout year.
2. Aggregate to branch as `best_cid_dev_max` and `best_cid_dev_sum`.
3. Rank by:
   - `0.4 * dev_sum_rank + 0.4 * dev_max_rank + 0.2 * shadow_rank`
4. Compare against:
   - `v0c`
   - `Bucket_6_20`
   - published V4.4
5. Inspect known dormant misses:
   - `MNTCELO`
   - `13866`
   - `AUST_TAYS`
   - `BURNHMUNST`

If this closes the gap on those names, then the feature family is likely the right direction.

## Decision Rule

This approach is worth promoting only if it does at least one of:

- materially improves overlap-only dormant ranking
- materially improves top dormant binder hit rate
- materially improves deployment NB capture without unacceptable SP loss

Promotion standard:

- NB / dormant prediction should be comparable to V4.4
- overall value capture should stay comparable to `Bucket_6_20`

If it does not, then the missing signal is likely outside what our density parquet can recover, and we should stop short of a full rebuild claim.
