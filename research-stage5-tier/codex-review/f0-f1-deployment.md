# f0/f1 Deployment Note

## Intent

The intended `7.0` rollout is:

- use the stage5 ML pipeline for `f0`
- use the stage5 ML pipeline for `f1`
- keep all other period types exactly equal to `6.2B`
- expose the result as one wrapped signal family

That intention makes sense.

It is achievable.

It is slightly broader than the old `f0`-only note, but it is on the same architectural line:

- ML where stage5 is reviewed and ready enough (`f0`, now also `f1`)
- passthrough `6.2B` where stage5 is not intended to replace production yet (`f2+`, quarterlies, annuals)

## What “7.0” Means Technically

The wrapped signal API is just partitioned parquet behind `ConstraintsSignal`:

- reads: `ConstraintsSignal.load_data(...)`
- writes: `ConstraintsSignal.save_data(...)`

The path contract is:

- `{rto}/constraints/{signal_name}/{auction_month}/{period_type}/{class_type}`

from [base.py](/home/xyz/workspace/pbase/src/pbase/data/dataset/signal/base.py).

That means a single signal family like `TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1` can contain:

- ML-generated `f0`
- ML-generated `f1`
- copied `6.2B` data for `f2`, `f3`, `q2`, `q3`, `q4`, etc.

So if you want one signal name that downstream code can load uniformly, the correct implementation is a **hybrid family** under one wrapper name.

## Recommended Deployment Shape

For each auction month and class type:

- `f0`: score with stage5 `v10e-lag1` logic
- `f1`: score with the updated stage5 `f1 v2` logic
- all other period types: copy `6.2B` output unchanged into the `7.0` path

So `7.0` becomes:

- `f0` = stage5 ML
- `f1` = stage5 ML
- `everything else` = exact `6.2B` passthrough

This is the cleanest way to satisfy your requirement that:

- `f0/f1` match research-stage5 behavior
- non-`f0/f1` remain exactly `6.2B`
- downstream consumers only see one signal name

## Can It Match Stage5 Exactly

Yes in the practical deployment sense, with two caveats.

What can match exactly:

- same feature construction
- same safe training window logic
- same blend weights
- same model configuration
- same rank/tier transformation
- same output schema

What still needs discipline:

1. The deployment code must call the same scoring logic as stage5, not a rewritten approximation.
2. Exact equality of ML output depends on using the same data inputs and the same library/runtime behavior.

For the stage5 logic itself:

- `f0` should match current `v10e-lag1`
- `f1` should match current `v2`

The main operational requirement is to reuse the existing code paths closely enough that deployment is a packaging layer, not a reimplementation.

## Confirmed 6.2B Compatibility

The wrapped `6.2B` signal name is:

- `TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1`

For `rank_ori`, `6.2B` matches the current `v0` formula exactly:

- `rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value`

This is supported by:

- [ml/v62b_formula.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/v62b_formula.py)
- [production-migration/assessment.md](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/production-migration/assessment.md)

So passthrough for non-`f0/f1` is straightforward:

- load the `6.2B` parquet partition
- save the same rows under the new `7.0` signal name

No modeling work is needed for those untouched period types.

## What Must Exist Before Shipping

### 1. Enough realized DA cache

For ML periods:

- `f0` needs realized DA through `M-2`
- `f1` needs realized DA through the correct `f1` delivery-time cutoff

Without this:

- training rows are incomplete
- `binding_freq_*` cannot be computed correctly

### 2. Inference-only loaders for target months

Current shared loading is evaluation-oriented and joins target-month GT.

For deployment, `f0` and `f1` target-month scoring must:

- load target-month V6.2B features
- load target-month Spice6 features
- compute lagged `binding_freq_*`
- not require target-month `realized_sp`

This is still a small but necessary code change.

### 3. A hybrid signal builder

The deployment job should:

1. build ML signal partitions for `f0`
2. build ML signal partitions for `f1`
3. enumerate all other tradable period types for that auction month
4. copy those partitions from `6.2B` into `7.0`
5. save every partition through `ConstraintsSignal.save_data()`

This is the key addition beyond the old `f0`-only deployment note.

### 4. Rank / tier generation for ML partitions

For ML `f0` and `f1`, the output must include at least:

- `constraint_id`
- `shadow_sign`
- `shadow_price`
- `rank_ori`
- `rank`
- `tier`

Practical mapping:

- `rank_ori` = model score
- `rank` = dense-rank-normalized score
- `tier` = 5 buckets from rank

What we know confidently:

- `6.2B rank` is dense-rank normalized
- `6.2B tier` is 5 roughly equal buckets

The best practical implementation remains:

- `rank = dense_rank(score) / max_dense_rank`
- `tier = qcut(rank, 5)` with output `0..4`

### 5. Signal-shaped output writing

The final job must write the same wrapped signal schema that downstream code expects.

This is operational plumbing, not research.

## What Does Not Need To Change

These do not need new modeling work for `7.0`:

- `f2`
- `f3`
- `q2`
- `q3`
- `q4`
- any other untouched period/class slices

They can be exact `6.2B` passthrough.

## Recommended Runtime Contract

The clean production entrypoint should do this for one auction month:

1. determine tradable period types for that auction month
2. for `f0`, run stage5 ML scoring
3. for `f1`, run stage5 ML scoring
4. for all remaining period types, copy `6.2B`
5. write every partition under one signal name, e.g. `...V7.0.R1`

That is the right deployment shape if you want one wrapped signal family instead of multiple mixed signal names in downstream configs.

## Runtime and Parallelism

This section is a recommendation, not a hard requirement.

The useful parallelism unit is not one giant dataframe. It is each independent:

- `(auction_month, period_type, class_type)`

For one auction month, the expensive work is only the ML slices:

- `f0/onpeak`
- `f0/offpeak`
- `f1/onpeak`
- `f1/offpeak`

All other period types are just passthrough copy from `6.2B`, so they should be much cheaper.

Recommended execution shape:

1. copy passthrough `6.2B` slices in parallel if needed
2. run ML scoring for `f0` and `f1` with light parallelism
3. write all partitions under the `7.0` signal name

Why only light parallelism for ML:

- the codebase already notes that storage reads can dominate runtime
- LightGBM uses internal threads
- the benchmark code explicitly warns that aggressive multiprocessing can be slower because of spawn overhead and NFS contention

So the practical recommendation is:

- parallelize passthrough copy jobs more freely
- keep ML jobs to roughly `1-2` workers per machine unless performance testing shows more helps
- group incoming data by `(auction_month, period_type, class_type)` and process each group independently

If the input arrives as one dataframe covering many months and slices, that is still workable:

- split it by `(auction_month, period_type, class_type)`
- process each slice separately
- avoid assuming one giant dataframe should be scored in one model call

The main point is:

- hybrid `7.0` should be fast enough in practice because only a small number of slices require real ML work
- the rest is mostly IO and format-preserving passthrough

## Main Remaining Risks

### 1. Upstream snapshot provenance

This remains the main shared unresolved risk.

Meaning:

- if `6.2B` / Spice6 monthly snapshots are true pre-auction artifacts, then both ML and passthrough slices are fine on that part
- if those upstream snapshots were regenerated later with post-cutoff information, then both `6.2B` passthrough and stage5 ML inherit that issue

This is a shared upstream-data assumption, not a new deployment-only bug.

### 2. f1 artifact discipline

The updated `f1` path is much better, but deployment should use the corrected current logic:

- correct `f1` row timing
- correct `delivery_month` GT semantics
- correct class-type-specific blend weights

Do not freeze an older `f1` snapshot of the code.

## Bottom Line

Your intention is coherent and achievable.

The right `7.0` interpretation is:

- `f0` = stage5 ML
- `f1` = stage5 ML
- all other ptypes = exact `6.2B`
- all saved under one wrapped signal name

That is fully consistent with my deployment understanding now.

Compared with the old `f0`-only note, the only real extension is:

- add `f1` ML generation
- add hybrid-family assembly under one signal name instead of separate signal names by period type

So the recommended deployment strategy is:

- build a hybrid `7.0` family
- reuse stage5 scoring logic for `f0` and `f1`
- copy untouched `6.2B` partitions for everything else
- do not re-model `f2+` just to ship `7.0`

data cached has been refreshed.
for f0 onpeak 2025-03 is the same for auction month & market month right?
what about f1? for 6.2b, which "month" is the latest


## 
**crucial thing on speed**==> after "caching" what constraints are there for any month, and use 7.0 pipeline to calculate the rank -> then divide into tiers using pcut, then there is nothing Machine Learning to run anymore right? because ALL constraints in our universe are pre-loaded, and ML speed is not a constraint at all?

am i over-worrying on the speed issue?

Clear pattern: starting from 2026-01, spice6 renamed score_df.parquet to score.parquet. Our loader only
  looks for score_df.parquet, so it silently returns empty for 2026-01+.   ==> is this true? write this into a core .md.