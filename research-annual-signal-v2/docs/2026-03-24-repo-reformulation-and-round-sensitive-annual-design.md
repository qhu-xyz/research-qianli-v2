# Repo Reformulation + Round-Sensitive Annual Design

**Date**: 2026-03-24  
**Scope**: MISO annual immediately, PJM annual after the reformulation  
**Purpose**: Combine the repo/product reorganization with a concrete design for round-sensitive annual features and publishing.

---

## 1. Problem statement

Two changes are now needed at the same time:

1. The repo needs to be reformulated around durable product contracts rather than phase-driven research scripts.
2. The annual signal needs to become round-sensitive.

Today the annual pipeline has two limitations:

- history is built from **month-level** realized-DA caches, so a few extra days before R2/R3 cannot change features
- auction-side data loaders are hardcoded to `market_round=1`, so the pipeline ignores round-specific density, limits, SF, and branch metadata even though round 2 and round 3 partitions exist

The result is that:

- `7.0b` is identical across annual rounds
- `bucket_6_20` is identical across annual rounds
- the repo structure makes it hard to express this limitation cleanly

This design addresses both issues in one plan.

---

## 2. Current-state facts

### 2.1 What is currently round-insensitive

Annual auction-side loaders use `market_round=1` today:

- density distribution
- constraint limits
- SF matrices
- pbase branch metadata used by publishing

History uses month-level DA cache:

- one monthly parquet per `YYYY-MM` and ctype
- each file contains one `realized_sp` per `constraint_id` for the whole month

### 2.2 What this means operationally

Current annual features can differ by planning year, quarter, and ctype, but not by round.

They are insensitive to:

- different annual round partitions
- partial-month DA history between R1, R2, and R3

### 2.3 What can change if redesigned

If the pipeline is made round-aware:

- density/limits/SF/branches can differ across rounds immediately
- history can also differ across rounds if DA is rebuilt at daily resolution or month-to-date partial resolution

---

## 3. Design goals

### 3.1 Functional goals

- support annual rounds `R1`, `R2`, `R3` as explicit pipeline inputs
- use exact round cutoff dates from pbase/calendar sources
- include all history available strictly before each round cutoff under the chosen cutoff granularity
- use round-specific annual auction partitions where available
- preserve ex-ante validity
- keep `onpeak` and `offpeak` explicit everywhere

### 3.2 Repo/product goals

- standardize universes, feature recipes, label recipes, model specs, policy specs, and release specs
- separate benchmarks from internal models
- separate research scripts from production code
- prepare a clean market boundary for PJM

### 3.3 Non-goals

- training separate annual models per round in the first implementation
- redesigning the entire annual feature set at the same time
- rewriting all historical research scripts into the new framework before `7.1b`

---

## 4. First-class objects to add

The existing migration plan already introduces:

- `Universe`
- `FeatureRecipe`
- `LabelRecipe`
- `ModelSpec`
- `PolicySpec`
- `ReleaseSpec`

Round sensitivity adds three more first-class objects.

### 4.1 RoundSpec

Defines:

- round id: `R1`, `R2`, `R3`
- official bidding close timestamp
- last valid history timestamp
- eligible auction-side partitions

Example:

- `miso_annual_round_r1_v1`
- `miso_annual_round_r2_v1`
- `miso_annual_round_r3_v1`

### 4.2 CutoffSpec

Defines how feature history is truncated.

Fields:

- timestamp granularity: monthly / daily / sub-daily
- inclusion rule
- month-to-date policy if used

Implementation decision for v1:

- use **day-level** cutoff granularity
- include dates strictly before the round close date
- exclude the entire close date

Reason:

- this is implementable with a daily DA cache
- it still gives R2/R3 additional valid history days
- it avoids pretending we have sub-day precision when the proposed cache does not preserve it

Possible future upgrade:

- move to hourly or timestamped DA history if true intra-day cutoff precision becomes necessary

### 4.3 CacheManifest

Defines versioned local cache coverage.

This applies at least to:

- daily DA cache
- monthly DA cache
- NB/model cache
- collapsed/universe cache

---

## 5. Target data model

### 5.1 Round-aware run key

Every annual run key becomes:

- `planning_year`
- `aq_quarter`
- `class_type`
- `market_round`

This applies to:

- model table building
- benchmark comparison
- publication
- release manifests

This is the canonical annual base grain.

That means:

- every annual model, benchmark, and policy result is first built and evaluated at `(planning_year, aq_quarter, class_type, market_round)`
- any aggregate across quarters, ctypes, or rounds is secondary and must be derived from that base grain

Immediate v1 policy:

- `class_type` separation is mandatory
- `market_round` separation is mandatory
- separate per-round training is optional in v1
- separate per-round feature/build/eval artifacts are mandatory in v1

### 5.2 Daily DA history cache

Add a new cache family alongside the existing monthly cache.

Recommended layout:

```text
data/realized_da_daily/
  2025-04-01.parquet
  2025-04-01_offpeak.parquet
  ...
```

Each file should contain:

- `constraint_id`
- `realized_sp`
- date metadata implied by filename

Aggregation rule:

- sum within day + ctype
- abs after netting within that day + ctype

This preserves enough resolution to build:

- full-month aggregates
- month-to-date partial aggregates
- exact pre-close **days**

Source requirement:

- daily DA cache must be built from a source that provides trade-date-level DA rows
- the implementation must document the exact upstream loader used
- acceptable first sources are a pbase daily DA loader or an equivalent daily AP-tools extraction

### 5.3 Derived monthly cache

The current month-level cache can remain as a derived convenience layer.

Rule:

- monthly cache must be derivable from daily cache
- daily cache is the source of truth for round-sensitive history

### 5.4 Round-aware auction-side data

All annual auction-side loaders must accept `market_round`.

This includes:

- density distribution
- constraint limits
- density signal score / flow direction
- SF matrices
- pbase annual branches

Round-aware loader contract:

- if round partition exists, use it
- if it does not exist, fail explicitly
- do not silently fall back to round 1

---

## 6. New annual pipeline design

### 6.1 Inputs

Each run takes:

- `planning_year`
- `aq_quarter`
- `class_type`
- `market_round`
- `universe_id`
- `feature_recipe_id`
- `label_recipe_id`
- `policy_id`

No annual production or benchmark run should be defined without both:

- `class_type`
- `market_round`

### 6.2 Step A: Resolve round cutoff

Look up round close timestamp from pbase/calendar.

Derive:

- `round_close_ts`
- `history_cutoff_date`

Required rule:

- in v1, no history observation on or after the **round close date** may be included
- the entire close date is excluded

This is a deliberate v1 simplification.

If sub-day history is introduced later, the cutoff spec can be upgraded to exact timestamps.

### 6.3 Step B: Build round-aware history

From daily DA cache:

- include all days strictly before `history_cutoff_date`
- aggregate to branch-level monthly or rolling windows as needed by feature recipe

Features affected:

- `bf_*`
- `shadow_price_da`
- `da_rank_value`
- recency features
- recent max SP
- cross-class BF / SPDA features

Implementation note:

The feature recipe may still choose to expose 6/12/24-month windows, but the underlying history source must be day-aware so R2/R3 can incorporate additional pre-close days.

### 6.4 Step C: Load round-aware auction-side data

Use the requested `market_round` when loading:

- density distribution
- limits
- flow direction source
- SF
- annual branch metadata

This is the immediate path for round sensitivity even before training changes.

### 6.5 Step D: Build round-aware model table

The model table must be keyed by:

- `planning_year`
- `aq_quarter`
- `class_type`
- `market_round`

The same model spec may be used across all rounds, but the feature values can change by round.

### 6.6 Step E: Score

Initial recommendation:

- keep one model spec per ctype
- do not train separate models per round yet
- score round-specific features with the same fitted model

Reason:

- simplest first implementation
- lets us measure whether round-aware features create useful deltas before multiplying model count

### 6.7 Step F: Publish

Publishing becomes round-aware in both metadata and output path.

Recommended release path shape:

```text
.../signal_name/{planning_year}/{aq_quarter}/{class_type}/round={R}/signal.parquet
```

Implementation decision for v1:

- publish all three rounds as distinct outputs
- release manifests may additionally designate one round as canonical for downstream consumers if needed

Do not hide round-specific outputs behind a single shared path in the first implementation.

---

## 7. Model strategy for round sensitivity

### 7.1 Phase 1 recommendation

Use the same model weights across rounds, but recompute features per round.

Apply to:

- `7.0b` / `v0c`
- `bucket_6_20`

This isolates the value of:

- round-specific density/limits/SF
- extra pre-close history days

### 7.2 Phase 2 recommendation

Only consider round-specific model retraining if phase 1 shows material deltas.

Possible future variants:

- one model per round
- one model with `market_round` as a feature
- one model with cutoff-relative recency features

### 7.3 Expected benefit by component

Likely strongest first-order effect:

- round-specific density / limits / SF

Likely second-order effect:

- extra few days of DA history

Reason:

- auction-side annual partitions are explicitly round-specific
- current history features are fairly coarse, so only some branches will move from a few extra days

Still, daily DA is worth implementing because:

- it makes the history cutoff principled
- it removes a known structural limitation
- it generalizes to future products and PJM

---

## 8. Repo reformulation changes needed to support this

### 8.1 New product contracts

The migration plan remains in force:

- universes are explicit
- feature recipes are explicit
- label recipes are explicit
- models are explicit
- policies are explicit
- releases are explicit

Round sensitivity adds:

- `market_round` to run keys
- `RoundSpec`
- `CutoffSpec`
- cache manifests

### 8.2 New target layout

Recommended additions to the target structure:

```text
ml/
  core/
    calendars.py
    cache_manifest.py
  markets/
    miso/
      annual_calendar.py
      realized_da_daily.py
      annual_round_data.py
  products/
    annual/
      rounds/
      feature_recipes/
      label_recipes/
      universes/

registry/
  miso/
    annual/
      models/
      benchmarks/
      releases/
```

### 8.3 Release manifests

Every annual release must state:

- whether it is round-sensitive
- supported rounds
- cutoff source
- cutoff granularity
- cache manifest references
- whether a canonical consumer round exists in addition to full round outputs

Example:

- `miso_annual_7_1b` may be the first round-sensitive annual release

---

## 9. Concrete implementation plan

### Phase 0: Contracts and calendar

Deliverables:

- `RoundSpec` schema
- `CutoffSpec` schema
- annual round calendar extractor from pbase
- documented rule for exact cutoff handling

### Phase 1: Daily DA cache

Deliverables:

- `scripts/fetch_realized_da_daily.py`
- `ml/markets/miso/realized_da_daily.py`
- cache manifest for daily DA coverage
- documented upstream daily DA source

Rules:

- source of truth is daily cache
- monthly cache becomes derived

### Phase 2: Round-aware history builder

Deliverables:

- new history builder that accepts `cutoff_ts`
- month-to-date partial aggregation logic
- round-aware BF/SPDA/recency features

Design rule:

- exact inclusion boundary must be testable and auditable

### Phase 3: Round-aware auction-side loaders

Deliverables:

- density loader with `market_round`
- limits loader with `market_round`
- density signal score / flow-direction loader with `market_round`
- SF loader with `market_round`
- publish metadata loader with `market_round`

Hard rule:

- no silent fallback to `market_round=1`

### Phase 4: Round-aware model table

Deliverables:

- `build_class_model_table(..., market_round=...)`
- updated cache keys including round
- updated spec/registry keys including round

### Phase 5: Round-aware scoring and publish

Deliverables:

- round-aware `v0c`
- round-aware `bucket_6_20`
- round-aware release/publish runner
- hard failure if any published constraint is missing SF

### Phase 6: Repo reformulation execution

Deliverables:

- standardized registry layout
- research split from production surface
- MISO/PJM market boundary
- release manifests

---

## 10. Backward compatibility strategy

The repo reformulation and round-sensitive change should not break active scripts midstream.

Recommended approach:

1. Add new round-aware modules first.
2. Keep old entrypoints as thin wrappers.
3. Migrate production runners to new APIs.
4. Remove wrappers only after release verification passes.

This is safer than a single giant rewrite.

---

## 11. Risks

### 11.1 Data leakage risk

If round cutoffs are implemented loosely, it becomes easy to accidentally include post-close days.

Mitigation:

- exact timestamp cutoff contract
- explicit daily-cache inclusion tests

### 11.2 Cache coverage risk

Daily history introduces more cache coverage requirements.

Mitigation:

- cache manifests
- preflight coverage checks in every release run

### 11.3 Release complexity risk

Three annual rounds can triple release combinations.

Mitigation:

- one model spec per ctype at first
- round-specific inference first, not round-specific training

### 11.4 Repo migration risk

Market split can break imports.

Mitigation:

- compatibility shims
- phased migration

---

## 12. Success criteria

This design is successful when:

- annual runs can differ legitimately by `R1`, `R2`, `R3`
- round-specific differences come from explicit data and cutoff logic, not ad hoc copies
- daily DA history is the source of truth for history features
- `v0c` and successor models can be evaluated round by round
- release manifests make round coverage explicit
- the repo is structured so MISO and PJM can coexist cleanly

---

## 13. Recommended first build

The first implementation batch should do only this:

1. Add round calendar + cutoff specs.
2. Add daily DA cache and manifest.
3. Make history builder cutoff-aware.
4. Make density/limits/SF loaders accept `market_round`.
5. Recompute `v0c` and `bucket_6_20` features per round using the same fitted model weights.
6. Measure whether round-aware deltas are material before training separate round models.

That is the smallest serious implementation that answers the business question cleanly.
