# Repo Migration Plan: Annual Signal Productization

**Date**: 2026-03-24  
**Scope**: MISO annual now, PJM annual next  
**Purpose**: Move the repo from phase/script-driven research layout to a durable product layout with explicit contracts, a standard registry, and market boundaries.

---

## Status update

As of 2026-03-24, the migration is partially implemented.

Completed:

- current-state freeze doc
- universe catalog
- feature recipe bindings for `v0c`, `Bucket_6_20`, and the current NB specialist
- registry schema contract
- non-breaking skeleton packages under:
  - `ml/core/`
  - `ml/markets/miso/`
  - `ml/markets/pjm/`
  - `ml/products/annual/`

Implemented but not yet fully normalized into the new registry/release surface:

- round-aware `v0c` vs `V4.4` comparison artifact under `registry/miso/annual/comparisons/round_comparison_v1/`
- daily DA cache manifest and round-aware cutoff plumbing

Still pending:

- normalized `spec.json` / `metrics.json` backfill for existing promoted annual entries
- release manifests for `7.0b` and `7.1b`
- research/production script split
- migration of active production code into the new package layout with compatibility shims

---

## 1. Why this migration is needed

The current repo shape reflects research history, not the product we are actually shipping.

Current failure modes:

- champion status lives partly in docs, partly in memory, partly in registry
- model identity is ambiguous (`v0`, `v0c`, `Opt3`, `bucket_6_20`, `7.0b`, `7.1b`)
- metrics are mixed across native, overlap, and deployment views
- universe definitions are implicit in scripts instead of named, versioned objects
- feature sets are embedded in scripts instead of reusable recipes
- external benchmarks like `V4.4` are treated too much like internal models
- production code and exploratory research are mixed in the same surface area
- a second worktree can easily miss recent progress because there is no single canonical promotion path

This is manageable for one MISO-only repo, but it will break down once `PJM annual` enters the same codebase.

---

## 2. Target operating model

The repo should be organized around 6 first-class objects.

### 2.1 Universe

A named candidate population with explicit inclusion rules.

Examples:

- `miso_annual_branch_active_v1`
- `miso_annual_nb_hist12_v1`
- `miso_annual_v44_published_overlap_v1`

Every model and every comparison must name its universe explicitly.

### 2.2 FeatureRecipe

A named, versioned recipe describing:

- source tables
- allowed data horizon
- transforms
- fill policy
- feature eligibility rules

Examples:

- `v0c_core_v1`
- `nb_top2_density_v1`
- `annual_bucket_features_v1`

### 2.3 LabelRecipe

A named, versioned target definition.

Examples:

- `annual_bind_tiered_v1`
- `annual_bucket_6_20_v1`
- `annual_danger_binary_10k_v1`

### 2.4 ModelSpec

A model is one specific combination of:

- universe
- feature recipe
- label recipe
- objective
- training window rule
- ctype scope
- round scope

Examples:

- `miso_annual_v0c_formula_v1`
- `miso_annual_nb_opt3_v1`
- `miso_annual_bucket_6_20_v1`

### 2.5 PolicySpec

A policy converts model scores into final picks.

Examples:

- `annual_rank_only_v1`
- `annual_r30_nb_v1`
- `annual_r50_nb_v1`
- `annual_dynamic_tau_v1`

### 2.6 ReleaseSpec

A release is what actually gets published.

Examples:

- `miso_annual_7_0b`
- `miso_annual_7_1b`

Each release must point to exact model/policy code, signal names, supported slices, and validation gates.

---

## 3. Core contracts to standardize

### 3.1 Model contract

Every model must have:

- `onpeak` and `offpeak` explicitly represented
- annual `market_round` explicitly represented
- named universe
- named feature recipe
- named label recipe
- named training window rule
- named objective
- reproducible artifact path

If a model is not full-universe, that must be explicit in the model spec.

For annual products, the canonical base grain is:

- `(planning_year, aq_quarter, ctype, market_round)`

All annual models, benchmarks, and policies must be specifiable at that grain.

Examples:

- general ranker on full branch universe
- NB specialist on dormant-only universe
- benchmark adapter on published external universe

### 3.2 Registry contract

Every run directory must have the same required files:

- `spec.json`
- `metrics.json`
- `summary.md`
- `artifacts/`

Optional:

- `predictions.parquet`
- `feature_importance.json`
- `case_study.json`

`spec.json` must always include:

- `market`
- `product`
- `ctype`
- `market_round`
- `planning_years_train`
- `planning_years_eval`
- `universe_id`
- `feature_recipe_id`
- `label_recipe_id`
- `objective`
- `training_window_rule`
- `policy_id`
- `benchmark_id` if applicable
- `code_version`
- `data_version`

### 3.3 Metric contract

Do not mix metric views.

Every comparison must declare one of these views:

1. `native_topk`
2. `overlap_topk`
3. `deployment_policy`
4. `coverage`

Standard metrics to report:

- `SP@50/100/200/400`
- `Binders@50/100/200/400`
- `NB_SP@50/100/200/400`
- `NB_Binders@50/100/200/400`
- coverage metrics
- year × ctype breakdown
- standardized top true binder hit tables
- standardized case study output

For annual products, those metrics must first exist at:

- `(planning_year, aq_quarter, ctype, market_round)`

Any higher-level summary must explicitly declare how it aggregates that base grain.

Forbidden practices:

- mixing native and overlap rankings in the same headline
- presenting projected-to-our-universe results as native standalone results
- using ambiguous “absolute rank” without naming the candidate set
- treating external benchmarks as internal model families

### 3.4 Feature recipe contract

Features must be defined as recipes, not script-local column lists.

Each recipe must state:

- exact source columns
- transformations
- fill rules
- ex-ante validity horizon
- whether the recipe is valid for full-universe, NB-only, or both

Promotion rule:

- no model may be promoted, declared champion, or attached to a release without a bound feature recipe ID

Example:

- `Bucket_6_20` must be referred to as:
  - model spec: `miso_annual_bucket_6_20_v1`
  - feature recipe: `miso_annual_bucket_features_v1`

This prevents ambiguous cases where the same model name is rebuilt with different columns or fill rules.

### 3.5 Data cache contract

Data caches are versioned run inputs, not invisible local state.

This applies at minimum to:

- `data/nb_cache/`
- `data/realized_da/`
- any collapsed/universe cache used to build model tables

Each cache family must have a manifest that records:

- cache name
- schema version
- covered months / planning years / ctypes
- build timestamp
- source data snapshot
- code commit used to build it

If the cache is annual and depends on round-specific auction-side data, the manifest must also record:

- `market_round`
- whether the cache is round-sensitive or legacy-R1-only

Required rule:

- no publish or model run should rely on an unstated cache dependency

This is especially important because the `BF_FLOOR_MONTH` issue was operationally a cache coverage problem, not just a code/config problem.

### 3.6 Rank direction contract

Any signal or benchmark artifact with a rank-like column must declare its sort direction explicitly.

Examples:

- lower is better
- higher is better
- score, not rank

This belongs either in:

- the universe catalog
- the benchmark adapter spec
- or the model spec

The goal is to prevent bugs caused by assuming the wrong direction for external ranks or published benchmark fields.

### 3.7 Release contract

Each release must define:

- release name
- output signal name/path
- scoring model(s)
- policy
- supported planning years / quarters / ctypes
- validation gates
- known exclusions

Example:

- `7.0b` = `v0c` publish path + supplement matching + zero-SF filter
- `7.1b` = successor release, with exact changes stated explicitly

---

## 4. Separate models, policies, and benchmarks

This is the minimum conceptual split the repo needs.

### 4.1 Internal production models

- `v0c` or its successor general ranker
- NB specialist model(s)

### 4.2 Policies

- rank-only
- `R30`
- `R50`
- dynamic threshold policy

### 4.3 External benchmarks

- `V4.4` published signal

`V4.4` should be modeled as a benchmark adapter, not as a first-class internal model family.

---

## 5. Proposed target repo layout

```text
ml/
  core/
    registry.py
    metrics.py
    policies.py
    release_manifest.py
    model_api.py
  markets/
    miso/
      config.py
      bridge.py
      data_loader.py
      ground_truth.py
      history_features.py
      signal_publisher.py
    pjm/
      config.py
      bridge.py
      data_loader.py
      ground_truth.py
      history_features.py
      signal_publisher.py
  products/
    annual/
      universes/
      feature_recipes/
      label_recipes/
      comparisons/
  models/
    v0c/
    nb/
    bucket/
  benchmarks/
    v44_published.py

scripts/
  releases/
  training/
  evaluation/
  research/

registry/
  miso/
    annual/
  pjm/
    annual/

releases/
  miso/
    annual/
      7.0b/
      7.1b/
  pjm/
    annual/

docs/
  contracts/
  releases/
  migration/
  archive/

research/
  miso/
    annual/
      nb/
      bucket/
      ablations/
      comparisons/
```

Notes:

- `core` is market-agnostic.
- `markets/*` own loaders, bridge/entity resolution, GT construction, publishing adapters.
- `products/annual` owns universe and comparison contracts.
- `research/` is not part of the production surface.

---

## 6. Naming rules

Stop mixing research aliases with product identities.

Use both, but keep them separate.

### 6.1 Research alias

Examples:

- `nb_v3/+top2`
- `bucket_6_20`
- `danger_binary_10k`

### 6.2 Production identity

Examples:

- `miso_annual_nb_opt3_v1`
- `miso_annual_bucket_6_20_v1`
- `miso_annual_v0c_formula_v1`

### 6.3 Release identity

Examples:

- `miso_annual_7_0b`
- `miso_annual_7_1b`

Research names are allowed in `research/` and reports. Production names are required in registry and release manifests.

---

## 7. Promotion workflow

There must be one canonical state flow:

1. `research_run`
2. `candidate_model`
3. `champion_model`
4. `release_candidate`
5. `published_release`

Promotion requires:

- required registry files exist
- metric contract satisfied
- direct comparison against current champion
- deployment-view evaluation
- release compatibility
- validation gates pass

Champion status must be machine-readable, not doc-only.

Required files:

- `registry/.../champion.json` for model promotion
- `releases/.../manifest.json` for publish promotion

---

## 8. Standard case-study contract

Case studies should stop being bespoke.

Every model comparison report should include:

- top true overall binders hit table
- top true NB binders hit table
- top missed binders table
- one overlap-only heavy-binder rank table
- one deployment-policy hit table

Each table must state the ranking universe explicitly:

- `native_full`
- `overlap_shared`
- `deployment_shortlist`

Never label overlap-only ranks as generic “absolute rank”.

---

## 9. Release management changes

Releases need their own manifest layer.

Each release directory should contain:

- `manifest.json`
- `validation.json`
- `notes.md`

`manifest.json` should include:

- release name
- signal path
- source model ids
- source policy id
- supported slices
- output schema version
- code commit
- input data snapshot

This is what makes another worktree able to “see” the current state without depending on informal context.

---

## 10. Migration phases

### Phase 0: Freeze current state

Goal: stop further entropy while migration is happening.

Actions:

- freeze current release semantics for `7.0b`
- freeze current candidate semantics for `7.1b`
- mark exploratory scripts as research-only in docs
- identify current champions explicitly, even if temporary

Deliverables:

- `docs/contracts/current-state.md`
- temporary `champion.json` files for current promoted models

### Phase 1: Define contracts

Goal: lock naming and schema before moving code.

Actions:

- finalize model contract
- finalize metric contract
- finalize feature recipe contract
- finalize release manifest contract
- define universe IDs for existing annual workflows

Deliverables:

- `docs/contracts/model-contract.md`
- `docs/contracts/metric-contract.md`
- `docs/contracts/feature-recipe-contract.md`
- `docs/contracts/release-contract.md`
- `docs/contracts/universe-catalog.md`

### Phase 2: Split research from the production surface

Goal: reduce immediate confusion without breaking the active release path.

Actions:

- move ablations and ad hoc scripts to `research/`
- keep only release/training/evaluation entrypoints in `scripts/`
- keep only reusable production code in `ml/`
- preserve compatibility aliases or wrappers while scripts are moved

Deliverables:

- reduced production script surface
- research alias index mapping old script names to new locations

### Phase 3: Standardize registry

Goal: make all promoted runs discoverable and reproducible.

Actions:

- create standard `spec.json` / `metrics.json` / `summary.md` shape
- backfill current promoted runs into the standard schema
- separate benchmark runs from internal model runs

Deliverables:

- normalized registry layout under `registry/miso/annual/`
- migrated entries for:
  - `v0c`
  - current NB champion candidate
  - current bucket candidate
- `V4.4` benchmark adapter

### Phase 4: Introduce explicit product modules

Goal: organize by product instead of phase.

Actions:

- create `products/annual/universes`
- create `products/annual/feature_recipes`
- create `products/annual/label_recipes`
- create `products/annual/comparisons`

Deliverables:

- reusable universe definitions
- reusable feature recipes
- reusable label recipes
- reusable comparison runners

### Phase 5: Split market-specific code

Goal: prepare for PJM without duplicating chaos.

Actions:

- move MISO-specific loaders and bridge logic into `markets/miso`
- create matching empty interfaces in `markets/pjm`
- keep common logic in `core`
- preserve import compatibility during migration

Deliverables:

- clear MISO/PJM market boundary
- PJM skeleton ready for implementation
- temporary compatibility shims or one-shot atomic move plan documented

### Phase 6: Add release manifests

Goal: make releases first-class.

Actions:

- create release manifests for `7.0b`
- create release candidate manifest for `7.1b`
- record supported slice coverage and known blockers

Deliverables:

- `releases/miso/annual/7.0b/manifest.json`
- `releases/miso/annual/7.1b/manifest.json`

### Phase 7: Tighten tests

Goal: ensure publication invariants are enforced.

Required tests:

- no duplicate published constraint ids
- no duplicate `constraint_id|shadow_sign|spice` indices
- branch cap enforcement
- no published constraint missing SF
- grouped dedup invariants
- onpeak/offpeak parity of publish path
- release manifest validation

---

## 11. Immediate priorities

These should happen before PJM work starts.

1. Freeze one canonical metric contract.
2. Make universe definitions explicit and versioned.
3. Separate benchmark adapters from internal models.
4. Add release manifests for `7.0b` and `7.1b`.
5. Standardize the registry schema.
6. Move research scripts out of the production surface.
7. Create the market boundary for MISO vs PJM.

---

## 12. Backward compatibility strategy

There are many existing imports from:

- `ml/config.py`
- `ml/features.py`
- `ml/phase6/...`

Moving directly to `ml/markets/miso/...` will break scripts unless the migration is done carefully.

Allowed strategies:

### Option A: Atomic move

- perform the market split in one commit
- update all imports in the same commit
- run the full test and script verification sweep before merge

### Option B: Compatibility shims

- introduce new modules under `ml/markets/miso/`
- keep old import paths temporarily as thin re-export wrappers
- migrate callers incrementally
- remove wrappers only after the repo is fully cut over

Recommendation:

- use compatibility shims first unless there is a strong reason to do an atomic move

This lowers migration risk and avoids breaking active release and research scripts mid-transition.

---

## 13. Non-goals

This migration should not try to do all of these at once:

- rewrite every historical experiment into the new system
- fully generalize every model runner before `7.1b`
- create a large shared abstraction layer before MISO/PJM interfaces are stable
- rename every legacy alias immediately

The goal is to stabilize the product surface first, then migrate legacy research incrementally.

---

## 14. Cache and data readiness checks

Before any release or champion promotion, the run should verify:

- required cache families are present
- required month coverage exists
- cache manifest versions match the run spec
- no hidden local cache dependency is being relied on

At minimum, release manifests should reference:

- realized DA cache coverage
- NB/model cache coverage
- source data snapshot or extraction date

---

## 15. Success criteria

The migration is successful when:

- another worktree can identify current champions from registry + release manifests alone
- every promoted model has an explicit universe, feature recipe, label recipe, and metric view
- every promoted model and benchmark declares rank direction explicitly
- every promoted run declares cache inputs explicitly
- every release is reproducible from a manifest
- production entrypoints are easy to identify
- research artifacts no longer masquerade as production artifacts
- PJM can be added as a new market adapter instead of a second copy of MISO chaos

---

## 16. Recommended first implementation batch

Initial recommended batch and current status:

1. Create contract docs and a universe catalog. Completed.
2. Add cache manifests and rank-direction metadata for current active models/benchmarks. Partial.
3. Add release manifests for `7.0b` and `7.1b`. Not started.
4. Move exploratory NB/bucket scripts into `research/`. Not started.
5. Normalize registry entries for current MISO annual champions/candidates. Not started.
6. Create `ml/core`, `ml/markets/miso`, and empty `ml/markets/pjm`, with compatibility shims if needed. Completed as a non-breaking skeleton pass.

The first normalized annual model entries should explicitly bind feature recipes for:

- `v0c`
- current NB candidate
- `Bucket_6_20`

This gives immediate organizational benefit without blocking current annual release work.
