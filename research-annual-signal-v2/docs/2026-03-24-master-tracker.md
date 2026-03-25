# Master Tracker: Annual Signal Release + Repo Reorganization

**Date**: 2026-03-24  
**Scope**: MISO annual release work, round-sensitive pipeline, repo reorganization, and next modeling steps

---

## 1. Overall status

### Release track: `7.1b` (`v0c`, round-sensitive)

Status: **release-candidate**

Completed:

- round-sensitive `v0c` feature pipeline
- daily DA cutoff integration
- matched-round `v0c` vs `V4.4` comparison
- normalized comparison registry entry
- `7.1b` manifest
- `R1/R2/R3 × onpeak/offpeak` publish smoke test
- publish hard-fails on missing SF
- explicit published output contract

Open:

- small artifact metadata cleanup
- rerun the publish smoke test once after the `constraint_limit` schema fix
- final go / no-go publish decision

### Repo reorganization track

Status: **in progress**

Completed:

- contracts
- universe catalog
- feature recipe bindings
- registry schema
- package skeletons under `ml/core`, `ml/markets`, `ml/products`

Open:

- migrate active production code from flat `ml/`
- split scripts into production / evaluation / research
- backfill older registry entries
- add compatibility shims as the new package layout becomes authoritative

### Model-development track

Status: **in progress**

Completed:

- `Bucket_6_20` champion confirmation
- NB top-tail diagnostics
- `V4.4` notebook reverse-engineering
- `V4.4`-like rebuild design from our own data

Open:

- round-aware `Bucket_6_20`
- specialist branch productionization
- `V4.4`-like internal rebuild implementation

---

## 2. Current decisions

- `7.1b` is a natural extension of `7.0b` for `v0c`
- formula remains the same
- the main change is round-sensitive inputs and daily-cutoff history
- annual base grain is:
  - `(planning_year, aq_quarter, class_type, market_round)`
- repo migration should use compatibility shims, not a large atomic move
- external worktrees are out of scope for repo cleanup and should not be touched

---

## 3. Completed work

### 3.1 Round-sensitive pipeline

- round calendar / cutoff helpers
- explicit `market_round` in core annual APIs
- round-aware loader plumbing
- round-aware cache keys for collapsed density and CID mapping
- daily DA cache with manifest
- partial-month history inclusion
- round-aware bridge use in history
- explicit `--market-round` in publish CLI

### 3.2 `7.1b` validation artifacts

- [manifest.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/releases/miso/annual/7.1b/manifest.json)
- [smoke_test.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/releases/miso/annual/7.1b/smoke_test.json)
- [spec.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/registry/miso/annual/comparisons/round_comparison_v1/spec.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/registry/miso/annual/comparisons/round_comparison_v1/metrics.json)
- [analysis.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/registry/miso/annual/comparisons/round_comparison_v1/analysis.json)
- [output-schema.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/output-schema.md)

### 3.3 Repo-organization groundwork

- [current-state.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/current-state.md)
- [universe-catalog.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/universe-catalog.md)
- [feature-recipes.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/feature-recipes.md)
- [registry-schema.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/registry-schema.md)
- skeleton packages in:
  - [ml/core](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/core)
  - [ml/markets/miso](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/markets/miso)
  - [ml/markets/pjm](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/markets/pjm)
  - [ml/products/annual](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/products/annual)

---

## 4. In-progress items

### 4.1 `7.1b` release finalization

Remaining:

- update round-comparison artifact wording / metadata so the 7-PY summary is precise
- make final publish decision

Notes:

- results are real
- be explicit that `504` is the raw metric-cell count and `84` is the aggregated head-to-head cell count

### 4.2 Repo reorganization

Current state:

- design exists
- contracts exist
- package skeletons exist
- active code has **not** moved yet

Meaning:

- the repo is in transition
- reorganization is real, but not complete

---

## 5. Blockers and known gaps

### 5.1 Not blockers for `7.1b`

- `Bucket_6_20` is still R1-only
- `data/nb_cache/` is still legacy
- specialist branch is not productionized
- full code migration is not done

### 5.2 Actual open gaps

- stale wording in some comparison-analysis outputs
- rerun the publish smoke test after the `constraint_limit` schema fix
- registry backfill for older promoted annual entries
- script surface still mixes production and research
- active production code still lives in flat `ml/`

---

## 6. Next actions

### Immediate

1. Clean the `7.1b` comparison artifact wording/metadata.
2. Make the final `7.1b` publish decision.

### After `7.1b`

1. Start the first low-risk migration batch:
   - `ml/bridge.py`
   - `ml/realized_da.py`
   - split `ml/config.py` into core calendar vs MISO config
2. Backfill standardized registry entries.
3. Split the script surface into:
   - `production`
   - `evaluation`
   - `research`
4. Migrate the remaining active MISO production code.

### Modeling after migration starts

1. round-aware `Bucket_6_20`
2. specialist branch productionization
3. `V4.4`-like rebuild from our data

---

## 7. Recommended sequence

1. finalize `7.1b` evidence
2. publish `7.1b` or explicitly defer it
3. run first migration batch
4. backfill registry
5. split script surface
6. migrate remaining production code
7. resume broader model-development work

---

## 8. Related docs

Primary planning docs:

- [2026-03-24-repo-migration-plan.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-repo-migration-plan.md)
- [2026-03-24-post-7.1b-next-steps.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-post-7.1b-next-steps.md)
- [2026-03-24-repo-reformulation-and-round-sensitive-annual-design.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-repo-reformulation-and-round-sensitive-annual-design.md)

Contracts:

- [current-state.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/current-state.md)
- [universe-catalog.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/universe-catalog.md)
- [feature-recipes.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/feature-recipes.md)
- [registry-schema.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/contracts/registry-schema.md)

Specialist / modeling handoff docs:

- [2026-03-24-round-modeling-handoff-note.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-round-modeling-handoff-note.md)
- [2026-03-24-v44-notebook-gap-analysis.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-v44-notebook-gap-analysis.md)
- [2026-03-24-v44-like-rebuild-from-our-data.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/2026-03-24-v44-like-rebuild-from-our-data.md)
