# PJM Annual Signal — Concrete Implementation Plan

**Date**: 2026-03-25
**Status**: Execution plan after Phase 0 and Phase 1
**Goal**: build the PJM annual-signal pipeline with the same repo shape, experiment discipline, and artifact structure as `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2`, while keeping all market behavior PJM-native and treating released `V4.6` as the benchmark baseline to beat

---

## 1. Fixed Inputs

This plan assumes the following are already frozen and must not be re-opened during Phase 2:

- Base grain contract: `docs/phase0-pjm-base-grain-contract.md`
- Data-source contract: `docs/phase1-data-source-contract.md`
- Coverage proof and known anomalies: `docs/phase1-coverage-report.md`
- Principal handoff and sequence: `human-input/2026-03-25-pjm-handoff.md`
- Reconstruction charter: `docs/pjm-reconstruction-charter.md`

Operational facts already proved:

- Publish grain: `(planning_year, market_round, class_type)` with annual `period_type=a`
- Internal GT bridge layer: quarterly `aq1-aq4`, aggregated back to annual
- Current annual data is available under the audited canonical root in the Phase 1 contract; do not assume older dual-root notes are still valid
- GT and model-universe coverage must remain separate metrics
- Benchmark rank direction is low-rank-is-best
- 2022-06 exists and is usable where coverage exists; do not globally exclude it
- 2025-06 is holdout and incomplete
- exact V4.6 flow/deviation reconstruction from density is not the objective
- baseline source policy is reproducible density + DA history only
- the real baseline is released `V4.6`, not an internal challenger label

---

## 2. Target Repo Shape

The target structure should match the MISO repo shape. Generic orchestration stays generic; PJM behavior stays under `ml/markets/pjm`.

```text
research-annual-signal-pjm/
├── ml/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── calendars.py
│   │   ├── metrics.py
│   │   ├── registry.py
│   │   └── specs.py
│   ├── products/
│   │   └── annual/
│   │       ├── __init__.py
│   │       └── output_schema.py
│   ├── markets/
│   │   └── pjm/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── bridge.py
│   │       ├── realized_da.py
│   │       ├── ground_truth.py
│   │       ├── data_loader.py
│   │       ├── features.py
│   │       ├── history_features.py
│   │       ├── scoring.py
│   │       └── signal_publisher.py
│   ├── __init__.py
│   ├── config.py
│   ├── evaluate.py
│   ├── features.py
│   ├── registry.py
│   └── train.py
├── data/
│   ├── collapsed/
│   ├── realized_da/
│   └── benchmark_cache/
├── registry/
│   └── pjm/
│       └── annual/
├── releases/
│   └── pjm/
│       └── annual/
├── scripts/
└── docs/
```

Rules:

- Reuse generic modules from the MISO repo where behavior is truly market-agnostic.
- Do not place PJM business rules in top-level generic files if they belong under `ml/markets/pjm`.
- Keep one source of truth per responsibility:
  - bridge mapping in `ml/markets/pjm/bridge.py`
  - DA loading in `ml/markets/pjm/realized_da.py`
  - GT mapping in `ml/markets/pjm/ground_truth.py`
  - branch-universe construction in `ml/markets/pjm/data_loader.py`

---

## 3. Execution Model

The repo should mirror MISO's separation of concerns:

- `ml/core/`: registry schema, metrics schema, calendar helpers
- `ml/markets/pjm/`: all PJM-native loading, mapping, collapse, history, scoring, publishing
- `ml/products/annual/`: annual output schema and release contract
- top-level `ml/*.py`: generic train/evaluate/registry wrappers that delegate to market modules

The project must be branch-level end to end:

- SPICE constraints map to `branch_name` for features
- DA constraints map to `branch_name` for labels
- many constraints collapse onto one branch
- model rows are branch rows, not constraint rows

PJM-specific differences live inside the mapping and loading ladder, not in the model grain.

Additional modeling rule:

- the first reconstruction baseline must be buildable from reproducible density + DA inputs only
- do not design the initial pipeline around hidden `flow_memo`-style artifacts

---

## 4. What Should Stay the Same vs Change

The top-level repo structure should stay MISO-like. The internal PJM API should not blindly copy MISO's quarter-first assumptions.

### 4.1 Keep the same

- `ml/core`, `ml/markets/{market}`, `ml/products/annual` separation
- config-driven splits and registry artifacts
- one source of truth for bridge / GT / universe modules
- branch-level modeling
- separate coverage and ranking metrics
- benchmark and model comparison through normalized `spec.json` / `metrics.json`

### 4.2 Change for PJM

- Headline evaluation grain is annual publish cell:
  `(planning_year, market_round, class_type)`
- Quarter is an internal bridge / GT resolution layer, not the primary public result surface
- `class_type` set is PJM-native:
  `onpeak`, `dailyoffpeak`, `wkndonpeak`
- `wkndonpeak` is not a direct bridge partition and should be handled in DA/filtering and publication logic, not copied from MISO ctype assumptions
- PJM root resolution is mandatory in loaders; MISO-style single-root partition scans are wrong
- PJM DA does not need the same monthly/daily cache design as MISO
- PJM history features and dangerous/NB thresholds must be calibrated independently
- baseline signal design is density-proxy-first, not V4.6 formula-clone-first

### 4.3 Practical rule

Replicate MISO's package boundaries and artifact discipline.
Do not replicate MISO's internal evaluation axis or business rules where PJM data shape differs.

---

## 5. Default Experimental Policy

This plan adopts the same config-driven experiment structure as MISO. Do not choose training years ad hoc inside notebooks or worker messages.

### 5.1 PY policy

- `holdout_pys = ["2025-06"]`
- `core_train_pys = ["2019-06", "2020-06", "2022-06", "2023-06", "2024-06"]`
- `optional_train_pys = ["2021-06"]`

Use `2021-06` only in explicit sensitivity runs because its behavior is weaker and should be validated separately.
Use `2022-06` only where the audited round coverage exists.

### 5.2 Build sequence

Expand in this order:

1. `2024-06 / R2 / onpeak`
2. `2024-06 / R2 / dailyoffpeak`
3. `2024-06 / R2 / wkndonpeak`
4. all core training PYs for `R2 / onpeak`
5. all core training PYs for `R2 / all class types`
6. full rounds `R1-R4` on core training PYs
7. holdout `2025-06`

This keeps one checkpoint slice as the first hard gate, but the expansion path is concrete.

### 5.3 Split contract

`ml/markets/pjm/config.py` should own the official split dictionary, analogous to MISO `EVAL_SPLITS`.

Recommended initial contract:

```python
EVAL_SPLITS = {
    "2023-06": {
        "train_pys": ["2019-06", "2020-06"],
        "eval_pys": ["2023-06"],
        "split": "dev",
    },
    "2024-06": {
        "train_pys": ["2019-06", "2020-06", "2022-06", "2023-06"],
        "eval_pys": ["2024-06"],
        "split": "dev",
    },
    "2025-06": {
        "train_pys": ["2019-06", "2020-06", "2022-06", "2023-06", "2024-06"],
        "eval_pys": ["2025-06"],
        "split": "holdout",
    },
}
```

Secondary sensitivity split:

- `2024-06+2021`: same as above, but include `2021-06` in `train_pys`

Do not hard-code exclusions that contradict the current audited data contract.

---

## 6. Module Contracts

### 6.1 `ml/markets/pjm/config.py`

Owner of:

- all PJM paths
- root-resolution logic per `(planning_year, market_round)`
- `PLANNING_YEARS`, `AQ_QUARTERS`, `CLASS_TYPES`, `VALID_ROUNDS`
- quarter-to-market-month mapping
- threshold constants
- holdout / dev split contract
- checkpoint slice constants

Must expose:

- `resolve_root(py: str, market_round: int) -> str`
- `get_market_months(planning_year: str, aq_quarter: str) -> list[str]`
- `CLASS_TYPES = ["onpeak", "dailyoffpeak", "wkndonpeak"]`
- `BRIDGE_CLASS_TYPES = ["onpeak", "dailyoffpeak"]`
- `CHECKPOINT = {"planning_year": "2024-06", "market_round": 2, "class_type": "onpeak"}`

Notes:

- `wkndonpeak` is publish/eval-visible even though bridge data does not carry it directly.
- `resolve_root` must be used by density, bridge, limit, and SF loaders.

### 6.2 `ml/markets/pjm/bridge.py`

Single source of truth for constraint-to-branch mapping.

Must implement:

- annual bridge load by `(PY, aq_quarter, round)`
- monthly `f0` bridge fallback
- ambiguity detection and drop policy
- mapping provenance diagnostics

Required mapping ladder:

1. annual bridge exact `constraint_id -> branch_name`
2. monthly `f0` bridge fallback where available
3. approved PJM monitored-line fallback
4. optional future `cons_name_mapping/pjm` layer only if proven time-safe

Important:

- step 3 is a fallback, not the primary semantics
- ambiguity policy must be explicit and shared by GT and feature loaders
- all mapping functions return both mapped data and diagnostics

### 6.3 `ml/markets/pjm/realized_da.py`

Owner of realized DA load and class-type filtering.

Must implement:

- load DA for one quarter or one annual cell
- normalize `ACTUAL -> BASE`
- construct:
  - `constraint_id`
  - `monitored_line`
  - `realized_sp`
- apply class-type hour filtering
- aggregate hourly DA to CID-level totals

Required behavior:

- no monthly cache initially; Phase 1 proved load cost is already low
- year-partition scan only
- all-negative PJM DA convention handled explicitly
- replacement logic treated as irrelevant for constraint-level DA unless new evidence appears

Open point to freeze during implementation:

- exact `onpeak`, `dailyoffpeak`, `wkndonpeak` hour definitions in code, with one explicit test fixture

### 6.4 `ml/markets/pjm/ground_truth.py`

Owner of DA-to-branch GT construction.

Must implement:

- per-quarter GT build
- full fallback ladder through `bridge.py`
- branch-level aggregation
- annual aggregation from aq1-aq4
- positive-binding filtering
- label-tier assignment
- detailed coverage diagnostics

Outputs:

- quarter GT parquet/dataframe
- annual GT parquet/dataframe
- diagnostics dict with:
  - `total_da_sp`
  - `annual_mapped_sp`
  - `monthly_recovered_sp`
  - `fallback_recovered_sp`
  - `still_unmapped_sp`
  - `ambiguous_sp`

Rule:

- GT coverage metrics must always be reported against total DA SP, not filtered-universe SP

PJM adjustment:

- Public API should expose annual-cell GT builders first, with quarter builders as internal helpers.
- The MISO quarter-first UX should not be copied into the top-level PJM experiment interface.

### 6.5 `ml/markets/pjm/data_loader.py`

Owner of branch-universe construction and branch-level feature collapse.

Must implement:

- load density by `(PY, aq_quarter, round)` using root resolution
- load limits by `(PY, aq_quarter, round)`
- load SF by `(PY, aq_quarter)` if needed by publishing / later features
- compute `right_tail_max`
- apply CID-level active/universe filter
- map all CIDs to branches
- collapse to branch-level density and limit features
- cache both:
  - branch-level collapsed features
  - CID-to-branch mapping

Initial feature contract:

- selected density bins
- branch max/min over CID-level collapsed bins
- limit min/mean/max/std
- count of CIDs and active CIDs

Threshold policy:

- start with the current Phase 1 threshold used in coverage work
- add threshold calibration as a separate artifact later
- do not bake MISO's threshold in as an unexplained constant

PJM adjustment:

- The public entry point should build annual branch features per `(planning_year, market_round, class_type)`.
- Internal quarter-specific collapse artifacts are allowed, but the training-table interface should be annual-first.

### 6.6 `ml/markets/pjm/features.py`

Owner of branch training-table assembly.

Must implement the same join discipline as MISO:

- universe branch table as left side
- GT joined in
- historical features joined in
- non-binding branches added after left join

This is where branch rows become model rows.

### 6.7 `ml/markets/pjm/history_features.py`

Phase 4b module. Do not block Phase 2/3 on it.

Planned responsibilities:

- DA rank / BF-like history features
- recency windows consistent with PJM round timing
- no leakage across auction cutoff

Baseline v0a may skip this module; baseline v0b/v0c should add it.

PJM adjustment:

- Do not assume MISO BF windows or NB cohort definitions survive unchanged.
- History features should start minimal:
  DA rank / historical branch SP / binding frequency by PJM class type.

### 6.8 `ml/markets/pjm/scoring.py`

Phase 5/6 module.

Must convert model outputs into branch scores consumable by:

- benchmark comparison
- publication assembly
- release artifacts

### 6.9 `ml/markets/pjm/signal_publisher.py`

Phase 5+ module.

Owner of final publish assembly at annual grain:

- `(planning_year, market_round, class_type)`
- V4.6-compatible comparison surface
- output schema compliance

The internal quarterly GT layer must not leak into the published contract.

---

## 7. Feature Policy

PJM should not start with the full MISO feature inventory.

### 7.1 Baseline feature set for v0a

Use only features already supported by the audited PJM inputs:

- density right-tail bins and selected density bins
- constraint limit statistics
- count of total / active CIDs per branch

Interpretation:

- these are the first reproducible substitutes for the V4.6 flow/deviation component
- they are not required to match V4.6 feature columns one-for-one

This baseline is intentionally rebuildable and low-risk.

### 7.2 Add in v0b

- branch-level DA history rank
- historical branch SP aggregates by class type
- simple binding-frequency features

This is the first density + DA reconstruction baseline.

### 7.3 Delay until later

- pnode-sensitive features
- complex NB specialist features
- any feature that depends on unproven replacement or leak-prone helper logic
- anything that requires unreproducible simulator outputs

### 7.4 Feature contract difference from MISO

MISO's feature surface was built around its own BF conventions, ctype semantics, and cache structure.
PJM should keep the same feature assembly pattern, but not the same feature list.

---

## 8. Generic Modules to Port Early

These should be copied or ported from the MISO repo with minimal edits because they are structural, not market-specific:

- `ml/core/specs.py`
- `ml/core/metrics.py`
- `ml/core/registry.py`
- `ml/core/calendars.py`
- `ml/evaluate.py`
- `ml/train.py`
- `ml/registry.py`
- `ml/products/annual/output_schema.py`

Porting rule:

- preserve file shape and registry contract
- replace MISO imports with PJM-compatible entry points
- keep benchmark-agnostic metrics generic

---

## 9. Caches and Artifacts

Use the same style as MISO.

### 9.1 Data caches

Recommended directories:

- `data/collapsed/pjm/`
- `data/realized_da/pjm/`
- `data/benchmark_cache/pjm/`

Initial cache policy:

- collapsed branch features: yes
- CID mapping cache: yes
- DA cache: optional interface only, disabled by default
- benchmark cache: yes if benchmark load becomes repetitive

### 9.2 Registry

Use MISO-style normalized registry layout:

```text
registry/
  pjm/
    annual/
      models/
      benchmarks/
      policies/
```

Every evaluated model version writes:

- `spec.json`
- `metrics.json`
- optional `analysis.json`

### 9.3 Releases

Use:

```text
releases/pjm/annual/{version_id}/
```

Release contents should eventually match the MISO discipline:

- manifest
- publish parquet(s)
- benchmark comparison summary
- release notes

---

## 10. Evaluation Policy

PJM evaluation should preserve MISO's honesty, but the reporting surface should be slightly different.

### 10.1 Same as MISO

- keep `Abs_SP@K` as the honest headline metric
- keep `VC@K` as a conditional in-universe metric
- keep dangerous-branch and dormant/NB diagnostics
- keep overlap-only reranking as a diagnostic view

### 10.2 Different for PJM

- headline results are annual cell metrics, not quarter headlines
- quarter results are diagnostics supporting GT construction and anomaly explanation
- benchmark comparison must use V4.6 annual publish cells directly
- any quarter aggregation must be explicit because aq4 behavior can differ materially

### 10.3 Baseline metric set

For each annual cell:

- `SP_Captured@200`, `SP_Captured@400`
- `Abs_SP@200`, `Abs_SP@400`
- `VC@200`, `VC@400`
- `Recall@200`, `Recall@400`
- `Binders@200`, `Binders@400`
- `Dang_Recall@200`, `Dang_Recall@400`
- `NB12_SP@200`, `NB12_SP@400`
- `gt_mapping_coverage`
- `model_universe_coverage`
- `combined_capture_ceiling`

Baseline benchmark rule:

- every baseline candidate is compared directly against V4.6 on the same annual cell
- the first optimization target is not rank similarity to V4.6
- the first optimization target is higher annual-cell `Abs_SP@200/400`

### 10.4 Deferred metrics

Keep these in diagnostics or later-phase reports, not as baseline gate metrics:

- `NDCG`
- `Spearman`
- full overlap-only binder rank studies
- reserved-slot fill-rate metrics

---

## 11. Hard Gates by Phase

### Gate A: repo bootstrap complete

Must exist:

- `ml/core/*`
- `ml/markets/pjm/config.py`
- package imports resolve
- one smoke command loads config and quarter month mapping

### Gate B: data contracts executable

For checkpoint slice `2024-06 / aq1 / R2 / onpeak`:

- density loads from correct root
- bridge loads quarter partition
- DA loads and filters correctly
- one GT dataframe builds
- one collapsed branch dataframe builds

### Gate C: checkpoint slice parity

For `2024-06 / R2 / onpeak`:

- GT annual recovery is approximately Phase 1 result
- aq4 weakness is reproduced, not hidden
- universe coverage is approximately Phase 1 result
- benchmark path loads and low-rank-is-best is preserved

If these are not true, do not move to model training.

### Gate D: annual branch-table assembly

For `2024-06 / R2 / onpeak`:

- branch universe
- GT labels
- feature columns
- benchmark overlap

must all exist in one reproducible training/eval table.

### Gate E: baseline model

First baseline is intentionally simple:

- LightGBM
- class-specific
- round-aware
- density + limits only for v0a
- annual-cell training / eval output

Only after v0a is stable:

- add history features for v0b/v0c

PJM-specific interpretation:

- `v0a` = density-only reproducible baseline
- `v0b` = density + DA-history reproducible baseline
- success means beating or matching V4.6 on at least some checkpoint metrics, not cloning its internal formula

### Gate F: full-grid expansion

Only after checkpoint slice is stable:

- expand across core training PYs
- expand across class types
- expand across rounds
- run holdout last

---

## 12. Concrete Build Order

### Step 1: bootstrap repo structure

Create:

- `ml/`
- `ml/core/`
- `ml/markets/`
- `ml/markets/pjm/`
- `ml/products/annual/`
- `data/`
- `registry/`
- `releases/`

Port generic files from MISO first.

### Step 2: implement `ml/markets/pjm/config.py`

Freeze:

- paths
- root resolution
- PY policy
- class types
- round list
- quarter month map
- threshold
- checkpoint constants
- experiment splits

### Step 3: implement `ml/markets/pjm/bridge.py`

Deliver:

- annual bridge load
- `f0` monthly fallback load
- ambiguity handling
- provenance counters
- one checkpoint-slice smoke test

### Step 4: implement `ml/markets/pjm/realized_da.py`

Deliver:

- quarter DA load
- class hour filters
- normalized `constraint_id`
- `monitored_line` extraction
- branch-mapping-ready CID totals

### Step 5: implement `ml/markets/pjm/ground_truth.py`

Deliver:

- quarter GT build
- annual GT build
- full diagnostics output
- parity check against Phase 1 coverage report

### Step 6: implement `ml/markets/pjm/data_loader.py`

Deliver:

- density/limit root resolution
- branch collapse
- active universe filter
- cache outputs
- parity check against Phase 1 universe report

### Step 7: implement generic feature assembly

Deliver:

- market-specific `ml/markets/pjm/features.py`
- generic `ml/features.py` glue if needed
- one training table for checkpoint slice

### Step 8: baseline v0a

Deliver:

- one LightGBM training run
- one eval table
- one benchmark comparison against V4.6
- one registry entry

Before that, add a pre-model sweep:

- simple density-only score candidates
- direct benchmark comparison against V4.6 on checkpoint slice
- pick the best density-proxy baseline before training the first model

### Step 9: history features

Deliver:

- leakage-safe round-aware history windows
- baseline v0b/v0c
- compare against v0a

### Step 10: release path

Deliver:

- annual publish assembly
- output schema validation
- release artifact layout

---

## 13. Tests and Smoke Checks

At minimum, add one reproducible smoke script or test for each:

- `config.resolve_root("2024-06", 1) == newer`
- `config.resolve_root("2024-06", 2) == legacy`
- `bridge` drops ambiguous mappings deterministically
- `realized_da` reproduces all-negative DA convention
- `ground_truth` reproduces checkpoint annual recovery within tolerance
- `data_loader` reproduces checkpoint universe coverage within tolerance
- benchmark loader confirms rank direction on a known cell

Recommended tolerances:

- coverage parity: within `+/- 1.0pp`
- branch counts: exact or explain any drift
- benchmark row counts: exact

---

## 14. Explicit Non-Goals

Do not do these before baseline parity:

- hybrid model
- specialist policy
- constraint-level final policy
- aggressive fuzzy matching beyond approved fallback ladder
- tuning-heavy modeling
- full-grid training before checkpoint slice parity

---

## 15. Immediate Next Action

Start Phase 2 with this exact scope:

1. bootstrap `ml/`, `data/`, `registry/`, `releases/`
2. port generic `ml/core` and top-level wrappers from MISO
3. implement `ml/markets/pjm/config.py`
4. implement `ml/markets/pjm/bridge.py`
5. implement `ml/markets/pjm/realized_da.py`
6. implement `ml/markets/pjm/ground_truth.py`
7. implement `ml/markets/pjm/data_loader.py`
8. prove parity on `2024-06 / R2 / onpeak`

That is the Phase 2 objective. Model training starts only after those eight items are complete.
