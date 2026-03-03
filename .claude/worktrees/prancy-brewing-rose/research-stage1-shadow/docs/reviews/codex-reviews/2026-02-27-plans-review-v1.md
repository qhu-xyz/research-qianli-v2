# Review: docs/plans (2026-02-27)

## Scope
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`
- `docs/plans/2026-02-26-verification-plan.md`
- `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`

## Findings

### 1. Critical: Implementation plan collapses reviewer states into `REVIEWING`, violating the documented state machine
- Evidence:
  - Implementation skeleton uses `WORKER_RUNNING -> REVIEWING -> ORCHESTRATOR_SYNTHESIZING` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:801`, `:805`).
  - Design state machine and run loop require separate `REVIEW_CLAUDE` and `REVIEW_CODEX` transitions (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:105`, `:106`, `:573`, `:576`).
  - Verification requires handoff mappings for both reviewer states (`docs/plans/2026-02-26-verification-plan.md:238`, `:246`, `:247`).
- Impact:
  - Breaks `STATE_TO_HANDOFF` and `get_expected_artifact()` contracts.
  - Breaks watchdog timeout naming and reviewer timeout handling.
  - Will fail verification tests that assert separate reviewer states.
- Required fix:
  - Update Task 17 skeleton/snippets to use explicit `REVIEW_CLAUDE` then `REVIEW_CODEX` transitions and artifacts.

### 2. High: End-of-iteration state contract is inconsistent (`IDLE` vs `ORCHESTRATOR_PLANNING`)
- Evidence:
  - Implementation Task 17 step 18 sets `ORCHESTRATOR_SYNTHESIZING -> IDLE` for `N < 3` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:784`, `:811`).
  - Design/Tier-2 integrity criteria expect final state `ORCHESTRATOR_PLANNING` after 1 iteration (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1306`, `:1309`; `docs/plans/2026-02-26-verification-plan.md:405`).
- Impact:
  - Integration tests and actual controller behavior cannot both be correct under current text.
- Required fix:
  - Pick one canonical post-iteration state for `N < 3` and align all three docs (design, verification, implementation).

### 3. High: v0 baseline source is inconsistent; implementation currently instructs smoke baseline for gate calibration
- Evidence:
  - Implementation Task 26 explicitly builds v0 with `SMOKE_TEST=true` and commits it as baseline (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:1011`, `:1016`, `:1038`).
  - Design defines v0 as retraining with source-repo-equivalent config before pipeline runs (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:644`, `:645`, `:648`).
  - Verification Part 1 still describes a different baseline interpretation (manual extraction framing) (`docs/plans/2026-02-26-verification-plan.md:22`, `:25`).
- Impact:
  - If smoke metrics are used to populate gate floors, gate thresholds become detached from real-data performance.
- Required fix:
  - Separate smoke-only baseline from real baseline explicitly.
  - Forbid `populate_v0_gates.py` against smoke artifacts (or mark smoke baselines and prevent promotion-floor initialization from them).
  - Reconcile baseline language between design and verification docs.

### 4. Medium: Task 2 bootstrap instructions conflict on `handoff/` tracking
- Evidence:
  - Task includes creating `handoff/.gitkeep` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:125`).
  - `.gitignore` then ignores entire `handoff/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:132`).
  - Commit command still includes `handoff/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:200`).
- Impact:
  - Bootstrap step is non-deterministic/confusing (`.gitkeep` may not be added unless forced).
- Required fix:
  - Either track `handoff/.gitkeep` via `!handoff/.gitkeep` in `.gitignore`, or remove `.gitkeep` from required committed artifacts.

## Open Questions
- For iterations 1-2, should the controller end in `IDLE` or immediately in `ORCHESTRATOR_PLANNING`?
- Is v0 for production gates allowed to be smoke-generated, or must it always be generated from real data path with Ray-enabled loading?
