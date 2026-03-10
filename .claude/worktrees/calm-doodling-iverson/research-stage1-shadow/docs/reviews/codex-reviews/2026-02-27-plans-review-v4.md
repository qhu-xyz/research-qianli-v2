# Review: docs/plans (fourth pass, 2026-02-27)

## Scope
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v10)
- `docs/plans/2026-02-26-verification-plan.md`
- `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`

## Findings (new + still-blocking)

### 1. Critical (still): Implementation plan references a non-existent `REVIEWING` state
- Evidence: Task 17 skeleton uses `WORKER_RUNNING → REVIEWING → ORCHESTRATOR_SYNTHESIZING` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:801`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:805`).
- Conflicts with: design and verification require `REVIEW_CLAUDE` then `REVIEW_CODEX` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:105`, `docs/plans/2026-02-26-verification-plan.md:238`).

### 2. Critical (still): Post-iteration state mismatch (`IDLE` vs `ORCHESTRATOR_PLANNING`)
- Evidence:
  - Implementation ends iter with `... → IDLE` for `N < 3` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:784`).
  - Design + verification Tier 2 require final state `ORCHESTRATOR_PLANNING` after 1 iteration (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1306`, `docs/plans/2026-02-26-verification-plan.md:405`).

### 3. Critical (new): Design’s `run_pipeline.sh` sketch is not runnable as written (`PROJECT_DIR` referenced before set; bad default arg syntax)
- Evidence:
  - Sketch sources config via `source "${PROJECT_DIR}/agents/config.sh"` before establishing `PROJECT_DIR` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1193`).
  - Sketch uses `BATCH_NAME="${1:---batch-...}"`, which is not the valid bash default-substitution form (should be `${1:-...}`) and also doesn’t match the `--batch-name` contract used elsewhere (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1196`).
- Impact:
  - Anyone copying the design sketch will get an immediate error (`unbound variable` with `set -u` or empty path), before any pipeline logic executes.
- Fix:
  - Source config relative to the script location (e.g., `source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"`) or hardcode `PROJECT_DIR` in `config.sh` and just `source agents/config.sh` after `cd`.
  - Replace the batch-name handling in the design sketch with the same `--batch-name` parser specified in the implementation plan.

### 4. Critical (still from v3): Design’s gates.json example omits `v0_offset` even though v0-floor initialization requires an offset
- Evidence:
  - Design describes computing `v0_metric − offset` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:660`), but the example gates omit offsets (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:672`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:676`).
  - Implementation plan includes offsets while claiming it is “Exact content from design doc” (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:142`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:151`).

### 5. High (still): `.gitignore` prevents committing artifacts the implementation plan says to commit
- Evidence:
  - `.gitignore` ignores `.logs/` and `handoff/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:131`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:132`).
  - Tasks then instruct committing those directories (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:200`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:259`).

## Suggested next edits (smallest set that unblocks implementers)
1. Fix `run_pipeline.sh` design sketch so it’s runnable and matches the implementation plan’s `--batch-name` contract.
2. Align state names and end-of-iter state between design/verification/implementation.
3. Make gates.json schema canonical (decide whether `v0_offset` is required or has defaults) and make “Exact content” claims true.
