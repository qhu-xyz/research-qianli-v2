# Codex Review: Implementation Plan (v1)

## Scope
- `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md` (primary)
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (cross-check)
- `docs/plans/2026-02-26-verification-plan.md` (cross-check)

## Findings (Ordered by Severity)

1. **[CRITICAL] Plan forbids modifications that it later requires, creating an execution deadlock**
- Evidence: Task 1’s `CLAUDE.md` content marks `registry/gates.json` and `ml/evaluate.py` as "agents must NEVER modify" (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:43-46`).
- Conflicting required tasks: Task 2 explicitly requires writing `registry/gates.json` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:124-145`), and Task 10 requires creating `ml/evaluate.py` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:498-542`).
- Impact: A strict task-by-task executor will violate its own just-created constraints or halt.
- Fix: Explicitly scope "HUMAN-WRITE-ONLY" to runtime agents only (worker/reviewers/orchestrator), or move creation/bootstrap of these files to a clearly human-only pre-step.

2. **[HIGH] Synthetic fixture sizes are inconsistent and will break training tests as written**
- Evidence: `synthetic_features()` is specified as shape `(100, 14)` while `synthetic_labels()` is specified as length `200` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:328-329`).
- Tests then train with both fixtures together (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:413-420`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:469-473`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:509-517`).
- Impact: Model training should fail with inconsistent sample counts before meaningful assertions execute.
- Fix: Set `synthetic_labels()` length to 100 (or make both fixtures share a single `n_samples` constant).

3. **[HIGH] Prompt/launcher contract for Codex is self-contradictory**
- Evidence A: Task 19 states all launchers must use stdin redirect and "never `$(cat ...)`" (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:801`).
- Evidence B: Task 23 says Codex launch relies on the `$(cat ...)` pattern for variable expansion (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:895`).
- Cross-doc note: design v10 currently shows Codex launch with `"$(cat .../reviewer_codex.md)"` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:487`).
- Impact: Implementers cannot satisfy both constraints simultaneously; this also re-opens shell-safety risk that prior reviews tried to remove.
- Fix: Choose one contract and specify it concretely (recommended: keep stdin-safe launch flow and define a separate explicit variable-substitution step such as `envsubst` into a temp prompt file).

4. **[MEDIUM] `PipelineConfig` spec and test usage disagree on `registry_dir`**
- Evidence: Task 5’s required `PipelineConfig` fields do not include `registry_dir` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:324-325`).
- But Task 13 test instantiates `PipelineConfig(..., registry_dir=...)` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:628`).
- Impact: Either config construction fails or implementations diverge ad hoc.
- Fix: Add `registry_dir` to the formal `PipelineConfig` spec (with default), or remove it from the test and pass registry path elsewhere.

5. **[MEDIUM] Task 14 requests new tests but commit command omits the edited test file**
- Evidence: Task 14 Step 2 says to add tests to `ml/tests/test_registry.py` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:674-683`).
- Commit only stages `ml/populate_v0_gates.py` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:692-693`).
- Impact: The plan can leave test changes unstaged/uncommitted, breaking checkpoint reproducibility.
- Fix: Stage both files (or update the step to state tests are intentionally committed later).

6. **[LOW] Phase/task numbering is inconsistent**
- Evidence: Phase F title says "Tasks 25–27" (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:923`) but includes Task 28 (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:991`).
- Impact: Minor tracking confusion during execution/checklists.
- Fix: Rename heading to "Tasks 25–28".

## Open Questions
1. Should `registry/gates.json` / `ml/evaluate.py` be immutable only after initial bootstrap, or truly human-only for the full lifecycle including initial creation?
2. For Codex prompt substitution, do you want a safe temp-file render flow (`envsubst`/template expansion) or an argument-based prompt assembly model?

## Summary
The implementation plan is close, but Findings 1-3 are execution blockers and should be resolved before task-by-task execution. Findings 4-6 are consistency/reproducibility fixes that should be cleaned up in the same edit pass.
