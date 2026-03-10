# Critical Review v5: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Canonical handoff schema is impossible to satisfy at `ORCHESTRATOR_PLANNING` and still inconsistent across sections (Critical)
- Evidence:
  - Canonical schema says all agents must conform and includes `version_id`: lines 603-616.
  - Orchestrator planning handoff is emitted before version allocation: lines 274, 477-480.
  - Reviewer schema examples omit `version_id` (and also omit `producer`): lines 415, 797.
  - Codex handoff writer omits both `version_id` and `producer`: line 443.
- Why this is a bug:
  - At planning completion, `version_id` does not exist yet, so strict schema compliance is impossible.
  - Inconsistent field sets across producers invite parser drift and brittle controller logic.
- Required fix:
  - Define per-state schema (or make `version_id` optional before worker phase).
  - Keep one canonical key set and update all examples/prompts/`jq` writers to match it.

### 2) Synthesis phase launch is under-specified and likely uses the wrong prompt (High)
- Evidence:
  - Two orchestrator prompts are declared: `orchestrator_plan.md` and `orchestrator_synthesize.md`: lines 186-187.
  - Orchestrator launch command is hardcoded to planning prompt: line 262.
  - Iteration flow reuses `launch_orchestrator.sh` for synthesis: line 492.
- Why this is a bug:
  - Without explicit phase selection, synthesis can run with planning instructions and miss required synthesis artifacts/state transitions.
- Required fix:
  - Make `launch_orchestrator.sh` accept `--phase plan|synthesize` and select prompt deterministically.
  - Add an integrity-test assertion that synthesis writes `orchestrator_synth_done.json` using the synth prompt.

### 3) Prompt path resolution is brittle (cwd-dependent) for orchestrator/reviewers (High)
- Evidence:
  - Orchestrator and both reviewer launches read `prompts/...` via relative paths: lines 262, 370, 433.
  - Worker launch uses absolute `${PROJECT_DIR}/agents/prompts/worker.md`: line 301.
- Why this is a bug:
  - Running scripts from repo root (or cron/other cwd) can make `prompts/...` unresolved, causing silent launch failures.
- Required fix:
  - Use `${PROJECT_DIR}/agents/prompts/...` consistently for all launch scripts.
  - Alternatively `cd "${PROJECT_DIR}/agents"` before any prompt reads.

### 4) Codex timeout/failure contract is internally inconsistent (Medium)
- Evidence:
  - Spec says timeout writes a notice into the review file: lines 455-456.
  - Controller flow expects a timeout notice file: line 490.
  - Watchdog writes `handoff/.../timeout_${state}.json`: lines 865-869.
- Why this is a bug:
  - Three different timeout artifacts are implied; controller behavior becomes ambiguous and error-prone.
- Required fix:
  - Standardize on one timeout artifact contract (recommended: handoff timeout JSON).
  - Explicitly define handling for non-timeout Codex failures (auth error/process crash) so the loop degrades immediately, not only after timeout.

### 5) Worker write-scope rules conflict between architecture and prompt contract (Medium)
- Evidence:
  - Worker sandbox rules allow edits broadly under project/worktree: line 324.
  - Worker prompt restricts edits to `ml/` and `registry/${VERSION_ID}/`: line 760.
- Why this is a bug:
  - Ambiguous authority can lead to inconsistent behavior and weak enforcement in audits.
- Required fix:
  - Unify write-scope policy in one authoritative rule and mirror it verbatim in both architecture and prompt sections.

## Missing tests / validation gaps

1. No test that validates handoff schema by state (especially planning before `version_id` exists).
2. No test proving synthesis phase selects `orchestrator_synthesize.md` rather than planning prompt.
3. No test that launch scripts succeed regardless of shell cwd.
4. No explicit failure-path test for Codex non-timeout errors (e.g., auth/model error) with graceful degradation.
5. No policy test to enforce the final worker write scope.

## Overall assessment

v4 improves several earlier blockers, but this revision still has one critical contract issue (handoff schema timing/consistency) and two high-severity execution risks (synthesis prompt selection and cwd-dependent prompt paths). These should be resolved before implementation to avoid brittle orchestration and false-negative handoff validation.
