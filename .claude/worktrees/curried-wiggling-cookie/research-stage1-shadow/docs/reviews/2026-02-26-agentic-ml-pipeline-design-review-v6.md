# Critical Review v6: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`  
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Worktree isolation breaks artifact visibility, code review fidelity, and iterative learning (Critical)
- Evidence:
  - Worker runs in isolated worktree (`cd "${WORKTREE}"`): lines 311-319.
  - Worker writes version artifacts under relative `registry/${VERSION_ID}/...`: lines 329-334.
  - Controller comparison step reads root `registry/v*/metrics.json`: lines 349-357.
  - Reviewers are instructed to read `ml/` codebase and worker outputs from root paths: lines 395-399, 813-822.
  - Iteration flow has no explicit merge/sync step from worker worktree back to main workspace: lines 503-525.
- Why this is a bug:
  - Artifacts and code changes produced in the worker worktree are not guaranteed to be visible at the controller/reviewer paths.
  - Even if one iteration succeeds locally, next iterations can start from stale main-branch code with no accumulated improvements.
- Required fix:
  - Define a mandatory sync step after worker completion (merge/cherry-pick branch and copy/promote artifacts), or
  - Run comparison/review against the same worktree and explicitly base the next iteration on the prior iteration branch.

### 2) Worker failure contract is undefined and can deadlock the state machine (High)
- Evidence:
  - Worker prompt says: if tests fail 3x, write failure handoff and stop: line 804.
  - Controller flow waits for `worker_done.json`: line 513.
  - Handoff verification for `WORKER_RUNNING` requires `registry/${VERSION_ID}/metrics.json`: lines 665, 679.
  - Canonical example only shows `status: "done"` and no failure schema: lines 636-647.
- Why this is a bug:
  - On test/training failure, expected success artifact may never exist, leaving controller to timeout instead of handling deterministic failure.
- Required fix:
  - Specify an explicit failure artifact/schema (`status: "failed"`, error fields), and define controller transition behavior for that case.

### 3) Codex crash/error handling claims immediate degradation, but watchdog logic only degrades at timeout (High)
- Evidence:
  - Failure table states crash/auth error is detected by watchdog and handled on next cycle: lines 480-483.
  - Watchdog currently only writes timeout artifact when `elapsed > max_s`: lines 907-910.
  - Dead session detection is logged but does not trigger failure artifact creation: lines 887-899.
- Why this is a bug:
  - A fast Codex failure can stall the pipeline for up to the full 1200s timeout despite documented graceful degradation.
- Required fix:
  - Add immediate dead-session + missing-handoff detection path that writes `timeout_REVIEW_CODEX.json` (or a dedicated failure artifact) without waiting full timeout.

### 4) Handoff verification table omits `ORCHESTRATOR_SYNTHESIZING` despite required verification for all state transitions (Medium)
- Evidence:
  - Controller is expected to verify handoffs before transitioning state: lines 630-631.
  - Iteration flow polls for `orchestrator_synth_done.json`: line 523.
  - `EXPECTED_ARTIFACT_PATTERN` has no `ORCHESTRATOR_SYNTHESIZING` entry: lines 664-669.
- Why this is a bug:
  - Either synthesis handoff is not verified (security/integrity gap) or verification behavior is undefined.
- Required fix:
  - Add explicit expected artifact rule for synthesis state and include it in the same verification path as other agents.

### 5) Integrity test expectations conflict with synthesis output rules (Medium)
- Evidence:
  - Synthesis writes batch summary only when `iteration == 3`: lines 295-303.
  - Integrity test is described as a 1-iteration loop: lines 1022-1023.
  - Same integrity checklist expects synthesis to write an executive summary and end in `HUMAN_SYNC`: lines 1037-1042.
- Why this is a bug:
  - Test contract and runtime contract diverge; this will produce flaky or misleading pass/fail outcomes.
- Required fix:
  - Either run a true 3-iteration integrity test, or redefine 1-iteration smoke semantics and expected synthesis artifacts/state accordingly.

## Missing tests / validation gaps

1. No test that verifies worker-produced artifacts are visible to controller/reviewers when worker runs in a separate worktree.
2. No test for worker failure handoff path (test failure, train failure) and deterministic controller transition.
3. No test for immediate Codex crash/auth-error handling without waiting for full timeout.
4. No test that synthesis handoff goes through the same artifact-path/hash verification as other states.
5. No test enforcing iterative code continuity across iterations (iter2 must include iter1 accepted changes).

## Overall assessment

The design is much tighter than earlier revisions, but one critical architecture gap remains: the worktree isolation model is not integrated with artifact/review/continuity flow. Resolve finding #1 before implementation; findings #2-#5 should be addressed to avoid avoidable stalls and false integrity signals.
