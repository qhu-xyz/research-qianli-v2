# Critical Review v7: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`  
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Worker-to-main code continuity still depends on an unstated git commit contract (Critical)
- Evidence:
  - Controller syncs only registry artifacts from worktree (`rsync ... registry/${VERSION_ID}/`): lines 515-516.
  - Controller then merges worker branch into main (`git merge ... iter${N}-${BATCH_ID}`): lines 517-518.
  - Worker responsibilities and prompt do not require any `git add/commit`: lines 323-335, 834-840.
- Why this is a bug:
  - If worker edits are left uncommitted in the worktree (the default unless explicitly instructed), branch merge will not carry those `ml/` changes into main.
  - That breaks review fidelity (reviewers may inspect stale code) and iteration continuity (iter N+1 may not build on iter N code changes).
- Required fix:
  - Add an explicit worker commit contract (message template + required staged paths), and make controller verify branch head moved before merge.
  - Or replace merge dependency with deterministic file sync/cherry-pick logic that does not rely on implicit worker git behavior.

### 2) Watchdog crash detection checks the wrong handoff filenames and can emit false failure artifacts (High)
- Evidence:
  - Canonical handoff filenames are `worker_done.json`, `claude_reviewer_done.json`, `codex_reviewer_done.json`: lines 208-211.
  - Watchdog checks `"${HANDOFF_DIR}/${state,,}_done.json"`: line 958.
  - For active states, this expands to names like `review_codex_done.json` or `worker_running_done.json`, which do not match canonical files.
- Why this is a bug:
  - A successful agent run can be misclassified as crash (`session dead + HANDOFF_EXISTS=false`), producing `timeout_${state}.json` spuriously.
  - Controller explicitly polls timeout files alongside handoffs (line 979), so false timeouts can trigger incorrect degradation or skip logic.
- Required fix:
  - Replace `${state,,}_done.json` with a state-to-handoff map used by both controller and watchdog.
  - Add guard: if any valid success handoff exists, never write crash timeout artifact.

### 3) Worker-failure transition path conflicts with synthesis input requirements (High)
- Evidence:
  - On `worker_done.status == "failed"`, controller skips comparison and jumps directly to synthesis: lines 514, 672-674.
  - Synthesis phase currently specifies reading both review files and the comparison report: lines 289-291.
- Why this is a bug:
  - In the failure branch, those review/comparison artifacts do not exist by design, so synthesis has an underspecified/contradictory input contract.
  - This can stall or fail the very recovery path meant to keep the pipeline progressing.
- Required fix:
  - Define a dedicated synthesis-on-failure input contract (e.g., worker failure handoff + prior memory only).
  - Update prompt and verification logic so synthesis does not require nonexistent artifacts when worker fails.

### 4) Iteration-3 synthesis artifact path is inconsistent with archive directory contract (Medium)
- Evidence:
  - Archive structure uses `memory/archive/batch-{id}/...`: lines 232-233, 756-760; synthesis phase also writes to `memory/archive/batch-{id}/`: line 297.
  - Handoff verification expects iteration-3 synthesis artifact at `memory/archive/${batch_id}/executive_summary.md`: line 701.
- Why this is a bug:
  - Path mismatch will fail synthesis handoff verification on iteration 3, blocking transition to `HUMAN_SYNC`.
- Required fix:
  - Normalize on one path convention (`batch-{id}` is already used elsewhere) and use the same helper in both writer and verifier code paths.

## Missing tests / validation gaps

1. Test that worker code changes are present in main after step 7c when worker does and does not create commits.
2. Watchdog unit test for each active state proving correct handoff filename detection and no false crash artifacts.
3. Integration test for `worker_done.status="failed"` that reaches synthesis and exits cleanly without review/comparison artifacts.
4. Iteration-3 synthesis verification test asserting archive artifact path matches expected handoff pattern and state advances to `HUMAN_SYNC`.

## Overall assessment

v6 fixed major earlier issues, but the remaining contract gaps are still execution-critical. Finding #1 must be resolved before implementation; findings #2 and #3 are high-priority reliability blockers; finding #4 is a deterministic iteration-3 breakage.
