# Critical Review v9: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`  
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Codex reviewer is launched in read-only sandbox while its contract requires file writes (Critical)
- Evidence:
  - Codex launch uses `--sandbox read-only` (line 469).
  - Reviewer contract requires writing `reviews/{batch_id}_iter{N}_{reviewer_id}.md` and `handoff/{batch_id}/iter{N}/{reviewer_id}_done.json` (lines 906-910, 913).
- Why this is a bug:
  - The reviewer instructions and runtime permissions conflict. In read-only mode, the Codex agent cannot satisfy its declared write contract.
  - This creates deterministic non-compliance risk and likely timeout/degradation paths for REVIEW_CODEX.
- Required fix:
  - Pick one consistent model:
    - Use `--sandbox workspace-write` plus strict path constraints, or
    - Keep read-only and change the reviewer prompt to stdout-only output while controller/wrapper writes artifacts.

### 2) `verify_handoff()` cannot validate documented `status:"failed"` handoffs (Critical)
- Evidence:
  - Failure schema sets `artifact_path: null` and `sha256: null` for worker failure (lines 682-695).
  - `verify_handoff()` unconditionally enforces path equality and sha256 of an on-disk artifact (lines 735-748).
- Why this is a bug:
  - A valid failed handoff (by documented schema) cannot pass this verifier as written.
  - If implemented literally, worker failure handling can dead-end at verification instead of transitioning to synthesis.
- Required fix:
  - Make verification status-aware:
    - `status == done`: enforce path + hash.
    - `status == failed`: require `error` field, skip artifact hash/path checks, allow transition to failure synthesis flow.

### 3) Orchestrator/Claude reviewer launches depend on caller CWD, so relative paths can break artifact IO (High)
- Evidence:
  - Worker launch explicitly `cd`'s to worktree before invoking agent (line 335).
  - Orchestrator launch has no `cd "${PROJECT_DIR}"` (lines 280-281).
  - Claude reviewer launch also has no `cd "${PROJECT_DIR}"` (lines 407-410).
  - Codex wrapper writes `REVIEW_FILE="reviews/..."` via shell redirection (lines 463, 473), which is shell-CWD relative.
- Why this is a bug:
  - Prompts and contracts use relative paths (`memory/...`, `reviews/...`, `handoff/...`).
  - If scripts are invoked from a non-root CWD, agents can read/write wrong locations, causing missing handoffs and timeouts.
- Required fix:
  - Standardize launch commands to run from project root (`cd "${PROJECT_DIR}" && ...`).
  - Use absolute output paths in wrappers (e.g., `REVIEW_FILE="${PROJECT_DIR}/reviews/..."`).

### 4) Integrity test narrative still encodes obsolete pre-merge artifact sync behavior (Medium)
- Evidence:
  - Main iteration flow now states merge-first with explicit "No pre-merge rsync" (lines 542-544).
  - Integrity test still lists `4a. Artifacts synced from worktree to main` before merge (lines 1130-1131).
- Why this is a bug:
  - Test expectations are now out of sync with the corrected design.
  - This can produce false failures or reintroduce the previously fixed merge/sync sequencing mistake.
- Required fix:
  - Update integrity test spec/assertions to check merge-only propagation of `ml/` and `registry/${VERSION_ID}/` artifacts.

## Missing tests / validation gaps

1. Contract test for Codex reviewer mode consistency:
   - read-only mode must use stdout-only prompt, or
   - workspace-write mode must prove writes are constrained to `reviews/` + `handoff/`.
2. Unit tests for `verify_handoff()` covering `status=done` and `status=failed` with null artifact fields.
3. Launch-script test that runs from a non-project CWD and verifies all orchestrator/reviewer artifacts are still written under `${PROJECT_DIR}`.
4. Integration test assertion update to enforce merge-only registry propagation (no rsync step).

## Overall assessment

v8 resolves the previously reported major blockers, but two contract-level execution blockers remain: Codex write-permission mismatch and failure-handoff verification incompatibility. Address findings #1-#2 before implementation; #3-#4 should be fixed to prevent brittle runtime behavior and regression of the merge-flow correction.
