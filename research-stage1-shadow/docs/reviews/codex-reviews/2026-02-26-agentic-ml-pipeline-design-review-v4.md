# Critical Review v4: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Version ID scheme is incompatible with additive registry and multi-batch execution (Critical)
- Evidence:
  - Registry is defined as additive-only (`registry/` comment): line 191.
  - Worker writes `--version-id v{N}` and artifacts under `registry/v{N}/`: lines 311-313, 731-733.
  - Comparison reads *all* `registry/v*/metrics.json`: line 337.
- Why this is a bug:
  - `N` is iteration index (1..3 per batch), so batch 2 will attempt to reuse `v1..v3`.
  - That collides with prior versions or forces overwrite, which violates additive-only and corrupts historical comparisons.
- Required fix:
  - Separate `iteration` from `version_id`.
  - Use globally unique version IDs (e.g., `v0001`, `v0002`, ... or `v<batch>-<iter>`).
  - Persist `iteration -> version_id` mapping in `state.json` and include in handoff schema.

### 2) “Isolated git worktree” is specified but not actually enforced in launch command (Critical)
- Evidence:
  - Worker identity claims isolated worktree: line 291.
  - Worktree is created: lines 296-297.
  - Claude invocation does not `cd` into worktree and does not pass `--worktree`: lines 298-302.
  - CLI reference explicitly lists `-w/--worktree`: line 886.
- Why this is a bug:
  - The worker may operate in the main checkout, defeating isolation guarantees.
  - Any assumption that worker changes are quarantined becomes false.
- Required fix:
  - Launch with an explicit working directory: `cd "$WORKTREE" && claude ...`.
  - Or enforce CLI worktree flag and validate cwd in prompt preamble.

### 3) Codex reviewer independence requirement contradicts its stated objective (High)
- Evidence:
  - Codex reviewer is said to emphasize “cross-checking Claude's conclusions”: line 446.
  - But both global memory policy and reviewer prompt require independence and prohibit reading other reviewer output: lines 674, 758.
- Why this is a bug:
  - The design asks Codex to verify something it is explicitly prevented from seeing.
  - This creates ambiguous behavior and inconsistent reviewer expectations.
- Required fix:
  - Pick one model:
  - A) Independent parallel reviews (current prompt constraints) and remove “cross-checking Claude”.
  - B) Sequential meta-review where Codex explicitly reads Claude review (new state + prompt changes).

### 4) Handoff contract is internally inconsistent and weakly validated (High)
- Evidence:
  - Reviewer schema omits `producer`: line 411; also line 773.
  - Example handoff includes `producer`: line 601.
  - Verification function trusts `artifact_path` from handoff and only checks hash equality: lines 610-616.
- Why this is a bug:
  - Schema mismatch invites parsing drift.
  - A compromised/misbehaving agent can point `artifact_path` to any file it created and still pass hash validation.
- Required fix:
  - Freeze one canonical schema.
  - In controller, validate `(agent,state) -> expected artifact path pattern` before hash check.
  - Reject handoff if artifact path does not match expected location for that state.

### 5) Watchdog test spec contradicts watchdog runtime behavior (Medium)
- Evidence:
  - Watchdog exits immediately in `IDLE`/`HUMAN_SYNC`: line 800.
  - Component test for idle mode expects a tail entry from `.logs/audit.jsonl`: line 984.
- Why this is a bug:
  - Test expectation is incompatible with the script behavior.
  - This will either fail consistently or pass spuriously based on stale prior logs.
- Required fix:
  - Either log idle probes before exit, or change test to assert “no new probe written in idle”.

### 6) Cron installer is not idempotent and can duplicate watchdog jobs (Medium)
- Evidence:
  - Installer appends entry unconditionally: line 788.
- Why this is a bug:
  - Re-running install adds duplicate cron lines; watchdog can run multiple times per minute.
- Required fix:
  - Use an idempotent install pattern (`crontab -l | grep -F ... || echo ...`).

### 7) Disk limit config is duplicated and one path ignores configured variable (Low)
- Evidence:
  - Config defines `REGISTRY_DISK_LIMIT_MB=10240`: line 878.
  - Watchdog hard-codes `10240`: line 834.
- Why this matters:
  - A future config change can silently be ignored by watchdog logic.
- Required fix:
  - Replace literal with `${REGISTRY_DISK_LIMIT_MB}` in watchdog.

### 8) Prompt spec has internal section-count inconsistency (Low)
- Evidence:
  - Reviewer prompt says “all five sections are mandatory” but lists six numbered sections including verdict: lines 760-767.
- Why this matters:
  - Small but avoidable ambiguity in automation checks.
- Required fix:
  - Update wording to “all six sections are mandatory”.

## Missing tests / validation gaps

1. No explicit test proving version IDs remain unique across multiple batches (not only one smoke iteration).
2. No negative test for forged handoff paths (`artifact_path` mismatch with state expectations).
3. No test that verifies worker actually runs in intended worktree/cwd.
4. No test covering idempotent cron installation.
5. No test ensuring reviewer independence policy is enforced by controller (not only prompt text).

## Overall assessment

The architecture is directionally strong, but the current v3 spec has two blocking design flaws (version namespace and unenforced worktree isolation) that can invalidate provenance and reproducibility. Resolve findings 1-2 before implementation.
