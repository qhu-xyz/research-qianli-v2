# Implementation Plan Review (v4)

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`
**Cross-referenced**: Design doc v10, Verification plan, prior reviews (2026-02-26, v2, v3)
**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-27

---

## Executive Summary

All v3 findings (CB-1 through CB-4, HG-1 through HG-3) are incorporated into the current
plan. The worktree isolation fixes (commit-before-worktree, absolute handoff paths), CAS
corrections (IDLE at step 18, HUMAN_SYNC reset), and specification gaps (--overrides,
SMOKE_TEST data, degenerate test data) are all addressed.

This review focuses on **error recovery paths** and **operational robustness** — angles
that prior reviews did not cover. The v3 review found bugs in the *happy path*; this review
examines what happens when agents crash, polls expire, or system state becomes
inconsistent.

**2 high-priority bugs found** (pipeline will leave state stuck on first agent failure),
**3 medium-priority gaps**, **4 observations**.

---

## High-Priority Bugs

### HP-1: Unhandled poll timeout for orchestrator and Claude reviewer leaves state stuck

**Where**: Task 17 steps 3/4, 10/11, 16/17; Task 16 (`poll_for_handoff`).

**Root cause**: `poll_for_handoff` returns 1 on timeout (Task 16, line 754). With
`set -euo pipefail` (design doc §12.3, line 1192), a non-zero return immediately
terminates `run_single_iter.sh`. But the script has already transitioned state.json to
`ORCHESTRATOR_PLANNING` (step 2), `REVIEW_CLAUDE` (step 9), or `ORCHESTRATOR_SYNTHESIZING`
(step 15). If the poll times out:

```
Step 2:   state → ORCHESTRATOR_PLANNING
Step 3:   launch orchestrator
Step 4:   poll_for_handoff ... returns 1 (timeout)
          set -e kills the script immediately
          state.json is stuck at ORCHESTRATOR_PLANNING
          No cleanup: tmux session still running, state never recovers
```

The watchdog detects the stuck state and writes `timeout_ORCHESTRATOR_PLANNING.json`, but
the controller is dead — nothing reads the timeout artifact. The pipeline is now wedged:
state is `ORCHESTRATOR_PLANNING`, `run_pipeline.sh` (the parent) exited because its child
died, and the flock on `state.lock` is released. A new `run_pipeline.sh` invocation will
find state is not IDLE/HUMAN_SYNC and refuse to start.

**Same issue at steps 10/11** (Claude reviewer poll timeout) and **steps 16/17** (synthesis
orchestrator poll timeout). Codex (steps 13/14) is the only poll with explicit timeout
handling — it checks for `timeout_REVIEW_CODEX.json` and continues.

**Contrast with worker failure path**: The worker has an explicit failure branch (step 7a
checks handoff status). The orchestrator and Claude reviewer have no equivalent — a crash
or timeout is an unhandled case.

**Fix**: Add explicit timeout/error handling after each non-Codex poll:

```bash
# Step 3-4: orchestrator plan
ORCH_SESSION=$(launch_orchestrator.sh --phase plan)
# ... write orchestrator_tmux to state.json ...
if ! poll_for_handoff "${HANDOFF_DIR}" "orchestrator_plan_done.json" "${MAX_SECONDS}" 30; then
  # Poll timed out or failed — kill tmux, reset state, abort batch
  tmux kill-session -t "${ORCH_SESSION}" 2>/dev/null || true
  cas_transition ORCHESTRATOR_PLANNING IDLE '{"error":"orchestrator plan timeout"}'
  echo "FATAL: orchestrator plan timed out at iter ${N}" >&2
  exit 1
fi
```

This resets state to IDLE so the next `run_pipeline.sh` invocation can start fresh. The
error field in state.json records what went wrong.

**Alternative** (less disruptive): Wrap `poll_for_handoff` calls in explicit error traps
instead of relying on `set -e`:

```bash
set +e  # disable errexit for this block
poll_for_handoff "${HANDOFF_DIR}" "orchestrator_plan_done.json" "${MAX_SECONDS}" 30
POLL_RC=$?
set -e
if (( POLL_RC != 0 )); then
  # handle timeout/error
fi
```

The implementation plan must specify which approach to use. Without this, the implementer
will likely not handle the timeout case and the first agent failure will wedge the pipeline.

---

### HP-2: Step 5a `git commit` fails with "nothing to commit" if orchestrator writes nothing

**Where**: Task 17 step 5a (CB-1 fix).

**Root cause**: The CB-1 fix adds:
```bash
git -C "${PROJECT_DIR}" add memory/ && \
  git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator plan"
```

If the orchestrator fails silently (handoff reports "done" but the direction file is
empty or not written), or if memory files are unchanged from the prior iteration, `git add`
stages nothing and `git commit` exits with code 1 ("nothing to commit"). With `set -e`,
this kills the script.

The same issue affects step 17a (synthesis commit) — if synthesis writes no changes to
memory files, the commit fails.

**Risk**: This is more likely than it appears. If the orchestrator's Claude session hits a
context limit and produces a truncated or empty direction file, `verify_handoff()` at
step 5 (sha256 check) will still pass as long as the agent wrote the file and reported its
correct hash. The commit then fails because `git add` of an empty file produces no diff
(if the file was already empty).

**Fix**: Use `--allow-empty` or check for staged changes:

```bash
# Step 5a: commit orchestrator outputs so worktree inherits them
git -C "${PROJECT_DIR}" add memory/
if ! git -C "${PROJECT_DIR}" diff --cached --quiet; then
  git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator plan"
fi
```

This skips the commit if nothing changed (no error) while still committing when there are
changes. The subsequent worktree creation still works — it just gets HEAD without new
memory content (which is the pre-fix behavior, degraded but not crashed).

Same pattern for step 17a.

---

## Medium-Priority Gaps

### MG-1: cas_transition temp file may not be atomic if `/tmp` is a different filesystem

**Where**: Task 16 (state_utils.sh `cas_transition` implementation).

**Problem**: The CAS pattern requires atomic write: write new state to a temp file, then
`mv` it over `state.json`. If the implementation writes to `/tmp/state.json.$$` and then
`mv`s to `${PROJECT_DIR}/state.json`, and `/tmp` is a different filesystem (common in
containers with tmpfs), `mv` falls back to `cp + rm` — which is NOT atomic. A concurrent
read (by the watchdog) could see a partial file.

The design doc says "single writer" so there's no concurrent writer risk, but the watchdog
reads `state.json` (§11.2, line 1043: `jq -r '.entered_at' "$STATE_FILE"`). A non-atomic
write could produce a truncated JSON that `jq` can't parse, causing the watchdog to
misinterpret the state.

**Fix**: Specify in Task 16 that the temp file must be in the same directory as state.json:

```bash
cas_transition() {
  local expected="$1" new_state="$2" new_fields="$3"
  local tmpfile="${STATE_FILE}.tmp.$$"
  # ... build new JSON ...
  printf '%s' "$new_json" > "$tmpfile"
  mv "$tmpfile" "$STATE_FILE"   # atomic on same filesystem
}
```

This is a one-line implementation detail, but if not specified, the implementer may use
`mktemp` (which defaults to `/tmp/`).

---

### MG-2: BATCH_NAME not sanitized for tmux session names

**Where**: Task 18 (run_pipeline.sh `--batch-name` parsing).

**Problem**: The user passes `--batch-name "my experiment"` and the plan constructs:
```bash
BATCH_ID="${BATCH_NAME}-$(date +%Y%m%d-%H%M%S)"
```

BATCH_ID is then used in tmux session names (e.g., `worker-${BATCH_ID}-iter${N}`). tmux
session names cannot contain `.`, `:`, or whitespace. If the user passes a batch name
with spaces or periods, all 4 launch scripts will fail with cryptic tmux errors.

**Risk**: Low (only triggered by user input), but a confusing failure mode.

**Fix**: Add sanitization in Task 18:
```bash
# Sanitize batch name: replace non-alphanumeric chars with hyphens
BATCH_NAME=$(echo "$BATCH_NAME" | tr -c '[:alnum:]-' '-' | sed 's/-\+/-/g; s/^-//; s/-$//')
```

Or validate and reject:
```bash
[[ "$BATCH_NAME" =~ ^[a-zA-Z0-9-]+$ ]] || { echo "ERROR: batch name must be alphanumeric+hyphens only"; exit 1; }
```

---

### MG-3: Promotion decision mechanism lacks structured handoff field

**Where**: Task 17 step 17 ("controller calls `promote_version()` if recommended"),
Task 23 (orchestrator_synthesize.md prompt).

**Problem**: The plan says the controller "applies synthesis decisions" including calling
`promote_version()` "if recommended." But the synthesis orchestrator's handoff schema
(`orchestrator_synth_done.json`) uses the same generic handoff schema as all agents:
`{status, artifact_path, sha256}`. There is no field for the orchestrator to communicate
*which decisions it made* — specifically whether it recommends promoting the current
version.

The controller would need to parse the synthesis orchestrator's natural language output
(the direction file or executive summary) to determine whether to promote. This is fragile:
the controller is a shell script, not an LLM. It cannot reliably extract structured
decisions from prose.

**Fix**: Extend the synthesis handoff schema with a structured `decisions` field:

```json
{
  "status": "done",
  "artifact_path": "memory/direction_iter2.md",
  "sha256": "...",
  "decisions": {
    "promote_version": null,
    "gate_change_requests": [],
    "next_hypothesis": "..."
  }
}
```

If `promote_version` is non-null (e.g., `"v0003"`), the controller calls
`promote_version()`. Otherwise it skips promotion. The controller does simple JSON
extraction (`jq -r '.decisions.promote_version // empty'`), not NLP.

This was flagged as DG-2 in v2 and marked as "resolved" because step 17 says "controller
calls promote_version() if recommended." But the *mechanism* for the recommendation is
still missing — the controller has no structured way to know what the orchestrator
recommended.

---

## Observations (not blocking)

### O-1: No stale worktree cleanup from crashed batches

**Where**: Task 18 (run_pipeline.sh), post-loop cleanup.

The plan adds post-loop worktree cleanup (lines 886-891):
```bash
for i in 1 2 3; do
  WT="${PROJECT_DIR}/.claude/worktrees/iter${i}-${BATCH_ID}"
  [[ -d "$WT" ]] && git worktree remove "$WT" --force 2>/dev/null || true
done
```

This only cleans worktrees from the *current batch*. If a previous batch crashed (OOM,
hardware failure, manual kill), its worktrees remain in `.claude/worktrees/`. Over time,
these accumulate and waste disk space (each worktree is a full checkout).

**Suggestion**: Add a pre-loop cleanup that removes worktrees from all prior batches:
```bash
# Pre-loop: clean stale worktrees from any prior crashed batches
git -C "${PROJECT_DIR}" worktree prune
for stale_wt in "${PROJECT_DIR}/.claude/worktrees"/iter*; do
  [[ -d "$stale_wt" ]] && git worktree remove "$stale_wt" --force 2>/dev/null || true
done
```

Not blocking because `.claude/worktrees/` is in `.gitignore` and doesn't affect pipeline
logic.

### O-2: VP Tier 2 final state expectation inconsistency

**Where**: Verification plan Part 4, Tier 2 (line 405):
```
**Final state must be `ORCHESTRATOR_PLANNING`** (not `HUMAN_SYNC`).
```

The implementation plan (Task 17, step 18 note at line 814) says the VP Tier 2 assertion
should be updated to expect `IDLE`. The implementation plan is correct (step 18 N<3
transitions to IDLE). But the verification plan document itself has not been updated — it
still says `ORCHESTRATOR_PLANNING`.

If the implementer follows the VP literally, Tier 2 will fail. The VP should be updated
to say `IDLE`.

### O-3: Watchdog timeout budget for orchestrator planning (600s) may be too tight

**Where**: Design doc §11.3, Task 17 step 2 (`max_seconds = 600`).

The orchestrator reads all of `memory/hot/`, `memory/warm/`, `memory/archive/index.md`,
`registry/gates.json`, and champion metrics. For iteration 1 this is light (~10KB). But by
iteration 3, `memory/warm/` may have grown to ~50KB of experiment/hypothesis/decision logs.
The orchestrator needs to read all of this, reason about 2 prior iterations of feedback,
formulate a hypothesis, and write a structured direction file.

Claude Opus 4.6 on large context windows routinely takes 2-4 minutes just for the API
response. Add tmux startup time, prompt loading, and file reads/writes, and 600s (10
minutes) is reasonable for iteration 1 but may be tight for iteration 3.

Not blocking — the watchdog timeout can be tuned during Tier 3 testing without plan changes.

### O-4: All v3 findings confirmed resolved in current plan

| v3 ID | Status | Evidence |
|-------|--------|----------|
| CB-1 | Fixed | Step 5a: `git add memory/ && git commit` (line 784-788) |
| CB-2 | Fixed | Step 6 note: "worker writes to ${PROJECT_DIR}/handoff/" (line 791-793) |
| CB-3 | Fixed | Step 18: `ORCHESTRATOR_SYNTHESIZING → IDLE` for N<3 (line 811) |
| CB-4 | Fixed | Pre-loop HUMAN_SYNC reset in run_pipeline.sh (line 877-882) |
| HG-1 | Fixed | Task 13 step 2: --overrides spec added (line 676) |
| HG-2 | Fixed | Task 6: actual_shadow_price specified (implied by v3 fix incorporation) |
| HG-3 | Fixed | Task 10: degenerate data replaced (implied by v3 fix incorporation) |
| IR-1 | Fixed | Task 13 step 1: `version_counter.json` scaffolding added (line 665) |
| O-1 | Fixed | Step 7d: pre-merge guard for HUMAN-WRITE-ONLY files (line 795) |
| O-3 | Fixed | Step 17a: synthesis commit added (line 798-802) |

---

## Summary Table

| ID | Severity | Category | Summary |
|----|----------|----------|---------|
| HP-1 | High | Error recovery | poll_for_handoff timeout kills script via set -e, state stuck forever |
| HP-2 | High | Error recovery | git commit fails with "nothing to commit" if orchestrator writes nothing |
| MG-1 | Medium | Atomicity | cas_transition temp file non-atomic if written to /tmp |
| MG-2 | Medium | Input validation | BATCH_NAME with spaces/dots breaks tmux session names |
| MG-3 | Medium | Handoff schema | Promotion decision has no structured field in synthesis handoff |
| O-1 | Note | Cleanup | Stale worktrees from crashed batches not cleaned |
| O-2 | Note | VP consistency | VP Tier 2 still says ORCHESTRATOR_PLANNING, should say IDLE |
| O-3 | Note | Tuning | 600s orchestrator timeout may be tight for iteration 3 |
| O-4 | Note | Prior reviews | All v3 findings confirmed resolved |

**Should fix before implementation**: HP-1, HP-2, MG-3
**Should fix during implementation**: MG-1, MG-2
**Address during testing**: O-1, O-2, O-3

---

## Recommended Fix Order

1. **HP-1** (5 minutes): Specify poll-timeout error handling pattern — either `set +e`
   wrapper or explicit trap. Reset state to IDLE with error field on timeout. Apply to
   steps 3/4, 10/11, and 16/17.

2. **HP-2** (2 minutes): Add `git diff --cached --quiet` guard before both commit steps
   (5a and 17a). Skip commit if nothing staged.

3. **MG-3** (5 minutes): Extend synthesis handoff schema with `decisions` field. Add
   controller logic to read `decisions.promote_version` via jq.

4. **MG-1** (1 minute): Specify temp file location as `${STATE_FILE}.tmp.$$` in Task 16.

5. **MG-2** (1 minute): Add BATCH_NAME validation regex in Task 18.

---

## Verdict

**FIX HP-1 AND HP-2 BEFORE IMPLEMENTING** — HP-1 is the most important finding: the
pipeline currently has no error recovery for the 3 most common poll points (orchestrator
plan, Claude review, orchestrator synthesis). The first time any agent crashes or times out,
the pipeline state wedges permanently. HP-2 is a direct consequence of the CB-1 fix from
v3 — the fix introduced a new failure mode that needs a guard.

MG-3 should also be addressed before implementation because it affects the handoff schema
(a cross-cutting contract), but the pipeline can technically function without it (the
orchestrator just can't trigger promotion).

After fixing these 3 issues, the plan is ready for implementation. The remaining medium-
priority gaps (MG-1, MG-2) and observations can be addressed during implementation.

---

## Cross-Review Lineage

| Review | Findings | Status |
|--------|----------|--------|
| 2026-02-26 (initial) | 3 spec bugs, 5 gaps, 4 structural concerns | All resolved in plan |
| v2 (2026-02-27) | 3 spec bugs, 4 gaps, 3 impl risks, 3 observations | All resolved in plan |
| v3 (2026-02-27) | 4 critical bugs, 3 high gaps, 1 medium, 4 observations | All resolved in plan |
| **v4 (2026-02-27)** | **2 high bugs, 3 medium gaps, 4 observations** | **Current** |

Total unique findings across all reviews: 46 (12 + 13 + 12 + 9).
Remaining open: 9 (from this review).
