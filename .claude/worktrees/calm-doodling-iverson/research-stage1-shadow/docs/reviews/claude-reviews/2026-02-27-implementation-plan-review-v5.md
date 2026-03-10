# Implementation Plan Review (v5)

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`
**Cross-referenced**: Design doc v10, Verification plan, prior reviews (2026-02-26, v2, v3, v4)
**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-27

---

## Executive Summary

This review adopts a **cross-document consistency** lens. Prior reviews focused on the
implementation plan in isolation (v2-v3) or on error recovery paths (v4). This pass
traces contracts across all three documents (implementation plan, design doc, verification
plan) and between shell scripts, CLAUDE.md, and prompt files — looking for places where
one document was updated for a prior fix but dependent documents were not.

**2 high-priority bugs found** (worker will read stale data; Codex prompt will be
corrupted), **2 medium-priority gaps**, **5 observations**.

All v4 findings (HP-1, HP-2, MG-1 through MG-3) are confirmed still open — the plan
has not been updated since v4 was written. These are NOT re-listed below.

---

## High-Priority Bugs

### HP-3: CLAUDE.md and prompt header instruct worker to read state.json via relative path — stale in worktree

**Where**: Task 1 (CLAUDE.md), line 88; Task 23 (prompt header), lines 1026-1031.

**Root cause**: The CB-2 fix from v3 established that the worker must use `${PROJECT_DIR}`
for shared state. Task 17 (step 6 note, lines 790-793) says: "Worker handoff must use
absolute path... Worker prompt must clearly distinguish relative paths (worktree) from
absolute paths (PROJECT_DIR, shared state)."

But two documents that the worker reads were NOT updated to match:

1. **CLAUDE.md Worker-Specific Rules** (Task 1, line 88):
   ```
   Read VERSION_ID from state.json: `jq -r '.version_id' state.json`
   ```
   This is a **relative path**. In the worktree, `state.json` exists (it's tracked — see
   HP-4 below) but contains the **committed** content: `{"state": "IDLE", "version_id": null}`.
   The worker reads `version_id = null` and passes `--version-id null` to `pipeline.py`,
   which creates `registry/null/` or crashes.

2. **Prompt header** (Task 23, lines 1026-1031):
   ```
   NOTE: Variables like {N}, {batch_id}, ${VERSION_ID} are NOT shell-substituted.
   Read them from state.json at the start of your task:
     jq -r '.iteration' state.json        → N
     jq -r '.batch_id' state.json         → batch_id
     jq -r '.version_id // empty' state.json → VERSION_ID
   ```
   Same issue: all paths are relative. In the worktree, the worker gets `iteration = 0`,
   `batch_id = null`, `version_id = null/empty` — all from the stale committed state.json.

**Trace (iteration 1)**:
```
Step 2:    Controller CAS → ORCHESTRATOR_PLANNING (state.json: iteration=1, version_id=null)
Step 5:    Controller allocates v0001 (state.json: version_id="v0001")
Step 5a:   Controller commits memory/ (state.json NOT committed — it's just modified in CWD)
Step 6:    Worktree created from HEAD → worktree's state.json = committed version:
           {"state": "IDLE", "iteration": 0, "version_id": null}
           Worker reads: iteration=0, batch_id=null, version_id=null
           Worker runs: python ml/pipeline.py --version-id null ...
           → ValueError or creates registry/null/
```

**Impact**: The worker gets the wrong version_id, batch_id, and iteration number. Every
operation that depends on these values fails or corrupts data.

**This is NOT the same as CB-1 or CB-2**: Those found that memory/direction files and
handoff writes needed absolute paths. This finding is about `state.json` itself — the
worker reads stale state metadata because CLAUDE.md and the prompt header use relative
paths for `jq` commands.

**Fix**: Two changes:

**(A) Update CLAUDE.md Task 1 Worker-Specific Rules (line 88)**:
```markdown
- Read VERSION_ID from state.json using the PROJECT_DIR path (NOT the worktree copy):
  `VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")`
  The worktree's state.json is a stale committed copy — always use the absolute path.
```

**(B) Update prompt header in Task 23 — worker-specific version**:
The generic prompt header (lines 1026-1031) works for orchestrator and reviewers (they
run in PROJECT_DIR). But the worker prompt (`worker.md`) must override it:
```
NOTE: You are running in a git worktree. The worktree's state.json is STALE.
Always read state from the PROJECT_DIR copy:
  VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")
  BATCH_ID=$(jq -r '.batch_id' "${PROJECT_DIR}/state.json")
  N=$(jq -r '.iteration' "${PROJECT_DIR}/state.json")
```

The verification plan's R4 fix (line 68-73) already says:
```
VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")
```
But this fix was applied to the implementation plan's worker prompt spec, NOT to CLAUDE.md
or the generic prompt header. The worker reads both CLAUDE.md AND the prompt — if CLAUDE.md
says one thing and the prompt says another, the agent may follow either instruction.

---

### HP-5: Codex `$(cat ...)` shell expansion corrupts literal `$` in prompt

**Where**: Task 19 (`launch_reviewer_codex.sh`), design doc §5.5 (lines 480-501).

**Root cause**: The Codex launch script uses:
```bash
"$(cat ${PROJECT_DIR}/agents/prompts/reviewer_codex.md)"
```

This is inside a double-quoted tmux command string. The `$(cat ...)` is evaluated by the
launching shell, which substitutes ALL `$` characters in the prompt — not just the intended
`${BATCH_ID}`, `${N}`, `${VERSION_ID}`.

Task 23 (lines 1032) confirms this is intentional for the three substitution targets:
```
the $(cat ...) pattern in the Codex launcher already does shell expansion of these vars
```

But the Codex prompt reviews ML code and references bash/Python variables. Likely content
in `reviewer_codex.md`:

```markdown
## Architecture and Code Quality
Check that the worker:
- Used `$VERSION_ID` correctly in pipeline.py
- Did not hardcode `$threshold` values
- Handles `$SMOKE_TEST` env var

## Constraints
- NEVER modify `registry/$VERSION_ID/` for versions other than the current one
```

Each of these `$` characters gets expanded by the shell:
- `$VERSION_ID` → `v0001` (correct for this one, but now the Codex prompt says
  "Used `v0001` correctly" — which is confusingly specific)
- `$threshold` → `` (empty — no such variable in the launcher)
- `$SMOKE_TEST` → `true` or `false` (from config.sh, NOT the intended reference)

The prompt is silently corrupted. Codex reviews code based on a garbled prompt.

**Fix**: Use targeted variable substitution instead of raw `$(cat ...)`:

```bash
# In launch_reviewer_codex.sh: expand ONLY the 3 intended variables
PROMPT_TEXT=$(BATCH_ID="${BATCH_ID}" N="${N}" VERSION_ID="${VERSION_ID}" \
  envsubst '${BATCH_ID} ${N} ${VERSION_ID}' \
  < "${PROJECT_DIR}/agents/prompts/reviewer_codex.md")

tmux new-session -d -s "${SESSION}" \
  "codex exec -m ${CODEX_MODEL} --sandbox read-only --full-auto \
   -C ${PROJECT_DIR} \
   \"${PROMPT_TEXT}\" > ${REVIEW_FILE_ABS} 2>&1 && ..."
```

`envsubst` with an explicit variable list expands ONLY the named variables, leaving all
other `$` characters intact.

**Alternative**: Write the Codex prompt using `{BATCH_ID}` (no `$`) as placeholders and
use `sed` for substitution:
```bash
PROMPT_TEXT=$(sed "s/{BATCH_ID}/${BATCH_ID}/g; s/{N}/${N}/g; s/{VERSION_ID}/${VERSION_ID}/g" \
  "${PROJECT_DIR}/agents/prompts/reviewer_codex.md")
```

Either approach works. The current `$(cat ...)` approach does not.

---

## Medium-Priority Gaps

### MG-4: state.json is tracked by git but should be gitignored

**Where**: Task 2 step 0 (.gitignore), step 4 (state.json creation), step 6 (commit).

**Problem**: state.json is committed in Task 2 step 6 (`git add ... state.json`). It is
NOT in `.gitignore`. But the controller modifies state.json every few seconds during
pipeline execution (every CAS transition). This creates three problems:

1. **Stale copy in worktrees** (root cause of HP-3): Since state.json is tracked, every
   worktree gets the committed version. The committed version is always stale (it was last
   committed during Task 2 with `state: IDLE, version_id: null`). This is the direct
   cause of HP-3 — the worker reads the stale committed copy.

2. **Noisy `git status`**: Every `git status` shows state.json as modified. Every
   `git add` command must be careful to NOT include it. The step 5a command `git add memory/`
   is safe, but if an implementer uses `git add .` or `git add -A`, state.json gets
   committed with runtime state — breaking the next worktree that reads it.

3. **Merge interference**: Although `git merge --no-ff` doesn't touch untracked files, a
   tracked state.json that's modified in the working directory CAN cause merge issues if
   git detects the working directory is dirty during a merge operation.

**Fix**: Add `state.json` to `.gitignore` (Task 2 step 0):
```gitignore
state.lock
state.json
.logs/
...
```

And change Task 2 step 6 to NOT commit state.json:
```bash
git add .gitignore registry/ reports/ reviews/ handoff/
git commit -m "foundation: add .gitignore, registry JSON files, and directory scaffolding"
```

state.json is created by `run_pipeline.sh` at first launch (or by a one-time setup
script). It's purely runtime state — never read from git history.

If state.json is gitignored, worktrees won't have it at all. The worker's `jq -r
'.version_id' state.json` (relative) would fail with "No such file" — a loud error that
forces the worker to use the absolute path. This is defense-in-depth for HP-3.

---

### MG-5: Comparison table output path convention not cross-referenced

**Where**: Task 11 (compare.py CLI), Task 17 step 8, Task 23 (reviewer prompts).

**Problem**: Three components must agree on the comparison table output path:

1. **compare.py** (Task 11): CLI takes `--output path`. The design doc §5.3 (line 382)
   shows: `--output "reports/${BATCH_ID}_iter${N}_comparison.md"`
2. **Controller** (Task 17 step 8): "run compare.py, generate comparison table" — does
   NOT specify the `--output` argument
3. **Reviewer prompts** (Task 23): Must read the comparison table — the READ section
   must reference the same path

The implementation plan says "run compare.py" at step 8 but doesn't give the exact
command. The implementer must infer the output path from the design doc. If the implementer
uses a different path (e.g., `reports/comparison_${BATCH_ID}_iter${N}.md`), the reviewer
prompt won't find it.

**Fix**: Specify the exact command in Task 17 step 8:
```bash
# Step 8: run compare.py
python "${PROJECT_DIR}/ml/compare.py" \
  --batch-id "${BATCH_ID}" \
  --iteration "${N}" \
  --output "${PROJECT_DIR}/reports/${BATCH_ID}_iter${N}_comparison.md"
```

And in Task 23, specify the READ path for reviewer prompts:
```
READ: reports/{batch_id}_iter{N}_comparison.md
```

This ensures the producer and consumer agree on the path.

---

## Observations (not blocking)

### O-5: Watchdog crash detection excludes ORCHESTRATOR_PLANNING and ORCHESTRATOR_SYNTHESIZING

**Where**: Design doc §11.2 (line 1099-1103).

The immediate crash detection only fires for 3 of 5 active states:
```bash
if [[ "$SESSION_ALIVE" == "false" ]] && [[ "$HANDOFF_EXISTS" == "false" ]] \
   && [[ "$state" == "REVIEW_CODEX" || "$state" == "REVIEW_CLAUDE" || "$state" == "WORKER_RUNNING" ]]; then
```

If the orchestrator's tmux session crashes (e.g., Claude CLI OOM), the watchdog does NOT
write a crash artifact immediately. It waits for the full `max_s` timeout (600s for
planning, 600s for synthesis) before the elapsed timeout handler fires.

With HP-1 from v4 fixed (controller handles poll timeouts), this means a crashed
orchestrator delays the batch by up to 10 minutes (poll timeout) instead of being detected
within 2-5 minutes (next watchdog cycle).

**Why the design may have excluded orchestrator states**: The immediate crash detection
also kills the tmux session (line 1112-1114). For the orchestrator, the tmux session IS
the agent, and killing it is appropriate. There's no obvious reason to exclude orchestrator
states from crash detection.

**Suggestion**: Extend the crash detection condition to all 5 active states:
```bash
if [[ "$SESSION_ALIVE" == "false" ]] && [[ "$HANDOFF_EXISTS" == "false" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    echo "{\"timeout\": false, \"crash\": true, ...}" > "$TIMEOUT_FILE"
  fi
fi
```

The `state == "IDLE" || state == "HUMAN_SYNC"` states already exit early (line 1041),
so no guard is needed for those.

### O-6: Worker sandbox in CLAUDE.md allows `handoff/{batch_id}/iter{N}/worker_done.json` — relative path contradicts CB-2

**Where**: Task 1 (CLAUDE.md), line 84.

CLAUDE.md Worker-Specific Rules say:
```
Only modify files under ml/, registry/${VERSION_ID}/, and
handoff/{batch_id}/iter{N}/worker_done.json
```

The `handoff/` path is relative. Per CB-2, the worker must write to
`${PROJECT_DIR}/handoff/...` (absolute). The CLAUDE.md instruction implies the worker can
write to the worktree's `handoff/` — which doesn't exist (gitignored).

An LLM agent following CLAUDE.md literally would attempt `handoff/...` (relative) and
get a "No such file or directory" error, then potentially create the directory in the
worktree. This wastes an attempt and confuses the agent.

**Fix**: Update CLAUDE.md Task 1 line 84:
```
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and
  `${PROJECT_DIR}/handoff/{batch_id}/iter{N}/worker_done.json` (absolute path — handoff/
  is gitignored and does not exist in the worktree)
```

### O-7: Per-Agent Context Slices table references `changes_summary` for reviewers but path not specified

**Where**: Task 1 (CLAUDE.md), line 75.

The Claude Reviewer's required reads include "changes_summary". This refers to
`registry/${VERSION_ID}/changes_summary.md` (written by the worker, per Task 12
line 630). But the context slice table just says "changes_summary" without the full path.

The reviewer prompt (Task 23) must specify:
```
READ: registry/${VERSION_ID}/changes_summary.md
```

Where `${VERSION_ID}` is read from `state.json`. If the reviewer reads the wrong path
(e.g., looking for `changes_summary.md` in the root), it won't find the file.

### O-8: Orchestrator archive directory creation not in prompt spec

**Where**: Task 23 (`orchestrator_synthesize.md`), design doc §9 Tier 3.

At iteration 3, the synthesis orchestrator must write:
- `memory/archive/${batch_id}/executive_summary.md`
- `memory/archive/${batch_id}/experiment_log_full.md`
- `memory/archive/${batch_id}/all_critiques.md`
- Append to `memory/archive/index.md`

The directory `memory/archive/${batch_id}/` does not exist. The orchestrator must create
it (`mkdir -p memory/archive/${batch_id}/`). The orchestrator has Bash access so this is
possible, but the prompt should include the mkdir instruction to avoid a "No such file"
error on the first write.

### O-9: v4 findings status

All 9 findings from v4 remain open (plan not updated since v4):

| v4 ID | Status | Impact |
|-------|--------|--------|
| HP-1 | Open | Poll timeout wedges pipeline permanently |
| HP-2 | Open | git commit fails on empty staging area |
| MG-1 | Open | cas_transition atomicity risk |
| MG-2 | Open | BATCH_NAME breaks tmux |
| MG-3 | Open | Promotion decision unstructured |
| O-1 | Open | Stale worktree cleanup |
| O-2 | Open | VP Tier 2 state expectation |
| O-3 | Open | 600s timeout budget |
| O-4 | N/A | Confirmation of v3 fixes |

---

## Summary Table

| ID | Severity | Category | Summary |
|----|----------|----------|---------|
| HP-3 | High | Cross-doc consistency | CLAUDE.md + prompt header use relative state.json — stale in worktree |
| HP-5 | High | Shell expansion | Codex $(cat) expands ALL $ in prompt, corrupting code references |
| MG-4 | Medium | Git hygiene | state.json tracked but should be gitignored (root cause of HP-3) |
| MG-5 | Medium | Cross-doc consistency | compare.py output path not specified in step 8 or reviewer prompts |
| O-5 | Note | Watchdog | Crash detection excludes ORCHESTRATOR_* states (delayed detection) |
| O-6 | Note | Cross-doc consistency | CLAUDE.md worker sandbox uses relative handoff/ path, contradicts CB-2 |
| O-7 | Note | Prompt completeness | Reviewer context slice says "changes_summary" but no path specified |
| O-8 | Note | Prompt completeness | Orchestrator archive mkdir -p not in iter 3 prompt spec |
| O-9 | Note | Prior reviews | All v4 findings remain open |

**Fix before implementation**: HP-3, HP-5 (plus HP-1, HP-2, MG-3 from v4)
**Should fix before implementation**: MG-4, MG-5
**Address during implementation**: O-5, O-6, O-7, O-8

---

## Recommended Fix Order

1. **HP-3 + O-6** (5 minutes): Update CLAUDE.md Worker-Specific Rules to use
   `${PROJECT_DIR}/state.json`. Update worker prompt header to use absolute state.json
   path. Update CLAUDE.md worker sandbox constraint to use absolute handoff path.
   These share the same root cause (CB-2 fix not propagated to CLAUDE.md).

2. **MG-4** (2 minutes): Add `state.json` to `.gitignore`. Remove `state.json` from
   Task 2 step 6 commit. This provides defense-in-depth for HP-3 (worktree won't have
   a stale copy at all).

3. **HP-5** (3 minutes): Replace `$(cat ...)` with `envsubst` in Codex launcher spec
   (Task 19). Update design doc §5.5 to match.

4. **MG-5** (2 minutes): Add exact compare.py command to Task 17 step 8. Add READ path
   for comparison table in Task 23 reviewer prompt specs.

---

## Verdict

**FIX HP-3 AND HP-5 BEFORE IMPLEMENTING** — HP-3 causes the worker to read stale state
metadata from the worktree's committed state.json, giving it the wrong version_id,
batch_id, and iteration number. Every worker operation fails or corrupts data. HP-5 causes
the Codex reviewer to receive a garbled prompt with unintended shell expansions.

Both are consequences of the v3 worktree isolation fixes being partially applied: the
implementation plan was updated (step 6 notes, handoff paths) but CLAUDE.md and prompt
headers — which the worker agent ALSO reads — were not updated to match. The fix is to
propagate the absolute-path convention to all documents the worker reads.

Combined with v4's open HP-1 and HP-2, there are now **4 high-priority bugs** that should
be fixed before implementation. All are small, targeted fixes (total estimated: ~15 minutes
of plan editing).

After these fixes, the plan is ready for implementation.

---

## Cross-Review Lineage

| Review | Findings | Status |
|--------|----------|--------|
| 2026-02-26 (initial) | 3 spec bugs, 5 gaps, 4 structural | All resolved in plan |
| v2 (2026-02-27) | 3 spec bugs, 4 gaps, 3 impl risks, 3 obs | All resolved in plan |
| v3 (2026-02-27) | 4 critical, 3 high, 1 medium, 4 obs | All resolved in plan |
| v4 (2026-02-27) | 2 high, 3 medium, 4 obs | **Open (plan not updated)** |
| **v5 (2026-02-27)** | **2 high, 2 medium, 5 obs** | **Current** |

Total unique findings across all reviews: 55 (12 + 13 + 12 + 9 + 9).
Resolved in plan: 37.
Open (v4 + v5): 18 (9 + 9).

---

## Diminishing Returns Assessment

After 5 review passes, the remaining findings are increasingly subtle (cross-document
consistency, shell expansion edge cases, git tracking hygiene). The core architecture,
state machine, ML pipeline, and test infrastructure are sound. The open findings are all
fixable with targeted plan edits totaling ~20 minutes.

**Recommendation**: Fix the 4 high-priority bugs (HP-1, HP-2, HP-3, HP-5) and the 2
highest-impact medium gaps (MG-3, MG-4), then proceed to implementation. Remaining medium
and low findings can be addressed during implementation as the implementer encounters them.
Further review passes are unlikely to find issues of comparable severity.
