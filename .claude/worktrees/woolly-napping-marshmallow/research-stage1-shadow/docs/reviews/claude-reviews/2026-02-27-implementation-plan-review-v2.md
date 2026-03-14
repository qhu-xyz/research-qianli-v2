# Implementation Plan Review (v2)

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`
**Cross-referenced**: Design doc v10, Verification plan, prior review (2026-02-26)
**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-27

---

## Executive Summary

The plan has been revised since the prior review (2026-02-26). All 12 findings from that review (3 spec bugs, 5 gaps, 4 structural concerns) are now incorporated: synthetic_labels is 100, PipelineConfig includes registry_dir, SMOKE_TEST has a fixed seed, tests exist for data_loader/features/compare, Step 18 is specified, WORKER_FAILED injection is in the launch script spec, .gitignore covers transient artifacts, Task 17 has a code skeleton, poll_for_handoff checks timeouts, and worktree cleanup is in run_pipeline.sh.

This review focuses on **new issues** found in the current plan. I found **3 spec bugs** (will cause failures), **4 design-level gaps** (ambiguities between the plan and the design doc), and **3 implementation risks** (things likely to trip up the implementer).

**Recommendation**: Fix SB-1 (state name) and SB-2 (step ordering) in the skeleton before implementation. SB-3 is a design doc bug to fix in both documents. The gaps and risks can be resolved during implementation if the implementer is aware of them.

---

## Spec Bugs (will cause failures)

### SB-1: Skeleton uses non-existent REVIEWING state

**Where**: Task 17, step skeleton (lines 801, 805):
```
# Step 8:  cas_transition WORKER_RUNNING -> REVIEWING
...
# Step 12: cas_transition REVIEWING -> ORCHESTRATOR_SYNTHESIZING
```

**Problem**: The state machine defined in design doc Section 3 has six pipeline states: `ORCHESTRATOR_PLANNING`, `WORKER_RUNNING`, `REVIEW_CLAUDE`, `REVIEW_CODEX`, `ORCHESTRATOR_SYNTHESIZING`, `HUMAN_SYNC`. There is no `REVIEWING` state. Every infrastructure component uses the two-state model:

- `get_expected_artifact()` (design doc line 770-778): separate cases for REVIEW_CLAUDE and REVIEW_CODEX
- `STATE_TO_HANDOFF` (design doc line 1086-1092): separate entries for REVIEW_CLAUDE and REVIEW_CODEX
- Watchdog case statement (design doc line 1048-1059): separate cases for REVIEW_CLAUDE and REVIEW_CODEX
- Verification plan tests 3.2 and 3.3: test both REVIEW_CLAUDE and REVIEW_CODEX explicitly

If the implementer follows the skeleton and creates a single REVIEWING state, `get_expected_artifact()`, `STATE_TO_HANDOFF`, and the watchdog will all fail to match it.

**Fix**: Replace the skeleton with the design doc's step numbering. The two review states need separate CAS transitions:
```bash
# Step 8:  run compare.py, generate comparison table
# Step 9:  cas_transition WORKER_RUNNING -> REVIEW_CLAUDE
# Step 10: launch Claude reviewer, poll for handoff
# Step 11: verify Claude review sha256
# Step 12: cas_transition REVIEW_CLAUDE -> REVIEW_CODEX
# Step 13: launch Codex reviewer, poll for handoff (with timeout)
# Step 14: (poll complete or timeout detected)
# Step 15: cas_transition REVIEW_CODEX -> ORCHESTRATOR_SYNTHESIZING
```

**Severity**: Will cause runtime failures in state_utils.sh and watchdog.sh.

---

### SB-2: Compare.py runs after state transition instead of before

**Where**: Task 17, step skeleton (lines 801-802):
```
# Step 8:  cas_transition WORKER_RUNNING -> REVIEWING
# Step 9:  run compare.py, generate comparison table
```

**Problem**: The design doc Section 6.2 specifies the opposite order:
```
8. Run ml/compare.py (non-AI comparison table). sha256 already verified in step 7c.
9. CAS: state = REVIEW_CLAUDE, max_s = 1200.
```

Compare.py is a non-AI deterministic script that should run while the state is still `WORKER_RUNNING`. Transitioning to a review state first has two consequences:

1. **Watchdog confusion**: In the `REVIEW_CLAUDE` state, the watchdog expects a Claude reviewer tmux session to exist (it reads `claude_reviewer_tmux` from state.json). But compare.py runs as a synchronous subprocess, not a tmux session. If the watchdog fires while compare.py is running, it will see no tmux session and may write a spurious crash artifact.

2. **Timeout tracking**: The max_seconds for the new state starts ticking at the CAS transition. If compare.py takes 30 seconds and the Claude reviewer gets 1200s, the reviewer effectively has 1170s. Minor, but semantically wrong.

**Fix**: Swap steps 8 and 9 to match the design doc: compare.py runs first, then CAS transition.

---

### SB-3: Watchdog elapsed-timeout handler ignores existing handoff files

**Where**: Design doc Section 11.2 (lines 1107-1116), inherited by Task 20.

**Problem**: The watchdog has two detection paths:

1. **Crash detection** (line 1099-1103): Checks `SESSION_ALIVE=false AND HANDOFF_EXISTS=false` -- correctly guarded.
2. **Elapsed timeout** (line 1107-1110): Checks only `elapsed > max_s AND ! -f TIMEOUT_FILE` -- **no check for HANDOFF_EXISTS**.

The verification plan Section 3.7 tests exactly this scenario:
```
Scenario: agent completed (handoff exists) but tmux session already exited.
Expected: watchdog must NOT write a timeout/crash artifact.
```

In this test, `entered_at` is 1 hour ago, `max_seconds` is 1800. So `elapsed` (3600) > `max_s` (1800). The crash detection correctly does nothing (HANDOFF_EXISTS=true). But the elapsed timeout handler fires unconditionally and writes a timeout artifact -- violating the test's expected outcome.

This means the VP test 3.7 will **fail** if implemented exactly as the design doc specifies the watchdog. The controller's `poll_for_handoff()` would then see both the normal handoff AND a spurious timeout artifact. Since `poll_for_handoff()` returns different exit codes for each (0 vs 1), this creates a race condition on which file is detected first.

**Fix**: Add HANDOFF_EXISTS guard to the elapsed timeout handler (fix in both design doc and implementation plan):
```bash
if (( elapsed > max_s )) && [[ "$HANDOFF_EXISTS" == "false" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    echo "{...}" > "$TIMEOUT_FILE"
  fi
  ...
fi
```

**Severity**: VP test 3.7 will fail; spurious timeout artifacts in production.

---

## Design-Level Gaps

### DG-1: Missing REVIEW_CLAUDE -> REVIEW_CODEX transition changes the observable state machine

**Where**: Task 17 skeleton, between steps 10 and 11.

**Problem**: The design doc has Claude and Codex reviews as sequential states with an explicit CAS transition between them (step 12: `CAS: state = REVIEW_CODEX`). The implementation plan's skeleton runs them back-to-back under the single "REVIEWING" pseudo-state with no intermediate transition.

This matters for:
- **Watchdog**: Needs to know which reviewer to probe. Under REVIEW_CLAUDE it checks `claude_reviewer_tmux`; under REVIEW_CODEX it checks `codex_reviewer_tmux`. Without separate states, it can't distinguish which session should be alive.
- **Timeout isolation**: Claude reviewer and Codex reviewer each get 1200s. With separate states, each timeout starts fresh at the CAS transition. Without them, a slow Claude reviewer steals time from the Codex reviewer.
- **Auditability**: The history array in state.json should show each state transition for debugging. Losing the intermediate transition makes post-mortem harder.

**Recommendation**: The skeleton must include the intermediate CAS transition. See SB-1 fix above.

---

### DG-2: Promotion logic -- who calls promote_version()?

**Where**: Task 17, Step 15 ("apply synthesis decisions (promote, update memory)"); ml/registry.py `promote_version()`.

**Problem**: The design doc never explicitly states who runs promotion. The pieces:
- `registry.py` has `promote_version(registry_dir, version_id, champion_path)` (Task 12)
- `compare.py` has `check_gates()` which returns per-gate pass/fail (Task 11)
- The orchestrator synthesis phase reads reviews and comparison tables
- Step 15 of the skeleton says "apply synthesis decisions (promote, update memory)"

Two possible designs:

**(A) Orchestrator decides, controller executes**: The synthesis orchestrator writes a handoff with `"promote": true/false`. The controller reads this and calls `promote_version()` from the shell (via `python -c "from ml.registry import promote_version; ..."`). This preserves the single-writer rule for registry files.

**(B) Orchestrator decides and executes**: The synthesis orchestrator calls `promote_version()` directly from its Claude session. This is simpler but means the orchestrator modifies `registry/champion.json`, which is outside the worker sandbox but inside the orchestrator's broader permissions.

The design doc's orchestrator constraints say "Do NOT modify any ML code or registry/ files" (Section 10.1). This prohibits option (B). But Step 15 says the controller applies promotion decisions, implying option (A) without specifying the mechanism.

**Recommendation**: Specify in Task 17 step 15:
```bash
# Step 15: read orchestrator synthesis handoff
# If synthesis recommends promotion: python -c "from ml.registry import promote_version; ..."
# Update memory/hot/champion.md with new champion metrics
```

Also clarify the handoff schema for synthesis: does `orchestrator_synth_done.json` include a `promote_version_id` field?

---

### DG-3: Worker failure skip range is ambiguous in the skeleton

**Where**: Task 17, skeleton step 7a:
```
# Step 7a: check worker handoff -- if failed, WORKER_FAILED=1, skip 7b-7d
```

**Problem**: The skeleton says "skip 7b-7d" but the text above (line 777) says "transition directly to step 15 (ORCHESTRATOR_SYNTHESIZING)". The design doc Section 8 says: "skip artifact sync (steps 7b-7c), skip comparison (step 8), write failure note to `memory/hot/progress.md`, transition directly to `ORCHESTRATOR_SYNTHESIZING`."

So the actual skip range is steps 7b through 14 (merge, compare, Claude review, Codex review). The skeleton only annotates "skip 7b-7d" in the step 7a comment, which could mislead the implementer into thinking steps 8-14 still run on the failure path.

**Recommendation**: Update the skeleton comment:
```bash
# Step 7a: check worker handoff -- if failed, WORKER_FAILED=1, jump to step 15
#          (skip 7b-7d merge, steps 8-14 compare+reviews)
```

---

### DG-4: Ray init in real-data mode missing local `ml` package

**Where**: Task 6 (data_loader.py, real-data branch).

**Problem**: The design doc Section 14 specifies:
```python
import ml as shadow_ml
init_ray(address=RAY_ADDRESS, extra_modules=[pmodel, shadow_ml])
```

The implementation plan's Task 6 doesn't mention including the local `ml` package in `extra_modules`. If `ml` code is used in Ray remote functions, the package won't be serialized to the cluster, causing `ModuleNotFoundError` on workers.

For SMOKE_TEST mode this doesn't matter (no Ray). But for real-data mode, the omission would cause silent failures the first time the pipeline runs against real data.

**Recommendation**: Add to Task 6's real-data branch specification:
```
Real branch: ... Ray init with extra_modules=[pmodel, ml] ...
```

---

## Implementation Risks

### IR-1: test_pipeline.py missing registry scaffolding

**Where**: Task 13, test code (lines 651-658):
```python
config = PipelineConfig(version_id="v0001", registry_dir=str(tmp_path / "registry"))
(tmp_path / "registry").mkdir()
metrics = run_pipeline(config)
```

**Problem**: `run_pipeline()` calls `register_version()` which needs:
- `registry/version_counter.json` (for `allocate_version_id()` -- though version_id is passed in, this may still be read)
- `registry/gates.json` (if compare.py is called within the pipeline, or if GateConfig is loaded)
- `registry/champion.json` (if comparison references the champion)

The test only creates the empty `registry/` directory. Depending on how `run_pipeline()` is implemented, it may fail with `FileNotFoundError` on any of these JSON files.

**Recommendation**: Either:
- (a) `run_pipeline()` should only need PipelineConfig fields (no global registry reads) -- comparison and promotion happen outside the pipeline
- (b) The test should scaffold all required JSON files:
```python
(tmp_path / "registry" / "version_counter.json").write_text('{"next_id": 1}')
(tmp_path / "registry" / "gates.json").write_text('{"version": 1, "gates": {}}')
(tmp_path / "registry" / "champion.json").write_text('{"version": null}')
```

Option (a) is cleaner -- `pipeline.py` should handle load/train/eval/register, while `compare.py` runs separately as a controller step.

---

### IR-2: test_pipeline.py uses module-level os.environ mutation

**Where**: Task 13, test code (lines 648-649):
```python
import os
os.environ["SMOKE_TEST"] = "true"
```

**Problem**: This mutates the process environment at import time, persisting across all tests in the pytest session. If a test file is imported but never cleaned up, other tests that should NOT run in SMOKE_TEST mode will silently get synthetic data. This is especially dangerous because `conftest.py` is loaded early and could interact with this.

**Recommendation**: Use pytest's `monkeypatch` fixture or `mock.patch.dict`:
```python
@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")
```

Or scope the env var in conftest.py for the entire test suite (since all ML tests run in SMOKE_TEST mode anyway).

---

### IR-3: No dry-run test for launch_worker.sh

**Where**: Task 19, test specs.

**Problem**: Task 19 says "All must support `--dry-run`" and provides test cases for `launch_orchestrator.sh` (test_arg_parser.sh, VP 3.4), `launch_reviewer_claude.sh` (VP 3.6), and `launch_reviewer_codex.sh` (VP 3.6). But there are no dry-run tests for `launch_worker.sh`.

The worker launcher has a unique side effect: `git worktree add`. In `--dry-run` mode, this should be skipped (creating a worktree is not idempotent and leaves state behind). Without a test, the dry-run flag may be implemented for the other three launchers but forgotten for the worker.

**Recommendation**: Add to test_arg_parser.sh:
```bash
# launch_worker.sh --dry-run should NOT create a worktree
OUTPUT=$(bash agents/launch_worker.sh --dry-run 2>&1)
echo "$OUTPUT" | grep -q "git worktree add" || echo "FAIL: worker dry-run missing worktree command"
[[ ! -d "${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}" ]] \
  || echo "FAIL: worker dry-run created actual worktree"
```

---

## Observations (not blocking)

### O-1: Inconsistent TDD application

Tasks 5, 8, 10, 12, 13 follow TDD (write failing test, implement, verify). Tasks 6, 7, 11 implement first, then add tests. This is a pragmatic choice (data_loader and features are porting tasks where the shape of the code is known), but worth noting for the implementer: don't skip the test step on non-TDD tasks.

### O-2: Prior review findings all resolved

The 2026-02-26 review's 12 findings (SB-1 through SC-4) are all incorporated in the current plan. Specifically:
- SB-1 (labels size): now 100 at line 343
- SB-2 (registry_dir): now in PipelineConfig at line 337
- SB-3 (fixed seed): now specified at line 371
- G-1 (data_loader/features tests): Task 7 Step 2a, lines 406-409
- G-2 (compare.py tests): Task 11 Step 1a, lines 581-587
- G-3 (Step 18 logic): Task 17, lines 779-786
- G-4 (WORKER_FAILED injection): Task 19, lines 873-880
- G-5 (.gitignore): Task 2 Step 0, lines 128-135
- SC-1 (Task 17 skeleton): lines 789-812
- SC-3 (poll_for_handoff timeout): Task 16, line 744
- SC-4 (worktree cleanup): Task 18, lines 846-852

### O-3: Codex launcher quoting

The Codex launcher in design doc Section 5.5 nests shell quoting three levels deep (tmux string > codex exec > `$(cat ...)` with shell expansion of `$BATCH_ID`, `$N`, `$VERSION_ID`). This is the single most error-prone shell command in the entire pipeline. The dry-run test (VP 3.6) helps but only checks the final command string. Consider adding a guard in the launcher:
```bash
[[ -n "$BATCH_ID" && -n "$N" && -n "$VERSION_ID" ]] \
  || { echo "ERROR: BATCH_ID, N, VERSION_ID must be set"; exit 1; }
```

---

## Summary Table

| ID | Severity | Category | Summary |
|----|----------|----------|---------|
| SB-1 | Bug | Task 17 skeleton | Non-existent REVIEWING state; should be REVIEW_CLAUDE + REVIEW_CODEX |
| SB-2 | Bug | Task 17 skeleton | compare.py runs after state transition; should run before |
| SB-3 | Bug | Design doc + Task 20 | Watchdog elapsed-timeout ignores existing handoff; VP 3.7 will fail |
| DG-1 | Gap | Task 17 | Missing REVIEW_CLAUDE -> REVIEW_CODEX CAS transition |
| DG-2 | Gap | Task 17 step 15 | Promotion logic flow unspecified (who calls promote_version?) |
| DG-3 | Gap | Task 17 step 7a | Worker failure skip range unclear (says 7b-7d, actually 7b-14) |
| DG-4 | Gap | Task 6 | Ray init missing local ml package in extra_modules for real-data mode |
| IR-1 | Risk | Task 13 test | test_pipeline.py missing registry JSON scaffolding |
| IR-2 | Risk | Task 13 test | Module-level os.environ mutation leaks across tests |
| IR-3 | Risk | Task 19 tests | No dry-run test for launch_worker.sh |
| O-1 | Note | Plan-wide | Inconsistent TDD application (pragmatic, not blocking) |
| O-2 | Note | Prior review | All 12 prior findings resolved in current plan |
| O-3 | Note | Task 19 | Codex launcher quoting complexity -- add env var guards |

**Blocking**: SB-1, SB-2 (skeleton fixes, straightforward); SB-3 (design doc + watchdog fix)
**Should fix before implementation**: DG-1, DG-2, DG-3
**Address during implementation**: DG-4, IR-1, IR-2, IR-3

---

## Verdict

**PROCEED WITH FIXES** -- The plan is mature and well-structured. The prior review's findings are fully resolved. Fix the skeleton's state names (SB-1/DG-1) and step ordering (SB-2) -- these are the same root issue (the skeleton diverges from the design doc's step numbering). Fix the watchdog handoff guard (SB-3) in both the design doc and implementation plan. Then implement.
