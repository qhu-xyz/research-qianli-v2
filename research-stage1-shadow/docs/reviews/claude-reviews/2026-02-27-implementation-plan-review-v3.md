# Implementation Plan Review (v3)

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`
**Cross-referenced**: Design doc v10, Verification plan, prior reviews (2026-02-26 + v2)
**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-27

---

## Executive Summary

Both prior reviews' findings (12 from 2026-02-26, 13 from v2) have been incorporated
into the current plan. SB-1/SB-2/SB-3/DG-1 through DG-4 from v2 are all resolved.
The skeleton now uses REVIEW_CLAUDE and REVIEW_CODEX, compare.py runs before the CAS
transition, the watchdog HANDOFF_EXISTS guard is added, and launch scripts validate env
vars.

This review identifies **4 critical bugs** that all prior reviews missed. They share a
root cause: **the worktree isolation model is incompatible with the shared-state model
for memory and handoff files.** Additionally, **2 CAS state transition mismatches** will
prevent iterations 2-3 from starting.

These 6 issues will cause the pipeline to fail on its very first run. They must be fixed
in the plan before implementation.

---

## Critical Bugs (pipeline will not function)

### CB-1: Worker cannot read orchestrator's direction file from its worktree

**Where**: Task 17 step 6 (launch worker), Task 23 (worker prompt), design doc Section 5.2.

**Root cause**: The orchestrator writes `memory/direction_iter{N}.md` to `${PROJECT_DIR}` in
step 3, but this write is **never committed**. The worker's worktree is created from HEAD in
step 6 via `git worktree add`. Uncommitted files in the main working directory do not appear
in worktrees.

**Trace**:
```
Step 3:  Orchestrator writes ${PROJECT_DIR}/memory/direction_iter1.md (uncommitted)
Step 4:  CAS → WORKER_RUNNING
Step 5:  Allocate version_id
Step 6:  git worktree add .claude/worktrees/iter1-batch123 -b iter1-batch123
         → Worktree starts from HEAD. direction_iter1.md is NOT in HEAD. File is absent.
Step 6:  Worker starts in worktree, prompt says "Read memory/direction_iter1.md"
         → File not found. Worker has no instructions.
```

**Same issue affects**:
- `memory/hot/progress.md` (updated by orchestrator, uncommitted)
- `memory/warm/decision_log.md` (appended by orchestrator, uncommitted)
- For iteration 2+: `memory/hot/champion.md` (updated by synthesis, uncommitted)
- For iteration 2+: `memory/hot/learning.md` (updated by synthesis, uncommitted)

The only memory files the worker can reliably read are the **initial committed stubs** from
Task 3. After any orchestrator phase modifies them without committing, they become invisible
to subsequent worktrees.

**Fix**: Two options (pick one):

**(A) Controller commits orchestrator outputs before creating the worktree** (recommended):
Add to `run_single_iter.sh` between steps 5 and 6:
```bash
# Step 5a: commit orchestrator outputs so worktree inherits them
git -C "${PROJECT_DIR}" add memory/ && \
  git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator plan"
```
Then the worktree, created from the updated HEAD, has the direction file and all memory
updates. This adds ~1 commit per iteration to the git history (3 per batch + worker
merges = ~6 per batch total), but ensures correctness.

**(B) Worker reads shared files from PROJECT_DIR using absolute paths**:
Change the worker prompt to use `${PROJECT_DIR}` for all READ paths:
```
READ:
- ${PROJECT_DIR}/memory/direction_iter{N}.md
- ${PROJECT_DIR}/memory/hot/champion.md
- ${PROJECT_DIR}/memory/hot/learning.md
- ${PROJECT_DIR}/memory/hot/runbook.md
```
The worker agent has Bash access and can read any path. This avoids extra commits but
makes the prompt depend on absolute paths.

Option (A) is more robust because it doesn't rely on the Claude agent correctly resolving
mixed absolute/relative paths.

---

### CB-2: Worker handoff writes are invisible to the controller

**Where**: Task 17 step 7 (poll for worker handoff), Task 2 (.gitignore), design doc Section 5.2.

**Root cause**: `handoff/` is in `.gitignore` (Task 2, line 131). Gitignored directories do
not exist in worktrees. The worker writes `handoff/{batch_id}/iter{N}/worker_done.json`
using a relative path (design doc Section 5.2 step 7). Since the worker CWD is the worktree,
this creates:

```
${WORKTREE}/handoff/${BATCH_ID}/iter${N}/worker_done.json
```

The controller polls:
```
${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}/worker_done.json
```

These are different filesystem paths. The controller never sees the handoff. It polls
forever until the watchdog timeout fires.

**Same issue affects the failure path**: If the worker fails and writes `status: "failed"` to
the handoff JSON, the controller still can't see it. The WORKER_FAILED path never triggers.

**Fix**: The worker prompt must use `${PROJECT_DIR}` for the handoff write:
```
WRITE handoff:
  ${PROJECT_DIR}/handoff/{batch_id}/iter{N}/worker_done.json
```

If using CB-1 fix option (A), the handoff still needs an absolute path because `handoff/` is
gitignored and won't exist in the worktree even after committing memory files.

The worker prompt should clearly distinguish:
- **Relative paths** (worktree, committed via git): `ml/`, `registry/${VERSION_ID}/`
- **Absolute paths** (PROJECT_DIR, shared state): `${PROJECT_DIR}/memory/...`,
  `${PROJECT_DIR}/handoff/...`, `${PROJECT_DIR}/state.json`

---

### CB-3: Step 18 CAS conflict -- iterations 2 and 3 cannot start

**Where**: Task 17, step 18 (line 788) and step 1 (line 796).

**Conflict**:
```
Step 18 (N<3):  cas_transition ORCHESTRATOR_SYNTHESIZING → ORCHESTRATOR_PLANNING
Step 1 (next):  cas_transition IDLE → ORCHESTRATOR_PLANNING
```

After iteration 1's step 18, the state is `ORCHESTRATOR_PLANNING`. Iteration 2's step 1
does `cas_transition IDLE → ORCHESTRATOR_PLANNING`. The CAS check finds the current state
is `ORCHESTRATOR_PLANNING`, not `IDLE`. Mismatch. **Abort.**

Iterations 2 and 3 can never run. The pipeline completes exactly one iteration and then
deadlocks.

**Fix**: Change step 18 to transition to `IDLE` for N<3:
```bash
if (( N == 3 )); then
  cas_transition ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC '{}'
else
  cas_transition ORCHESTRATOR_SYNTHESIZING IDLE '{}'
fi
```

This is the cleaner option because:
- Step 1 always starts from `IDLE` (single expected state, not conditional)
- The brief IDLE window between iterations is invisible (flock held by run_pipeline.sh)
- The watchdog exits immediately for IDLE (`[[ "$state" == "IDLE" ]] && exit 0`)

The design doc Section 6.2 lists "IDLE" as an option for step 18: "CAS: state = IDLE
(or ORCHESTRATOR_PLANNING for next iter, or HUMAN_SYNC after iter 3)". The plan chose
ORCHESTRATOR_PLANNING without updating step 1 to match.

---

### CB-4: HUMAN_SYNC to first iteration CAS mismatch

**Where**: Task 18 (run_pipeline.sh), Task 17 step 1.

**Conflict**: `run_pipeline.sh` accepts `HUMAN_SYNC` as a valid starting state (design
doc Section 12.3):
```bash
if [[ "$current_state" != "IDLE" && "$current_state" != "HUMAN_SYNC" ]]; then
  echo "Cannot start"; exit 1
fi
```

But it does not reset the state to `IDLE` before entering the iteration loop.
`run_single_iter.sh` step 1 does `cas_transition IDLE → ORCHESTRATOR_PLANNING`. If the
state is `HUMAN_SYNC`, the CAS check fails. **Starting a new batch after a completed
3-iteration batch always fails.**

**Fix**: Add state reset in `run_pipeline.sh` before the loop:
```bash
if [[ "$current_state" == "HUMAN_SYNC" ]]; then
  source agents/state_utils.sh
  cas_transition HUMAN_SYNC IDLE '{"batch_id":null,"iteration":0}'
fi
```

---

## High-Priority Gaps

### HG-1: --overrides mechanism is unspecified

**Where**: Task 13 (pipeline.py), Task 23 (worker prompt Section 10.2).

**Problem**: The pipeline CLI takes `--overrides '{"n_estimators": 300}'` and the worker
prompt says to use it, but the plan never specifies:

1. **What can be overridden**: Only HyperparamConfig fields? FeatureConfig (add/remove
   features)? ThresholdConfig (beta, scaling_factor)? PipelineConfig fields?
2. **How overrides are applied**: JSON merge into a specific dataclass? Nested keys?
   What happens on invalid keys?
3. **Whether overrides are recorded**: For reproducibility, `registry/${VERSION_ID}/config.json`
   should include both the base config and the applied overrides. Otherwise you can't
   reproduce a specific version's training run.

**Risk**: Without specification, the implementer will make arbitrary choices that may not
match what the orchestrator/worker prompts expect. The orchestrator plans changes like
"increase n_estimators to 300" and writes them in the direction file. The worker translates
this to `--overrides '{"n_estimators": 300}'`. If the override mechanism doesn't support
the field, the change is silently ignored and the iteration is wasted.

**Recommendation**: Add to Task 13:
```
--overrides accepts a flat JSON dict. Keys are matched against HyperparamConfig fields
first, then PipelineConfig fields (threshold_beta, scale_pos_weight_auto). Unknown keys
cause a ValueError. The resolved config (base + overrides) is written to
registry/${VERSION_ID}/config.json for reproducibility.
```

---

### HG-2: SMOKE_TEST data generator missing actual_shadow_price column

**Where**: Task 6 (data_loader.py), Task 7 (features.py `compute_binary_labels`).

**Problem**: `compute_binary_labels(df, threshold=0.0)` uses `actual_shadow_price > threshold`
to create binary labels. The SMOKE_TEST data generator is specified as "100 rows, 14 features
+ label + metadata columns" (Task 6, line 374) but does not explicitly name `actual_shadow_price`
as a required column.

Similarly, `evaluate_classifier()` (Task 10) takes a `shadow_prices` argument for Value Capture
metrics. The pipeline needs to extract this from the DataFrame.

If the implementer generates synthetic data without an `actual_shadow_price` column (using
just a binary `label`), both `compute_binary_labels()` and the evaluate step will fail.

**Fix**: Specify in Task 6 SMOKE_TEST branch:
```
Columns: 14 feature columns (from FeatureConfig) + actual_shadow_price (continuous,
lognormal ~ mean=50, mostly 0 for non-binding) + constraint_id + auction_month
```

And in Task 7:
```
compute_binary_labels uses actual_shadow_price > threshold (not a pre-existing binary label)
```

---

### HG-3: test_evaluate.py uses degenerate shadow price data

**Where**: Task 10, test code (line 538):
```python
fake_sp = synthetic_labels.astype(float) * 100  # fake shadow prices
```

**Problem**: This creates shadow prices of exactly 0 or 100. With only two distinct SP
values:

- **VCAP@K**: If K exceeds the number of binding cases (~7 of 100), the metric degenerates.
  All binding cases have SP=100, all non-binding have SP=0. The "value" dimension collapses
  to a count, making VCAP identical to recall.
- **NDCG**: With binary relevance (0 or 100), NDCG degenerates to binary NDCG. The graded
  relevance signal (which catches model miscalibration) is not tested.
- **CAP@K**: With identical SP values for all binding cases, the "top K by actual SP" set is
  arbitrary among binding cases. The metric tests list membership, not ranking quality.

The test verifies that metric keys exist in the output dict, but cannot catch computation
bugs in the ranking metric implementations (e.g., off-by-one in top-K selection, incorrect
normalization in NDCG).

**Risk**: evaluate.py is HUMAN-WRITE-ONLY. Once written and committed, bugs in ranking
metrics persist until a human manually reviews the implementation. The degenerate test data
provides no safety net.

**Fix**: Use continuous shadow prices in the test:
```python
rng = np.random.RandomState(42)
fake_sp = np.where(synthetic_labels == 1,
                   rng.lognormal(mean=3, sigma=1.5, size=synthetic_labels.shape),
                   0.0)
```

And add a basic sanity check:
```python
# VCAP@100 should be 1.0 when there are fewer than 100 binding cases
# and the model ranks all binding cases in its top 100
assert metrics["S1-VCAP@100"] > 0  # at minimum, some value is captured
```

---

## Remaining from v2 (still unfixed)

### IR-1: test_pipeline.py missing registry scaffolding

**Where**: Task 13 test code.

**Status**: Still only creates `(tmp_path / "registry").mkdir()`. If `run_pipeline()` reads
`gates.json`, `champion.json`, or `version_counter.json`, the test fails with
`FileNotFoundError`. Since Task 13 step 2 says "pipeline.py does NOT call compare.py",
the minimum scaffolding is `version_counter.json` (for `register_version()`):

```python
(tmp_path / "registry").mkdir()
(tmp_path / "registry" / "version_counter.json").write_text('{"next_id": 1}')
```

---

## Observations (not blocking)

### O-1: No pre-merge guard for HUMAN-WRITE-ONLY files

**Where**: Task 17 step 7d (merge worker branch).

The worker sandbox constraint says "NEVER modify evaluate.py or gates.json." This is
enforced by the prompt, not by the merge process. If the worker agent ignores the
constraint (LLM agents are not deterministic), the merge brings the change into main
silently.

Defense-in-depth: add a pre-merge check in step 7d:
```bash
# Verify HUMAN-WRITE-ONLY files unchanged on worker branch
for PROTECTED in ml/evaluate.py registry/gates.json; do
  MAIN_SHA=$(git -C "${PROJECT_DIR}" show HEAD:"${PROTECTED}" | sha256sum | cut -d' ' -f1)
  BRANCH_SHA=$(git -C "${PROJECT_DIR}" show "iter${N}-${BATCH_ID}:${PROTECTED}" | sha256sum | cut -d' ' -f1)
  [[ "$MAIN_SHA" == "$BRANCH_SHA" ]] || { echo "ERROR: worker modified protected file: ${PROTECTED}"; exit 1; }
done
```

### O-2: --from-phase crash recovery is untested and underspecified

**Where**: Task 13 (pipeline.py).

The plan says pipeline.py saves intermediates to parquet between phases and supports
`--from-phase N`, but:
- No test exercises `--from-phase`
- Phase numbers are not defined (which phase is 1? 2?)
- Intermediate parquet file paths are not specified (where does each phase save?)

If `--from-phase` breaks, crash recovery during real runs fails silently (the pipeline
restarts from scratch, wasting 5-30 minutes per attempt). This is acceptable for v0 but
should be specified before real-data runs.

### O-3: Synthesis orchestrator commit needed for iteration 2+ memory files

Even after applying the CB-1 fix (commit orchestrator plan outputs), the **synthesis
orchestrator** (step 16) also writes memory updates:
- `memory/hot/critique_summary.md`
- `memory/hot/gate_calibration.md`
- `memory/warm/experiment_log.md`
- `memory/warm/hypothesis_log.md`
- `memory/warm/decision_log.md`
- `memory/direction_iter{N+1}.md` (for N<3)

These are also uncommitted. For iteration 2's worker to read the updated critique_summary
or the new direction file, the controller needs a second commit after synthesis:
```bash
# Step 17a: commit synthesis outputs
git -C "${PROJECT_DIR}" add memory/ && \
  git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator synthesis"
```

This should be included in the CB-1 fix. The pattern is: **commit orchestrator outputs
before any step that creates a worktree.**

### O-4: All v2 review findings resolved

Confirming the current plan incorporates all 13 findings from the v2 review:
- SB-1 (REVIEWING state) → skeleton now uses REVIEW_CLAUDE + REVIEW_CODEX (lines 808-814)
- SB-2 (compare ordering) → step 8 is compare, step 9 is CAS (lines 807-808)
- SB-3 (watchdog) → Task 20 includes HANDOFF_EXISTS guard (lines 931-938)
- DG-1 (intermediate CAS) → skeleton has steps 9, 12, 15 (lines 808, 811, 814)
- DG-2 (promotion) → step 17 says "controller calls promote_version()" (line 817)
- DG-3 (skip range) → step 7a says "jump to step 15, skip 7b-14" (lines 802-803)
- DG-4 (Ray ml package) → Task 6 says "extra_modules=[pmodel, ml]" (line 375)
- IR-1 (registry scaffolding) → **NOT fixed** (still only mkdir)
- IR-2 (os.environ) → Task 13 uses monkeypatch fixture (lines 651-653)
- IR-3 (worker dry-run) → Task 19 includes launch_worker.sh dry-run test (line 902)
- O-1 (TDD inconsistency) → acknowledged, pragmatic choice
- O-2 (prior review status) → confirmed, all resolved
- O-3 (Codex env var guards) → Task 19 includes env var validation (lines 891-896)

---

## Summary Table

| ID | Severity | Category | Summary |
|----|----------|----------|---------|
| CB-1 | Critical | Worktree isolation | Worker can't read orchestrator's uncommitted direction file |
| CB-2 | Critical | Worktree isolation | Worker handoff writes to worktree, invisible to controller |
| CB-3 | Critical | CAS mismatch | Step 18 ORCHESTRATOR_PLANNING conflicts with step 1 expecting IDLE |
| CB-4 | Critical | CAS mismatch | HUMAN_SYNC not reset to IDLE before first iteration |
| HG-1 | High | Task 13 | --overrides mechanism unspecified (what, how, recording) |
| HG-2 | High | Task 6 | SMOKE_TEST data missing actual_shadow_price column |
| HG-3 | High | Task 10 | test_evaluate.py degenerate data hides ranking metric bugs |
| IR-1 | Medium | Task 13 | test_pipeline.py missing version_counter.json (carryover from v2) |
| O-1 | Note | Task 17 | No pre-merge guard for HUMAN-WRITE-ONLY files |
| O-2 | Note | Task 13 | --from-phase crash recovery untested/underspecified |
| O-3 | Note | Task 17 | Synthesis orchestrator outputs also need commit before next worktree |
| O-4 | Note | Prior reviews | All v2 findings resolved except IR-1 |

**Blocking (fix before implementation)**: CB-1, CB-2, CB-3, CB-4
**Should fix before implementation**: HG-1, HG-2, HG-3
**Address during implementation**: IR-1, O-1, O-2, O-3

---

## Recommended Fix Order

1. **CB-3 + CB-4** (2 minutes): Change step 18 N<3 to `IDLE`. Add HUMAN_SYNC reset in run_pipeline.sh.
2. **CB-1 + CB-2 + O-3** (10 minutes): Add controller commit steps after orchestrator phases. Update worker prompt to use `${PROJECT_DIR}` for handoff writes. These three issues share the root cause (worktree doesn't see uncommitted state) and should be fixed together.
3. **HG-1** (5 minutes): Specify --overrides target fields and recording.
4. **HG-2** (2 minutes): Add actual_shadow_price to SMOKE_TEST data spec.
5. **HG-3** (5 minutes): Replace degenerate shadow prices with continuous values in test.

---

## Verdict

**FIX BEFORE IMPLEMENTING** -- The 4 critical bugs (CB-1 through CB-4) will cause the
pipeline to fail on its first run. CB-1 and CB-2 are a fundamental design issue: the
worktree isolation model assumes all shared state is committed, but orchestrator outputs
are never committed. CB-3 and CB-4 are straightforward CAS logic errors. All are fixable
with small, targeted changes to the plan.

After fixing these 4 issues plus the 3 high-priority gaps, the plan is ready for
implementation.
