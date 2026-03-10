# Implementation Plan Review

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`
**Cross-referenced**: Design doc v10, Verification plan, v10 design review
**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-26

---

## Executive Summary

The implementation plan is well-structured: 28 tasks across 6 phases, TDD where it matters, phased checkpoints, and explicit commit boundaries. It faithfully translates the v10 design doc into an implementable sequence. The prior v10 review's findings (#1 batch-name, #2 early exit, #3 placeholder substitution) are all addressed in the implementation plan.

However, I found **3 spec bugs** that will cause test or runtime failures if not fixed before implementation, **5 gaps** where important behavior is underspecified, and **4 structural concerns** that affect maintainability but don't block implementation.

**Recommendation**: Fix the 3 spec bugs (trivial), then proceed. The gaps can be addressed during implementation.

---

## Spec Bugs (will cause failures)

### SB-1: `synthetic_labels()` fixture size mismatch (Task 5)

**Where**: Task 5, Step 3, conftest.py specification:
> `synthetic_labels()` fixture -> numpy array **200** with ~7% positive rate

**Problem**: `synthetic_features()` produces shape `(100, 14)`. If labels are length 200, every test that pairs features with labels (Tasks 8, 9, 10) will fail with a dimension mismatch. XGBoost's `model.fit(X, y)` requires `len(y) == X.shape[0]`.

**Fix**: Change `200` to `100`:
```
synthetic_labels() fixture -> numpy array 100 with ~7% positive rate
```

**Severity**: Will cause immediate test failures in Tasks 8-10.

---

### SB-2: `PipelineConfig` missing `registry_dir` field (Task 13 vs Task 5)

**Where**: Task 13, Step 1, test code:
```python
config = PipelineConfig(version_id="v0001", registry_dir=str(tmp_path / "registry"))
```

**Problem**: Task 5 defines `PipelineConfig` with these fields:
> `auction_month, class_type, period_type, version_id, train_months=10, val_months=2, threshold_beta=0.7, threshold_scaling_factor=1.0, scale_pos_weight_auto=True`

No `registry_dir` field. The test will fail at dataclass construction with `TypeError: unexpected keyword argument 'registry_dir'`.

**Fix**: Either:
- (a) Add `registry_dir: str = "registry"` to PipelineConfig in Task 5's spec, or
- (b) Change the test to pass `registry_dir` via a separate mechanism (env var, function arg to `run_pipeline()`)

Option (a) is simpler and consistent with `pipeline.py` needing to know where to write.

---

### SB-3: Smoke-test v0 baseline non-determinism (Task 26)

**Where**: Task 26 creates v0 baseline with `SMOKE_TEST=true`, then Task 26 Step 2 populates gate floors from v0 metrics.

**Problem**: Task 6 specifies the SMOKE_TEST data loader as "returns synthetic polars DataFrames (100 rows, 14 features + label + metadata columns)" but does not specify a random seed. If `data_loader.py` uses `np.random.rand()` without a seed, each invocation produces different synthetic data, meaning:
- v0 metrics are non-reproducible
- Gate floors change on re-run
- Tier 2+ smoke tests may pass/fail non-deterministically

The `HyperparamConfig` has `random_state=42` for XGBoost, but the input data itself is unseeded.

**Fix**: Specify in Task 6 that the SMOKE_TEST branch must use a fixed seed:
```python
rng = np.random.RandomState(42)
```
This also applies to `conftest.py` fixtures (Task 5).

---

## Gaps (underspecified behavior)

### G-1: No test files for `data_loader.py` and `features.py` (Tasks 6-7)

**Where**: Tasks 6 and 7 have only manual "Verify" one-liners, not proper `test_data_loader.py` or `test_features.py` files.

**Problem**: If `data_loader.py` schema changes (column names, dtypes) or `features.py` breaks (wrong column selection, NaN handling), there's no regression guard. These are the foundation of the ML pipeline — a silent schema change propagates to every downstream module.

**Recommendation**: Add minimal test files:
- `test_data_loader.py`: verify SMOKE_TEST returns correct shape, correct columns, correct dtypes, no NaNs in features
- `test_features.py`: verify `prepare_features()` output shape, `compute_binary_labels()` output range, `compute_scale_pos_weight()` correctness on known input

This doesn't need to be TDD — just add them after implementation, before the Phase C checkpoint.

---

### G-2: No tests for `compare.py` (Task 11)

**Where**: Task 11 creates `ml/compare.py` with no test file.

**Problem**: The comparison table is the primary artifact reviewers use. It implements gate-checking logic (noise_tolerance, direction-aware pass/fail, v0-relative floors). If `check_gates()` gets the Brier direction wrong (higher instead of lower), it silently approves bad models or blocks good ones.

**Recommendation**: Add `test_compare.py` with:
- Correct pass/fail for higher-direction gates
- Correct pass/fail for lower-direction gates (Brier: v0=0.089, floor=0.109, candidate=0.095 should PASS because 0.095 < 0.109)
- Noise tolerance edge cases (delta exactly at noise_tolerance boundary)
- Missing v0 metrics handling
- Markdown output format validation (no broken `|` in cells per CLAUDE.md rule)

---

### G-3: `run_single_iter.sh` Step 18 state transition logic unspecified (Task 17)

**Where**: Task 17 says "Implement all 18 steps from design doc §6.2" but step 18 has conditional logic:
> CAS: state = IDLE (or ORCHESTRATOR_PLANNING for next iter, or HUMAN_SYNC after iter 3)

**Problem**: `run_single_iter.sh` needs to decide between three target states based on iteration number. The implementation plan doesn't specify the branching logic:
```
if N == 3: target = HUMAN_SYNC
elif N < 3: target = ORCHESTRATOR_PLANNING  # (or IDLE — design is ambiguous)
```

Looking at `run_pipeline.sh` (Task 18): the loop checks `[[ "$current_state" == "HUMAN_SYNC" ]] && break` after each iteration. This means:
- Iter 1-2: `run_single_iter.sh` should transition to `ORCHESTRATOR_PLANNING` (so the next iteration starts directly)
- Iter 3: `run_single_iter.sh` should transition to `HUMAN_SYNC`

But alternatively, `run_single_iter.sh` could always transition to IDLE, and `run_pipeline.sh` handles the next CAS in the loop's next iteration. The design doc is ambiguous here — both patterns work but the state machine semantics differ.

**Recommendation**: Specify explicitly in Task 17:
```bash
# Step 18
if (( N == 3 )); then
  cas_transition ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC '{}'
else
  cas_transition ORCHESTRATOR_SYNTHESIZING IDLE '{}'
fi
```
Then `run_pipeline.sh`'s loop handles the IDLE -> ORCHESTRATOR_PLANNING transition at the start of the next iteration.

---

### G-4: `launch_orchestrator.sh` WORKER_FAILED injection mechanism (Task 19 vs Task 23)

**Where**: Task 23 specifies that the synthesis prompt must receive WORKER_FAILED via stdin prepending:
```bash
{ echo "WORKER_FAILED=${WORKER_FAILED}"; cat "${PROMPT}"; } | claude --print --model opus ...
```

But Task 19 (launch_orchestrator.sh) doesn't reference this mechanism. The design doc §5.1 shows plain `< "${PROMPT}"` stdin redirect.

**Problem**: If the implementer follows Task 19 in isolation, they'll build the launch script with simple file redirect. Task 23's requirement to inject WORKER_FAILED arrives later and requires retroactively modifying the launch script. This creates a dependency that isn't tracked.

**Recommendation**: Add to Task 19's implementation spec:
> `launch_orchestrator.sh` must support WORKER_FAILED injection when `--phase synthesize`:
> ```bash
> if [[ "$PHASE" == "synthesize" ]]; then
>   { echo "WORKER_FAILED=${WORKER_FAILED}"; cat "${PROMPT}"; } | claude ...
> else
>   claude ... < "${PROMPT}"
> fi
> ```

---

### G-5: No `.gitignore` for transient artifacts

**Where**: Not addressed anywhere in the plan.

**Problem**: After running the pipeline, `git status` will show noise from:
- `.logs/sessions/` (append-only JSONL)
- `.logs/audit.jsonl`
- `state.lock`
- `.claude/worktrees/` (worker worktrees)
- `handoff/` (ephemeral per-batch)
- Parquet intermediates from `--from-phase` crash recovery

**Recommendation**: Add a Task 2.5 or append to Task 2:
```gitignore
state.lock
.logs/sessions/
handoff/
.claude/worktrees/
ml/**/*.parquet
```

Alternatively, `.logs/audit.jsonl` and `handoff/` could be tracked — but the implementer should decide and create the `.gitignore` explicitly.

---

## Structural Concerns

### SC-1: Task 17 (`run_single_iter.sh`) is underspecified relative to its complexity

**Where**: Task 17 is the most complex task in the plan — it implements the entire 18-step iteration controller with CAS transitions, flock allocation, sha256 verification, merge, and error handling. But the implementation spec is just:
> "Implement all 18 steps from design doc §6.2"

Compare to Task 18 (`run_pipeline.sh`) which provides a full `while` arg parser code block, or Task 16 (`state_utils.sh`) which lists all functions explicitly.

**Risk**: The implementer will need to cross-reference §6.2, the verification plan's tests (§3.1-3.3, §3.5), and the handoff schema simultaneously. Without a code skeleton in the implementation plan, there's high risk of missing steps or getting ordering wrong (e.g., exporting WORKER_FAILED before vs after the failure check).

**Recommendation**: Add a code skeleton for Task 17 that shows the 18 steps as commented shell blocks:
```bash
#!/usr/bin/env bash
set -euo pipefail
# Step 0: guards and exports
[[ -n "${PIPELINE_LOCKED:-}" ]] || { echo "ERROR: must be called via run_pipeline.sh"; exit 1; }
export BATCH_ID N VERSION_ID PROJECT_DIR
export WORKER_FAILED=0
# Step 1: flock + CAS ...
# Step 2: mkdir + CAS -> ORCHESTRATOR_PLANNING ...
# ... etc
```

---

### SC-2: Commit batching creates large, unreviewable commits

**Where**: Several tasks batch commits: Task 6+7 (data_loader + features), Task 8+9 (train + threshold), Task 11+12 (compare + registry).

**Problem**: Batched commits combine unrelated modules. If a bug is later traced to `threshold.py`, the commit also contains `train.py` changes, making `git bisect` less precise.

**Recommendation**: Not blocking, but prefer one commit per module. The plan already does this for most tasks — the batched ones are likely just for convenience. The implementer should use their judgment.

---

### SC-3: No explicit error handling spec for `poll_for_handoff()` edge cases

**Where**: Task 16, `state_utils.sh`, `poll_for_handoff()` function.

**Problem**: The plan says "poll loop" but doesn't specify:
- What happens if the handoff file exists but is empty (partial write)?
- What happens if jq parsing fails on the handoff JSON?
- Whether to check for `timeout_${state}.json` alongside the normal handoff file

The verification plan §3.9 and §3.10 imply the controller must poll for both normal handoff AND timeout artifacts. The `poll_for_handoff()` function should accept both filenames.

**Recommendation**: Specify that `poll_for_handoff()` checks for BOTH:
```bash
poll_for_handoff() {
  local handoff_dir="$1" filename="$2" timeout_filename="timeout_${state}.json" ...
  while true; do
    [[ -f "${handoff_dir}/${filename}" ]] && return 0
    [[ -f "${handoff_dir}/${timeout_filename}" ]] && return 1  # timeout detected
    ...
  done
}
```

---

### SC-4: Worktree accumulation not addressed

**Where**: Open Decision #3 in design doc, not mentioned in implementation plan.

**Problem**: Each iteration creates a worktree at `.claude/worktrees/iter${N}-${BATCH_ID}`. After 3 iterations, 3 worktrees exist. After 3 batches, 9 worktrees exist. Each is a full repo checkout. If the repo is 500MB, that's 4.5GB in worktrees alone.

The implementation plan should at minimum add worktree cleanup to the HUMAN_SYNC distillation (Task 23, orchestrator_synthesize.md iter==3 branch) or as a post-batch step in `run_pipeline.sh`.

**Recommendation**: Add to Task 18 (`run_pipeline.sh`), after the loop completes:
```bash
# Cleanup worktrees from this batch
for N in 1 2 3; do
  WORKTREE="${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}"
  if [[ -d "$WORKTREE" ]]; then
    git worktree remove "$WORKTREE" --force 2>/dev/null || true
  fi
done
```

---

## Verification Plan Cross-Check

The implementation plan correctly incorporates all verification plan tests:

| VP Section | Implementation Plan Task | Status |
|------------|------------------------|--------|
| 3.1 verify_handoff (6 cases) | Task 16 self-test | Covered |
| 3.2 get_expected_artifact (6 cases) | Task 16 self-test | Covered |
| 3.3 STATE_TO_HANDOFF (5 entries) | Task 16 self-test | Covered |
| 3.4 arg parser | Task 19 test_arg_parser.sh | Covered |
| 3.5 PIPELINE_LOCKED guard | Task 19 test_guards.sh | Covered |
| 3.6 CWD independence | Task 19 test_arg_parser.sh | Covered |
| 3.7 Watchdog false-positive | Task 19 test_guards.sh | Covered |
| 3.8 populate_v0_gates | Task 14 test_registry.py | Covered |
| 3.9 Worker failure path | Task 28 (Tier 4) | Covered |
| 3.10 Codex timeout path | Task 28 (Tier 5) | Covered |
| Tier 0 pre-flight | Task 21 check_clis.sh | Covered |
| Tier 1 unit tests | Task 27 Step 1 | Covered |
| Tier 2 1-iter smoke | Task 27 Step 2 | Covered |
| Tier 3 3-iter smoke | Task 28 Step 1 | Covered |
| Tier 4 worker failure | Task 28 Step 2 | Covered |
| Tier 5 Codex timeout | Task 28 Step 3 | Covered |

**No verification plan items are missing from the implementation plan.**

---

## Design Doc Fidelity Check

The implementation plan matches the design doc v10 on all critical contracts:

| Contract | Design Doc | Implementation Plan | Match? |
|----------|-----------|-------------------|--------|
| 14 features, 8 monotone | Section 2 | Task 5 | YES |
| XGBoost + scale_pos_weight | Section 2 | Task 8 | YES |
| F-beta threshold (beta=0.7) | Section 2 | Task 9 | YES |
| 10 gate metrics | Section 7.2 | Task 10 | YES |
| Brier direction (lower) | Section 7.2 | Task 14 (explicit test) | YES |
| v0-relative floors | Section 7.3 | Task 14 | YES |
| CAS state transitions | Section 8 | Task 16 | YES |
| Single writer rule | Section 3 | Task 17 | YES |
| Codex read-only sandbox | Section 5.5 | Task 19 | YES |
| Codex stdout capture | Section 5.5 | Task 19 | YES |
| WORKER_FAILED flow | Section 6.2, step 7a | Task 17 + Task 23 | YES |
| sha256 before merge | Section 6.2, step 7c | Task 17 | YES |
| HUMAN-WRITE-ONLY evaluate.py | Section 7.1 | Task 10 | YES |
| Memory tiers (hot/warm/archive) | Section 9 | Task 3 | YES |
| 3-iteration loop | Section 12.3 | Task 18 | YES |
| Watchdog READ-ONLY | Section 11.2 | Task 20 | YES |
| SMOKE_TEST env override | Section 12.1 (deviation) | Task 3 (noted) | YES (intentional) |
| --batch-name fix (v10 finding #1) | v10 review | Task 18 | YES |
| Prompt placeholder resolution (v10 finding #3) | v10 review | Task 23 | YES |

**One intentional deviation noted and justified** (SMOKE_TEST env override in config.sh).

---

## Summary of Findings

| ID | Severity | Category | Summary |
|----|----------|----------|---------|
| SB-1 | Bug | conftest.py | synthetic_labels size 200 should be 100 |
| SB-2 | Bug | Task 13 test | PipelineConfig missing registry_dir field |
| SB-3 | Bug | Task 26 | SMOKE_TEST data loader needs fixed random seed |
| G-1 | Gap | Tasks 6-7 | No test files for data_loader.py and features.py |
| G-2 | Gap | Task 11 | No tests for compare.py (gate-checking logic) |
| G-3 | Gap | Task 17 | Step 18 state transition branching unspecified |
| G-4 | Gap | Task 19/23 | WORKER_FAILED stdin injection not in launch script spec |
| G-5 | Gap | Task 2 | No .gitignore for transient artifacts |
| SC-1 | Structure | Task 17 | Most complex task has thinnest spec |
| SC-2 | Structure | Various | Commit batching reduces bisect precision |
| SC-3 | Structure | Task 16 | poll_for_handoff edge cases and timeout file polling |
| SC-4 | Structure | Task 18 | Worktree cleanup not addressed |

**Blocking**: SB-1, SB-2, SB-3 (trivial fixes, but will cause test failures if not addressed)
**Should fix before implementation**: G-1, G-2, G-3, G-4
**Address during implementation**: G-5, SC-1, SC-2, SC-3, SC-4

---

## Verdict

**PROCEED WITH FIXES** — The plan is implementable. Fix the 3 spec bugs (each is a one-liner), then implement. Address the 4 gaps as you encounter each task. The structural concerns are minor and can be handled by the implementer's judgment.
