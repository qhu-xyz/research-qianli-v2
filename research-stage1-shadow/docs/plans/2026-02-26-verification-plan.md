# Verification Plan: Agentic ML Pipeline

**Design doc**: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v9)
**Date**: 2026-02-26

This document serves two purposes:
1. **Final design review** — remaining gaps found after 9 Codex rounds + 1 Claude review
2. **Verification spec** — explicit test assertions for every contract in the design

---

## Part 1: Final Design Review

### Remaining findings (not previously flagged)

---

#### R1 — v0 baseline creation is unspecified (Critical)

**Where**: Section 7.3 (`registry/v0/metrics.json` assumed to exist); `populate_v0_gates.py`; comparison table; `memory/hot/champion.md`.

**Problem**: `registry/v0/` is referenced throughout as the fixed baseline and the source for v0-relative gate floors. The design never specifies how v0 is created. Options: (a) manually extracted from source repo before the pipeline runs, (b) a special `--run-baseline` flag on `run_pipeline.sh`, (c) iteration 0 run by the controller before starting iteration 1. Without this, `populate_v0_gates.py` has nothing to read and the comparison table will fail on first run.

**Required fix**: Add an explicit Section 0 or pre-pipeline bootstrap step:
- v0 is the baseline model from `research-spice-shadow-price-pred-qianli` (manual extraction)
- Before running any iteration, the implementer must: `python ml/pipeline.py --version-id v0 --auction-month 2021-07 --class-type onpeak --period-type f0`
- This creates `registry/v0/metrics.json`, `registry/v0/config.json`, etc.
- Only then does `run_pipeline.sh` (and `populate_v0_gates.py`) have a valid baseline
- Add a guard in `run_pipeline.sh`: if `registry/v0/metrics.json` does not exist, abort with "Run baseline extraction first"

---

#### R2 — Env vars for launch scripts are never explicitly passed (High)

**Where**: All launch scripts (`launch_orchestrator.sh`, `launch_worker.sh`, `launch_reviewer_claude.sh`, `launch_reviewer_codex.sh`) use `${BATCH_ID}`, `${N}`, `${VERSION_ID}` without showing how these reach the script.

**Problem**: `run_single_iter.sh` calls e.g. `CODEX_SESSION=$(launch_reviewer_codex.sh)` but the launch script references `${BATCH_ID}` and `${N}` as if they were in scope. If not exported, the variables are empty and all session names, paths, and handoff filenames will be malformed (e.g., `rev-codex--iter.json`).

**Required fix**: `run_single_iter.sh` must export all shared variables before calling any launch script:
```bash
export BATCH_ID N VERSION_ID PROJECT_DIR
```
Document this in the design — either as an export block at the top of `run_single_iter.sh`, or as explicit arguments to each launch script (e.g., `launch_worker.sh --batch "${BATCH_ID}" --iter "${N}" --version "${VERSION_ID}"`). Either pattern is fine; one must be chosen and stated.

---

#### R3 — `WORKER_FAILED` env var referenced but never set in the iteration flow (High)

**Where**: Section 5.1 synthesis phase preamble: "controller sets `WORKER_FAILED` env var before launching [the synthesis orchestrator]". Section 6.2 step 7a: "skip 7b-7d, transition to ORCHESTRATOR_SYNTHESIZING" — but no step sets `WORKER_FAILED`.

**Problem**: The synthesis orchestrator's conditional input contract (worker succeeded vs failed) depends on this env var. If never set, the orchestrator will always take the success path and try to read review/comparison files that don't exist on a failure iteration.

**Required fix**: Add an explicit export step in the iteration flow between 7a and step 16:
```
7a. Worker status == "failed": set WORKER_FAILED=1 and export; skip 7b-7d.
    Worker status == "done": WORKER_FAILED=0 (default).
...
16. export WORKER_FAILED; ORCH_SESSION=$(launch_orchestrator.sh --phase synthesize)
```
The synthesis prompt reads this env var (or a file flag) to choose its input contract.

---

#### R4 — Worker prompt does not specify how to obtain `${VERSION_ID}` (Medium)

**Where**: Section 10.2 (Worker prompt structure), step 3:
```
python ml/pipeline.py ... --version-id ${VERSION_ID} ...
```

**Problem**: The worker operates in a worktree with no parent shell context. `${VERSION_ID}` is not defined for it — it must read `state.json["version_id"]` explicitly. If the worker guesses or leaves it blank, the registry write will land in the wrong path.

**Required fix**: Add to worker prompt step 3:
```
VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/../state.json")
# Note: state.json is in PROJECT_DIR (the main repo root), not the worktree.
```
(Worktree is at `PROJECT_DIR/.claude/worktrees/iter{N}-{BATCH_ID}/`, so `state.json` is two levels up from the worktree root, or accessible via `${PROJECT_DIR}/state.json` if PROJECT_DIR is exported.)

---

#### R5 — `check_clis.sh` uses hardcoded model ID (Minor)

**Where**: Section 13.1 line: `codex exec -m gpt-5.3-codex ...`

**Problem**: `CODEX_MODEL` was moved to `config.sh` (S4 fix) specifically to avoid hardcoded model IDs. `check_clis.sh` sources `agents/config.sh` but then ignores `${CODEX_MODEL}`.

**Required fix**: Change to `codex exec -m ${CODEX_MODEL} ...`.

---

### Design strengths confirmed by final read-through

- State machine CAS + single writer: consistent and correct throughout
- sha256 verification before merge (step 7c before 7d): correct ordering
- `get_expected_artifact()` dynamic path computation: correct design
- `verify_handoff()` status-aware branching: correct
- `STATE_TO_HANDOFF` map in watchdog: correct, covers all 5 states
- Tiered memory + HUMAN_SYNC distillation: well specified
- Worker failure path through synthesis: now has explicit input contract
- Codex graceful degradation: timeout artifact → controller skips → synthesis proceeds
- Worktree isolation + branch merge: clean, no rsync collision risk
- Gate auto-population (`populate_v0_gates.py`): correctly assigned and idempotent

---

## Part 2: Implementation Cross-Reference Checklist

Before claiming any component is done, verify each item below.

### Shell infrastructure (`agents/`)

| File | Key design sections | Verify |
|------|---------------------|--------|
| `config.sh` | §12.1 | All 6 vars present; `CODEX_MODEL` not hardcoded elsewhere |
| `run_pipeline.sh` | §12.3, §7.3 | Lock acquired; `PIPELINE_LOCKED` exported; `direction_iter*.md` cleaned; v0 guard; `populate_v0_gates.py` called; 3-iter loop calls `run_single_iter.sh` |
| `run_single_iter.sh` | §6.2 | `PIPELINE_LOCKED` guard; all 18 steps present; `BATCH_ID`/`N`/`VERSION_ID` exported; `WORKER_FAILED` set on step 7a |
| `launch_orchestrator.sh` | §5.1 | `while` arg parser; `--phase plan\|synthesize`; stdin redirect; `cd "${PROJECT_DIR}"` |
| `launch_worker.sh` | §5.2 | `git worktree add`; `cd "${WORKTREE}"`; stdin redirect; `echo "${SESSION}"` |
| `launch_reviewer_claude.sh` | §5.4 | `cd "${PROJECT_DIR}"`; stdin redirect; Write+Bash tools |
| `launch_reviewer_codex.sh` | §5.5 | `--sandbox read-only`; absolute `REVIEW_FILE`; absolute `HANDOFF`; `${CODEX_MODEL}` not hardcoded; shell writes handoff JSON |
| `state_utils.sh` | §8 | `get_expected_artifact()` (all 5 states, iter 1/2/3); `verify_handoff()` (done/failed branches); CAS read/write helpers |
| `watchdog.sh` | §11.2 | `STATE_TO_HANDOFF` map (5 entries); immediate crash detection (3 states); elapsed timeout; disk check; audit log |
| `check_clis.sh` | §13.1 | Sources config.sh; `${CODEX_MODEL}` used |
| `test_pipeline_integrity.sh` | §13.2 | 12 assertions; `--iterations 3` flag; final state check; orphan tmux check |
| `install_cron.sh` | §11.1 | Idempotent cron install |

### ML code (`ml/`)

| File | Key design sections | Verify |
|------|---------------------|--------|
| `config.py` | §2 | 14 features; 8 monotone constraints; F-beta β=0.7 |
| `data_loader.py` | §12.4, §14 | SMOKE_TEST branch (100 rows synthetic); polars lazy scan; Ray init; `mem_mb()` printed |
| `features.py` | §14 | polars; no pandas |
| `train.py` | §2 | XGBoost; `scale_pos_weight`; monotone constraints |
| `threshold.py` | §2 | F-beta threshold optimization; stores threshold in metrics |
| `evaluate.py` | §7.1 | HUMAN-WRITE-ONLY — never modified by agents |
| `compare.py` | §5.3 | Reads all `registry/v*/metrics.json`; produces Markdown table with all 10 gates; writes `registry/comparisons/{batch_id}_iter{N}.json` |
| `pipeline.py` | §5.2 | `--version-id`; `--from-phase N`; saves intermediates to parquet; `mem_mb()` at each stage; `ray.shutdown()` after data load |
| `registry.py` | §4 | `exist_ok=False` on `registry/v*/` mkdir; `champion.json` update; `version_counter.json` flock |
| `populate_v0_gates.py` | §7.3 | Reads `registry/v0/metrics.json`; writes `gates.json` floors for `pending_v0: true` entries; idempotent |
| `tests/conftest.py` | §4 | Synthetic data fixture; mock config |

### Prompts (`agents/prompts/`)

| File | Key design sections | Verify |
|------|---------------------|--------|
| `orchestrator_plan.md` | §10.1 | Reads hot/+warm/+index+gates+champion; writes `direction_iter{N}.md`+`progress.md`+`decision_log.md`+handoff; no ML code edits |
| `orchestrator_synthesize.md` | §5.1, §10 | Conditional on `WORKER_FAILED`; reads reviews independently; writes warm/ updates; iter<3: writes `direction_iter{N+1}.md`; iter==3: archives + `executive_summary.md` |
| `worker.md` | §10.2 | Reads `direction_iter{N}.md`+champion+learning+runbook; `VERSION_ID` from `state.json`; commit mandatory; sandbox constraints verbatim; failure handoff on 3× test failure |
| `reviewer_claude.md` | §10.3 | All 6 review sections; writes file directly; sha256+handoff at end |
| `reviewer_codex.md` | §5.5, §10.3 | Same 6 sections; **prints to stdout** (not writes file); no file-write instructions; shell wrapper captures output |

---

## Part 3: Unit Test Specifications

Each test below has explicit pass/fail criteria. Implement these before the smoke test.

### 3.1 `state_utils.sh test` — `verify_handoff()` cases

```bash
# Test 1: done handoff — correct path and sha256
echo '{"status":"done","artifact_path":"memory/direction_iter1.md","sha256":"<actual>"}' > /tmp/t1.json
echo "test content" > memory/direction_iter1.md
SHA=$(sha256sum memory/direction_iter1.md | cut -d' ' -f1)
echo "{\"status\":\"done\",\"artifact_path\":\"memory/direction_iter1.md\",\"sha256\":\"${SHA}\"}" > /tmp/t1.json
# Seed state.json: {"state":"ORCHESTRATOR_PLANNING","iteration":1,"batch_id":"b1","version_id":null}
verify_handoff /tmp/t1.json ORCHESTRATOR_PLANNING
# Expected: PASS (exit 0)

# Test 2: done handoff — sha256 mismatch
echo "{\"status\":\"done\",\"artifact_path\":\"memory/direction_iter1.md\",\"sha256\":\"wrong\"}" > /tmp/t2.json
verify_handoff /tmp/t2.json ORCHESTRATOR_PLANNING
# Expected: FAIL (exit nonzero, message "sha256")

# Test 3: done handoff — path mismatch
echo "{\"status\":\"done\",\"artifact_path\":\"memory/direction_iter9.md\",\"sha256\":\"x\"}" > /tmp/t3.json
verify_handoff /tmp/t3.json ORCHESTRATOR_PLANNING
# Expected: FAIL (exit nonzero, message "Path mismatch")

# Test 4: failed handoff — has error field
echo '{"status":"failed","artifact_path":null,"sha256":null,"error":"pytest failed 3x"}' > /tmp/t4.json
verify_handoff /tmp/t4.json WORKER_RUNNING
# Expected: PASS (exit 0) — failed handoffs bypass path/sha check

# Test 5: failed handoff — missing error field
echo '{"status":"failed","artifact_path":null,"sha256":null}' > /tmp/t5.json
verify_handoff /tmp/t5.json WORKER_RUNNING
# Expected: FAIL (exit nonzero, message "missing 'error'")

# Test 6: unknown state
verify_handoff /tmp/t1.json UNKNOWN_STATE
# Expected: FAIL (exit nonzero, message "Unknown state")
```

### 3.2 `get_expected_artifact()` — all states and iterations

```bash
# State: ORCHESTRATOR_PLANNING, iter=1
# state.json: {"state":"ORCHESTRATOR_PLANNING","iteration":1,"batch_id":"b1","version_id":null}
result=$(get_expected_artifact ORCHESTRATOR_PLANNING)
[[ "$result" == "memory/direction_iter1.md" ]] || echo "FAIL: $result"

# State: WORKER_RUNNING, iter=2, version_id=v0007
# state.json: {"state":"WORKER_RUNNING","iteration":2,"batch_id":"b1","version_id":"v0007"}
result=$(get_expected_artifact WORKER_RUNNING)
[[ "$result" == "registry/v0007/metrics.json" ]] || echo "FAIL: $result"

# State: ORCHESTRATOR_SYNTHESIZING, iter=2 (mid-batch → direction_iter3.md)
# state.json: {"state":"ORCHESTRATOR_SYNTHESIZING","iteration":2,"batch_id":"b1"}
result=$(get_expected_artifact ORCHESTRATOR_SYNTHESIZING)
[[ "$result" == "memory/direction_iter3.md" ]] || echo "FAIL: $result"

# State: ORCHESTRATOR_SYNTHESIZING, iter=3 (HUMAN_SYNC → executive_summary)
# state.json: {"state":"ORCHESTRATOR_SYNTHESIZING","iteration":3,"batch_id":"batch-20260226-001"}
result=$(get_expected_artifact ORCHESTRATOR_SYNTHESIZING)
[[ "$result" == "memory/archive/batch-20260226-001/executive_summary.md" ]] || echo "FAIL: $result"
```

### 3.3 `STATE_TO_HANDOFF` map completeness

```bash
# All 5 pipeline states must have a mapping
for state in WORKER_RUNNING REVIEW_CLAUDE REVIEW_CODEX ORCHESTRATOR_PLANNING ORCHESTRATOR_SYNTHESIZING; do
  fname="${STATE_TO_HANDOFF[$state]:-}"
  [[ -n "$fname" ]] || echo "FAIL: no mapping for $state"
done
# Expected: no FAIL lines

# Spot-check values
[[ "${STATE_TO_HANDOFF[REVIEW_CODEX]}" == "codex_reviewer_done.json" ]] || echo "FAIL REVIEW_CODEX"
[[ "${STATE_TO_HANDOFF[ORCHESTRATOR_PLANNING]}" == "orchestrator_plan_done.json" ]] || echo "FAIL ORCH_PLAN"
```

### 3.4 `launch_orchestrator.sh` arg parser

```bash
# --phase plan (two-arg style)
PHASE=$(bash agents/launch_orchestrator.sh --phase plan --dry-run 2>&1 | grep "^PHASE=" | cut -d= -f2)
[[ "$PHASE" == "plan" ]] || echo "FAIL: two-arg plan: $PHASE"

# --phase=synthesize (equals style)
PHASE=$(bash agents/launch_orchestrator.sh --phase=synthesize --dry-run 2>&1 | grep "^PHASE=" | cut -d= -f2)
[[ "$PHASE" == "synthesize" ]] || echo "FAIL: equals synthesize: $PHASE"

# Invalid phase → exits nonzero
bash agents/launch_orchestrator.sh --phase invalid 2>/dev/null && echo "FAIL: should reject invalid phase"

# No args → defaults to plan
PHASE=$(bash agents/launch_orchestrator.sh --dry-run 2>&1 | grep "^PHASE=" | cut -d= -f2)
[[ "$PHASE" == "plan" ]] || echo "FAIL: default: $PHASE"
```
*(Requires a `--dry-run` flag that prints `PHASE=...` and exits without launching tmux.)*

### 3.5 `PIPELINE_LOCKED` guard

```bash
# Direct invocation without lock → exits nonzero with "must be called via run_pipeline.sh"
unset PIPELINE_LOCKED
bash agents/run_single_iter.sh 2>&1 | grep -q "must be called via" || echo "FAIL: guard not triggered"

# Via run_pipeline.sh (PIPELINE_LOCKED exported) → guard passes, reaches next check
export PIPELINE_LOCKED=1
bash agents/run_single_iter.sh 2>&1 | grep -v "must be called via" || echo "FAIL: guard fires when locked"
```

### 3.6 CWD independence of agent launchers

```bash
# Launch scripts invoked from /tmp must still write artifacts under PROJECT_DIR
cd /tmp
BATCH_ID=test-cwd N=1 VERSION_ID=v0001 bash "${PROJECT_DIR}/agents/launch_reviewer_claude.sh" --dry-run
# Verify: session command contains "cd \"${PROJECT_DIR}\"" as first token after tmux
# Verify: REVIEW_FILE in Codex launch resolves to ${PROJECT_DIR}/reviews/... (absolute path)
```

### 3.7 Watchdog false-positive guard

```bash
# Scenario: agent has completed (handoff file exists) but tmux session already exited
# Watchdog must NOT write a timeout/crash artifact
mkdir -p handoff/b1/iter1
echo '{"status":"done"}' > handoff/b1/iter1/worker_done.json
# Set state.json: state=WORKER_RUNNING, entered_at=1 hour ago
# Kill tmux session "worker-b1-iter1" (or don't create it)
bash agents/watchdog.sh
[[ ! -f "handoff/b1/iter1/timeout_WORKER_RUNNING.json" ]] || echo "FAIL: false crash artifact created"
```

### 3.8 `populate_v0_gates.py` — idempotency and correctness

```python
# Test 1: populates pending_v0: true entries from v0 metrics
v0_metrics = {"S1-AUC": 0.695, "S1-VCAP@100": 0.412, "S1-BRIER": 0.089, ...}
# After populate: S1-VCAP@100 floor = 0.412 - 0.05 = 0.362; pending_v0 = false
# After second run (no pending entries): gates.json unchanged (idempotent)

# Test 2: aborts cleanly if registry/v0/metrics.json does not exist
# Expected: exits nonzero, no partial writes to gates.json
```

### 3.9 Worker failure path through synthesis

```bash
# Inject a failed worker handoff
echo '{"status":"failed","error":"pytest failed 3x","artifact_path":null,"sha256":null}' \
  > handoff/b1/iter1/worker_done.json
# Run synthesis step with WORKER_FAILED=1
export WORKER_FAILED=1 BATCH_ID=b1 N=1
bash agents/launch_orchestrator.sh --phase synthesize
# Assert: synthesis prompt reads handoff + memory/hot/ + memory/warm/ only
# Assert: no attempt to read reviews/ or reports/ (files don't exist)
# Assert: orchestrator_synth_done.json written successfully
# Assert: direction_iter2.md written (not empty)
```

### 3.10 Codex timeout → Claude-only degradation path

```bash
# Inject a timeout artifact instead of Codex handoff
echo '{"timeout":true,"state":"REVIEW_CODEX","elapsed_s":1201}' \
  > handoff/b1/iter1/timeout_REVIEW_CODEX.json
# Controller should detect this on its polling loop and proceed to ORCHESTRATOR_SYNTHESIZING
# Assert: state transitions REVIEW_CODEX → ORCHESTRATOR_SYNTHESIZING (not stuck)
# Assert: synthesis prompt notes Codex was unavailable
# Assert: pipeline completes iteration normally without Codex review
```

---

## Part 4: Integration Smoke Tests

Run in order. Each tier must pass before the next.

### Tier 0 — Pre-flight (< 30 seconds)

```bash
bash agents/check_clis.sh
# Expected: "Claude: OK" and "Codex: OK"
```

### Tier 1 — Shell unit tests (< 2 minutes)

Run all tests from Part 3, sections 3.1–3.8:
```bash
bash agents/state_utils.sh test        # covers 3.1–3.3
bash agents/test_arg_parser.sh         # covers 3.4
bash agents/test_guards.sh             # covers 3.5, 3.7
pytest ml/tests/ -v                    # ML unit tests
python -m pytest ml/tests/test_registry.py::test_populate_v0_gates   # 3.8
```
**Pass criteria**: all green, no skips.

### Tier 2 — 1-iteration smoke loop (< 3 minutes)

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 1
```
12 assertions per Section 13.2. **Final state must be `ORCHESTRATOR_PLANNING`** (not `HUMAN_SYNC`).

### Tier 3 — Full 3-iteration smoke loop (< 10 minutes)

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 3
```
Additional assertions over 1-iter:
- `memory/archive/${batch_id}/executive_summary.md` written
- `memory/archive/index.md` has new entry
- `memory/hot/learning.md` updated
- `memory/warm/` files reset to empty stubs
- Final state == `HUMAN_SYNC`
- No orphaned tmux sessions (`tmux ls` returns only non-pipeline sessions)

### Tier 4 — Failure path integration (< 5 minutes)

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --inject-worker-failure
```
Assertions:
- Steps 7b–7d skipped (no merge, no comparison)
- Synthesis runs and reaches `direction_iter2.md` (or exits HUMAN_SYNC if iter 3)
- `memory/hot/progress.md` contains failure note
- No crash of run_single_iter.sh — pipeline completes iteration normally

### Tier 5 — Codex degradation integration (< 5 minutes)

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --inject-codex-timeout
```
Assertions:
- Timeout artifact detected by controller polling
- State transitions `REVIEW_CODEX → ORCHESTRATOR_SYNTHESIZING`
- Synthesis runs with Claude review only
- `memory/archive/${batch_id}/all_critiques.md` contains "Codex unavailable" note (if iter 3)

---

## Part 5: Failure Triage Protocol

When any smoke test assertion fails, determine root cause before touching code:

```
1. Which assertion number failed?
2. Is the design spec for that assertion unambiguous?
   - YES → implementation bug. Fix code, rerun that tier.
   - NO  → design gap. Update this doc + design doc FIRST, then implement fix.
3. If a design gap: check if the fix changes the contract for other assertions.
   If yes, rerun from Tier 1.
```

**Common failure signatures**:

| Symptom | Likely cause |
|---------|--------------|
| Session name empty (e.g., `rev-codex--iter`) | R2: env vars not exported |
| `populate_v0_gates.py` fails on first run | R1: v0 not created before running pipeline |
| Synthesis reads missing review files | R3: `WORKER_FAILED` not set |
| Worker uses wrong VERSION_ID | R4: state.json path wrong from worktree |
| `verify_handoff()` fails with "Path mismatch" on iter 3 synthesis | Confirm `iter==3` branch in `get_expected_artifact()` |
| Watchdog writes false crash artifact | Handoff file not found before session exit — check naming vs `STATE_TO_HANDOFF` |
| Pipeline re-runs after restart (OOM loop) | Never use `claude -r` per CLAUDE.md |

---

## Part 6: Pre-Run Checklist (Final Gate)

Before running the first real (non-smoke) batch:

- [ ] R1 resolved: `registry/v0/` created with real data; guard in `run_pipeline.sh` verified
- [ ] R2 resolved: `BATCH_ID`, `N`, `VERSION_ID` exported in `run_single_iter.sh`
- [ ] R3 resolved: `WORKER_FAILED` set in step 7a of `run_single_iter.sh`
- [ ] R4 resolved: worker prompt explicitly instructs `VERSION_ID=$(jq -r .version_id ${PROJECT_DIR}/state.json)`
- [ ] R5 resolved: `check_clis.sh` uses `${CODEX_MODEL}`
- [ ] Codex model ID confirmed on subscription (Open Decision #1)
- [ ] `check_clis.sh` passes (Tier 0)
- [ ] All Tier 1 shell unit tests pass
- [ ] Tier 2 (1-iter smoke) passes
- [ ] Tier 3 (3-iter smoke) passes — `HUMAN_SYNC` reached and exited cleanly
- [ ] Tier 4 (worker failure path) passes
- [ ] Tier 5 (Codex degradation) passes
- [ ] `registry/v0/metrics.json` reviewed; gate floors after `populate_v0_gates.py` look sane
- [ ] HUMAN_SYNC notification mechanism decided (Open Decision #4)
