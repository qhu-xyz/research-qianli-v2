# Design: Agentic ML Research Pipeline for Shadow Price Classification

**Date**: 2026-02-26
**Version**: v10 (v9 fixes + Codex handoff relative paths, reviewer_codex.md spec, WORKER_FAILED reset, handoff dir creation, run_pipeline.sh loop spec, v0 baseline clarification)
**Author**: Claude (brainstorming session with user)

---

## 1. Purpose

Build an agentic ML research pipeline that autonomously iterates on a shadow price
classification model (XGBoost binary classifier: "will this transmission constraint bind?").
The pipeline runs 3 iterations autonomously, then pauses for human review.

This ports the OpenClaw "Software Engineering" agent pattern into a "Machine Learning
Research" pattern: Code PRs become Model Iterations; Evaluation Reports replace UI Screenshots.

### Goals

1. **Autonomous iteration**: 3-iteration batches with orchestrator, worker, and reviewer
   agents coordinated via file-based state.
2. **Apple-to-apple comparison**: v0 benchmark always present. Metrics/gates frozen within
   a 3-iteration batch; only updated during human-in-the-loop sync.
3. **Traceability**: Every version is registered with full config, metrics, and provenance.
   Nothing is deleted (except post-3-iter cleanup of temporary files).
4. **Two interaction modes**: (a) ask questions / modify pipeline without triggering a run,
   (b) say "start" to kick off a 3-iteration batch.
5. **Cost efficiency**: Claude Max subscription (Opus 4.6) for orchestrator/worker/Claude
   reviewer. Codex CLI for second reviewer.

### Non-goals

- Stage 2 (regression) — extracted and ported later.
- Multi-node distributed training — single-machine XGBoost is sufficient.
- Web UI — all interaction via Claude Code CLI.

---

## 2. Source Model: What We're Extracting

From `research-spice-shadow-price-pred-qianli`, we extract only the Stage 1 classifier:

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (`XGBClassifier`) |
| Task | Binary classification: P(constraint binds) |
| Features | 14 features (5 exceedance probs, 3 below-threshold probs, expected overload, 3 distribution moments, 2 historical DA features) |
| Monotone constraints | Yes — 8 of 14 features have directional constraints |
| Training window | Rolling 12-month: 10-month fit, 2-month validation (threshold opt) |
| Threshold optimization | F-beta (beta=0.7) on validation set |
| Class imbalance | ~5-7% binding rate, handled via `scale_pos_weight` |

### Initial scope (minimal slice for pipeline development)

- 1 auction month (e.g., 2021-07), 1 class type (onpeak), 1 period type (f0)
- Default classifier only (no per-branch models)
- Completes in ~2-5 minutes — fast enough to validate agent infrastructure

---

## 3. Architecture Overview

```
                +----------+
                |  Human   |  (via Claude Code CLI)
                +----+-----+
                     |
         "start 3 iters" or "lower threshold to 0.4"
                     |
                     v
              +------+------+
              | run_pipeline |  master shell script
              |    .sh       |  acquires state.lock (flock)
              +------+------+
                     |
         +-----------+-----------+
         |           |           |
         v           v           v
   +----------+ +--------+ +-----------+
   |Orchestrat| | Worker | | Reviewers |
   | (Claude) | |(Claude | | (Claude + |
   |          | |in wktree| |  Codex)  |
   +----------+ +--------+ +-----------+
         |           |           |
         +-----------+-----------+
                     |
           File-based coordination:
           state.json (single source of truth)
           handoff/{batch_id}/iter{N}/*.json
           memory/*.md
           registry/vN/
                     |
                     v
              +------+------+
              | watchdog.sh |  cron (every 60s), READ-ONLY observer
              +-------------+
```

### State Machine

```
IDLE
  --> ORCHESTRATOR_PLANNING
  --> WORKER_RUNNING
  --> REVIEW_CLAUDE
  --> REVIEW_CODEX
  --> ORCHESTRATOR_SYNTHESIZING
  --> [repeat for iterations 2, 3]
  --> HUMAN_SYNC
```

**Single writer rule**: Only the controller script (`run_single_iter.sh`) writes `state.json`.
Agents write completion signal files (`handoff/.../agent_done.json`). Watchdog never writes
`state.json`. This eliminates race conditions.

**Compare-and-swap**: Before any state transition, controller checks
`current_state == expected_state`. Mismatch causes abort + audit log entry.

### `state.json` schema

```json
{
  "batch_id": "batch-20260226-001",
  "iteration": 1,
  "version_id": "v0004",
  "state": "WORKER_RUNNING",
  "entered_at": "2026-02-26T14:30:00Z",
  "max_seconds": 1800,
  "orchestrator_tmux": "orch-batch20260226001-iter1",
  "worker_tmux": "worker-batch20260226001-iter1",
  "claude_reviewer_tmux": "rev-claude-batch20260226001-iter1",
  "codex_reviewer_tmux": "rev-codex-batch20260226001-iter1",
  "history": [
    {"state": "ORCHESTRATOR_PLANNING", "entered_at": "...", "exited_at": "...", "duration_s": 120}
  ],
  "human_input": null,
  "error": null
}
```

---

## 4. Directory Structure

```
research-stage1-shadow/
├── human-input/                   # User requirements (read-only for agents)
│   ├── mem.md
│   ├── requirement.md
│   └── reference.md
├── docs/plans/                    # Design docs
├── CLAUDE.md                      # Agent sandbox rules
│
├── ml/                            # ML code (extracted from source repo)
│   ├── config.py                  # FeatureConfig, HyperparamConfig, GateConfig
│   ├── data_loader.py             # Load density parquets + historical DA (requires Ray)
│   ├── features.py                # Feature engineering
│   ├── train.py                   # XGBoost training + feature selection
│   ├── threshold.py               # F-beta threshold optimization
│   ├── evaluate.py                # Standardized eval harness — HUMAN-WRITE-ONLY
│   ├── compare.py                 # Deterministic cross-version comparison tables
│   ├── pipeline.py                # End-to-end: load -> train -> eval -> register
│   ├── registry.py                # Version registration, promotion logic
│   └── tests/
│       ├── test_config.py
│       ├── test_train.py
│       ├── test_evaluate.py
│       ├── test_pipeline.py
│       ├── test_registry.py
│       └── conftest.py            # Fixtures: synthetic data, mock config
│
├── agents/                        # Agent infrastructure
│   ├── config.sh                  # All env-specific paths and settings
│   ├── run_pipeline.sh            # Master: acquires lock, runs 3-iteration batch
│   ├── run_single_iter.sh         # Runs 1 iteration; single state writer
│   ├── launch_orchestrator.sh     # Spawns orchestrator in manual tmux
│   ├── launch_worker.sh           # Spawns worker (worktree + manual tmux)
│   ├── launch_reviewer_claude.sh  # Spawns Claude reviewer in manual tmux
│   ├── launch_reviewer_codex.sh   # Spawns Codex reviewer in manual tmux
│   ├── watchdog.sh                # Read-only health/timeout checker
│   ├── install_cron.sh            # Installs watchdog into crontab
│   ├── state_utils.sh             # CAS helpers: read/write/check state.json
│   ├── check_clis.sh              # Pre-flight: verify claude + codex CLIs respond
│   ├── test_pipeline_integrity.sh # Integration test: full loop in smoke mode
│   └── prompts/
│       ├── orchestrator_plan.md
│       ├── orchestrator_synthesize.md
│       ├── worker.md
│       ├── reviewer_claude.md
│       └── reviewer_codex.md
│
├── registry/                      # Version registry (additive-only)
│   ├── version_counter.json       # {"next_id": 5} — controller increments atomically per iteration
│   ├── champion.json              # {"version": "v0004", "promoted_at": "..."}
│   ├── gates.json                 # Promotion gate definitions — HUMAN-WRITE-ONLY
│   ├── v0/
│   │   ├── meta.json
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── changes_summary.md     # What changed vs prior version
│   │   └── model/                 # Saved XGBoost .ubj files (gzipped)
│   └── comparisons/
│       └── {batch_id}_iter{N}.json
│
├── handoff/                       # Agent completion signals (ephemeral per batch)
│   └── {batch_id}/
│       └── iter{N}/
│           ├── orchestrator_plan_done.json
│           ├── worker_done.json
│           ├── claude_reviewer_done.json
│           └── codex_reviewer_done.json
│
├── memory/
│   ├── direction_iter{N}.md       # Per-iteration task for Worker (ephemeral, cleaned at HUMAN_SYNC)
│   ├── human_input.md             # Human direction for current batch
│   │
│   ├── hot/                       # Tier 1 — always loaded (~10KB total, context-window safe)
│   │   ├── progress.md            # Current batch / iter / state
│   │   ├── champion.md            # Best model config + metrics + how to beat it
│   │   ├── critique_summary.md    # Compact: last 2 iters of reviewer feedback
│   │   ├── gate_calibration.md    # Rolling per-gate reviewer assessments
│   │   ├── learning.md            # Curated top-20 hard-won insights (capped ~2KB)
│   │   └── runbook.md             # Safety rules (static)
│   │
│   ├── warm/                      # Tier 2 — orchestrator + reviewers load on demand (~50KB)
│   │   ├── experiment_log.md      # Rolling: last 2 batches full, older = one-liners + archive pointer
│   │   ├── hypothesis_log.md      # Hypothesis → result → insight, last 2 batches
│   │   └── decision_log.md        # Why each change was chosen, last 2 batches
│   │
│   └── archive/                   # Tier 3 — long-term per-batch snapshots (never deleted)
│       ├── index.md               # One-liner per batch: date, best version, key insight
│       └── ${batch_id}/           # e.g. batch-20260226-001/ — batch_id is the full variable
│           ├── executive_summary.md
│           ├── experiment_log_full.md
│           ├── all_critiques.md       # Both reviewer outputs verbatim
│           ├── insights_extracted.md  # Key learnings from this batch
│           └── hypothesis_results.md
│
├── reviews/                       # Individual review outputs
│   └── {batch_id}_iter{N}_claude.md
│   └── {batch_id}_iter{N}_codex.md
│
├── reports/                       # Cross-version comparison tables + exec summaries
│   ├── {batch_id}_iter{N}_comparison.md   # Auto-generated by ml/compare.py
│   └── {batch_id}_summary.md              # Executive summary (after iter 3)
│
├── .logs/                         # Append-only archival logs
│   ├── audit.jsonl                # Per-watchdog-cycle agent probes
│   └── sessions/YYYY-MM-DD/
│       └── {batch_id}.jsonl       # Per-batch structured logs
│
└── state.json                     # Pipeline state machine (single source of truth)
```

---

## 5. Agent Specifications

### 5.1 Orchestrator

**Identity**: Claude Opus 4.6 (Max subscription)
**Launch** (`launch_orchestrator.sh --phase plan|synthesize` — phase selects prompt deterministically):
```bash
# launch_orchestrator.sh
PHASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)   PHASE="$2"; shift 2 ;;
    --phase=*) PHASE="${1#--phase=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
PHASE="${PHASE:-plan}"
case "$PHASE" in
  plan)       PROMPT="${PROJECT_DIR}/agents/prompts/orchestrator_plan.md" ;;
  synthesize) PROMPT="${PROJECT_DIR}/agents/prompts/orchestrator_synthesize.md" ;;
  *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac
SESSION="orch-${BATCH_ID}-iter${N}-${PHASE}"
# Use stdin redirect — never embed prompt via $(cat): LLM output in memory/ files
# can contain quotes, backticks, or $() that corrupt shell parsing.
tmux new-session -d -s "${SESSION}" \
  "cd \"${PROJECT_DIR}\" && claude --print --model opus \
   --allowedTools \"Read,Write,Edit,Glob,Grep,Bash\" \
   < \"${PROMPT}\""
echo "${SESSION}"  # controller writes orchestrator_tmux to state.json
```
**Timeout**: 600s

**Planning phase** (`ORCHESTRATOR_PLANNING`):
1. Read `memory/hot/` (all files), `memory/warm/` (all files), `memory/archive/index.md`
2. If iteration 1 and human input: read `memory/human_input.md`
3. Read `registry/gates.json`
4. Read `registry/{champion}/metrics.json` to understand current best baseline
5. Decide what change to try. Write `memory/direction_iter{N}.md` with:
   - What to change and why
   - Specific code changes (file paths, parameters, values)
   - Expected metric impact
6. Update `memory/hot/progress.md`
7. Write completion signal: `handoff/{batch_id}/iter{N}/orchestrator_plan_done.json`

**Synthesis phase** (`ORCHESTRATOR_SYNTHESIZING`):

*Input contract is conditional on worker outcome (controller sets `WORKER_FAILED` env var before launching):*

- **Worker succeeded** (normal path):
  1. Read `reviews/{batch_id}_iter{N}_claude.md` and `reviews/{batch_id}_iter{N}_codex.md` independently
  2. Read `reports/{batch_id}_iter{N}_comparison.md` (the cross-version table)
  3. Update `memory/warm/experiment_log.md`, `memory/warm/hypothesis_log.md`, `memory/warm/decision_log.md`

- **Worker failed** (failure path — review/comparison artifacts do not exist):
  1. Read `handoff/{batch_id}/iter{N}/worker_done.json` (contains error summary)
  2. Read `memory/hot/` and `memory/warm/` only
  3. Update `memory/warm/experiment_log.md` with failure entry (no hypothesis/decision update)

*Steps below apply to both paths:*
4. Update `memory/hot/critique_summary.md` and `memory/hot/gate_calibration.md`
5. If iteration < 3:
   a. Write `memory/direction_iter{N+1}.md` — synthesized plan for the next iteration (this is the verified artifact)
   b. Write `handoff/{batch_id}/iter{N}/orchestrator_synth_done.json` with sha256 of `direction_iter{N+1}.md`
6. If iteration == 3 (HUMAN_SYNC distillation):
   a. Write `memory/archive/${batch_id}/executive_summary.md` (full batch narrative — this is the primary artifact)
   b. Archive: copy `warm/` + both raw reviews + comparison reports to `memory/archive/${batch_id}/`
   c. Update `memory/archive/index.md` (append one-liner for this batch)
   d. Rewrite `memory/hot/learning.md`: distill top-20 insights from ALL archive batches
   e. Reset `memory/warm/` files to empty stubs
   f. Clean up `memory/direction_iter*.md` and other ephemeral files
   g. Write `handoff/{batch_id}/iter3/orchestrator_synth_done.json`

### 5.2 Worker

**Identity**: Claude Opus 4.6 (Max subscription), isolated git worktree
**Launch**:
```bash
# launch_worker.sh — prints session name to stdout; controller writes to state.json
SESSION="worker-${BATCH_ID}-iter${N}"
WORKTREE="${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}"   # batch_id prevents cross-batch collision
git worktree add "${WORKTREE}" -b "iter${N}-${BATCH_ID}"
tmux new-session -d -s "${SESSION}" \
  "cd \"${WORKTREE}\" && claude --print --model opus \
   --allowedTools \"Read,Write,Edit,Glob,Grep,Bash\" \
   --permission-mode default \
   < \"${PROJECT_DIR}/agents/prompts/worker.md\""
# cd into worktree BEFORE launching claude — enforces isolation; worker operates on its branch only
# Use stdin redirect (< file) not -p "$(cat ...)" — worker.md is static but consistent with other launches
echo "${SESSION}"   # controller captures this and writes worker_tmux to state.json
```
**Timeout**: 1800s

**Responsibilities** (`WORKER_RUNNING`):
1. Read `memory/direction_iter{N}.md`
2. Implement directed changes in worktree
3. Run: `pytest ml/tests/ -v` (up to 3 attempts on failure)
4. Run: `python ml/pipeline.py --auction-month 2021-07 --class-type onpeak --period-type f0 --version-id ${VERSION_ID} --overrides '{...}'`
   where `${VERSION_ID}` is read from `state.json["version_id"]` — globally unique across all batches
5. Validate: `registry/${VERSION_ID}/metrics.json` exists and has all required fields
6. Write `registry/${VERSION_ID}/changes_summary.md`:
   - What changed vs direction file
   - Tests run and result
   - Any anomalies during training
6a. Commit all changes to the worktree branch (local only — never push):
    `git add ml/ registry/${VERSION_ID}/ && git commit -m "iter${N} ${VERSION_ID}: <one-line summary>"`
    This is mandatory — the controller merges this branch; uncommitted changes are invisible to main.
7. Write completion signal: `handoff/{batch_id}/iter{N}/worker_done.json` with sha256 of `metrics.json`

**Sandbox constraints** (authoritative — must match prompt verbatim):
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and `handoff/{batch_id}/iter{N}/worker_done.json`
- NEVER modify `registry/v0/` (baseline is immutable)
- NEVER modify `registry/gates.json` or `ml/evaluate.py`
- NEVER modify any other `registry/v*/` directory
- NEVER run `rm -rf`
- NEVER delete any `registry/v*/` directory

### 5.3 Comparison Step (deterministic, non-AI)

After worker completes and before reviewers launch, the controller runs:

```bash
python ml/compare.py \
  --batch-id "${BATCH_ID}" \
  --iteration "${N}" \
  --output "reports/${BATCH_ID}_iter${N}_comparison.md"
```

`ml/compare.py` reads ALL `registry/v*/metrics.json` files and produces a standardized
Markdown table:

```
| Gate       | Floor  | v0(base) | v1    | v2    | vN(curr) | Champ | Pass? |
|------------|--------|----------|-------|-------|----------|-------|-------|
| S1-AUC     | 0.65   | 0.695    | 0.701 | 0.708 | 0.715    | 0.708 | YES   |
| S1-AP      | 0.12   | 0.218    | 0.220 | 0.215 | 0.225    | 0.220 | YES   |
| S1-VCAP@100| v0-0.05| 0.412    | 0.415 | 0.409 | 0.421    | 0.415 | YES   |
| S1-VCAP@500| v0-0.05| 0.388    | 0.391 | 0.385 | 0.395    | 0.391 | YES   |
| S1-VCAP@1k | v0-0.05| 0.365    | 0.369 | 0.362 | 0.374    | 0.369 | YES   |
| S1-NDCG    | v0-0.05| 0.541    | 0.548 | 0.537 | 0.555    | 0.548 | YES   |
| S1-BRIER   | v0+0.02| 0.089    | 0.087 | 0.091 | 0.085    | 0.087 | YES   |
| S1-REC     | 0.40   | 0.481    | 0.490 | 0.478 | 0.495    | 0.490 | YES   |
| S1-CAP@100 | v0-0.05| 0.520    | 0.528 | 0.514 | 0.535    | 0.528 | YES   |
| S1-CAP@500 | v0-0.05| 0.487    | 0.495 | 0.481 | 0.501    | 0.495 | YES   |
```

This is the primary artifact reviewers read for cross-model comparison. Numbers come
directly from `metrics.json` — no AI interpretation. Also writes
`registry/comparisons/{batch_id}_iter{N}.json` (machine-readable gate pass/fail).

### 5.4 Claude Reviewer

**Identity**: Claude Opus 4.6 (Max subscription)
**Launch**:
```bash
SESSION="rev-claude-${BATCH_ID}-iter${N}"
# cd to PROJECT_DIR so relative paths in prompt (memory/, reviews/, handoff/) resolve correctly.
# Stdin redirect avoids shell injection from LLM-generated memory content.
tmux new-session -d -s "${SESSION}" \
  "cd \"${PROJECT_DIR}\" && claude --print --model opus \
   --allowedTools \"Read,Glob,Grep,Write,Bash\" \
   < \"${PROJECT_DIR}/agents/prompts/reviewer_claude.md\""
# Write + Bash needed to write review file, handoff JSON, and compute sha256
# Prompt constrains writes to reviews/ and handoff/ only
echo "${SESSION}"  # controller writes claude_reviewer_tmux to state.json
```
**Timeout**: 1200s

**Review scope**:
1. **Implementation fidelity**: Read `memory/direction_iter{N}.md` and `registry/${VERSION_ID}/changes_summary.md`. Did the worker implement exactly what was directed?
2. **Results analysis**: Read `reports/{batch_id}_iter{N}_comparison.md`. Analyze trends across all versions. Flag anomalies (e.g., VCAP improving while BRIER worsens = possible overfit).
3. **Gate calibration** (mandatory, not optional): For each gate — is the floor too strict, too loose, or missing signal? Are Group B gates still apple-to-apple under the current threshold method?
4. **Architecture and code quality**: Read modified files. Bugs, inefficiencies, risks.
5. **Suggestions**: What should iteration N+1 try?

**Output format** (`reviews/{batch_id}_iter{N}_claude.md`):
```markdown
# Claude Review — Batch {id} Iteration {N}

## Implementation Fidelity
[Did worker implement exactly what orchestrator directed?]

## Results Analysis
[Cross-version trends from comparison table. Anomalies?]

## Gate Calibration Assessment
[For each gate: too strict / too loose / appropriate / missing signal]
[Specific floor adjustment suggestions (enforced only at HUMAN_SYNC)]

## Architecture and Code Quality
[Bugs, issues, risks]

## Suggestions for Next Iteration
[Ranked by expected impact]

## Verdict
[PASS / PASS_WITH_NOTES / FAIL]
[If FAIL: what must be fixed before proceeding]
```

After writing review, the reviewer computes sha256 and writes the unified handoff:
```bash
SHA=$(sha256sum reviews/${BATCH_ID}_iter${N}_claude.md | cut -d' ' -f1)
# Write handoff JSON matching the unified schema (same as all other agents)
```
Schema: `{agent, batch_id, iteration, artifact_path, sha256, created_at, status}`
This is enforced in the reviewer prompt's WHEN DONE section.

### 5.5 Codex Reviewer

**Identity**: gpt-5.3-codex via Codex CLI
**Launch** (corrected — `codex exec` has no `-o` flag; stdout redirect used):
```bash
SESSION="rev-codex-${BATCH_ID}-iter${N}"
# Relative paths for JSON (must match verify_handoff() get_expected_artifact() output).
# Absolute paths used only for actual file I/O — CWD of the tmux wrapper may vary.
REVIEW_FILE_REL="reviews/${BATCH_ID}_iter${N}_codex.md"
REVIEW_FILE_ABS="${PROJECT_DIR}/${REVIEW_FILE_REL}"
HANDOFF_REL="handoff/${BATCH_ID}/iter${N}/codex_reviewer_done.json"
HANDOFF_ABS="${PROJECT_DIR}/${HANDOFF_REL}"
# reviewer_codex.md MUST instruct Codex to print the full review to stdout (not write a file).
# --sandbox read-only means Codex cannot write files; the shell wrapper captures stdout into REVIEW_FILE_ABS.
# This is intentionally different from reviewer_claude.md which instructs Claude to write a file directly.
tmux new-session -d -s "${SESSION}" \
  "codex exec \
    -m ${CODEX_MODEL} \
    -c model_reasoning_effort=high \
    --sandbox read-only \
    --full-auto \
    -C ${PROJECT_DIR} \
    \"$(cat ${PROJECT_DIR}/agents/prompts/reviewer_codex.md)\" \
    > ${REVIEW_FILE_ABS} 2>&1 && \
   SHA=\$(sha256sum ${REVIEW_FILE_ABS} | cut -d' ' -f1) && \
   jq -n \
     --arg agent codex_reviewer \
     --arg producer codex_reviewer \
     --arg batch ${BATCH_ID} \
     --arg iter ${N} \
     --arg vid ${VERSION_ID} \
     --arg path \"${REVIEW_FILE_REL}\" \
     --arg sha \"\$SHA\" \
     --arg ts \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \
     '{agent:\$agent,producer:\$producer,batch_id:\$batch,iteration:(\$iter|tonumber),version_id:\$vid,artifact_path:\$path,sha256:\$sha,created_at:\$ts,status:\"done\"}' \
     > ${HANDOFF_ABS}"
echo "${SESSION}"  # controller writes codex_reviewer_tmux to state.json
```

**Note on `-p` flag**: In Claude CLI, `-p` = `--print` (non-interactive mode). In Codex CLI,
`-p` = `--profile`. These are different; do not confuse them.

**Review scope**: Same format and mandatory sections as the Claude reviewer. Codex provides
a fully independent perspective — statistical rigor, alternative interpretations, and
issues a same-family model might miss. Codex does NOT see Claude's review; independence
is the point.

**Timeout**: 1200s.

**Failure handling** — one artifact contract for all failure modes:

| Failure mode | Detection | Action |
|---|---|---|
| Timeout | Watchdog writes `handoff/.../timeout_REVIEW_CODEX.json` | Controller detects, proceeds to ORCHESTRATOR_SYNTHESIZING with Claude review only |
| Process crash | tmux session exits without handoff file | Watchdog detects dead session + no handoff, writes timeout artifact |
| Auth/model error | Codex exits non-zero, no handoff file | Same as crash path — watchdog catches on next cycle |

The review file is NOT used as a failure signal. Only the handoff JSON matters to the
controller. If Codex fails for any reason, the synthesizing orchestrator notes the absence
of Codex review and proceeds with Claude review alone.

---

## 6. Iteration Loop

### 6.1 Starting a batch

User types in Claude Code: `"start 3 iterations"` or `"lower threshold to 0.4 and run"`

Your interactive Claude Code session does NOT become the orchestrator. It:
1. Writes human direction to `memory/human_input.md` (if provided)
2. Runs `agents/run_pipeline.sh --batch-name "lower-threshold-04"` in background
3. Reports: "Pipeline started. Batch ID: batch-20260226-001. I'll notify you when done."

### 6.2 Single iteration flow (run_single_iter.sh)

```
0. Export shared vars (available to all child scripts):
   export BATCH_ID N VERSION_ID PROJECT_DIR
   export WORKER_FAILED=0   # explicit reset each iteration — bash exports persist across loop iterations
1. Acquire state.lock (flock). Verify state == expected_state (CAS check).
2. Set state = ORCHESTRATOR_PLANNING, entered_at = now, max_seconds = 600.
   Create handoff directory for this iteration (before any agent can write to it):
   mkdir -p "${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}"
3. ORCH_SESSION=$(launch_orchestrator.sh --phase plan); write orchestrator_tmux to state.json (controller-owned write)
4. Poll for handoff/.../orchestrator_plan_done.json (30s interval, timeout = max_seconds)
5. Verify sha256 of direction_iter{N}.md.
   Allocate globally unique version_id: flock registry/version_counter.json, read next_id,
   format as "v{next_id:04d}", increment and write back. Store version_id in state.json.
   CAS: state = WORKER_RUNNING, max_s = 1800.
6. WORKER_SESSION=$(launch_worker.sh); write worker_tmux to state.json (controller-owned write)
7. Poll for handoff/.../worker_done.json
7a. Read worker_done.json status.
    If "failed": export WORKER_FAILED=1; skip 7b-7d, transition directly to step 15 (ORCHESTRATOR_SYNTHESIZING).
    If "done":   WORKER_FAILED remains 0 (already reset in step 0); continue to 7b.
7b. Verify worker committed (branch head must differ from main HEAD — else fail fast):
    MAIN_HEAD=$(git -C "${PROJECT_DIR}" rev-parse HEAD)
    BRANCH_HEAD=$(git -C "${PROJECT_DIR}" rev-parse "iter${N}-${BATCH_ID}")
    [[ "$MAIN_HEAD" != "$BRANCH_HEAD" ]] || { echo "ERROR: worker made no commits"; exit 1; }
7c. Verify sha256 of metrics.json IN THE WORKTREE BRANCH before merging (fail fast before touching main):
    METRICS="${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}/registry/${VERSION_ID}/metrics.json"
    EXPECTED_SHA=$(jq -r '.sha256' "handoff/${BATCH_ID}/iter${N}/worker_done.json")
    ACTUAL_SHA=$(sha256sum "${METRICS}" | cut -d' ' -f1)
    [[ "$EXPECTED_SHA" == "$ACTUAL_SHA" ]] || { echo "ERROR: metrics.json sha256 mismatch"; exit 1; }
7d. Merge worker branch into main (brings ml/ + registry/${VERSION_ID}/ into main in one step):
    git -C "${PROJECT_DIR}" merge --no-ff "iter${N}-${BATCH_ID}" -m "Merge iter${N} worker: ${VERSION_ID}"
    # No pre-merge rsync — worker committed registry/${VERSION_ID}/ to its branch, merge handles it cleanly.
8. Run ml/compare.py (non-AI comparison table). sha256 already verified in step 7c.
9. CAS: state = REVIEW_CLAUDE, max_s = 1200.
10. CLAUDE_SESSION=$(launch_reviewer_claude.sh); write claude_reviewer_tmux to state.json
11. Poll for handoff/.../claude_reviewer_done.json
12. CAS: state = REVIEW_CODEX, max_s = 1200.
13. CODEX_SESSION=$(launch_reviewer_codex.sh); write codex_reviewer_tmux to state.json
14. Poll for handoff/.../codex_reviewer_done.json (or timeout notice file)
15. CAS: state = ORCHESTRATOR_SYNTHESIZING, max_s = 600.
16. ORCH_SESSION=$(launch_orchestrator.sh --phase synthesize); write orchestrator_tmux to state.json
17. Poll for handoff/.../orchestrator_synth_done.json
18. CAS: state = IDLE (or ORCHESTRATOR_PLANNING for next iter, or HUMAN_SYNC after iter 3)
```

### 6.3 Both shell script scenarios

**Scenario A — Human-triggered command**:
1. Human writes direction to `memory/human_input.md`
2. Controller starts at `IDLE`, proceeds through full loop
3. Orchestrator reads `human_input.md` at planning phase
4. After reviewers complete, orchestrator synthesizes and plans next iteration

**Scenario B — Continue from prior round**:
1. State is `HUMAN_SYNC` (after previous 3-iter batch)
2. Human clears `HUMAN_SYNC` by running `./agents/run_pipeline.sh`
3. Orchestrator reads `memory/hot/` and `memory/warm/` for accumulated context
4. No human input file required — orchestrator decides next change autonomously

---

## 7. Promotion Gates

### 7.1 Human-write-only files

`registry/gates.json` and `ml/evaluate.py` are **never modified by any agent**. Changes
require explicit human approval at HUMAN_SYNC. Reviewers may critique and recommend
changes; enforcement is a human action before the next batch starts.

### 7.2 Comprehensive gate set

**Group A — Threshold-independent (always apple-to-apple)**

| Gate ID | Metric | Direction | Absolute Floor | Notes |
|---------|--------|-----------|---------------|-------|
| S1-AUC | AUC-ROC | higher | 0.65 | Standard rank quality |
| S1-AP | Avg Precision | higher | 0.12 | Imbalanced-class rank quality |
| S1-VCAP@100 | Value Capture@100 (prob-ranked) | higher | v0 − 0.05 | Of top-100 actual SP, how much value captured by top-100 predicted |
| S1-VCAP@500 | Value Capture@500 (prob-ranked) | higher | v0 − 0.05 | |
| S1-VCAP@1000 | Value Capture@1000 (prob-ranked) | higher | v0 − 0.05 | Primary money metric |
| S1-NDCG | NDCG@1000 (prob-ranked vs actual SP) | higher | v0 − 0.05 | Ranking fidelity |
| S1-BRIER | Brier Score | lower | v0 + 0.02 | Calibration; must not worsen |

**Group B — Threshold-dependent (apple-to-apple when threshold method is frozen)**

| Gate ID | Metric | Direction | Absolute Floor | Notes |
|---------|--------|-----------|---------------|-------|
| S1-REC | Recall at optimized threshold | higher | 0.40 | Prevents precision-only collapse |
| S1-CAP@100 | Capture@100 (binary pred) | higher | v0 − 0.05 | Of top-100 by actual SP, fraction predicted binding |
| S1-CAP@500 | Capture@500 (binary pred) | higher | v0 − 0.05 | |

**Apple-to-apple rule for Group B**: If the threshold optimization method changes
(e.g., beta changes from 0.7 to 0.5), v0 must be re-evaluated under the new method
before Group B comparison is valid. This is a HUMAN_SYNC action item flagged by reviewers.

**Monitoring-only** (tracked in every `metrics.json`, never gate): Precision at threshold,
F1, F-beta (β=2.0), CAP@250, CAP@1000, VCAP@250.

### 7.3 Floor initialization

After v0 runs, the controller auto-populates v0-relative floors in `gates.json` (one-time
initialization). Until then, v0-relative floors are `null` with `"pending_v0": true`.
No candidate can be promoted until all `null` floors are resolved.

**v0 baseline creation** (one-time bootstrap, before any pipeline run):
`registry/v0/` is created by re-training with the **source repo's config** using the ported
pipeline code. This produces a config-equivalent baseline without copying pre-trained weights:
```bash
python ml/pipeline.py --version-id v0 --auction-month 2021-07 --class-type onpeak \
  --period-type f0
```
Set `random_state` in `ml/config.py` to a fixed seed so v0 is reproducible. Minor metric
deviations from the source repo's historical result are expected and acceptable — v0 is the
new reference point for all gate floors in this pipeline.

**`run_pipeline.sh` guard**: `run_pipeline.sh` aborts if `registry/v0/metrics.json` is absent
(see Section 12.3 loop sketch).

**Explicit assignment**: `run_pipeline.sh` calls `python ml/populate_v0_gates.py` before
launching iteration 1 of the very first batch. This script reads `registry/v0/metrics.json`,
computes `v0_metric − offset` for each pending gate (e.g., `v0_AUC − 0.05`), writes back
to `registry/gates.json`, and sets `"pending_v0": false`. It is idempotent — re-running
when no `pending_v0: true` entries exist is a no-op.

```json
{
  "version": 1,
  "effective_since": "2026-02-26",
  "noise_tolerance": 0.02,
  "gates": {
    "S1-AUC":      {"floor": 0.65,  "direction": "higher", "pending_v0": false},
    "S1-AP":       {"floor": 0.12,  "direction": "higher", "pending_v0": false},
    "S1-VCAP@100": {"floor": null,  "direction": "higher", "pending_v0": true},
    "S1-VCAP@500": {"floor": null,  "direction": "higher", "pending_v0": true},
    "S1-VCAP@1000":{"floor": null,  "direction": "higher", "pending_v0": true},
    "S1-NDCG":     {"floor": null,  "direction": "higher", "pending_v0": true},
    "S1-BRIER":    {"floor": null,  "direction": "lower",  "pending_v0": true},
    "S1-REC":      {"floor": 0.40,  "direction": "higher", "pending_v0": false, "group": "B"},
    "S1-CAP@100":  {"floor": null,  "direction": "higher", "pending_v0": true,  "group": "B"},
    "S1-CAP@500":  {"floor": null,  "direction": "higher", "pending_v0": true,  "group": "B"}
  }
}
```

### 7.4 Promotion logic

A candidate version is promotable when:
1. All gate floors pass (absolute and v0-relative)
2. No gate regresses more than `noise_tolerance = 0.02` vs champion
3. At least one gate improves more than `noise_tolerance` vs champion

**Corrected example** (v1 delta):
- S1-AUC: +0.015 → passes (beats champion by more than noise_tol)
- S1-AP: −0.025 → FAILS (regresses more than noise_tol 0.02)
- S1-AP: −0.013 → passes (|−0.013| = 0.013 < 0.02, within tolerance)

### 7.5 Gate modification protocol

1. Reviewer suggests gate change in their review
2. Orchestrator includes it in HUMAN_SYNC executive summary
3. Human approves and manually edits `registry/gates.json` before next batch
4. If a Group B metric changes its method: v0 re-evaluated, floors updated, prior
   versions marked `"group_b_stale": true` in their comparison records

---

## 8. Handshake Contracts

Each agent writes a completion signal file as its last action. The controller verifies
it before transitioning state.

**Canonical handoff schema** — all agents use the same keys; `version_id` is `null` for
pre-worker states (planning/synthesis) where no version has been allocated yet:

Success case:
```json
{
  "agent": "worker",
  "producer": "worker",
  "batch_id": "batch-20260226-001",
  "iteration": 1,
  "version_id": "v0004",
  "artifact_path": "registry/v0004/metrics.json",
  "sha256": "abc123def456...",
  "created_at": "2026-02-26T14:45:00Z",
  "status": "done"
}
```

Failure case (tests fail 3×, train error, or unrecoverable exception):
```json
{
  "agent": "worker",
  "producer": "worker",
  "batch_id": "batch-20260226-001",
  "iteration": 1,
  "version_id": "v0004",
  "artifact_path": null,
  "sha256": null,
  "created_at": "2026-02-26T14:45:00Z",
  "status": "failed",
  "error": "pytest failed 3 times: <last error summary>"
}
```

Controller behavior on `status: "failed"`: skip artifact sync (steps 7b-7c), skip comparison (step 8),
write failure note to `memory/hot/progress.md`, transition directly to `ORCHESTRATOR_SYNTHESIZING`.
Synthesis orchestrator reads the failure handoff and includes a failure analysis in its report.

**Per-state version_id policy**:

| State | version_id in handoff |
|-------|----------------------|
| ORCHESTRATOR_PLANNING | `null` — version not yet allocated |
| WORKER_RUNNING | required — allocated by controller before worker launch |
| REVIEW_CLAUDE | required — same version_id as worker |
| REVIEW_CODEX | required — same version_id as worker |
| ORCHESTRATOR_SYNTHESIZING | `null` — synthesis spans the full iteration, not one version |

Controller verification (status-aware, paths computed dynamically at call time):
```bash
# In state_utils.sh

# Compute expected artifact path at call time (never from a pre-declared array —
# bash declare -A evaluates values at declaration time, so ${VERSION_ID} would be empty).
get_expected_artifact() {
  local state="$1"
  local batch_id=$(jq -r '.batch_id' "$STATE_FILE")
  local iter=$(jq -r '.iteration' "$STATE_FILE")
  local version_id=$(jq -r '.version_id // empty' "$STATE_FILE")
  case "$state" in
    WORKER_RUNNING)            echo "registry/${version_id}/metrics.json" ;;
    REVIEW_CLAUDE)             echo "reviews/${batch_id}_iter${iter}_claude.md" ;;
    REVIEW_CODEX)              echo "reviews/${batch_id}_iter${iter}_codex.md" ;;
    ORCHESTRATOR_PLANNING)     echo "memory/direction_iter${iter}.md" ;;
    ORCHESTRATOR_SYNTHESIZING)
      if (( iter == 3 )); then echo "memory/archive/${batch_id}/executive_summary.md"
      else                     echo "memory/direction_iter$((iter+1)).md"
      fi ;;
    *) echo ""; return 1 ;;
  esac
}

verify_handoff() {
  local handoff_file="$1" state="$2"
  local status=$(jq -r '.status' "$handoff_file")

  # Failure handoffs: artifact_path and sha256 are null by design — only require error field
  if [[ "$status" == "failed" ]]; then
    local error=$(jq -r '.error // empty' "$handoff_file")
    [[ -n "$error" ]] || { echo "Failed handoff missing 'error' field"; return 1; }
    return 0
  fi

  # Success handoffs: enforce path match + sha256
  local reported_path=$(jq -r '.artifact_path' "$handoff_file")
  local expected_path
  expected_path=$(get_expected_artifact "$state") || { echo "Unknown state: $state"; return 1; }
  # Step 1: path must match controller-derived expectation (not agent-reported)
  [[ "$reported_path" == "$expected_path" ]] || { echo "Path mismatch: $reported_path != $expected_path"; return 1; }
  # Step 2: hash must match
  local expected_sha=$(jq -r '.sha256' "$handoff_file")
  local actual_sha=$(sha256sum "$expected_path" | cut -d' ' -f1)
  [[ "$expected_sha" == "$actual_sha" ]]
}
```

---

## 9. Memory System

Memory is tiered. Each tier trades recency and detail for context size.

### Tier 1 — Hot (`memory/hot/`, ~10KB total, always loaded)

| File | Owner | Contents |
|------|-------|----------|
| `progress.md` | Orchestrator | Current batch, iter, state, next action |
| `champion.md` | Orchestrator | Best model config + all gate metrics + identified weaknesses |
| `critique_summary.md` | Orchestrator | Compact synthesis of last 2 iterations of reviewer feedback |
| `gate_calibration.md` | Orchestrator | Per-gate: history of reviewer assessments (too strict / loose / appropriate) |
| `learning.md` | Orchestrator | Curated top-20 hard-won insights and anti-patterns; never grows beyond ~2KB |
| `runbook.md` | Orchestrator | Safety rules, operating procedures (static) |

### Tier 2 — Warm (`memory/warm/`, ~50KB, loaded by orchestrator + reviewers)

| File | Owner | Contents |
|------|-------|----------|
| `experiment_log.md` | Orchestrator | Rolling: last 2 batches in full detail; older entries = one-liner + archive pointer |
| `hypothesis_log.md` | Orchestrator | Per-iter: hypothesis → result → insight, last 2 batches |
| `decision_log.md` | Orchestrator | Why each change was chosen, last 2 batches |

### Tier 3 — Archive (`memory/archive/`, never deleted)

| File | Contents |
|------|----------|
| `index.md` | One-liner per batch: date, versions run, best result, key insight, link |
| `${batch_id}/executive_summary.md` | Full executive summary from HUMAN_SYNC |
| `${batch_id}/experiment_log_full.md` | Complete experiment log snapshot at batch end |
| `${batch_id}/all_critiques.md` | Both reviewer outputs verbatim (not aggregated); if Codex failed all 3 iterations, file includes a "Codex unavailable" note instead of missing |
| `${batch_id}/insights_extracted.md` | What was learned from this batch |
| `${batch_id}/hypothesis_results.md` | All hypotheses and outcomes |

### Distillation at HUMAN_SYNC

At the end of every 3-iteration batch the orchestrator runs a distillation step:
1. Reads all of `warm/` + both raw review files + comparison reports
2. Archives everything to `archive/${batch_id}/`
3. Updates `archive/index.md` (one new line)
4. Rewrites `hot/learning.md`: curate top-20 insights across ALL archive batches — drop superseded lessons, add new ones
5. Resets `warm/` files to empty stubs (fresh for next batch)
6. Updates `hot/critique_summary.md` and `hot/gate_calibration.md` from this batch

This keeps hot/ permanently compact and smart regardless of how many batches have run.

### Per-agent context slices

| Agent | Memory loaded |
|-------|--------------|
| Orchestrator (plan) | hot/ (all) + warm/ (all) + archive/index.md |
| Worker | direction_iter{N}.md + hot/champion.md + hot/learning.md + hot/runbook.md |
| Claude Reviewer | direction_iter{N}.md + changes_summary + comparison table + warm/experiment_log + hot/gate_calibration + warm/decision_log + gates.json + codebase |
| Codex Reviewer | Same as Claude reviewer — does NOT see Claude's review |
| Orchestrator (synth) | Both raw reviews (read independently) + comparison table + warm/ (all) |

**Archival logs**: Full structured JSONL lives in `.logs/sessions/` (append-only). Memory
files are narrative summaries built from archival content; truncating them does not lose
traceability.

---

## 10. Prompt Design

### 10.1 Orchestrator prompt (planning phase) — structure

```
IDENTITY: You are the Orchestrator for an ML research pipeline.

READ (in order):
- memory/hot/ (all: progress, champion, critique_summary, gate_calibration, learning, runbook)
- memory/warm/ (all: experiment_log, hypothesis_log, decision_log)
- memory/archive/index.md
- memory/human_input.md (if exists — human direction for this batch)
- registry/gates.json (current promotion gates)
- registry/{champion}/metrics.json

YOUR TASK:
Decide what change to try in iteration {N}. Consider reviewer suggestions, human input,
what has not been tried, expected gate impact.

WRITE:
- memory/direction_iter{N}.md (specific instructions for Worker: file, function, parameter,
  expected value, rationale)
- memory/warm/decision_log.md (append: why this change was chosen)
- memory/hot/progress.md (updated position)
- handoff/{batch_id}/iter{N}/orchestrator_plan_done.json

CONSTRAINTS:
- Do NOT modify any ML code or registry/ files
- Do NOT modify registry/gates.json or ml/evaluate.py (human-write-only)
- Do NOT run training
```

### 10.2 Worker prompt — structure

```
IDENTITY: You are the Worker for an ML research pipeline.

READ:
- memory/direction_iter{N}.md (what to implement — your primary instruction)
- memory/hot/champion.md (baseline model config — understand what v0 looks like)
- memory/hot/learning.md (hard-won insights and anti-patterns — read before coding)
- memory/hot/runbook.md (safety rules — mandatory)
- ml/ codebase

YOUR TASK:
1. Implement changes described in direction_iter{N}.md
2. Run: pytest ml/tests/ -v (fix and retry up to 3x)
3. Run: python ml/pipeline.py --auction-month 2021-07 --class-type onpeak
         --period-type f0 --version-id ${VERSION_ID} --overrides '{...}'
4. Write registry/${VERSION_ID}/changes_summary.md (what changed, tests run, anomalies)
5. Commit all changes locally (MANDATORY — controller merges this branch):
   git add ml/ registry/${VERSION_ID}/ && git commit -m "iter${N} ${VERSION_ID}: <one-line summary>"
6. Write handoff/{batch_id}/iter{N}/worker_done.json with sha256 of metrics.json

CONSTRAINTS:
- Only modify files under ml/, registry/${VERSION_ID}/, and handoff/{batch_id}/iter{N}/worker_done.json
- NEVER touch registry/v0/, registry/gates.json, ml/evaluate.py
- NEVER delete any registry/v*/ directory
- If tests fail 3x or training raises an unrecoverable exception:
    write handoff/{batch_id}/iter{N}/worker_done.json with status "failed" and error summary, then stop
    (do NOT commit on failure — nothing to preserve)
```

### 10.3 Reviewer prompt — structure

```
IDENTITY: You are a Reviewer for an ML research pipeline.

READ:
- memory/direction_iter{N}.md (what was supposed to happen)
- registry/${VERSION_ID}/changes_summary.md (worker's self-report)
- reports/{batch_id}_iter{N}_comparison.md (cross-version table — primary reference)
- registry/${VERSION_ID}/metrics.json (raw metrics)
- registry/gates.json (current gate definitions)
- memory/hot/gate_calibration.md (prior gate assessments across batches)
- memory/warm/experiment_log.md (full history of versions and outcomes)
- memory/warm/decision_log.md (why this change was chosen)
- ml/ codebase (for code quality review)

DO NOT read the other reviewer's output — reviews must be fully independent.

REVIEW (all six sections are mandatory):
1. Implementation Fidelity: worker did what orchestrator asked?
2. Results Analysis: cross-version trends, anomalies in the comparison table
3. Gate Calibration: for each gate — too strict / too loose / appropriate?
   Suggest specific adjustments (enforcement deferred to HUMAN_SYNC)
4. Architecture and Code Quality
5. Suggestions for Next Iteration (ranked)
6. Verdict: PASS / PASS_WITH_NOTES / FAIL

WRITE: reviews/{batch_id}_iter{N}_{reviewer_id}.md

WHEN DONE: compute sha256 of your review file and write the unified handoff JSON:
  handoff/{batch_id}/iter{N}/{reviewer_id}_done.json
  Schema: {agent, batch_id, iteration, artifact_path, sha256, created_at, status:"done"}

CONSTRAINTS:
- Only write to reviews/ and handoff/ — no other file modifications
- You MAY freely critique gates as stale, insufficient, or miscalibrated
- Enforcement of gate changes requires human approval at HUMAN_SYNC
```

This is `reviewer_claude.md` — Claude writes files directly and computes sha256 itself.

### 10.4 Codex reviewer prompt — structure (`reviewer_codex.md`)

**Diverged from `reviewer_claude.md`**: Codex runs in `--sandbox read-only` and cannot write
files. The shell wrapper captures stdout and writes the review file and handoff JSON on Codex's
behalf. The prompt must NOT instruct Codex to write files.

```
IDENTITY: You are a Reviewer for an ML research pipeline.

READ:
- memory/direction_iter{N}.md (what was supposed to happen)
- registry/${VERSION_ID}/changes_summary.md (worker's self-report)
- reports/{batch_id}_iter{N}_comparison.md (cross-version table — primary reference)
- registry/${VERSION_ID}/metrics.json (raw metrics)
- registry/gates.json (current gate definitions)
- memory/hot/gate_calibration.md (prior gate assessments across batches)
- memory/warm/experiment_log.md (full history of versions and outcomes)
- memory/warm/decision_log.md (why this change was chosen)
- ml/ codebase (for code quality review)

DO NOT read the other reviewer's output — reviews must be fully independent.

REVIEW (all six sections are mandatory):
1. Implementation Fidelity: worker did what orchestrator asked?
2. Results Analysis: cross-version trends, anomalies in the comparison table
3. Gate Calibration: for each gate — too strict / too loose / appropriate?
   Suggest specific adjustments (enforcement deferred to HUMAN_SYNC)
4. Architecture and Code Quality
5. Suggestions for Next Iteration (ranked)
6. Verdict: PASS / PASS_WITH_NOTES / FAIL

WHEN DONE:
Print the full review to stdout in the format above. Do NOT write any files.
The shell wrapper that launched you will capture your stdout and write:
  - reviews/{batch_id}_iter{N}_codex.md
  - handoff/{batch_id}/iter{N}/codex_reviewer_done.json
You do not need to take any file-write action.

CONSTRAINTS:
- Do NOT write any files (sandbox is read-only)
- You MAY freely critique gates as stale, insufficient, or miscalibrated
- Enforcement of gate changes requires human approval at HUMAN_SYNC
```

---

## 11. Watchdog / Cron Job

### 11.1 Installation

```bash
# Idempotent — will not add duplicate entry if watchdog is already installed
CRON_ENTRY="* * * * * ${PROJECT_DIR}/agents/watchdog.sh >> /tmp/watchdog.log 2>&1"
crontab -l 2>/dev/null | grep -qF "${PROJECT_DIR}/agents/watchdog.sh" \
  || (crontab -l 2>/dev/null; echo "${CRON_ENTRY}") | crontab -
```

### 11.2 Watchdog logic (READ-ONLY — never writes state.json)

```bash
# agents/watchdog.sh
source agents/config.sh
STATE_FILE="${PROJECT_DIR}/state.json"
AUDIT_LOG="${PROJECT_DIR}/.logs/audit.jsonl"

state=$(jq -r '.state' "$STATE_FILE")
[[ "$state" == "IDLE" || "$state" == "HUMAN_SYNC" ]] && exit 0

# Structured probe per agent
entered_at=$(jq -r '.entered_at' "$STATE_FILE")
max_s=$(jq -r '.max_seconds' "$STATE_FILE")
elapsed=$(( $(date +%s) - $(date -d "$entered_at" +%s) ))
# Probe the active session for the current state (all agents tracked)
case "$state" in
  ORCHESTRATOR_PLANNING|ORCHESTRATOR_SYNTHESIZING)
    ACTIVE_SESSION=$(jq -r '.orchestrator_tmux // empty' "$STATE_FILE") ;;
  WORKER_RUNNING)
    ACTIVE_SESSION=$(jq -r '.worker_tmux // empty' "$STATE_FILE") ;;
  REVIEW_CLAUDE)
    ACTIVE_SESSION=$(jq -r '.claude_reviewer_tmux // empty' "$STATE_FILE") ;;
  REVIEW_CODEX)
    ACTIVE_SESSION=$(jq -r '.codex_reviewer_tmux // empty' "$STATE_FILE") ;;
  *)
    ACTIVE_SESSION="" ;;
esac
SESSION_ALIVE=false
[[ -n "$ACTIVE_SESSION" ]] && tmux has-session -t "$ACTIVE_SESSION" 2>/dev/null && SESSION_ALIVE=true

# Build audit entry
probe=$(jq -n \
  --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg st "$state" \
  --arg el "$elapsed" \
  --arg sess "$ACTIVE_SESSION" \
  --argjson alive "$SESSION_ALIVE" \
  '{timestamp: $ts, state: $st, elapsed_s: ($el|tonumber), active_session: $sess, tmux_alive: $alive}')
echo "$probe" >> "$AUDIT_LOG"

# Disk check
registry_mb=$(du -sm "${PROJECT_DIR}/registry" | cut -f1)
if (( registry_mb > REGISTRY_DISK_LIMIT_MB )); then
  echo "ALERT: registry/ exceeds ${REGISTRY_DISK_LIMIT_MB} MB (actual: ${registry_mb} MB)" >> /tmp/watchdog_alerts.log
fi

# Derive timeout artifact path (shared by both crash detection and elapsed timeout)
ITER_N=$(jq -r '.iteration' "$STATE_FILE")
BATCH_ID_W=$(jq -r '.batch_id' "$STATE_FILE")
TIMEOUT_FILE="${PROJECT_DIR}/handoff/${BATCH_ID_W}/iter${ITER_N}/timeout_${state}.json"
HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID_W}/iter${ITER_N}"

# Canonical state-to-handoff filename map (must match controller and handoff/ directory)
declare -A STATE_TO_HANDOFF=(
  ["WORKER_RUNNING"]="worker_done.json"
  ["REVIEW_CLAUDE"]="claude_reviewer_done.json"
  ["REVIEW_CODEX"]="codex_reviewer_done.json"
  ["ORCHESTRATOR_PLANNING"]="orchestrator_plan_done.json"
  ["ORCHESTRATOR_SYNTHESIZING"]="orchestrator_synth_done.json"
)

# Immediate crash/auth-error detection: session dead + no handoff file yet
# This fires on every watchdog cycle — no need to wait full timeout
HANDOFF_EXISTS=false
HANDOFF_FNAME="${STATE_TO_HANDOFF[$state]:-}"
[[ -n "$HANDOFF_FNAME" && -f "${HANDOFF_DIR}/${HANDOFF_FNAME}" ]] && HANDOFF_EXISTS=true
if [[ "$SESSION_ALIVE" == "false" ]] && [[ "$HANDOFF_EXISTS" == "false" ]] \
   && [[ "$state" == "REVIEW_CODEX" || "$state" == "REVIEW_CLAUDE" || "$state" == "WORKER_RUNNING" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    echo "{\"timeout\": false, \"crash\": true, \"state\": \"${state}\", \"elapsed_s\": ${elapsed}}" > "$TIMEOUT_FILE"
  fi
fi

# Elapsed timeout handling (writes to handoff/, NOT state.json)
if (( elapsed > max_s )); then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    echo "{\"timeout\": true, \"state\": \"${state}\", \"elapsed_s\": ${elapsed}}" > "$TIMEOUT_FILE"
  fi
  # Kill stuck tmux session for whatever agent is active
  if [[ -n "$ACTIVE_SESSION" ]] && "$SESSION_ALIVE"; then
    tmux capture-pane -t "$ACTIVE_SESSION" -p -S -100 > /tmp/stuck_${state}.log 2>/dev/null
    tmux kill-session -t "$ACTIVE_SESSION" 2>/dev/null
  fi
fi
```

The controller polls for `timeout_${state}.json` alongside normal handoff files.
On timeout detection, the controller decides next action (skip review, spawn debugger,
or escalate to HUMAN_SYNC).

### 11.3 Timeout budgets

| State | Timeout | Rationale |
|-------|---------|-----------|
| ORCHESTRATOR_PLANNING | 600s | File reads + reasoning |
| WORKER_RUNNING | 1800s | Train + eval + tests |
| REVIEW_CLAUDE | 1200s | Read code + write review |
| REVIEW_CODEX | 1200s | Same |
| ORCHESTRATOR_SYNTHESIZING | 600s | Read reviews + write summary |

---

## 12. Infrastructure Configuration

### 12.1 `agents/config.sh`

```bash
# All environment-specific settings. Source this at top of every script.
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage1-shadow"
RAY_ADDRESS="ray://10.8.0.36:10001"
DATA_ROOT="/opt/temp/tmp/pw_data/spice6"
VENV_ACTIVATE="/home/xyz/workspace/pmodel/.venv/bin/activate"
SMOKE_TEST=false          # true = synthetic data, no Ray, full loop in <30s
REGISTRY_DISK_LIMIT_MB=10240
CODEX_MODEL="gpt-5.3-codex"   # confirm model ID on subscription before first run (Open Decision #1)
```

### 12.2 CLI invocation reference

| Flag | Claude CLI | Codex CLI |
|------|-----------|-----------|
| `-p` | `--print` (non-interactive) | `--profile` (config profile) |
| `-w` | `--worktree [name]` | N/A |
| `-m` | `--model [name]` | `--model [name]` |
| `--sandbox` | N/A | `read-only`, `workspace-write` |
| `--full-auto` | N/A | Automated sandbox |

Claude Max subscription auth: OAuth via `~/.claude/.credentials.json` (no API key needed).
Codex auth: `codex auth login` or `OPENAI_API_KEY` env var.

### 12.3 Batch lock

```bash
# run_pipeline.sh
LOCK_FILE="${PROJECT_DIR}/state.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Pipeline already running. Check state.json."; exit 1; }
export PIPELINE_LOCKED=1   # guards run_single_iter.sh against direct invocation

current_state=$(jq -r '.state' "${PROJECT_DIR}/state.json")
if [[ "$current_state" != "IDLE" && "$current_state" != "HUMAN_SYNC" ]]; then
  echo "Cannot start: state is ${current_state}. Wait for IDLE or HUMAN_SYNC."
  exit 1
fi

# Defensive cleanup: remove stale direction files from any prior crashed batch
# Orchestrator overwrites them anyway, but this eliminates the staleness window.
rm -f "${PROJECT_DIR}/memory/direction_iter"*.md
```

`run_single_iter.sh` guard — prevents direct invocation without the lock:
```bash
# Top of run_single_iter.sh
[[ -n "${PIPELINE_LOCKED:-}" ]] || { echo "ERROR: run_single_iter.sh must be called via run_pipeline.sh"; exit 1; }
```

**`run_pipeline.sh` iteration loop** (full sketch):
```bash
#!/usr/bin/env bash
set -euo pipefail
source "${PROJECT_DIR}/agents/config.sh"

# --- Batch setup ---
BATCH_NAME="${1:---batch-$(date +%Y%m%d-%H%M%S)}"   # optional name arg
BATCH_ID="batch-$(date +%Y%m%d-%H%M%S)"
export PROJECT_DIR BATCH_ID

# --- Acquire lock ---
LOCK_FILE="${PROJECT_DIR}/state.lock"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Pipeline already running. Check state.json."; exit 1; }
export PIPELINE_LOCKED=1

# --- Pre-run checks ---
current_state=$(jq -r '.state' "${PROJECT_DIR}/state.json")
if [[ "$current_state" != "IDLE" && "$current_state" != "HUMAN_SYNC" ]]; then
  echo "Cannot start: state is ${current_state}. Wait for IDLE or HUMAN_SYNC."
  exit 1
fi

# Guard: v0 baseline must exist before any iteration runs
[[ -f "${PROJECT_DIR}/registry/v0/metrics.json" ]] \
  || { echo "ERROR: registry/v0/metrics.json not found. Run baseline extraction first."; exit 1; }

# Defensive cleanup: remove stale direction files from any prior crashed batch
rm -f "${PROJECT_DIR}/memory/direction_iter"*.md

# Populate v0-relative gate floors (idempotent — no-op if already populated)
python "${PROJECT_DIR}/ml/populate_v0_gates.py"

# --- 3-iteration loop ---
for N in 1 2 3; do
  export N
  echo "=== Starting iteration ${N} of 3 (batch: ${BATCH_ID}) ==="
  bash "${PROJECT_DIR}/agents/run_single_iter.sh"

  # After iter 3, state will be HUMAN_SYNC — break
  current_state=$(jq -r '.state' "${PROJECT_DIR}/state.json")
  [[ "$current_state" == "HUMAN_SYNC" ]] && break
done

echo "=== Batch ${BATCH_ID} complete. State: $(jq -r '.state' "${PROJECT_DIR}/state.json") ==="
```

### 12.4 Smoke test mode

When `SMOKE_TEST=true` in `config.sh`:
- `ml/data_loader.py` returns 100 rows of synthetic data (14 features, random binary labels)
- Ray is not initialized
- Training completes in ~5 seconds
- Full agent loop (orchestrator → worker → 2 reviewers → orchestrator) completes in <2 minutes
- Used to validate the entire pipeline without infrastructure

### 12.5 Disk budget

XGBoost `.ubj` models for minimal slice: ~1–5 MB per version (gzipped ~0.5–1 MB).
100 versions ≈ 100–500 MB — not a real concern. Policy:
- Gzip models on write: `gzip registry/${VERSION_ID}/model/*.ubj`
- Log size each iteration: `du -sh registry/ >> .logs/audit.jsonl`
- Hard stop at 10 GB: watchdog alerts (see Section 11.2)

---

## 13. Pipeline Integrity Testing

> **See also**: [`docs/plans/2026-02-26-verification-plan.md`](2026-02-26-verification-plan.md) —
> comprehensive verification spec: final design review findings (R1–R5), explicit unit test
> assertions for every shell contract, integration smoke test tiers, failure triage protocol,
> and pre-run checklist.

Before the first real batch, and after any infrastructure change, run the integrity
test suite to validate each component works end-to-end.

### 13.1 Pre-flight CLI check (`agents/check_clis.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
source agents/config.sh

echo "=== Checking Claude CLI ==="
result=$(claude -p "Reply with exactly: CLAUDE_OK" --model opus 2>&1)
[[ "$result" == *"CLAUDE_OK"* ]] && echo "Claude: OK" || { echo "Claude: FAIL — $result"; exit 1; }

echo "=== Checking Codex CLI ==="
result=$(codex exec -m ${CODEX_MODEL} -s read-only --full-auto \
  "Print exactly the text: CODEX_OK and nothing else" 2>&1)
[[ "$result" == *"CODEX_OK"* ]] && echo "Codex: OK" || { echo "Codex: FAIL — $result"; exit 1; }

echo "All CLIs ready."
```

### 13.2 Full pipeline integrity test (`agents/test_pipeline_integrity.sh`)

Runs the complete 1-iteration loop in smoke mode (`SMOKE_TEST=true`). Asserts each
component works in sequence. Should complete in under 3 minutes.

```
What is tested (1-iteration smoke loop — iteration 1 of 3):
1. Batch lock acquired (state.lock)
2. Orchestrator launches, writes direction_iter1.md, writes handoff signal
3. State transitions IDLE → ORCHESTRATOR_PLANNING → WORKER_RUNNING (CAS verified)
4. Worker launches in worktree, writes metrics.json, writes changes_summary.md, commits, writes handoff
4a. sha256 of metrics.json verified against worktree branch (before merge)
4b. Worker branch merged into main — registry/${VERSION_ID}/ and ml/ now visible in PROJECT_DIR via git merge
5. ml/compare.py runs and produces comparison table
6. State transitions to REVIEW_CLAUDE
7. Claude reviewer launches, writes review file, writes handoff
8. State transitions to REVIEW_CODEX
9. Codex reviewer launches (or timeout stub fires), writes handoff
10. State transitions to ORCHESTRATOR_SYNTHESIZING
11. Orchestrator synthesis runs, writes direction_iter2.md
    (executive_summary.md and archive are only written at iteration 3, not tested here)
12. State transitions to ORCHESTRATOR_PLANNING (next iteration) — NOT HUMAN_SYNC
    (HUMAN_SYNC only triggers after iteration 3)

Pass criteria: all 12 assertions green, final state == ORCHESTRATOR_PLANNING, no orphaned tmux sessions.
Note: to test full 3-iteration path (archive, HUMAN_SYNC), run with --iterations 3 flag.
```

### 13.3 Component-level tests

| Component | Test command |
|-----------|-------------|
| ML unit tests | `pytest ml/tests/ -v` |
| Comparison script | `python ml/compare.py --batch-id test --iteration 1 --output /tmp/test_compare.md` |
| State utils CAS | `bash agents/state_utils.sh test` |
| Watchdog (idle) | `COUNT_BEFORE=$(wc -l < .logs/audit.jsonl); bash agents/watchdog.sh; COUNT_AFTER=$(wc -l < .logs/audit.jsonl); [[ $COUNT_BEFORE -eq $COUNT_AFTER ]] && echo PASS` — watchdog exits early in IDLE, writes NO audit entry |
| Watchdog (active) | Set fake WORKER_RUNNING state, run watchdog, verify audit probe written |

## 14. Memory Safety Compliance (from CLAUDE.md)

All ML code must comply with the workspace 128 GiB pod constraints:

1. **Use polars over pandas** in `data_loader.py` and `features.py`
2. **Lazy scan**: `pl.scan_parquet(...).filter(...).collect()` not `pl.read_parquet()`
3. **Print `mem_mb()`** at each pipeline stage
4. **Free intermediates**: `del df; gc.collect()` between stages
5. **Ray shutdown** immediately after data loading: `ray.shutdown()`
6. **Save intermediates** to parquet between phases for crash recovery
7. **`--from-phase N`** flag on pipeline for partial re-runs

Ray init pattern (must include the local `ml` package):
```python
from pbase.config.ray import init_ray
import pmodel
import ml as shadow_ml
init_ray(address=RAY_ADDRESS, extra_modules=[pmodel, shadow_ml])
```

---

## 15. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Concurrent pipeline starts | Batch lock (flock) on state.lock |
| Race condition on state.json | Single writer (controller), CAS expected-state check |
| Agent writes outside sandbox | CLAUDE.md directory constraints; worktree isolation |
| Agent deletes old versions | CLAUDE.md forbids; registry uses `exist_ok=False` on mkdir |
| Stale review artifact from prior batch | Artifact names include batch_id; controller polls by mtime |
| Partial/wrong artifact read | sha256 verification in handshake before state transition |
| Worker stuck/OOM | Watchdog timeout → kill tmux → timeout file → controller action |
| Codex reviewer fails | Graceful degradation: pipeline continues with Claude review only |
| Gate manipulation by agent | gates.json and evaluate.py are human-write-only |
| Metric drift across versions | Same `evaluate.py` for all versions; threshold stored per-version |
| Disk exhaustion | Gzipped models; watchdog size check; 10 GB hard stop |
| Context loss (cold-start reviewers) | Rich memory files + comparison table in every prompt |

---

## 16. Open Decisions (For Human Review at First HUMAN_SYNC)

1. **Codex model ID**: Design uses `gpt-5.3-codex`. Confirm model ID on your subscription.

2. **Capture@K K values**: Design uses 100 and 500 for gates, 1000 for monitoring. Adjust
   based on v0 results and business context.

3. **Worker worktree retention**: Current design keeps worktrees until HUMAN_SYNC cleanup.
   Can be deleted earlier to save disk if not needed for debugging.

4. **HUMAN_SYNC notification**: Current design writes to `state.json` + prints to stdout.
   Add Slack webhook or other notification in `config.sh` if desired.

5. **Gate floors after v0**: Once v0 runs, review auto-populated floors before iteration 1
   promotion gates take effect. The `v0 − 0.05` tolerance can be tightened or loosened.
