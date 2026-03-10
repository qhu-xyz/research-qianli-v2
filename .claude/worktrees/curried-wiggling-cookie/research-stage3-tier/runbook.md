# Runbook — Tier Classification Pipeline (Stage 3)

Tier classification pipeline using a single multi-class XGBoost classifier predicting 5 shadow-price tiers. Four autonomous agents (orchestrator, worker, Claude reviewer, Codex reviewer) coordinate via file-based handoffs. Runs 3 iterations per batch, fully autonomous after launch. Uses CLI subscriptions only (Claude Max/Pro + ChatGPT Pro) -- no API keys required.

For complete model settings, features, and metrics documentation, see `docs/tier-pipeline-settings.md`.

---

## Quick Start

```bash
# Activate venv first
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier

# 3-iteration batch (default)
bash agents/run_pipeline.sh --batch-name my-batch

# 1-iteration test
bash agents/run_pipeline.sh --batch-name test --max-iter 1

# With human guidance
echo "Focus on feature engineering for high-value constraints" > memory/human_input.md
bash agents/run_pipeline.sh --batch-name features

# Smoke test (synthetic n=20 data, fast)
SMOKE_TEST=true bash agents/run_pipeline.sh --batch-name smoke-test --max-iter 1

# Foreground mode (skip tmux, for debugging)
bash agents/run_pipeline.sh --batch-name debug --max-iter 1 --foreground
```

Auto-wraps in tmux by default. Prints attach and log commands:
```
Pipeline launched in tmux session: pipeline-20260303-142305
Attach:  tmux attach -t pipeline-20260303-142305
Log:     .logs/sessions/pipeline-20260303-142305.log
```

---

## Prerequisites

| Requirement | Install / Check |
|---|---|
| `claude` CLI | Claude Max or Pro subscription |
| `codex` CLI | ChatGPT Pro subscription |
| `tmux` | `apt install tmux` |
| `jq` | `apt install jq` |
| Python venv | `cd /home/xyz/workspace/pmodel && source .venv/bin/activate` |
| v0 baseline | `ls registry/v0/metrics.json` |

Run `bash agents/check_clis.sh` to verify all CLIs are available.

---

## Architecture

### State Machine

```
IDLE
  -> ORCHESTRATOR_PLANNING    (orchestrator writes direction + strategy)
  -> WORKER_RUNNING           (worker implements changes in git worktree)
  -> REVIEW_CLAUDE            (Claude reviews changes independently)
  -> REVIEW_CODEX             (Codex reviews changes independently)
  -> ORCHESTRATOR_SYNTHESIZING (orchestrator merges reviews, decides promotion)
  -> IDLE                     (next iteration, or HUMAN_SYNC after iter 3)
```

### Key Design Principles

- **File-based communication** -- agents exchange JSON handoff files, no inter-agent APIs or message queues
- **Tmux isolation** -- each agent runs in a detached tmux session (survives terminal close)
- **Git worktree isolation** -- worker operates in a separate worktree (no cross-iteration pollution)
- **Single-writer rule** -- only one agent modifies state at a time (CAS transitions + flock)
- **CLI subscriptions only** -- `claude --print` and `codex exec`, not API calls
- **Single model** -- one XGBoost multi:softprob classifier for all 5 tiers, no frozen components

---

## Stage 3 Identity: Tier Classification Pipeline

This pipeline replaces the two-stage approach (stage-1 classifier + stage-2 regressor) with a single multi-class XGBoost model that directly predicts 5 shadow price tiers:

| Tier | Range | Midpoint |
|------|-------|----------|
| 0 | [$3000, +inf) | $4000 |
| 1 | [$1000, $3000) | $2000 |
| 2 | [$100, $1000) | $550 |
| 3 | [$0, $100) | $50 |
| 4 | (-inf, $0) | $0 |

**EV Score** for ranking: `sum(P(tier=t) * midpoint[t])` — continuous signal for capital allocation.

**Current constraint**: Autonomous loop may only modify features and monotone constraints (see `memory/human_input.md`).

---

## Directory Structure

| Directory | Purpose | Who writes | Gitignored? |
|---|---|---|---|
| `agents/` | Shell scripts (launcher, controller, state machine) | Human | No |
| `agents/prompts/` | Agent prompt templates (5 files) | Human | No |
| `ml/` | Python ML code (features, training, evaluation) | Worker | No |
| `ml/tests/` | Pytest unit tests | Worker | No |
| `registry/` | Model versions, gates, champion tracking | Worker (own version only) | No |
| `registry/v0/` | Immutable baseline | Nobody (bootstrap only) | No |
| `registry/gates.json` | Gate definitions (HUMAN-WRITE-ONLY) | Human | No |
| `registry/comparisons/` | Batch comparison JSON outputs | Controller | No |
| `memory/hot/` | Current iteration working memory (6 files) | Orchestrator, Reviewers | No |
| `memory/warm/` | Batch history (append-only logs) | Orchestrator | No |
| `memory/archive/` | Completed batch summaries | Orchestrator | No |
| `memory/human_input.md` | Per-batch user guidance | Human | No |
| `reports/` | Gate comparison markdown per iteration | Controller | No |
| `reviews/` | Final reviewer outputs per iteration | Claude reviewer, Codex reviewer | No |
| `handoff/` | Agent handoff JSON signals | All agents | **Yes** |
| `.logs/sessions/` | Agent session logs | All agents | **Yes** |
| `.logs/audit.jsonl` | Watchdog audit trail | Watchdog | **Yes** |
| `.claude/worktrees/` | Git worktrees for worker isolation | Controller | **Yes** |
| `state.json` | Central state machine file | Controller (CAS) | **Yes** |
| `state.lock` | flock file for pipeline exclusion | Controller | **Yes** |
| `docs/` | Design docs, implementation plans | Human | No |

---

## Where Reports Live

| Report | Path |
|---|---|
| Gate comparison table | `reports/{batch}/iter{N}/comparison.md` |
| Claude review | `reviews/{batch}_iter{N}_claude.md` |
| Codex review | `reviews/{batch}_iter{N}_codex.md` |
| Synthesized critique | `memory/hot/critique_summary.md` |
| Accumulated learnings | `memory/hot/learning.md` |
| Final batch report (after iter 3) | `memory/archive/{batch}/executive_summary.md` |
| Pipeline log | `.logs/sessions/pipeline-*.log` |
| Per-agent logs | `.logs/sessions/{orch,worker,rev-claude,rev-codex,synth}-*.log` |
| Watchdog audit | `.logs/audit.jsonl` |

---

## Memory System

### `memory/hot/` -- Current state (read every iteration)

| File | Purpose |
|---|---|
| `progress.md` | Current iteration status and next steps |
| `champion.md` | Current champion version info |
| `learning.md` | Accumulated insights across iterations (cumulative) |
| `gate_calibration.md` | Gate analysis and calibration notes |
| `critique_summary.md` | Summary of reviewer critiques |
| `runbook.md` | Worker safety rules and guidelines (static) |

### `memory/warm/` -- Batch history (append-only)

| File | Purpose |
|---|---|
| `experiment_log.md` | All experiments tried |
| `hypothesis_log.md` | All hypotheses tested |
| `decision_log.md` | All decisions made and rationale |

### `memory/archive/` -- Completed batches

- `index.md` -- one-line summaries of all archived batches
- Per-batch subdirectories with executive summaries

### `memory/human_input.md` -- User guidance

Write guidance here before launching a batch to steer the orchestrator's strategy. Cleared between batches.

---

## Agent Roles & Constraints

| Agent | Role | Reads | Writes | Key Constraints |
|---|---|---|---|---|
| **Orchestrator (plan)** | Develops iteration strategy | `memory/hot/*`, `memory/warm/*`, `memory/archive/index.md`, `registry/gates.json`, champion metrics | `memory/hot/progress.md`, `memory/direction_iter{N}.md` | No ML code changes, no training |
| **Worker** | Implements ML changes | `memory/direction_iter{N}.md`, `memory/hot/champion.md`, `memory/hot/learning.md`, `memory/hot/runbook.md` | `ml/`, `registry/{VERSION_ID}/`, handoff JSON | Runs in worktree, must pass tests, must commit before handoff, must check human_input.md constraints |
| **Claude Reviewer** | Independent code + gate review | Direction, changes summary, comparison table, gates.json, `memory/warm/*`, `ml/` codebase | `reviews/{batch}_iter{N}_claude.md`, handoff JSON | Cannot see Codex review, may critique gates |
| **Codex Reviewer** | Independent second review | Same as Claude reviewer | `reviews/{batch}_iter{N}_codex.md`, handoff JSON | Cannot see Claude review, may critique gates |
| **Orchestrator (synth)** | Merges reviews, decides promotion | Both reviews, comparison table, `memory/warm/*` | `memory/hot/*`, `memory/warm/*`, handoff JSON (with `decisions` field) | No ML code changes, no training |

---

## Gate System -- Three-Layer Promotion Checks

### Overview

Models are evaluated across **12 primary eval months** using a rolling window (6-month train, 2-month val for early stopping per eval month). Promotion requires passing all three layers on all Group A gates.

### Three Layers

| Layer | What it checks | Formula | Purpose |
|---|---|---|---|
| **1. Mean Quality** | Average performance | `mean(metric) >= floor` | Basic quality bar |
| **2. Tail Safety** | Catastrophic month protection | `count(months below tail_floor) <= 1` | No single-month disasters |
| **3. Tail Non-Regression** | Worst-case improvement | `bottom_2_mean(new) >= bottom_2_mean(champ) - 0.02` | Worst months must not regress |

### Gate Groups (Tier Metrics)

| Group | Gates | Role |
|---|---|---|
| **A (hard)** | Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK | Must pass all 3 layers to promote |
| **B (monitor)** | Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1, Value-QWK | Tracked, don't block promotion |

### Cascade Stages

Evaluation follows a strict cascade: **f0 -> f1 -> f2+**
- f0 and f1 are **blocking** -- must pass before proceeding
- f2+ is **monitor only** -- tracked but doesn't block promotion

### metrics.json Structure

```json
{
  "per_month": {"2020-09": {"Tier-VC@100": 0.10, ...}, "2020-11": {...}, ...},
  "aggregate": {
    "mean": {"Tier-VC@100": 0.075, ...},
    "std": {"Tier-VC@100": 0.065, ...},
    "min": {"Tier-VC@100": 0.008, ...},
    "max": {"Tier-VC@100": 0.246, ...},
    "bottom_2_mean": {"Tier-VC@100": 0.012, ...}
  },
  "n_months": 12,
  "eval_config": {"eval_months": [...], "class_type": "onpeak", "ptype": "f0", ...}
}
```

### Running the Benchmark

```bash
# Smoke (single month, synthetic data)
SMOKE_TEST=true python ml/benchmark.py --version-id v0 --ptype f0 --eval-months 2021-07

# Real data (12 months via Ray, reads eval_months from gates.json)
python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak

# Populate gate floors from v0 baseline
python ml/populate_v0_gates.py
```

### All Metrics Higher-is-Better

Unlike stage 2 (which had lower-is-better C-RMSE and C-MAE), all tier metrics are higher-is-better. No direction inversions needed.

---

## Timeouts & Safety

### Poll Timeouts (controller waits this long for handoff)

| Agent | Poll Timeout | Hard Kill (`timeout` cmd) |
|---|---|---|
| Orchestrator (plan) | 7 min | 10 min |
| Worker | 50 min | 60 min |
| Claude Reviewer | 7 min | 10 min |
| Codex Reviewer | 7 min (non-fatal) | 10 min |
| Orchestrator (synth) | 20 min | 25 min |

Worker timeout is longer because tier benchmarks run ~20-40 min (12 months with early stopping via Ray). Synthesizer timeout is longer due to observed ~14 min runs that read ~15 files and write 8 files.

### Watchdog Recovery

- Runs via cron every 5 minutes (`bash agents/install_cron.sh` to set up)
- **Phase 1** (timeout detection): if state stuck past `max_seconds` and no handoff exists, writes timeout artifact -> controller's `poll_for_handoff` stops waiting
- **Phase 2** (recovery): if stuck for 2x `max_seconds`, kills orphaned tmux sessions, resets state to IDLE, releases lock
- Audit trail: `.logs/audit.jsonl`
- Manual run: `bash agents/watchdog.sh`

### Safety Mechanisms

- `flock` prevents concurrent pipeline runs
- CAS (compare-and-swap) state transitions prevent race conditions
- Worker runs in isolated git worktree
- HUMAN-WRITE-ONLY files (`evaluate.py`, `gates.json`) verified unchanged before merge
- `__pycache__/` gitignored to prevent merge conflicts
- All handoff JSON validated with `jq empty` before parsing
- Codex timeout is non-fatal (pipeline continues with Claude review only)
- **FE-only constraint**: `memory/human_input.md` restricts autonomous loop to feature changes only

---

## Configuration Reference

All tunables live in `agents/config.sh`:

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT_DIR` | `.../research-stage3-tier` | Absolute path to project root |
| `RAY_ADDRESS` | `ray://10.8.0.36:10001` | Ray cluster for data loading |
| `DATA_ROOT` | `/opt/temp/tmp/pw_data/spice6` | Production data directory |
| `VENV_ACTIVATE` | `.../pmodel/.venv/bin/activate` | Python venv activation script |
| `SMOKE_TEST` | `false` (env override) | Use synthetic n=20 data when `true` |
| `REGISTRY_DISK_LIMIT_MB` | `10240` | Max disk for model registry |
| `CODEX_MODEL` | `gpt-5.3-codex` | Codex model ID for reviewer |
| `STATE_FILE` | `${PROJECT_DIR}/state.json` | Central state file path |
| `TIMEOUT_ORCHESTRATOR` | `600` (10 min) | Hard kill timeout for orchestrator |
| `TIMEOUT_WORKER` | `3600` (60 min) | Hard kill timeout for worker |
| `TIMEOUT_REVIEWER_CLAUDE` | `600` (10 min) | Hard kill timeout for Claude reviewer |
| `TIMEOUT_REVIEWER_CODEX` | `600` (10 min) | Hard kill timeout for Codex reviewer |
| `TIMEOUT_SYNTHESIZER` | `1500` (25 min) | Hard kill timeout for synthesizer |

---

## Troubleshooting

### State stuck in non-IDLE state
```bash
# Check current state
jq . state.json

# Run watchdog manually (safe, idempotent)
bash agents/watchdog.sh

# Manual reset (only if no pipeline running)
echo '{"state":"IDLE","batch_id":null,"iteration":0,"version_id":null}' > state.json
```

### Agent timeout (controller kills tmux, logs error, moves on)
- Check per-agent log: `.logs/sessions/{agent}-{batch}-iter{N}.log`
- Look for `EXIT_CODE=` at end of log
- Codex timeout is non-fatal; others cause iteration failure

### `state.lock` held
```bash
# Confirm no pipeline is running
tmux ls | grep pipeline

# If no pipeline running, remove lock
rm state.lock
```

### Orphaned tmux sessions
```bash
# List all sessions
tmux ls

# Kill matching sessions
tmux kill-session -t orch-my-batch-iter1
tmux kill-session -t worker-my-batch-iter1
# etc.
```

### Worker merge failed
- Check if all code was committed before pipeline started (RT-7)
- Check if `__pycache__/` is in `.gitignore` (RT-7)
- Check `git checkout -- .` runs before merge (RT-5)
- Check worktree subdir path for monorepo (RT-8)

### Handoff JSON parse error
- Validate manually: `jq empty handoff/{batch}/iter{N}/{file}.json`
- RT-6 should catch this; check if validation was bypassed

### Worker can't find `ml/` modules
- PYTHONPATH must point to worktree project subdir, not worktree root (RT-8)
- Venv must be sourced inside tmux session (RT-2)

---

## Pitfalls & Lessons Learned (RT-1 through RT-12)

### HIGH severity

| Fix | One-liner |
|---|---|
| **RT-1** | Set PYTHONPATH and source venv BEFORE iteration loop -- Python scripts in pre-loop steps need it |
| **RT-2** | Set full environment (cd, PYTHONPATH, venv) inside every tmux command -- sessions don't inherit parent env |
| **RT-3** | Clean stale `memory/direction_iter*.md` before loop -- crashed batches leave orphaned files |
| **RT-4** | Run `verify_handoff` in worktree directory (`pushd $WT_PROJECT`) -- artifact paths are relative to worktree |
| **RT-5** | `git checkout -- .` before merge -- stale uncommitted files from prior runs cause conflicts |
| **RT-6** | Validate all handoff JSON with `jq empty` before parsing -- malformed JSON + `set -e` = instant death |
| **RT-12** | OS-level `timeout` on every agent process + watchdog recovery at 2x timeout -- agents can loop forever |

### MEDIUM severity

| Fix | One-liner |
|---|---|
| **RT-7** | Commit ALL code before pipeline start + gitignore `__pycache__/` -- worktrees created from HEAD miss untracked files |
| **RT-8** | Compute worktree project subdir (`git rev-parse --show-prefix`) -- in monorepo, git root != project dir |
| **RT-9** | Codex needs `workspace-write` sandbox -- `read-only` blocks handoff file writes, causing silent timeout |
| **RT-10** | Use `--full-auto` for Codex -- without it, Codex pauses for interactive approval in non-interactive tmux |
| **RT-11** | Auto-wrap `run_pipeline.sh` in tmux -- user can close terminal and pipeline keeps running |

---

## For Teammates Building Similar Pipelines

### What to Reuse

- **`state_utils.sh`** -- generic CAS state machine with built-in self-test; swap state names for your domain
- **`poll_for_handoff`** -- robust polling with timeout artifact detection; works for any file-based handoff
- **Watchdog pattern** -- `watchdog.sh` with cron; two-phase (signal then kill) is the right approach
- **Tmux launcher pattern** -- `launch_*.sh` scripts with `--dry-run`, env setup inside tmux, timeout wrapper
- **`run_pipeline.sh` auto-tmux** -- fire-and-forget with log tee; `--foreground` escape hatch for debugging

### What to Customize

- **`agents/prompts/`** -- agent identities, read/write instructions, constraints for your domain
- **`agents/config.sh`** -- timeouts, model IDs, paths, data locations
- **`ml/`** -- replace with your ML code; keep `evaluate.py` as the gate evaluation harness pattern
- **`registry/gates.json`** -- define your own quality gates and floor/ceiling thresholds

### Key Design Decisions

| Decision | Why |
|---|---|
| File-based handoff (not message queue) | Simplest coordination primitive; survives restarts; git-visible audit trail; no infra dependency |
| Tmux (not containers/k8s) | Survives terminal close; lightweight; matches CLI subscription model; `tmux ls` for instant status |
| Git worktrees (not containers) | Full git isolation without container overhead; merge = `git merge`; easy diffing |
| CAS transitions (not locks per state) | Prevents race conditions without deadlock risk; single atomic operation |
| Dual independent reviewers | Reduces blind spots; neither reviewer sees the other's output; synthesis resolves disagreements |
| Codex timeout non-fatal | Codex is less reliable; pipeline degrades gracefully to Claude-only review |
| Single model (no frozen stage) | Simpler architecture; one model output; aligns ML objective with downstream tier decisions |

### Common Mistakes to Avoid

1. **Forgetting tmux env setup** -- tmux sessions start with a clean environment; every variable, venv activation, and `cd` must be explicit inside the tmux command
2. **Relative paths in monorepo** -- `git worktree add` checks out the entire repo; your project is a subdirectory; always compute `WT_PROJECT` from `git rev-parse --show-prefix`
3. **Not committing before worktree** -- worktrees are created from HEAD; uncommitted files don't exist in the worktree
4. **Suppressing errors under `set -e`** -- `2>/dev/null || true` hides real failures; only use for genuinely idempotent cleanup
5. **Assuming agents write valid JSON** -- always validate with `jq empty` before parsing
6. **Interactive prompts in non-interactive sessions** -- use `--full-auto` or equivalent; agents in tmux can't answer "Are you sure?"
7. **Not testing with `--dry-run` first** -- every launcher supports it; use it before real runs to verify env and paths
8. **Ignoring human_input.md** -- always check per-batch constraints before modifying parameters
