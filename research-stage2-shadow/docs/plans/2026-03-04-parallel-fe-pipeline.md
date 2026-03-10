# Parallel FE Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clone `research-stage2-shadow` into a parallel directory and launch a 3-iteration autonomous FE-only batch starting from v0007 champion, using version namespace v1001+.

**Architecture:** Full directory clone with patched `config.sh` (PROJECT_DIR), reset state, pruned registry (keep only v0 + v0007), version counter starting at 1001, and a focused human prompt restricting iterations to feature engineering only.

**Tech Stack:** Bash (setup), existing autonomous pipeline (claude CLI + codex CLI + tmux)

---

### Task 1: Clone directory

**Files:**
- Create: `/home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/` (full copy)

**Step 1: Copy the directory**

```bash
cp -r /home/xyz/workspace/research-qianli-v2/research-stage2-shadow \
      /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel
```

**Step 2: Verify the copy**

```bash
ls /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/agents/run_pipeline.sh
```

Expected: file exists.

---

### Task 2: Patch config.sh

**Files:**
- Modify: `research-stage2-shadow-fe-parallel/agents/config.sh` (line 3)

**Step 1: Update PROJECT_DIR**

Change line 3 from:
```bash
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage2-shadow"
```
To:
```bash
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel"
```

**Step 2: Verify**

```bash
grep PROJECT_DIR /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/agents/config.sh
```

Expected: shows the new path.

---

### Task 3: Reset state.json

**Files:**
- Modify: `research-stage2-shadow-fe-parallel/state.json`

**Step 1: Write clean IDLE state**

```json
{
  "state": "IDLE",
  "batch_id": null,
  "iteration": 0,
  "version_id": "v0007",
  "entered_at": null,
  "max_seconds": 600
}
```

**Step 2: Remove stale lock**

```bash
rm -f /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/state.lock
touch /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/state.lock
```

---

### Task 4: Set version counter to 1001

**Files:**
- Modify: `research-stage2-shadow-fe-parallel/registry/version_counter.json`

**Step 1: Write new counter**

```json
{"next_id": 1001}
```

---

### Task 5: Prune registry — keep only v0 and v0007

**Files:**
- Delete: `registry/v0005/`, `registry/v0006/`, `registry/v0007-reeval/`, `registry/v0008/`, `registry/v0-reeval/`, `registry/v1/`
- Keep: `registry/v0/`, `registry/v0007/`, `registry/champion.json`, `registry/gates.json`, `registry/comparisons/`

**Step 1: Remove intermediate versions**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/registry
rm -rf v0005 v0006 v0007-reeval v0008 v0-reeval v1
```

**Step 2: Verify champion still points to v0007**

```bash
cat /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/registry/champion.json
```

Expected: `"version": "v0007"`

**Step 3: Verify v0 and v0007 exist**

```bash
ls /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/registry/v0/metrics.json
ls /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/registry/v0007/metrics.json
```

Expected: both exist.

---

### Task 6: Clean stale artifacts

**Files:**
- Clean: `handoff/`, `reviews/`, `reports/`, `memory/direction_iter*.md`, `.logs/`

**Step 1: Remove handoff files**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel
rm -rf handoff/*
```

**Step 2: Remove stale reviews and reports**

```bash
rm -rf reviews/*
rm -rf reports/*
```

**Step 3: Remove stale direction files**

```bash
rm -f memory/direction_iter*.md
```

**Step 4: Clean logs**

```bash
rm -rf .logs/sessions/*
```

**Step 5: Clean stale worktrees**

```bash
rm -rf .claude/worktrees/*
```

---

### Task 7: Write human prompt

**Files:**
- Modify: `research-stage2-shadow-fe-parallel/memory/human_input.md`

**Step 1: Overwrite with FE-only prompt**

Write the following content to `memory/human_input.md`:

```markdown
# Feature Engineering Focus — Parallel Batch

## Constraint
This batch is EXCLUSIVELY about feature engineering and feature selection for the **regressor**.
- Do NOT change hyperparameters (n_estimators, learning_rate, max_depth, subsample, etc.)
- Do NOT change the classifier (it is frozen)
- Do NOT change pipeline architecture or evaluation harness
- ONLY modify: regressor feature list, monotone_constraints, and interaction feature computation in ml/features.py

## Starting point
v0007 champion: 34 regressor features (see registry/v0007/config.json).

## Iteration protocol
Each iteration:
1. **Research**: Analyze feature importance from the current champion. Identify dead/low-value features and potential new features or interaction terms from available data columns in MisoDataLoader.
2. **Generate 2 hypotheses** using `--overrides` JSON. Examples:
   - H1: Drop the 5 lowest-importance features to reduce noise
   - H2: Add 3 new interaction terms (e.g., prob_exceed_100 * density_entropy)
3. **Screen both** on 2 months: 2022-06 (weak) and 2021-09 (strong)
4. **Implement winner** in code, run full 12-month benchmark

## Reporting
Report on **target month only** — val set is NOT used for reporting metrics.

## Feature ideas to explore
- Feature pruning: which of the 34 features contribute least? Remove noise.
- New interaction terms: ratios or products of exceedance/density/shift-factor features
- Temporal features: any untapped lagged or seasonal features in the data loader
- Monotone constraint audit: are current constraints correct per domain knowledge?
- Consider: exceed_severity_ratio = prob_exceed_110 / (prob_exceed_90 + 1e-6)
```

---

### Task 8: Update memory files for fresh batch context

**Files:**
- Modify: `research-stage2-shadow-fe-parallel/memory/hot/progress.md`
- Modify: `research-stage2-shadow-fe-parallel/memory/warm/experiment_log.md`
- Modify: `research-stage2-shadow-fe-parallel/memory/warm/hypothesis_log.md`
- Modify: `research-stage2-shadow-fe-parallel/memory/warm/decision_log.md`

**Step 1: Reset progress.md to reflect new batch**

Overwrite `memory/hot/progress.md` with:

```markdown
# Progress — FE-Parallel Batch

## Status
Starting fresh FE-only batch from v0007 champion. Version namespace: v1001+.

## Completed
- (none yet)

## Current
- Awaiting pipeline launch
```

**Step 2: Clear warm logs for fresh batch**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/memory/warm
> experiment_log.md
> hypothesis_log.md
> decision_log.md
```

---

### Task 9: Verify target-month-only reporting

**Files:**
- Read: `research-stage2-shadow-fe-parallel/ml/benchmark.py` (or `evaluate.py`)

**Step 1: Check that metrics are computed on test (target) month, not val split**

Read `ml/benchmark.py` and verify the evaluation loop:
- train on 6 months
- val on 2 months (threshold tuning only)
- metrics reported on the target month (month after val window)

If target-month reporting is NOT correct, fix it before launching.

---

### Task 10: Launch the pipeline

**Step 1: Activate venv**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

**Step 2: Launch**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel
bash agents/run_pipeline.sh --batch-name fe-parallel
```

**Step 3: Verify tmux session started**

```bash
tmux ls | grep pipeline
```

Expected: shows the new session alongside any existing pipeline session.

**Step 4: Tail the log**

```bash
tail -f .logs/sessions/pipeline-*.log
```

Verify iteration 1 starts, orchestrator plans, worker begins.

---

### Task 11: Monitor and compare (post-completion)

After both batches complete:

**Step 1: Compare champions**

```bash
# Main registry champion
cat /home/xyz/workspace/research-qianli-v2/research-stage2-shadow/registry/champion.json

# Parallel registry champion
cat /home/xyz/workspace/research-qianli-v2/research-stage2-shadow-fe-parallel/registry/champion.json
```

**Step 2: Compare metrics on blocking gates**

Compare EV-VC@100, EV-VC@500, EV-NDCG, Spearman (mean across 12 months) between both champions.

**Step 3: Port winner to main registry**

If the parallel batch's champion beats the main batch's champion:
1. Copy the champion's `registry/v1001/` (or whichever won) to main `registry/` as next sequential version
2. Run `python ml/compare.py` in the main registry
3. Update `champion.json` if gates pass
