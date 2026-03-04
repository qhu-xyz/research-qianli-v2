# Direction — Iteration 3 (ralph-v1-20260304-003317)

## ⚠️ CRITICAL: CLEAN THE CODEBASE FIRST

The main working tree has extensive UNCOMMITTED changes from a prior failed worker. These changes contaminate registry files, gates, and ML code. You MUST clean before doing ANYTHING else.

### Step 0: Revert dirty state (MANDATORY, DO THIS FIRST)

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow

# Revert ALL uncommitted changes to ml/ and registry/
git checkout -- ml/ registry/

# Verify clean state — should show NO ml/ or registry/ files
git diff --name-only | grep -E '^(ml/|registry/)' && echo "DIRTY — STOP" || echo "CLEAN — proceed"
```

**DO NOT proceed past Step 0 if the verification shows DIRTY.**

---

## ⚠️ DO NOT MODIFY these files (frozen / human-owned):
- `ml/evaluate.py` — HUMAN-WRITE-ONLY
- `ml/data_loader.py` — no changes needed
- `ml/features.py` — no changes needed
- `ml/pipeline.py` — no changes needed
- `ml/train.py` — no changes needed
- `ml/benchmark.py` — no changes needed
- `registry/gates.json` — HUMAN-WRITE-ONLY
- `registry/v0/` — baseline is immutable

### The classifier is FROZEN:
- **NEVER** modify `ClassifierConfig` in `ml/config.py`
- The ONLY class you may edit is `RegressorConfig` — and ONLY its default field values

---

## Goal

Salvage the valid v0003 results from the iter 2 worktree branch. The iter 2 worker produced correct results (reg_lambda=5.0, min_child_weight=25, full 12-month benchmark) but committed them on a worktree branch instead of main.

**Primary path**: Cherry-pick the v0003 commit to main.
**Fallback path**: If cherry-pick fails, re-apply the same config changes and re-run the benchmark.

---

## Step 1: Cherry-pick v0003 from worktree branch

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow

# Cherry-pick the worker's valid commit
git cherry-pick 01c22af --no-commit

# Verify the artifacts exist
ls registry/v0003/metrics.json registry/v0003/config.json registry/v0003/changes_summary.md

# Verify config.py changes are ONLY reg_lambda and min_child_weight
git diff --stat ml/config.py
# Should show ~4 lines changed (reg_lambda 1.0→5.0, min_child_weight 10→25)

# If verification passes, commit
git add ml/config.py ml/tests/test_config.py registry/v0003/
git commit -m "iter3: cherry-pick v0003 from worktree (reg_lambda=5.0, min_child_weight=25)"
```

**If cherry-pick fails** (conflict), proceed to Step 1B instead.

---

## Step 1B (FALLBACK ONLY — skip if Step 1 succeeded)

If cherry-pick fails, manually apply the changes and re-run:

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
source /home/xyz/workspace/pmodel/.venv/bin/activate

# Edit config.py: change ONLY these two values in RegressorConfig:
#   reg_lambda: float = 1.0  →  reg_lambda: float = 5.0
#   min_child_weight: int = 10  →  min_child_weight: int = 25

# Run full 12-month benchmark
python -m ml.benchmark --version-id v0003

# Verify output
cat registry/v0003/metrics.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('EV-VC@100 mean:', d['aggregate']['mean']['EV-VC@100'])"
# Expected: ~0.034 (should be close to the worktree results)

# Commit
git add ml/config.py ml/tests/test_config.py registry/v0003/
git commit -m "iter3: heavier L2 regularization (reg_lambda=5.0, min_child_weight=25)"
```

---

## Step 2: Run comparison

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
source /home/xyz/workspace/pmodel/.venv/bin/activate

python -m ml.benchmark --compare v0003 --baseline v0
```

---

## Step 3: Verify and write handoff

```bash
# Verify artifacts exist
test -f registry/v0003/metrics.json && echo "PASS" || echo "FAIL — metrics missing"
test -f registry/v0003/config.json && echo "PASS" || echo "FAIL — config missing"

# Verify last commit includes v0003
git log --oneline -1
git show --stat HEAD | head -20
```

**Only write worker_done.json AFTER all verifications pass.**

---

## v0 Baseline (COMMITTED, correct values — 10/2, 24 features)

| Metric | v0 Mean | v0 Bottom-2 |
|--------|---------|-------------|
| EV-VC@100 | 0.0303 | 0.0035 |
| EV-VC@500 | 0.1180 | 0.0488 |
| EV-NDCG | 0.7400 | 0.6735 |
| Spearman | 0.3921 | 0.3296 |
| C-RMSE | 3400.4 | 5967.6 |

---

## Expected v0003 Results (from iter 2 worktree)

| Metric | v0003 Mean | v0003 Bottom-2 | Δ vs v0 |
|--------|-----------|---------------|---------|
| EV-VC@100 | 0.0337 | 0.0048 | +0.0034 |
| EV-VC@500 | 0.1174 | 0.0429 | -0.0006 |
| EV-NDCG | 0.7435 | 0.6738 | +0.0035 |
| Spearman | 0.3921 | 0.3299 | +0.0000 |
| C-RMSE | 3377.7 | 5900.1 | -22.7 |

All Group A gates pass all 3 layers (committed gates). This version should be promoted.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cherry-pick conflict | Low | Fallback Step 1B re-applies manually |
| Dirty state not fully reverted | Low | Step 0 has explicit verification |
| Re-run produces different metrics | Very Low | Same config, same data, same pipeline → deterministic |
