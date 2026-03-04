# Direction — Iteration 2 (ralph-v1-20260304-003317)

## CRITICAL RULES — READ BEFORE ANYTHING ELSE

### DO NOT MODIFY these files (frozen / human-owned):
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
- **NEVER** add classifier features, change classifier hyperparams, or touch `threshold_beta`
- The ONLY class you may edit is `RegressorConfig` — and ONLY its default field values

### YOU MUST:
1. Revert ALL uncommitted changes before starting: `git checkout -- ml/`
2. Verify clean state: `git diff --name-only` should show NO ml/ files
3. Make ONLY the config.py edit described below
4. Commit before writing handoff
5. Verify `registry/{VERSION_ID}/metrics.json` exists before writing handoff

---

## Goal

Test whether a slower learning rate with more trees improves the regressor. This is a single-hypothesis experiment — no screening step.

## Hypothesis: Slower learning rate + more trees (smoother ensemble)

**Rationale**: v0 uses lr=0.05 with 400 trees. Reducing lr to 0.03 with 700 trees creates a smoother, higher-capacity ensemble that should reduce overfitting on training windows and improve tail months.

**Config change**: In `ml/config.py`, class `RegressorConfig`, change exactly 2 lines:
- `n_estimators: int = 400` → `n_estimators: int = 700`
- `learning_rate: float = 0.05` → `learning_rate: float = 0.03`

That's it. No other changes to any file.

---

## Exact Steps (copy-paste)

### Step 0: Clean up dirty state
```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
git checkout -- ml/
git diff --name-only  # verify: should NOT list any ml/ files
```

### Step 1: Read VERSION_ID
```bash
VERSION_ID=$(jq -r '.version_id' state.json)
echo "VERSION_ID=${VERSION_ID}"
```

### Step 2: Edit config.py (2 lines only)
In `ml/config.py`, find class `RegressorConfig` and change:
```python
    n_estimators: int = 700    # was 400
    learning_rate: float = 0.03  # was 0.05
```
Do NOT change anything else in this file. Do NOT touch ClassifierConfig.

### Step 3: Run tests
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
SMOKE_TEST=true python -m pytest ml/tests/ -v
```
If tests fail, fix only the test issue and retry (max 3 attempts). On 3rd failure, write failed handoff and stop.

### Step 4: Run full benchmark (f0)
```bash
python ml/benchmark.py --version-id ${VERSION_ID} --ptype f0 --class-type onpeak
```
This will:
- Evaluate all 12 months from gates.json
- Write `registry/${VERSION_ID}/metrics.json`
- Write `registry/${VERSION_ID}/config.json`

### Step 5: Run full benchmark (f1)
```bash
python ml/benchmark.py --version-id ${VERSION_ID} --ptype f1 --class-type onpeak
```

### Step 6: Run comparison
```bash
BATCH_ID=$(jq -r '.batch_id' state.json)
ITERATION=$(jq -r '.iteration' state.json)
python ml/compare.py --batch-id ${BATCH_ID} --iteration ${ITERATION}
```

### Step 7: Write changes summary
```bash
mkdir -p registry/${VERSION_ID}
cat > registry/${VERSION_ID}/changes_summary.md << 'EOF'
# Changes Summary — ${VERSION_ID}

## Hypothesis
Slower learning rate + more trees: lr 0.05→0.03, n_estimators 400→700.

## Changes
- `ml/config.py`: RegressorConfig defaults — n_estimators 400→700, learning_rate 0.05→0.03

## Files Modified
- ml/config.py (2 lines changed in RegressorConfig)

## Files NOT Modified
- ml/evaluate.py, ml/pipeline.py, ml/train.py, ml/features.py, ml/data_loader.py, ml/benchmark.py
EOF
```

### Step 8: Commit
```bash
git add ml/config.py registry/${VERSION_ID}/
git commit -m "iter2: lr 0.05→0.03, n_estimators 400→700"
```

### Step 9: Verify artifacts before handoff
```bash
# ALL of these must exist:
ls -la registry/${VERSION_ID}/metrics.json
ls -la registry/${VERSION_ID}/config.json
ls -la registry/${VERSION_ID}/changes_summary.md
# If ANY are missing, DO NOT write handoff. Debug and fix first.
```

### Step 10: Write handoff
Only after Step 9 confirms all artifacts exist:
```bash
HANDOFF_DIR="handoff/${BATCH_ID}/iter${ITERATION}"
mkdir -p "${HANDOFF_DIR}"
cat > "${HANDOFF_DIR}/worker_done.json" << EOF
{"status": "done", "artifact_path": "registry/${VERSION_ID}/changes_summary.md", "sha256": "$(sha256sum registry/${VERSION_ID}/changes_summary.md | cut -d' ' -f1)"}
EOF
```

---

## Expected Impact

| Gate | v0 Mean | Expected Δ | Reasoning |
|------|---------|------------|-----------|
| EV-VC@100 | 0.069 | +0.005 to +0.015 | Smoother predictions → better top-ranking |
| EV-VC@500 | 0.216 | +0.005 to +0.010 | Same mechanism, broader |
| EV-NDCG | 0.747 | +0.005 to +0.015 | Better ranking quality overall |
| Spearman | 0.393 | ±0.01 | Rank correlation is robust to ensemble changes |
| C-RMSE | 3133 | -50 to -200 | Less overfitting → lower error |

## If It Fails

If the benchmark fails to run (import error, data error, etc.):
1. Record the error message
2. Write a FAILED handoff with the error details
3. Do NOT modify any files to "fix" infrastructure — that's not your job

```bash
HANDOFF_DIR="handoff/${BATCH_ID}/iter${ITERATION}"
mkdir -p "${HANDOFF_DIR}"
cat > "${HANDOFF_DIR}/worker_done.json" << EOF
{"status": "failed", "error": "DESCRIBE THE ERROR HERE"}
EOF
```
