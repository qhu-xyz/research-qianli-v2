# Direction — Iteration 1

**Batch**: smoke-v7-20260227-191851
**Predecessor**: smoke-v6 iter1 confirmed pipeline determinism (v0001 = v0, zero delta). iter2 planned but not executed.

---

## Hypothesis

**H2 (carried forward): Lowering threshold_beta from 0.7 to 0.3 will produce positive predictions, fixing S1-REC.**

At beta=0.7, the F-beta optimal threshold is 0.82, which is too high for binding_rate=0.1 — the model predicts zero positives (pred_binding_rate=0.0). Reducing beta to 0.3 weights recall more heavily in the F-beta objective, which should lower the optimal threshold substantially, producing positive predictions and moving S1-REC from 0.0 toward the 0.4 floor.

Additionally: fix two HIGH/MEDIUM bugs found in prior reviews (from_phase crash recovery, Group B pass policy).

---

## Specific Changes

### Change 1: Run pipeline with threshold_beta=0.3

**File**: No code change needed — use `--overrides` flag.

**Pipeline command**:
```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow
cd /home/xyz/workspace/pmodel && source .venv/bin/activate && cd -
SMOKE_TEST=true python ml/pipeline.py \
  --version-id ${VERSION_ID} \
  --auction-month 2021-07 \
  --class-type onpeak \
  --period-type f0 \
  --overrides '{"threshold_beta": 0.3}'
```

The version_id will be assigned by the launcher. Read it from `${PROJECT_DIR}/state.json`.

### Change 2: Fix from_phase crash recovery (HIGH)

**File**: `ml/pipeline.py`, function `run_pipeline()`

**Problem**: When `from_phase > 1`, variables `train_df`, `val_df`, `X_train`, `y_train`, `X_val`, `y_val`, `model`, `val_proba`, `threshold` are referenced without being initialized. The `if from_phase <= N:` guards skip earlier phases but later phases assume those variables exist.

**Fix**: Add guards at the top of `run_pipeline()` that raise a clear error if `from_phase > 1` is requested but no saved intermediates exist. For now, the simplest safe fix is:

```python
# After line 58 (after feature_config check), add:
if from_phase > 1:
    raise NotImplementedError(
        f"from_phase={from_phase} requested but intermediate state loading is not yet implemented. "
        "Run from phase 1."
    )
```

This is better than silently referencing uninitialized variables. Full intermediate-state loading can be implemented when needed for real data.

### Change 3: Fix Group B pass policy in compare.py (MEDIUM)

**File**: `ml/compare.py`, function `build_comparison_table()`

**Problem**: Lines 142-162 compute `all_passed` by checking every gate. Group B gates (S1-REC, S1-CAP@100, S1-CAP@500) failing causes `all_passed = False` even though Group B gates are informational — they should NOT block overall pass.

**Fix**: In `build_comparison_table()`, replace the single `all_passed` with separate Group A and Group B tracking:

```python
# Replace the current all_passed logic (lines ~142-162):
group_a_passed = True
group_b_passed = True
for gate_name in gate_names:
    result = gate_results.get(gate_name, {})
    value = result.get("value")
    passed = result.get("passed")
    group = result.get("group", "A")

    if value is None:
        cells.append("--")
    elif isinstance(value, float) and (value != value):
        cells.append("NaN")
        if group == "A":
            group_a_passed = False
        else:
            group_b_passed = False
    else:
        mark = "P" if passed else "F"
        if passed is None:
            mark = "?"
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
        elif not passed:
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
        cells.append(f"{value:.4f} {mark}")

# Overall pass requires all Group A gates to pass.
# Group B status is shown but does not block.
pass_str = "YES" if group_a_passed else "NO"
group_b_str = "YES" if group_b_passed else "NO"
row = f"| {version_id} | " + " | ".join(cells) + f" | {pass_str} (B:{group_b_str}) |"
```

Also update `check_gates()` or add a helper function `evaluate_overall_pass()` that returns `(group_a_passed, group_b_passed)` so the JSON comparison output includes this distinction.

### Change 4: Gzip the model file after save

**File**: `ml/pipeline.py`, after line 181 (model save)

**Add**:
```python
import gzip
import shutil
model_ubj = model_dir / "classifier.ubj"
model_gz = model_dir / "classifier.ubj.gz"
with open(model_ubj, 'rb') as f_in:
    with gzip.open(model_gz, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
model_ubj.unlink()
print(f"[pipeline] model compressed to {model_gz}")
```

### Test protocol

After all changes, run:
```bash
python -m pytest ml/tests/ -v
```

Ensure all existing tests pass. Write additional tests for:
- `test_from_phase_gt1_raises()` — verify from_phase > 1 raises NotImplementedError
- `test_group_b_does_not_block_pass()` — verify Group A pass + Group B fail → overall YES

### Comparison

After pipeline completes, run:
```bash
python ml/compare.py \
  --batch-id ${BATCH_ID} \
  --iteration ${ITERATION} \
  --output registry/${VERSION_ID}/comparison.md
```

---

## Expected Impact

| Gate | v0 Value | Expected Direction | Notes |
|------|----------|-------------------|-------|
| S1-AUC | 0.7500 | unchanged | AUC is threshold-independent |
| S1-AP | 0.5909 | unchanged | AP is threshold-independent |
| S1-VCAP@100 | 1.0000 | unchanged | Saturated at n=20 |
| S1-VCAP@500 | 1.0000 | unchanged | Saturated at n=20 |
| S1-VCAP@1000 | 1.0000 | unchanged | Saturated at n=20 |
| S1-NDCG | 0.5044 | unchanged | Threshold-independent |
| S1-BRIER | 0.2021 | **may worsen** | Predicting positives changes calibration. 0.02 headroom — MONITOR CLOSELY |
| S1-REC (B) | 0.0000 | **should improve** → ≥0.4 | Primary target. beta=0.3 should yield positive predictions |
| S1-CAP@100 (B) | 0.0000 | may improve | Depends on which samples become positive |
| S1-CAP@500 (B) | 0.0000 | may improve | Depends on which samples become positive |

**Primary success criterion**: S1-REC ≥ 0.4 (Group B floor) while all Group A gates still pass.

**Secondary success**: Group B policy fix means overall pass is correctly determined by Group A gates only.

---

## Risk Assessment

1. **S1-BRIER flip (HIGH)**: With only 0.02 headroom (0.2021 vs floor 0.2221), any calibration change could push BRIER above the floor. At n=20, the Brier score is sensitive to even 1-2 prediction changes. **Mitigation**: If BRIER flips, the reviewers should analyze whether the floor is too tight for SMOKE_TEST and recommend recalibration at HUMAN_SYNC.

2. **Threshold too aggressive (MEDIUM)**: beta=0.3 is a significant shift from 0.7. The optimal threshold might drop very low, over-predicting positives and degrading precision substantially. **Mitigation**: Monitor pred_binding_rate in metrics — if it jumps above 0.5, the threshold is too aggressive. Consider beta=0.5 as a middle ground in iter2.

3. **Test breakage from compare.py changes (LOW)**: Changing the pass aggregation logic will likely need test updates. **Mitigation**: Worker should update `test_compare.py` alongside the code change.

---

## Priority Order

1. Fix from_phase (quick, safe, blocks nothing)
2. Fix Group B policy in compare.py + tests
3. Add model gzip
4. Run pipeline with threshold_beta=0.3
5. Run comparison
6. Write changes_summary.md in registry/${VERSION_ID}/
