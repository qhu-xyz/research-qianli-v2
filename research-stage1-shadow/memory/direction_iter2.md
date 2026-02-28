# Direction — Iteration 2

**Batch**: smoke-v7-20260227-191851
**Predecessor**: v0001 (H2 failed — beta=0.3 had no effect due to inverted F-beta understanding. All metrics identical to v0.)

---

## Hypothesis

**H3: Two independent fixes — (a) threshold_beta=2.0 and (b) threshold `>=` alignment — will produce positive predictions, fixing S1-REC.**

### H3a: Use beta > 1 to favor recall
At beta=0.3 (iter1), the F-beta formula weighted precision more, keeping the threshold at 0.82 and producing zero positives. In the F-beta formula F_β = (1 + β²) × P × R / (β² × P + R):
- **beta < 1 → weights precision** (higher threshold, fewer positives)
- **beta > 1 → weights recall** (lower threshold, more positives)

beta=2.0 gives F_2 = 5PR/(4P+R), which heavily weights recall. At n=20 with binding_rate=0.1 (2 positives), we need the optimizer to accept low precision to achieve non-zero recall.

### H3b: Fix threshold `>` vs `>=` mismatch
`precision_recall_curve` returns thresholds where predictions at that value are classified as positive (inclusive, `>=`). But `apply_threshold` (ml/threshold.py:64) uses strict `>`, making realized predictions more conservative than the optimized operating point. At discrete n=20, if the optimal threshold equals a sample's predicted probability, that sample gets excluded by strict `>` but would be included by `>=`. Fixing this aligns realized classification with the optimization target.

---

## Specific Changes

### Change 1: Fix threshold `>` to `>=` in apply_threshold (HIGH)

**File**: `ml/threshold.py`, line 64

**Current**:
```python
return (np.asarray(y_proba) > threshold).astype(int)
```

**Fix**:
```python
return (np.asarray(y_proba) >= threshold).astype(int)
```

Also update the docstring on line 63 from `"1 if proba > threshold"` to `"1 if proba >= threshold"`.

**Rationale**: Aligns with sklearn's `precision_recall_curve` semantics where thresholds are inclusive.

### Change 2: Fix misleading beta docstring (LOW)

**File**: `ml/threshold.py`, line 22

**Current**: `"Beta parameter for F-beta score (0.7 = moderate recall/precision balance)."`

**Fix**: `"Beta parameter for F-beta score. beta < 1 favors precision; beta > 1 favors recall; beta = 1 is standard F1."`

### Change 3: Run pipeline with threshold_beta=2.0

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
  --overrides '{"threshold_beta": 2.0}'
```

The version_id will be assigned by the launcher. Read it from `${PROJECT_DIR}/state.json`.

### Test protocol

After changes 1-2, run:
```bash
python -m pytest ml/tests/ -v
```

Ensure all existing tests pass. Update any tests that assert strict `>` behavior to assert `>=`. If `test_apply_threshold` tests exist, verify they reflect the new inclusive semantics.

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
| S1-AUC | 0.7500 | unchanged | Threshold-independent |
| S1-AP | 0.5909 | unchanged | Threshold-independent |
| S1-VCAP@100 | 1.0000 | unchanged | Saturated at n=20 |
| S1-VCAP@500 | 1.0000 | unchanged | Saturated at n=20 |
| S1-VCAP@1000 | 1.0000 | unchanged | Saturated at n=20 |
| S1-NDCG | 0.5044 | unchanged | Threshold-independent |
| S1-BRIER | 0.2021 | **likely worsens** | Predicting positives changes calibration. Only 0.02 headroom. HIGH risk of flip. |
| S1-REC (B) | 0.0000 | **should improve → ≥0.4** | Primary target. beta=2.0 + `>=` fix should produce positives. |
| S1-CAP@100 (B) | 0.0000 | may improve | Depends on which samples become positive |
| S1-CAP@500 (B) | 0.0000 | may improve | Depends on which samples become positive |

**Primary success criterion**: S1-REC ≥ 0.4 (Group B floor) while all Group A gates still pass.

**Secondary success**: pred_binding_rate > 0 (model produces any positive predictions at all).

---

## Risk Assessment

1. **S1-BRIER flip (HIGH)**: With only 0.02 headroom (0.2021 vs floor 0.2221), producing positive predictions will change Brier. At n=20 with 2 positives, even 1 false positive noticeably shifts Brier. **Mitigation**: If BRIER flips, document the actual value. If the increase is small (< 0.05 above floor) and proportionate to the recall gain, recommend BRIER floor recalibration at HUMAN_SYNC. Group A BRIER failure would prevent promotion, but this iteration's primary goal is proving the model CAN produce positive predictions.

2. **Overshoot (MEDIUM)**: beta=2.0 is a large swing from 0.3. The threshold might drop very low, over-predicting positives. **Mitigation**: Monitor pred_binding_rate — if it jumps above 0.5, the threshold is too aggressive. If so, iter3 could try beta=1.5 as a middle ground.

3. **Test breakage from >= change (LOW)**: Existing tests for `apply_threshold` may assert strict `>` behavior. **Mitigation**: Update test expectations to match `>=` semantics.

---

## Priority Order

1. Fix threshold `>` to `>=` in threshold.py (small, high impact)
2. Fix beta docstring in threshold.py (trivial)
3. Run tests — ensure all pass
4. Run pipeline with threshold_beta=2.0
5. Run comparison
6. Write changes_summary.md in registry/${VERSION_ID}/
