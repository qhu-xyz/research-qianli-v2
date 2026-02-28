# Direction — Iteration 2

**Batch**: smoke-v6-20260227-190225
**Iteration**: 2 of 3
**Objective**: Fix S1-REC gate failure + address code review findings

## Hypothesis

Lowering `threshold_beta` from 0.7 to 0.3 will shift the F-beta optimization to favor recall more heavily, producing a lower optimal threshold that allows the model to predict some positives. This should move S1-REC from 0.0 toward the 0.4 floor. Secondary: fix the `from_phase` crash recovery bug and implement Group B pass policy in `compare.py`.

## Specific Changes

The worker should make three categories of changes:

### A. ML Configuration Change (PRIMARY)
1. **Lower `threshold_beta`** from 0.7 to 0.3 in the pipeline config
   - F-beta with beta=0.3 weighs precision more but with a lower beta squared denominator, the optimal threshold point shifts
   - Actually, **lower beta = more precision weight**, higher beta = more recall weight
   - So to favor recall: **raise threshold_beta from 0.7 to 1.5** (or even 2.0)
   - At beta=2.0, recall is weighted 4x more than precision in the F-beta score
   - This should produce a lower optimal threshold, allowing positive predictions
   - **Use `threshold_beta=2.0`** — this strongly favors recall over precision

### B. Bug Fix: from_phase crash recovery
2. **Fix `from_phase` in `pipeline.py`**: When `from_phase > 1`, phases that are skipped leave variables uninitialized (train_df, val_df, model, X_val, val_proba, etc.). Either:
   - Persist phase artifacts to disk and reload them when resuming, OR
   - Raise a clear error if `from_phase > 1` is attempted without persisted artifacts (simpler, acceptable for now)

### C. Bug Fix: Group B pass policy
3. **Fix Group B aggregation in `compare.py`**: Currently `build_comparison_table` computes overall Pass as ALL gates passing including Group B. Change to:
   - Overall Pass = all Group A gates pass (Group B failures are non-blocking)
   - Still show Group B gate results in the table with their pass/fail status
   - Add a separate column or annotation for Group B status

### D. Run and Validate
4. **Run pipeline**: `SMOKE_TEST=true python ml/pipeline.py --version-id v0002 --auction-month 2021-07 --class-type onpeak --period-type f0`
   - Override threshold_beta=2.0 in the config or command line
5. **Run tests**: `python -m pytest ml/tests/ -v` — all must pass (update tests if Group B policy change affects existing assertions)
6. **Run comparison**: against v0 baseline

## Expected Impact

With threshold_beta=2.0, the F-beta optimization will favor recall heavily. Expected changes:

| Gate | v0 Value | Expected v0002 | Direction | Risk |
|------|----------|-----------------|-----------|------|
| S1-AUC | 0.75 | ~0.75 | Unchanged | AUC is threshold-independent |
| S1-AP | 0.5909 | ~0.5909 | Unchanged | AP is threshold-independent |
| S1-VCAP@100 | 1.0 | ~1.0 | Unchanged | Saturated at n=20 |
| S1-VCAP@500 | 1.0 | ~1.0 | Unchanged | Saturated at n=20 |
| S1-VCAP@1000 | 1.0 | ~1.0 | Unchanged | Saturated at n=20 |
| S1-NDCG | 0.5044 | ~0.5044 | Unchanged | Rank-based, threshold-independent |
| S1-BRIER | 0.2021 | ~0.2021 | Unchanged | **WATCH**: Brier uses probabilities, not threshold |
| S1-REC (B) | 0.0 | >0.0 | **Improve** | Goal: reach 0.4+ (pass Group B floor) |
| S1-CAP@100 (B) | 0.0 | ≥0.0 | Possible improve | May improve if positives are predicted correctly |
| S1-CAP@500 (B) | 0.0 | ≥0.0 | Possible improve | May improve if positives are predicted correctly |

**Key insight**: AUC, AP, NDCG, and Brier are all threshold-independent metrics (they use probabilities or rankings, not binary predictions). Only S1-REC and S1-CAP@K are affected by threshold changes. This means **lowering the threshold should be safe** for Group A gates.

**Caveat**: The SMOKE_TEST dataset has only 2 positives out of 20 samples. Even with a lower threshold, recall can only be 0.0, 0.5, or 1.0 (0/2, 1/2, or 2/2). We need at least recall=0.5 to pass the 0.4 floor.

## Success Criteria

1. S1-REC > 0.0 (ideally ≥ 0.5 to pass the 0.4 floor)
2. All Group A gates still pass (AUC, AP, VCAP, NDCG, BRIER)
3. from_phase either works correctly or raises a clear error
4. Group B pass policy implemented: overall Pass considers only Group A gates
5. All tests pass

## Risk Assessment

- **Low risk on Group A gates**: Threshold change only affects binary-prediction metrics (REC, CAP@K). Probability-based metrics (AUC, AP, NDCG, BRIER) are unaffected.
- **Medium risk on S1-REC**: With only 2 positives, the threshold must land between the 1st and 2nd positive's predicted probabilities. If both positives have very similar probabilities, we might get recall=1.0 or recall=0.0 with nothing in between.
- **Low risk on code fixes**: from_phase fix and Group B policy are localized changes.

## Notes for Reviewers

This iteration has both ML changes (threshold_beta) and code fixes (from_phase, Group B policy). Reviewers should:
- Verify threshold_beta=2.0 actually produces a lower threshold and non-zero recall
- Check that Group B policy implementation is correct and tests are updated
- Verify from_phase fix is sound (either full persistence or clean error)
- Confirm Group A gates are unaffected by threshold change
- Check that the comparison report correctly reflects the new Group B policy
