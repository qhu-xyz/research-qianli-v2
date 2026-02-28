# Changes Summary — v0001 (Iteration 1)

**Batch**: smoke-v7-20260227-191851
**Predecessor**: v0 (baseline)
**Hypothesis**: H2 — Lowering threshold_beta from 0.7 to 0.3 to produce positive predictions, fixing S1-REC.

---

## Changes Made

### 1. from_phase crash recovery guard (pipeline.py)
- Added `NotImplementedError` guard when `from_phase > 1` is requested
- Previously, phases > 1 silently referenced uninitialized variables (`train_df`, `model`, etc.)
- Now fails immediately with a clear message instead of crashing with `NameError`

### 2. Group B pass policy fix (compare.py)
- `build_comparison_table()` now tracks Group A and Group B gates separately
- Group A gates determine the overall pass/fail ("YES" / "NO")
- Group B gates are informational — shown as "(B:YES)" or "(B:NO)" but do not block promotion
- Added `evaluate_overall_pass()` helper that returns `(group_a_passed, group_b_passed)`
- `run_comparison()` JSON output now includes `pass_summary` with per-version group pass status

### 3. Model gzip (pipeline.py)
- After `model.save_model()`, the `.ubj` file is now compressed to `.ubj.gz` and the uncompressed file is removed
- Complies with the runbook requirement to gzip model files on write

### 4. Pipeline run with threshold_beta=0.3
- Ran pipeline via `--overrides '{"threshold_beta": 0.3}'`
- **Result**: Threshold unchanged at 0.8203, S1-REC remains 0.0

---

## Results

| Gate | v0 | v0001 | Delta | Status |
|------|-----|-------|-------|--------|
| S1-AUC | 0.7500 | 0.7500 | 0.0000 | P (unchanged) |
| S1-AP | 0.5909 | 0.5909 | 0.0000 | P (unchanged) |
| S1-VCAP@100 | 1.0000 | 1.0000 | 0.0000 | P (unchanged) |
| S1-VCAP@500 | 1.0000 | 1.0000 | 0.0000 | P (unchanged) |
| S1-VCAP@1000 | 1.0000 | 1.0000 | 0.0000 | P (unchanged) |
| S1-NDCG | 0.5044 | 0.5044 | 0.0000 | P (unchanged) |
| S1-BRIER | 0.2021 | 0.2021 | 0.0000 | P (unchanged) |
| S1-REC (B) | 0.0000 | 0.0000 | 0.0000 | F (unchanged) |
| S1-CAP@100 (B) | 0.0000 | 0.0000 | 0.0000 | P (unchanged) |
| S1-CAP@500 (B) | 0.0000 | 0.0000 | 0.0000 | P (unchanged) |

**Overall**: YES (B:NO) — All Group A gates pass. Group B (S1-REC) still fails.

---

## Analysis: Why H2 Failed

The hypothesis stated: "Reducing beta to 0.3 weights recall more heavily in the F-beta objective."

**This is incorrect.** In the F-beta formula:

```
F_β = (1 + β²) × (precision × recall) / (β² × precision + recall)
```

- **β < 1** weights **precision** more heavily (not recall)
- **β > 1** weights **recall** more heavily
- **β = 1** is the standard F1 score (equal weighting)

So β=0.3 (down from 0.7) actually makes the optimization MORE precision-oriented, which pushes the threshold **higher** (more conservative), not lower. The optimal threshold remained at 0.8203, producing zero positive predictions.

**To fix S1-REC**, the next iteration should try β > 1 (e.g., β=2.0 or β=3.0) to weight recall more heavily and drive the threshold down. At n=20 with only 2 positives, a high β is needed to overcome the extreme precision/recall imbalance.

---

## Tests

- 70/70 tests pass (including 4 new tests)
- `test_from_phase_gt1_raises` — validates from_phase guard
- `test_group_b_does_not_block_pass` — validates Group A pass + Group B fail = overall YES
- `test_group_b_does_not_block_table` — validates table rendering with Group B info
- `test_all_pass_shows_both_yes` — validates table when all gates pass
