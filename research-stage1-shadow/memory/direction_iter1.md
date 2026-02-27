# Direction — Iteration 1

**Batch**: smoke-v4-20260227-183323
**Date**: 2026-02-27
**Orchestrator**: Planning iteration 1

---

## Hypothesis

Running the pipeline with identical default hyperparameters to v0 will produce metrics within noise tolerance (±0.02), confirming pipeline determinism and end-to-end infrastructure correctness.

This is an **infrastructure validation** iteration, not an ML improvement iteration.

## Specific Changes

**No ML code changes.** The worker should:

1. **Run the pipeline** with default config (identical to v0):
   ```bash
   cd /home/xyz/workspace/pmodel && source .venv/bin/activate
   cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow
   SMOKE_TEST=true python ml/pipeline.py \
     --version-id ${VERSION_ID} \
     --auction-month 2021-07 \
     --class-type onpeak \
     --period-type f0
   ```
   No `--overrides` flag — use all defaults.

2. **Run tests** to verify no regressions:
   ```bash
   python -m pytest ml/tests/ -v
   ```

3. **Run evaluation** (compare against v0):
   ```bash
   python ml/compare.py --candidate ${VERSION_ID} --baseline v0
   ```

4. **Verify** the output metrics.json matches v0 values within noise tolerance (0.02).

## Expected Impact

| Gate | v0 Value | Expected | Gate Status |
|------|----------|----------|-------------|
| S1-AUC | 0.75 | 0.75 ± 0.02 | Pass (floor 0.65) |
| S1-AP | 0.5909 | 0.5909 ± 0.02 | Pass (floor 0.12) |
| S1-VCAP@100 | 1.0 | 1.0 | Pass (floor 0.95) |
| S1-VCAP@500 | 1.0 | 1.0 | Pass (floor 0.95) |
| S1-VCAP@1000 | 1.0 | 1.0 | Pass (floor 0.95) |
| S1-NDCG | 0.5044 | 0.5044 ± 0.02 | Pass (floor 0.4544) |
| S1-BRIER | 0.2021 | 0.2021 ± 0.02 | Pass (floor 0.2221) |
| S1-REC | 0.0 | 0.0 | **FAIL** (floor 0.4, Group B) |
| S1-CAP@100 | 0.0 | 0.0 | Pass (floor -0.05) |
| S1-CAP@500 | 0.0 | 0.0 | Pass (floor -0.05) |

**S1-REC will fail** — this is expected and known. The model's optimal threshold (0.82) produces zero positive predictions on SMOKE_TEST data (20 samples, 2 positive, binding_rate=0.1). This is a data limitation, not a model bug. Group B gates are informational for infrastructure validation.

**Primary success criteria**: All Group A gates pass, metrics match v0 within noise tolerance, and the full pipeline (version allocation → train → evaluate → compare → register) completes without errors.

## Risk Assessment

1. **Low risk**: Non-determinism from XGBoost — mitigated by `random_state=42`. Floating point differences should be negligible.
2. **Low risk**: Version allocation collision — the worker reads VERSION_ID from state.json, which should be set by the launcher.
3. **Low risk**: SMOKE_TEST data loader path — already validated in v0. If it fails, it's an infrastructure bug to fix.
4. **No risk**: No ML code changes, so no regression risk.

## What Reviewers Should Focus On

This iteration is about **infrastructure correctness**, not ML quality:
- Did the pipeline complete all 6 phases successfully?
- Are the metrics deterministic (match v0)?
- Is the version registry populated correctly (config.json, metrics.json, meta.json, model/)?
- Did the compare step produce a valid comparison table?
- Are there any architectural issues in the ML codebase that should be addressed in future iterations?
- Is the S1-REC failure expected given the data constraints? (Yes — it is.)
- Are the gate definitions sensible, or should any be recalibrated at HUMAN_SYNC?
