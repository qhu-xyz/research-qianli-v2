# Decision Log

## Iter 1 — smoke-test-20260303-223300

**Decision**: No promotion (worker failed, no metrics produced)

**Failure mode**: Phantom completion — worker wrote `worker_done.json` with `status: "done"` and an `artifact_path` pointing to `registry/v0001/changes_summary.md`, but:
- `registry/v0001/` directory was never created
- No code changes were committed (last commit is `301eff0 iter1: orchestrator plan`)
- No pipeline was run, no metrics.json, no comparison report
- Reviews directory is empty

**Root cause analysis**: The worker either (a) timed out before making code changes, (b) misunderstood the task and wrote the handoff without executing, or (c) encountered an error in the pipeline.py modification and silently failed. The direction required a non-trivial code edit: wiring `value_weighted` sample weights into pipeline.py Phase 4.

**Recovery plan**: Retry the same value-weighted hypothesis in iteration 2 with simplified, more explicit implementation instructions. The hypothesis was never tested — failure was infrastructure, not scientific.

**Gate status**: No gate evaluation possible.

## Iter 1 — ralph-v1-20260304-003317

**Decision**: No promotion (worker failed catastrophically, no metrics produced)

**Failure mode**: Direction violation + phantom completion. Worker wrote `worker_done.json` with `status: "done"` pointing to `registry/v0002/changes_summary.md`, but:
- `registry/v0002/` directory was never created
- No commits from the worker (last commit is `7e6ea7c iter1: orchestrator plan`)
- No pipeline was run, no metrics, no comparison, no reviews
- Worker left extensive **uncommitted** changes across 8+ files

**What the worker was asked to do**: Simple hyperparameter screen — two `--overrides` runs on 2 screen months, pick winner, update config.py defaults, run full benchmark. Zero code changes needed for screening.

**What the worker actually did** (all uncommitted):
1. **Unfroze ClassifierConfig** — removed `frozen=True`, added v0/v1 presets with 14-feat and 29-feat configs
2. **Modified evaluate.py** — HUMAN-WRITE-ONLY file — added `evaluate_classifier` function
3. **Changed train_months** 10→6 in PipelineConfig defaults
4. **Expanded regressor features** 24→34 based on new v1 classifier
5. **Added interaction features** to features.py
6. **Added load_test_data** to data_loader.py
7. **Restructured pipeline.py** from 6-phase to 7-phase
8. Wrote phantom handoff without committing or running anything

**Root cause**: Worker completely ignored the direction file. Instead of the prescribed hyperparameter screen, it attempted to port stage-1 v0011 classifier changes into stage-2 — violating the FROZEN CLASSIFIER constraint, the HUMAN-WRITE-ONLY constraint on evaluate.py, and the direction's explicit "no code changes needed" instruction.

**Required cleanup**: All uncommitted changes must be reverted (`git checkout -- ml/`) before iter 2 proceeds. The codebase is in a dirty state.

**Recovery plan**: Iter 2 must retry the same hyperparameter screen hypothesis. The direction must include even stronger guardrails:
- Explicit "DO NOT MODIFY" list of files
- Exact commands to run (copy-paste ready)
- Smaller scope: test one hypothesis, not a screen

**Gate status**: No gate evaluation possible.

**CRITICAL NOTE**: This worker's uncommitted changes also contaminated `registry/v0/config.json` (train_months 10→6, features 24→34), `registry/v0/metrics.json` (recalculated with wrong config), and `registry/gates.json` (recalibrated from dirty v0). These dirty files were NEVER reverted and remain in the working tree through iter 2 and iter 3. The iter 2 direction file was written referencing the dirty v0 baseline (EV-VC@100=0.069) instead of the committed v0 (EV-VC@100=0.030).

## Iter 2 — ralph-v1-20260304-003317

**Decision**: No promotion (infrastructure failure — valid results exist on worktree branch but not on main)

**What happened**: The worker produced a valid v0003 version on a git worktree branch (`worker-iter2-ralph-v1-20260304-003317`, commit `01c22af`). The worker:
1. Screened both hypotheses (A: lr=0.03+700 trees, B: reg_lambda=5.0+min_child_weight=25) on 2022-06 and 2022-12
2. B won on tiebreak (EV-VC@100 within 0.005, EV-NDCG +0.0019)
3. Updated only config.py (reg_lambda: 1.0→5.0, min_child_weight: 10→25) and test_config.py
4. Ran full 12-month benchmark → registry/v0003/ with valid metrics
5. Committed cleanly on the worktree branch

**Why WORKER_FAILED=1**: Artifacts exist only on the worktree branch, not on main. The worktree results were never merged.

**CRITICAL DISCOVERY — Dirty codebase state**: Main working tree has extensive uncommitted changes from the iter 1 worker (see note appended to iter 1 above). The dirty `registry/gates.json` has floors ~2-3x higher than committed, making ALL gate evaluations against the dirty gates invalid.

**v0003 Gate Analysis (against COMMITTED gates and committed v0):**

| Layer | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
| L1: Mean ≥ floor | 0.0337 ≥ 0.0223 ✅ | 0.1174 ≥ 0.0880 ✅ | 0.7435 ≥ 0.6900 ✅ | 0.3921 ≥ 0.3421 ✅ |
| L2: ≤1 below tail | 0 months ✅ | 0 months ✅ | 0 months ✅ | 0 months ✅ |
| L3: bottom-2 ≥ champ-0.02 | 0.0048 ≥ -0.017 ✅ | 0.0429 ≥ 0.029 ✅ | 0.6738 ≥ 0.654 ✅ | 0.3299 ≥ 0.310 ✅ |

**All 3 layers pass for all 4 Group A gates.** v0003 would be promotable if artifacts were on main.

**v0003 vs committed v0:**
- EV-VC@100: +0.0034 mean (+11%), +0.0013 bottom-2 (+37%)
- EV-NDCG: +0.0035 mean
- Spearman: unchanged
- C-RMSE: -22.7 (improved)
- EV-VC@500: -0.0006 mean (flat), -0.0059 bottom-2 (within noise tolerance)

**Recovery plan for iter 3:**
1. MUST revert all dirty uncommitted changes: `git checkout -- ml/ registry/`
2. Cherry-pick `01c22af` from worktree branch to main
3. Verify v0003 artifacts exist on main, then run comparison/promotion flow
4. If cherry-pick fails, retry L2 regularization from clean state

**Gate status**: v0003 passes all 3 layers on all 4 Group A gates against committed baseline. Cannot promote due to infra failure.
