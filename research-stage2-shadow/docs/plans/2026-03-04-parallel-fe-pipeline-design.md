# Design: Parallel Feature Engineering Pipeline

**Date**: 2026-03-04
**Goal**: Run a 3-iteration autonomous batch focused exclusively on feature engineering/selection for the stage-2 shadow price regressor, in parallel with the existing `feat-eng-3` batch.

---

## Architecture

```
research-stage2-shadow/              <- existing batch (feat-eng-3, v0008+)
research-stage2-shadow-fe-parallel/  <- new batch (FE-only, v1001+)
```

Both share:
- Ray cluster (`ray://10.8.0.36:10001`)
- Frozen stage-1 classifier (v0008)
- Data source (`/opt/temp/tmp/pw_data/spice6`)
- 12 eval months, same gates
- `claude` + `codex` CLI subscriptions (concurrent)

## Setup Steps

1. **Copy directory**: `cp -r research-stage2-shadow research-stage2-shadow-fe-parallel`
2. **Patch `agents/config.sh`**: Update `PROJECT_DIR` to new directory
3. **Reset `state.json`**: `{"state": "IDLE", "batch_id": null, "iteration": 0, "version_id": "v0007"}`
4. **Version namespace**: Set `registry/version_counter.json` to `{"counter": 1001}`. Keep `v0` (baseline) and `v0007` (champion). Delete intermediate versions (v0005, v0006, v0007-reeval, v1, v0-reeval).
5. **Champion**: Keep `champion.json` pointing at `v0007`
6. **Clean artifacts**: Remove stale handoff/, reviews/, reports/, memory/direction_iter*.md from prior batches
7. **Human prompt**: Write FE-only direction to `memory/human_input.md`

## Human Prompt

```
Focus: Feature engineering and feature selection ONLY for the regressor.
Do NOT change hyperparameters, architecture, or the classifier.

Starting point: v0007 champion (34 regressor features).

Each iteration:
1. Research: analyze feature importance from v0007, identify dead features
   and potential new features/interactions from available data columns.
2. Generate 2 hypotheses (e.g., "drop 5 lowest-importance features" vs
   "add 3 new interaction terms").
3. Screen both on 2 months (1 weak: 2022-06, 1 strong: 2021-09).
4. Implement winner, run full 12-month benchmark.

Reporting: target month only (not val set).

Ideas to explore:
- Feature pruning: remove low-importance features that add noise
- New interactions: ratios, products of existing density/exceedance features
- Temporal features: lagged values, rolling statistics
- Monotone constraint review: verify constraints match domain knowledge
```

## Scope Constraints

- **ONLY modify**: regressor feature list, monotone_constraints, interaction feature computation in `ml/features.py`
- **NEVER modify**: classifier config, eval months, gates, data loader, evaluation harness
- **NEVER change**: hyperparameters (n_estimators, learning_rate, max_depth, etc.)

## Evaluation & Merge

After both batches complete:
1. Compare final champions from each on the same 12 eval months (blocking gates: EV-VC@100, EV-VC@500, EV-NDCG, Spearman)
2. Port the better version to main `research-stage2-shadow/registry/` as next sequential version
3. Update `champion.json` in main registry if it beats v0007

## Resource Notes

- Both pipelines use Ray cluster concurrently — workers may briefly compete for resources but benchmarks are staggered by iteration timing
- Both use `claude` + `codex` CLI — concurrent usage is acceptable with Max/Pro subscriptions
- Memory: each benchmark ~2-4 GiB, well within pod budget
