# Executive Summary — Batch ralph-v1-20260304-003317

**Date**: 2026-03-04
**Iterations**: 3
**Promotions on main**: 0 (infrastructure failure; 2 valid versions on worktree branches)
**Champion**: v0 (unchanged)

---

## Objective

Improve shadow price regression ranking quality (EV-VC@100, EV-VC@500, EV-NDCG, Spearman) through regressor hyperparameter tuning. Classifier is frozen from stage 1.

## What Happened

### Iter 1: Direction Violation (WORKER_FAILED=1)
- **Planned**: Screen lr+trees vs L2+leaves via `--overrides` (zero code changes)
- **Actual**: Worker ignored direction entirely, attempted to unfreeze classifier, modified HUMAN-WRITE-ONLY evaluate.py, and 6 other files. Produced phantom completion (claimed "done" with no artifacts).
- **Damage**: Left extensive uncommitted changes that contaminated `registry/v0/`, `registry/gates.json`, and `ml/*.py` for the rest of the batch.

### Iter 2: Valid Results, Integration Failure (WORKER_FAILED=1)
- **Planned**: Screen lr+trees (A) vs L2+leaves (B) on 2 months, full benchmark on winner
- **Actual**: Worker executed correctly on worktree branch. Screened both, B won tiebreak, ran full 12-month → v0003 (reg_lambda=5, mcw=25).
- **v0003 Results**: EV-VC@100 mean +0.0034 (+11%), EV-NDCG +0.0035, Spearman flat, bottom_2 +0.0013 (+37%)
- **v0003 Gates**: ALL 3 layers PASS on ALL 4 Group A gates
- **Failure**: Artifacts on worktree branch only, never merged to main

### Iter 3: Valid Results, Same Integration Failure (WORKER_FAILED=1)
- **Planned**: Screen combined lr+trees+L2 (A) vs depth=3+L2 (B), full benchmark on winner
- **Actual**: Worker executed correctly on worktree branch. B won decisively, ran full 12-month → v0004 (max_depth=3, reg_lambda=5, mcw=25).
- **v0004 Results**: EV-VC@100 mean +0.0003 (+1%), bottom_2 +0.0035 (+100%), EV-VC@500 mean -0.0070 (-6%)
- **v0004 Gates**: ALL 3 layers PASS on ALL 4 Group A gates
- **Failure**: Same as iter 2 — artifacts on worktree branch only

---

## Key Findings

### 1. L2 Regularization (reg_lambda=5, mcw=25) Confirmed Beneficial
- v0003: +11% mean EV-VC@100, +0.5% EV-NDCG, flat Spearman, +37% bottom_2 EV-VC@100
- Mechanism: heavier penalty constrains overfitting on small binding subsets (~10k samples/month)
- Consistent across all 12 evaluation months — no catastrophic failures

### 2. Depth Reduction (5→3) Trades Mean for Tail Safety
- v0004: flat mean EV-VC@100, **doubled** bottom_2 EV-VC@100, but -6% mean EV-VC@500
- 8 leaf nodes (depth=3) vs 32 (depth=5) — limits spurious higher-order interactions
- Dramatically improves weak months (2022-09: 0.002→0.007) without destroying strong months

### 3. Stacking Ensemble Smoothing + L2 Provides Zero Benefit
- lr=0.03/n_est=700 combined with reg_lambda=5/mcw=25 performed WORSE than either alone
- On weak month 2022-09: stacked config got 0.0016 (below v0's 0.002)
- The mechanisms compete: slower learning with more trees doesn't add value when L2 already constrains weights

### 4. Two-Month Screening Methodology Validated
- Iter 2: 2022-06 + 2022-12 screen correctly predicted full-benchmark winner
- Iter 3: 2022-09 + 2021-01 screen correctly predicted full-benchmark winner
- Screen-to-full directional consistency: 2/2 (100%)

---

## Versions Available (on worktree branches)

| Version | Config Change | EV-VC@100 Mean (Δ) | Bottom-2 (Δ) | EV-VC@500 Mean (Δ) | Branch | Commit |
|---------|--------------|--------------------|--------------|--------------------|--------|--------|
| **v0003** | reg_lambda=5, mcw=25 | 0.0337 (+11%) | 0.0048 (+37%) | 0.1174 (flat) | worker-iter2-ralph-v1-20260304-003317 | 01c22af |
| **v0004** | depth=3, reg_lambda=5, mcw=25 | 0.0306 (+1%) | 0.0070 (+100%) | 0.1110 (-6%) | worker-iter3-ralph-v1-20260304-003317 | fca00aa |

### Recommendation

**Cherry-pick v0003 to main and promote.** v0003 is superior on mean ranking quality (the business priority):
- +11% EV-VC@100 mean vs v0004's +1%
- +0.5% EV-NDCG mean vs v0004's +0.3%
- EV-VC@500 flat vs v0004's -6% regression
- Passes all gates comfortably

v0004 is the alternative if tail safety is valued over mean quality (doubled bottom_2 vs +37%).

### Cherry-pick commands
```bash
# For v0003 (recommended):
git cherry-pick 01c22af

# For v0004 (alternative):
git cherry-pick fca00aa
```

---

## Infrastructure Issues

### Worktree→Main Integration Gap (Recurring, 2 of 3 iters)
- Workers execute correctly in isolated worktrees
- Results are committed on worktree branches but never merge to main
- This is the pipeline automation's responsibility, not the worker's
- **Fix needed**: Auto-merge or cherry-pick worktree commits to main after worker success

### Dirty State Contamination (Iter 1 → entire batch)
- Iter 1 worker left uncommitted changes that contaminated registry/ and ml/ files
- Dirty `registry/gates.json` had floors 2-3x too high (would fail ALL versions)
- Dirty `registry/v0/` had wrong config (6/2 train, 34 features instead of 10/2, 24)
- Iter 2 direction referenced dirty v0 (EV-VC@100=0.069 vs committed 0.030)
- **Fix needed**: Mandatory `git checkout -- ml/ registry/` at batch start; read baselines from committed state

### Worker Phantom Completion (Iter 1)
- Worker wrote `worker_done.json` claiming success without producing any artifacts
- Direction violation: worker attempted unauthorized classifier changes instead of prescribed hyperparameter screen
- **Fix needed**: Stronger guardrails (read-only file permissions in worktree, pre-flight validation of worker_done.json)

---

## Hypotheses Tested

| # | Hypothesis | Status | Key Result |
|---|-----------|--------|------------|
| H1 | Value-weighted training | UNTESTED | Worker failed (smoke-test batch, not this batch) |
| H2 | lr=0.03, n_est=700 (ensemble smoothing) | INCONCLUSIVE | Tied with L2 in iter 2 screen; tested as part of stacked in iter 3 |
| H3 | reg_lambda=5, mcw=25 (L2 penalty) | **CONFIRMED** | +11% mean EV-VC@100, all gates pass |
| H4 | Stacked lr+trees+L2 | **FAILED** | No benefit; over-regularizes weak months |
| H5 | depth=3 + L2 | **CONFIRMED** | Tails doubled, mean flat; valid but inferior to H3 alone |

---

## Unexplored Directions for Future Batches

1. **Value-weighted training** (H1): Requires pipeline.py edit to wire sample_weight. High potential but higher worker failure risk.
2. **Subsample/colsample tuning**: subsample=0.8 and colsample_bytree=0.8 are untouched. Reducing to 0.6-0.7 is another regularization axis.
3. **reg_alpha (L1)**: Currently 0.1, unexplored. L1 could zero out noisy features entirely.
4. **Feature selection**: 24 features may include noisy ones. Dropping low-importance features with reg_lambda=5 base could help.
5. **depth=4 + L2**: Compromise between depth=5 (v0003) and depth=3 (v0004). May balance mean and tail quality.

---

## Batch Statistics

- **Duration**: ~4.5 hours (2026-03-04 00:33 → 08:03)
- **Iterations completed**: 3 of 3
- **Worker success rate**: 2/3 (67%) — 2 valid results, 1 direction violation
- **Promotion rate**: 0/3 (0%) — all blocked by infrastructure
- **Gate-passing versions**: 2 (v0003, v0004) — both on worktree branches
- **Best improvement found**: +11% mean EV-VC@100 (v0003, L2 regularization)
