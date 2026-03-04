# Progress

## Current State
- **Champion**: v0 (baseline)
- **Iterations completed**: 0
- **Current iteration**: 1 (planning done, awaiting worker)
- **Batch**: tier-v1-20260304-145001

## Pipeline Status
- v0 baseline benchmark: DONE (12/12 months)
- Gate calibration: DONE (floors set from v0 means)
- Memory files: DONE
- **Iteration 1 plan: DONE** — direction_iter1.md written

## Iteration 1 Plan Summary
- **Focus**: Fix Tier-Recall@1 (0.098) and Tier-VC@100 (0.075)
- **Hypothesis A**: class_weights {0:15, 1:15, 2:3, 3:1, 4:0.5}, min_child_weight=10
- **Hypothesis B**: class_weights {0:20, 1:20, 2:3, 3:1, 4:0.3}, min_child_weight=5, n_estimators=600
- **Screen months**: 2022-06 (weak) + 2021-09 (strong)
- Awaiting worker screening and full benchmark

## Priority Improvement Areas (from v0 analysis)
1. Tier-Recall@1 catastrophically low (0.098) — needs aggressive class weight increase for tier 1
2. Tier-VC@100 very poor (0.075) — top-of-ranking quality must improve
3. High variance across months — regularization or feature engineering needed
4. Tier 4 has 0 samples — effectively a 4-class problem; consider in future iterations
