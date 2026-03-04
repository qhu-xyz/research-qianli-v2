# Progress

## Current State
- **Champion**: v0 (baseline)
- **Iterations completed**: 0
- **Current batch**: tier-fe-1-20260304-182037 (FE only)
- **Current iteration**: 1 — ORCHESTRATOR PLAN DONE, awaiting worker

## Iteration 1 Plan
- **Hypothesis A**: Swap 3 low-importance features for 3 existing interaction features (hist_physical_interaction, overload_exceedance_product, band_severity)
- **Hypothesis B**: Aggressive prune — remove 6 lowest-importance features (34→28)
- **Screen months**: 2021-11 (weak), 2021-09 (strong)
- **Winner criteria**: Higher mean Tier-VC@100, QWK safety check on strong month

## Pipeline Status
- v0 baseline benchmark: DONE (12/12 months, ~90 min runtime)
- Gate calibration: DONE (Value-QWK calibrated: floor=0.391, tail_floor=0.180)
- Memory files: DONE
- Registry: v0/ populated with metrics.json, feature_importance.json, config.json

## Priority Improvement Areas (from v0 analysis)
1. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
2. Tier-VC@100 very poor (0.071) — top-of-ranking quality must improve
3. High variance across months — 2021-11 and 2022-06 are disaster months
4. Tier 4 has 0 samples — effectively a 4-class problem
