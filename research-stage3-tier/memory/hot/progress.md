# Progress

## Current State
- **Champion**: v0 (baseline)
- **Iterations completed**: 1 (FAILED — no artifacts)
- **Current batch**: tier-fe-1-20260304-182037 (FE only)
- **Current iteration**: 2 — orchestrator plan written, awaiting worker

## Iteration History

### Iter 1 — FAILED
- **Hypothesis**: Feature swap (interaction features + light pruning) vs aggressive pruning
- **Outcome**: Worker claimed "done" but produced NO artifacts. registry/v0001/ doesn't exist.
- **Data collected**: None. Hypotheses untested.

### Iter 2 — PLANNED
- **Hypothesis A**: Add 3 dead interaction features (hist_physical_interaction, overload_exceedance_product, hist_seasonal_band) → 37 features
- **Hypothesis B**: Prune 6 lowest-importance features + add same 3 interactions → 31 features
- **Screen months**: 2021-11 (weak), 2021-09 (strong)
- **Key change from iter1**: Both hypotheses screenable via --overrides only (all features already exist as DataFrame columns). No code changes needed for screening phase.

## Pipeline Status
- v0 baseline benchmark: DONE
- Gate calibration: DONE
- Iter 1: FAILED (worker produced no artifacts)
- Iter 2: ORCHESTRATOR PLAN DONE — awaiting worker
- Remaining iterations: 1 (iter3)

## Priority Improvement Areas (unchanged from v0 analysis)
1. Tier-VC@100 below floor (0.071 vs 0.075) — only Group A gate failing Layer 1
2. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
3. High variance across months — 2021-11 and 2022-06 are disaster months
4. Tier 4 has 0 samples — effectively a 4-class problem
