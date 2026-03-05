# Progress

## Current State
- **Champion**: v0 (baseline)
- **Iterations completed**: 1 (FAILED — no artifacts)
- **Current batch**: tier-fe-1-20260304-182037 (FE only)
- **Current iteration**: 1 → synthesizing failure, planning iter2

## Iteration History

### Iter 1 — FAILED
- **Hypothesis**: Feature swap (interaction features + light pruning) vs aggressive pruning
- **Outcome**: Worker claimed "done" but produced NO artifacts. registry/v0001/ doesn't exist.
- **Data collected**: None. Hypotheses untested.
- **Recovery**: Simplify direction for iter2 — single hypothesis, explicit steps.

## Pipeline Status
- v0 baseline benchmark: DONE
- Gate calibration: DONE
- Iter 1: FAILED (worker produced no artifacts)
- Iter 2: PENDING (recovery direction being written)
- Remaining iterations: 2 (iter2, iter3)

## Priority Improvement Areas (unchanged from v0 analysis)
1. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
2. Tier-VC@100 very poor (0.071) — top-of-ranking quality must improve
3. High variance across months — 2021-11 and 2022-06 are disaster months
4. Tier 4 has 0 samples — effectively a 4-class problem
