# Progress

## Current State
- **Champion**: v0 (baseline, unchanged)
- **Iterations completed**: 2 (BOTH FAILED — no artifacts from either)
- **Current batch**: tier-fe-1-20260304-182037 (FE only)
- **Current iteration**: 3 (FINAL) — planning recovery direction
- **Version counter**: next_id=3 (leaked twice, no actual versions created beyond v0)

## Iteration History

### Iter 1 — FAILED
- **Hypothesis**: Feature swap (interaction features + light pruning) vs aggressive pruning
- **Outcome**: Worker claimed "done" but produced NO artifacts. registry/v0001/ doesn't exist.
- **Data collected**: None.

### Iter 2 — FAILED
- **Hypothesis A**: Add 3 interaction features → 37 features
- **Hypothesis B**: Prune 6 + add 3 interactions → 31 features
- **Outcome**: IDENTICAL failure mode to iter1. Worker claimed "done", no artifacts. registry/v0002/ doesn't exist.
- **Data collected**: None.

### Iter 3 — PLANNED (FINAL)
- **Hypothesis**: Add 3 interaction features (hist_physical_interaction, overload_exceedance_product, hist_seasonal_band) → 37 features. SINGLE hypothesis, NO screening, direct full benchmark.
- **Key change**: Eliminate all complexity. Worker must run benchmark.py FIRST, then write handoff LAST.

## Pipeline Status
- v0 baseline benchmark: DONE
- Gate calibration: DONE
- Iter 1: FAILED
- Iter 2: FAILED
- Iter 3: PLANNING — last chance

## Priority Improvement Areas (unchanged — no data to update)
1. Tier-VC@100 below floor (0.071 vs 0.075) — only Group A gate failing Layer 1
2. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
3. High variance across months — 2021-11 and 2022-06 are disaster months
4. Tier 4 has 0 samples — effectively a 4-class problem
