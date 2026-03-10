# Progress

## Current State
- **Batch**: tier-fe-2-20260305-001606 (FE only, 3 iterations max)
- **Iteration**: 2 PLANNED, awaiting worker
- **Champion**: v0 (baseline, unchanged — v0005 not promoted)
- **Iterations completed**: 1 successful (v0005)
- **Version counter**: next_id=6

## Iter 1 Result — v0005 (NOT PROMOTED)

- **Hypothesis A won**: Add 3 interaction features (34→37)
- **Features**: overload_x_hist, prob110_x_recent_hist, tail_x_hist
- **VC@100**: 0.0746 (+5.4%) — **FAILS L1** by 0.0004 (floor 0.0750)
- **All other Group A gates**: PASS all 3 layers
- **All metrics improved**: No regressions anywhere
- **Bottom_2_mean VC@100**: +64% (0.0103→0.0169) — interactions help weak months
- **Tier-Recall@1**: Still catastrophic (0.045) — structural, FE cannot fix

## Iter 2 Plan

- **Hypothesis A** (primary): Add 4 FE features (37→41) — log1p_hist_da, log1p_expected_overload, overload_x_recent_hist, prob_range_high
- **Hypothesis B** (alternative): Add 7 FE features (37→44) — A's 4 plus prob110_x_hist, prob105_x_hist, prob100_x_hist
- **Screen months**: 2021-11 (weak, worst VC@100=0.0082) + 2021-09 (strong, best VC@100=0.2489)
- **Goal**: Close the 0.0004 VC@100 gap
- **Risk**: Low (A) to medium (B) — all additive features

## Priority Improvement Areas
1. **Tier-VC@100** below floor by 0.0004 — ONLY blocking gate (closing this promotes)
2. Value-QWK barely passing (0.3918 vs 0.3914) — fragile, monitor
3. Tier-Recall@1 catastrophic (0.045) — BLOCKED by FE-only constraint
4. Macro-F1 structural failure — driven by Tier-Recall@1 collapse
5. 2021-11 worst month across most metrics — log transforms may help here
