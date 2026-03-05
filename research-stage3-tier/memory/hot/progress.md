# Progress

## Current State
- **Batch**: tier-fe-2-20260305-001606 (FE only, 3 iterations)
- **Iteration**: 1 — orchestrator plan DONE, awaiting worker
- **Champion**: v0 (baseline, unchanged)
- **Iterations completed**: 0 successful (all prior failed due to infrastructure bug, now fixed)
- **Version counter**: next_id=5 (leaked 4 times from failed iterations)

## Iter 1 Plan

- **Hypothesis A**: Add 3 interaction features (overload_x_hist, prob110_x_recent_hist, tail_x_hist) → 37 features
- **Hypothesis B**: Add same 3 interactions + prune 5 lowest-importance features → 32 features
- **Screen months**: 2022-06 (weak), 2021-09 (strong)
- **Direction file**: `memory/direction_iter1.md`

## Root Cause of All Prior Failures

ALL worker failures (tier-fe-1 iter1-2, tier-fe-2 iter1) had the same root cause:
**Uncommitted changes to HUMAN-WRITE-ONLY files** caused the pre-merge guard to reject worker output.

**Fix**: All changes to evaluate.py, gates.json, and registry/v0/ are now committed to main (commit a2a38c5).

## Metric Redesign (v2 gates) — COMMITTED

**Group A (blocking)** — all tier-count invariant:
- Tier-VC@100, Tier-VC@500, Tier0-AP (new), Tier01-AP (new)

**Group B (monitor)** — no hard gates:
- Tier-NDCG, QWK, Macro-F1, Value-QWK, Tier-Recall@0, Tier-Recall@1

**Removed**: Tier-Accuracy, Adjacent-Accuracy

## Priority Improvement Areas
1. Tier-VC@100 below floor (0.071 vs 0.075) — only Group A gate failing Layer 1
2. Tier0-AP mean 0.306 — high variance (0.114 to 0.594), worst months late 2022
3. Tier01-AP mean 0.311 — barely passing, worst months 2022-06 (0.195), 2022-12 (0.194)
4. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
5. High variance across months — 2022-06 is the worst month across most metrics
