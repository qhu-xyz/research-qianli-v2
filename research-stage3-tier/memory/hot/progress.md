# Progress

## Current State
- **Champion**: v0 (baseline, unchanged)
- **Current batch**: tier-fe-2-20260304-225923 (FE only, fresh batch)
- **Current iteration**: 1 — orchestrator plan complete, awaiting worker
- **Version counter**: next_id=3 (leaked from previous batch failures)

## Metric Redesign (v2 gates)

Completed redesign of metric structure for tier-count invariance:

**Group A (blocking)** — all tier-count invariant:
- Tier-VC@100 (ranking), Tier-VC@500 (ranking)
- Tier0-AP (threshold-free), Tier01-AP (threshold-free)

**Group B (monitor)** — no hard gates:
- Tier-NDCG, QWK, Macro-F1, Value-QWK, Tier-Recall@0, Tier-Recall@1

## Batch History

### Previous batch: tier-fe-1 (3 iterations, ALL FAILED)
- Worker execution failures — no artifacts produced in any iteration
- Hypotheses (interaction features + pruning) remain untested
- Version counter leaked to next_id=3

### Current batch: tier-fe-2, Iter 1 — PLANNED
- **Hypothesis A**: Add 3 interaction features (overload_x_hist, prob110_x_recent_hist, tail_x_hist) → 37 features
- **Hypothesis B**: Add same 3 + prune 4 lowest-importance (density_skewness, prob_exceed_90, density_cv, density_variance) → 33 features
- **Screen months**: 2022-06 (weak), 2021-09 (strong)
- **Code change required**: Update `compute_interaction_features()` in features.py before screening

## Priority Improvement Areas
1. Tier-VC@100 below floor (0.071 vs 0.075) — **only Group A gate failing Layer 1**
2. Tier0-AP mean 0.306 — high variance (0.114 to 0.594), worst months late 2022
3. Tier01-AP mean 0.311 — barely passing, worst months 2022-06 (0.195), 2022-12 (0.194)
4. Tier-Recall@1 catastrophically low (0.047) — missing most strongly binding constraints
5. High variance across months — 2022-06 is the worst month across most metrics
