# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260303-060938 |
| Iteration | 1 of 3 (synthesis complete, iter 2 next) |
| State | ORCHESTRATOR_SYNTHESIZING → next: WORKER (iter 2) |
| Champion | None (v0 baseline) |
| Last Hypothesis | H6: Combined window + interactions — PARTIALLY CONFIRMED (encouraging, not promoted) |
| Next Hypothesis | H7: train_months=18 + feature importance export |

## This Batch Strategy

From human input: test additivity of the two positive-signal levers, then refine or pivot.
- **Iter 1**: H6 — 14-month window + 3 interaction features → **DONE** (AUC +0.0015, 9W/3L, VCAP@100 +0.0056, 10W/2L. Not promoted.)
- **Iter 2**: H7 — train_months=18 with 17 features + feature importance export (tests continued window expansion + generates data for iter 3 decisions)
- **Iter 3**: Depends on iter 2 — if 18-month window helps → consider promotion; if diminishing returns → feature importance-guided pruning or new features

## v0004 Key Results (iter 1)

| Metric | v0 | v0004 | Delta | W/L | Stat Sig |
|--------|-----|-------|-------|-----|----------|
| S1-AUC | 0.8348 | 0.8363 | +0.0015 | 9W/3L | p=0.073 |
| S1-AP | 0.3936 | 0.3951 | +0.0015 | 6W/6L | p=1.0 |
| S1-VCAP@100 | 0.0149 | 0.0205 | +0.0056 | 10W/2L | **p=0.039** |
| S1-NDCG | 0.7333 | 0.7371 | +0.0038 | 7W/5L | p=0.39 |

All gates pass. Not promoted (AUC < 0.837, AP flat). VCAP@100 is first statistically significant improvement in pipeline history.

## Cumulative Evidence (all real-data experiments)

| Batch | Lever | AUC Δ | AUC W/L | Promoted |
|-------|-------|-------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| feat-eng-194243 iter1 | Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | No |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | +0.0015 | 9W/3L | No |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003-HP) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| feat-eng-20260302-194243 iter1 | Training window 10→14 (v0003) | H5 inconclusive — AUC +0.0013, 7W/4L/1T |
| feat-eng-20260303-060938 iter1 | Window + Interactions (v0004) | **H6 partially confirmed — AUC +0.0015, 9W/3L, VCAP@100 +0.0056, 10W/2L** |
