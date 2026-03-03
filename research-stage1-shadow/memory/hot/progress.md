# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260303-060938 |
| Iteration | 1 of 3 (planning complete, worker next) |
| State | ORCHESTRATOR_PLANNING → next: WORKER |
| Champion | None (v0 baseline) |
| Last Hypothesis | H5: Training window 10→14 months — inconclusive (weak positive, not promoted) |
| Current Hypothesis | H6: Combine 14-month window + 3 interaction features (tests additivity) |

## This Batch Strategy

From human input: test additivity of the two positive-signal levers, then refine or pivot.
- **Iter 1**: H6 — 14-month window + 3 interaction features (17 features total) + bug fixes (f2p crash, dual-default)
- **Iter 2-3**: Depends on iter 1 results:
  - If additive → refine (more interactions, or train_months=18)
  - If not additive → pivot to feature selection (drop shape features) or ratio features

## Cumulative Evidence (all real-data experiments)

| Batch | Lever | AUC Δ | AUC W/L | Promoted |
|-------|-------|-------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| feat-eng-194243 iter1 | Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | No |
| **feat-eng-060938 iter1** | **Window + Interactions (v0003)** | **Pending** | **—** | **—** |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003-HP) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| feat-eng-20260302-194243 iter1 | Training window 10→14 (v0003) | H5 inconclusive — AUC +0.0013, 7W/4L/1T (best so far) |
| **feat-eng-20260303-060938 iter1** | **Window + Interactions (v0003)** | **In progress** |
