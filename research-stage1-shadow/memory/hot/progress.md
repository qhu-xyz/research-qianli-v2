# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260302-194243 |
| Iteration | 1 of 3 (synthesis complete, iter2 pending) |
| State | ORCHESTRATOR_SYNTHESIZING → next: WORKER (iter2) |
| Champion | None (v0 baseline) |
| Last Hypothesis | H5: Training window 10→14 months — inconclusive (weak positive, not promoted) |
| Next Hypothesis | H6: Combine 14-month window + interaction features (tests additivity) |

## Iteration 1 Result Summary

- **v0003** (train_months 10→14, 14 base features): AUC +0.0013 (7W/4L/1T), not promoted
- Best result of 3 real-data iterations but below statistical significance
- 2022-12 improved substantially; 2022-09 unchanged
- Both reviewers agree: no promotion, retain 14-month window as new default

## Cumulative Evidence

| Batch | Lever | AUC Δ | AUC W/L | Promoted |
|-------|-------|-------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| **feat-eng-194243 iter1** | **Window 10→14 (v0003)** | **+0.0013** | **7W/4L/1T** | **No** |
| feat-eng-194243 iter2 | Combined (window + interactions) | Pending | — | — |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003-HP) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| feat-eng-20260302-194243 iter1 | Training window 10→14 (v0003) | H5 inconclusive — AUC +0.0013, 7W/4L/1T (best so far) |
