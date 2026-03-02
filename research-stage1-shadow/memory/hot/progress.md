# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260302-154125 |
| Iteration | 1 of 3 (planning complete, worker pending) |
| State | ORCHESTRATOR_PLANNING → next: WORKER |
| Champion | None (v0 baseline) |
| Current Hypothesis | H5: Training window expansion (10→14 months) + revert to v0 base features (14) |

## Iteration 1 Plan Summary

- **Objective**: Break AUC ceiling by addressing distribution shift via longer training window
- **Primary change**: `train_months` 10→14 (40% more training data per eval month)
- **Secondary change**: Revert to v0's 14 base features (remove 3 interaction features from v0002)
- **Code fix**: benchmark.py `train_months` plumbing bug (currently hardcoded to 10)
- **Code fix**: Schema guard for base feature columns (Codex MEDIUM from iter1)
- **Expected**: AUC +0.002–0.008, especially late-2022 months (2022-09, 2022-12)

## Cumulative Evidence

| Batch | Lever | Result | Conclusion |
|-------|-------|--------|------------|
| hp-tune-134412 | HP tuning (v0003) | AUC -0.0025, 0W/11L | Model not complexity-limited |
| hp-tune-144146 | Interaction features (v0002) | AUC +0.000, 5W/6L/1T | Information ceiling reached |
| **feat-eng** | **Training window (10→14)** | **Pending** | **Addresses distribution shift directly** |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| feat-eng-20260302-154125 iter1 | Training window expansion (10→14) | Worker pending |
