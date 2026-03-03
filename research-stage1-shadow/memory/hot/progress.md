# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-2-20260303-092848 |
| Iteration | 1 of 3 |
| State | ORCHESTRATOR_PLANNING → WORKER next |
| Champion | None (v0 baseline) |
| Best Version | v0004 (from prior batch, recommended for promotion at HUMAN_SYNC) |
| Current Hypothesis | H9: Add shift factor + constraint metadata features (6 new features, 19 total) |

## This Batch Strategy

**Human directive**: Aggressive feature engineering. The ~0.836 AUC ceiling is a feature ceiling. Source loader has 15+ unused features. Priority is adding many new features, not tweaking parameters. train_months=14 is HARD MAX.

### Planned Iterations
- **Iter 1**: H9 — Add 6 shift factor + constraint metadata features (sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac, is_interface, constraint_limit). Completely new signal class (network topology + structural).
- **Iter 2**: Depends on iter 1 results. If promising → add distribution + band features. If marginal → kitchen sink. If failed → pivot approach.
- **Iter 3**: Depends on iter 1+2 results.

## Prior Batch Results (feat-eng-20260303-060938)

- **Iter 1**: H6 — 14-month window + 3 interaction features → v0004. AUC +0.0015 (9W/3L), VCAP@100 +0.0056 (10W/2L, **p=0.039**). Best balanced version.
- **Iter 2**: H7 — 18-month window + feature importance → v0005. Diminishing returns confirmed.
- **Iter 3**: H8 — Prune to 13 features + revert to 14mo → v0006. NDCG +0.0227 and VCAP@100 +0.0121 (both p=0.039), but AP -0.0044 (3W/9L). Tradeoff discovery.

## Cumulative Evidence (all real-data experiments)

| Batch | Lever | AUC vs v0 | AUC W/L | Promoted |
|-------|-------|-----------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| feat-eng-194243 iter1 | Window 10-14 (v0003) | +0.0013 | 7W/4L/1T | No |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | +0.0015 | 9W/3L | No |
| feat-eng-060938 iter2 | Window 18 + importance (v0005) | +0.0013 | 7W/5L | No |
| feat-eng-060938 iter3 | Feature pruning 17-13 (v0006) | +0.0006 | 5W/7L | No |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed |
| hp-tune-134412 iter1 | HP tuning (v0003-HP) | H3 refuted |
| hp-tune-144146 iter1 | Interaction features (v0002) | H4 not supported |
| feat-eng-194243 iter1 | Training window 10-14 (v0003) | H5 inconclusive |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | H6 partially confirmed |
| feat-eng-060938 iter2 | Window 18 + importance (v0005) | H7 failed |
| feat-eng-060938 iter3 | Feature pruning 17-13 (v0006) | H8 tradeoff discovery |
| **feat-eng-2-092848 iter1** | **SF + metadata features (v0006)** | **H9 — in progress** |
