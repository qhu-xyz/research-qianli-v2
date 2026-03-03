# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-3-20260303-104101 |
| Iteration | 1 of 3 (planning complete, worker next) |
| State | ORCHESTRATOR_PLANNING → WORKER |
| Champion | **v0007** (19 features, AUC=0.8485, AP=0.4391) |
| Current Hypothesis | H10: Distribution shape + near-boundary band + seasonal historical features (19→26 features) |

## v0007 Champion Summary

| Metric | v0007 | v0 | Delta |
|--------|-------|-----|-------|
| S1-AUC | **0.8485** | 0.8348 | +0.0137 (12W/0L) |
| S1-AP | **0.4391** | 0.3936 | +0.0455 (11W/1L) |
| S1-VCAP@100 | 0.0247 | 0.0149 | +0.0098 (9W/3L) |
| S1-NDCG | 0.7333 | 0.7333 | +0.0000 (5W/7L) |
| S1-BRIER | **0.1395** | 0.1503 | -0.0108 |

**Bot2 (v0007 as champion baseline for L3)**:
- AUC: 0.8188, AP: 0.3685, VCAP@100: 0.0094, NDCG: 0.6562
- NDCG bot2 margin to L3 fail: **0.0046** — tightest constraint

## This Batch Strategy

**Human directive**: Aggressive feature engineering. The ~0.836 AUC ceiling was broken by v0007 (+0.014). Continue adding new signal categories from the source loader. 7 features available from distribution shape, near-boundary bands, and seasonal historical signal.

### Planned Iterations
- **Iter 1**: H10 — Add 7 features (density_mean, density_variance, density_entropy, tail_concentration, prob_band_95_100, prob_band_100_105, hist_da_max_season). Target: maintain AUC/AP, improve NDCG. **PLANNING COMPLETE.**
- **Iter 2**: Depends on iter 1 results. If NDCG improves → add derived interactions. If flat → monotone tuning or feature selection.
- **Iter 3**: Final optimization + HUMAN_SYNC preparation.

## Cumulative Evidence (all real-data experiments)

| Batch | Lever | AUC vs v0 | AUC W/L | Promoted |
|-------|-------|-----------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| feat-eng-194243 iter1 | Window 10-14 (v0003) | +0.0013 | 7W/4L/1T | No |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | +0.0015 | 9W/3L | No |
| feat-eng-060938 iter2 | Window 18 + importance (v0005) | +0.0013 | 7W/5L | No |
| feat-eng-060938 iter3 | Feature pruning 17-13 (v0006) | +0.0006 | 5W/7L | No |
| **feat-eng-2-092848 iter1** | **SF + metadata (v0007)** | **+0.0137** | **12W/0L** | **YES** |

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
| **feat-eng-2-092848 iter1** | **SF + metadata features (v0007)** | **H9 STRONGLY CONFIRMED — PROMOTED** |
| feat-eng-2-092848 iter2 | Orchestrator timeout | Did not execute |
