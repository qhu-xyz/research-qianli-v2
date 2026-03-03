# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-3-20260303-104101 |
| Iteration | 1 of 3 (synthesis complete, iter 2 planning next) |
| State | ORCHESTRATOR_SYNTHESIZING → iter 2 |
| Champion | **v0008** (26 features, AUC=0.8498, AP=0.4418, NDCG=0.7346) |
| Previous Champion | v0007 (19 features, AUC=0.8485, AP=0.4391) |
| Iter 1 Result | **PROMOTED** — H10 confirmed, NDCG bot2 +0.0101 |
| Next Hypothesis | H11: Derived interactions + colsample_bytree=0.9 |

## v0008 Champion Summary

| Metric | v0008 | v0007 | Delta | W/L |
|--------|-------|-------|-------|-----|
| S1-AUC | **0.8498** | 0.8485 | +0.0013 | 8W/4L |
| S1-AP | **0.4418** | 0.4391 | +0.0027 | 9W/3L |
| S1-VCAP@100 | 0.0240 | 0.0247 | -0.0007 | 4W/8L |
| S1-NDCG | **0.7346** | 0.7333 | +0.0013 | 8W/4L |
| S1-BRIER | **0.1383** | 0.1395 | -0.0012 | — |
| Precision | **0.5091** | 0.5020 | +0.0071 | — |

**Bot2 (v0008 as champion baseline for L3)**:
- AUC: 0.8199, AP: 0.3726, VCAP@100: 0.0061, NDCG: 0.6663
- VCAP@100 bot2 margin to L3 fail: **+0.0167** — tightest constraint
- NDCG bot2 margin: **+0.0301** — now comfortable (was 0.0046 in v0007)

## This Batch Strategy

**Human directive**: Aggressive feature engineering. Continue adding signal from the source loader and derived interactions.

### Iteration Results
- **Iter 1**: H10 — Added 7 features (density_mean, density_variance, density_entropy, tail_concentration, prob_band_95_100, prob_band_100_105, hist_da_max_season). **PROMOTED as v0008.** NDCG bot2 +0.0101, precision +0.007.
- **Iter 2**: H11 — Derived interactions targeting VCAP@100 + colsample_bytree=0.9. **NEXT.**
- **Iter 3**: TBD based on iter 2 results. Final optimization + HUMAN_SYNC preparation.

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
| **feat-eng-3-104101 iter1** | **Distrib + band + seasonal (v0008)** | **+0.0150** | **8W/4L** | **YES** |

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
| **feat-eng-3-104101 iter1** | **Distrib + band + seasonal (v0008)** | **H10 CONFIRMED — PROMOTED** |
