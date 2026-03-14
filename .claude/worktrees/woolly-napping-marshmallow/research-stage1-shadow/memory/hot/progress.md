# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-3-20260303-104101 |
| Iteration | 3 of 3 — **COMPLETE** |
| State | BATCH COMPLETE — ready for HUMAN_SYNC |
| Champion | **v0009** (29 features, AUC=0.8495, AP=0.4445, NDCG=0.7359, VCAP@100=0.0266, BRIER=0.1376) |
| Iter 1 Result | **PROMOTED v0008** — H10 confirmed, NDCG bot2 +0.0101 |
| Iter 2 Result | **PROMOTED v0009** — H11 confirmed, VCAP@100 bot2 +0.0028, AP pipeline high |
| Iter 3 Result | **NOT PROMOTED v0010** — H12 null, all metrics within noise |

## v0009 Champion Summary

| Metric | v0009 | v0 Baseline | Improvement |
|--------|-------|-------------|-------------|
| S1-AUC | 0.8495 | 0.8348 | +0.0147 |
| S1-AP | **0.4445** | 0.3936 | **+0.0509** |
| S1-VCAP@100 | **0.0266** | 0.0149 | +0.0117 |
| S1-NDCG | 0.7359 | 0.7333 | +0.0026 |
| S1-BRIER | **0.1376** | 0.1503 | **-0.0127** (better) |
| Precision | 0.503 | 0.442 | +0.061 |
| Features | 29 | 14 | +15 |

**Bot2 (v0009 — L3 baselines)**:
- AUC: 0.8189, AP: 0.3712, VCAP@100: 0.0089, NDCG: 0.6648

## This Batch Summary (feat-eng-3-20260303-104101)

### Iteration Results
- **Iter 1**: H10 — 7 features (distribution shape, near-boundary bands, seasonal historical). **PROMOTED v0008.** NDCG bot2 +0.0101, relieving tightest L3 constraint.
- **Iter 2**: H11 — 3 derived interactions + colsample_bytree=0.9. **PROMOTED v0009.** VCAP@100 bot2 +0.0028, AP at pipeline high 0.4445.
- **Iter 3**: H12 — n_estimators 200→300, learning_rate 0.1→0.07. **NOT PROMOTED (null).** Confirmed model at capacity ceiling.

### Key Outcomes
1. Champion improved from v0007 (AUC=0.8485, AP=0.4391) to v0009 (AUC=0.8495, AP=0.4445)
2. Feature count grew from 19 to 29 with 17.13% interaction importance
3. Optimization frontier confirmed: AUC~0.850, AP~0.445, NDCG~0.736
4. Feature engineering and hyperparameter tuning search spaces exhausted
5. Pipeline ready for HUMAN_SYNC

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
| **feat-eng-3-104101 iter2** | **Interactions + colsample (v0009)** | **+0.0147** | **4W/8L** | **YES** |
| feat-eng-3-104101 iter3 | More trees + slower LR (v0010) | +0.0148 | 6W/5L/1T | No (null) |

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
| **feat-eng-3-104101 iter2** | **Interactions + colsample (v0009)** | **H11 CONFIRMED — PROMOTED** |
| feat-eng-3-104101 iter3 | More trees + slower LR (v0010) | **H12 NULL — batch complete** |
