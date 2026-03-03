# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-2-20260303-092848 |
| Iteration | 1 of 3 (synthesis complete, iter 2 next) |
| State | ORCHESTRATOR_SYNTHESIZING → iter 2 planning next |
| Champion | **v0007** (PROMOTED — first champion) |
| Current Hypothesis | H9 strongly confirmed → H10 (NDCG recovery + feature expansion) |

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

**Human directive**: Aggressive feature engineering. The ~0.836 AUC ceiling is a feature ceiling. Source loader has 15+ unused features. Priority is adding many new features, not tweaking parameters. train_months=14 is HARD MAX.

### Planned Iterations
- **Iter 1**: H9 — Add 6 shift factor + constraint metadata features. **DONE — PROMOTED v0007.**
- **Iter 2**: H10 — NDCG recovery via distribution shape features + monotone constraint tuning on v0007 base. Target: maintain AUC/AP gains, recover NDCG.
- **Iter 3**: Depends on iter 2 results.

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
