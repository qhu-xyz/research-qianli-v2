# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260303-060938 |
| Iteration | 3 of 3 (**COMPLETE** — final synthesis) |
| State | ORCHESTRATOR_SYNTHESIZING → next: HUMAN_SYNC |
| Champion | None (v0 baseline) |
| Best Version | v0004 (recommended for HUMAN_SYNC promotion consideration) |
| Last Hypothesis | H8: Feature pruning 17→13 + revert window — TRADEOFF DISCOVERY |

## This Batch Results

- **Iter 1**: H6 — 14-month window + 3 interaction features → v0004. AUC +0.0015 (9W/3L), VCAP@100 +0.0056 (10W/2L, **p=0.039**). Not promoted but best balanced version.
- **Iter 2**: H7 — 18-month window + feature importance → v0005. Diminishing returns confirmed. Feature importance data collected. Not promoted.
- **Iter 3**: H8 — Prune to 13 features + revert to 14mo → v0006. NDCG +0.0227 and VCAP@100 +0.0121 (**both p=0.039**), but AP -0.0044 (3W/9L). Tradeoff discovery — not promoted.

## v0006 Key Results (iter 3)

| Metric | v0 | v0004 (best balanced) | v0006 (iter 3) | vs v0 | vs v0004 |
|--------|-----|----------------------|----------------|-------|----------|
| S1-AUC | 0.8348 | 0.8363 | 0.8354 | +0.0006 | -0.0009 |
| S1-AP | 0.3936 | 0.3951 | 0.3892 | **-0.0044** | **-0.0059** |
| S1-VCAP@100 | 0.0149 | 0.0205 | **0.0270** | **+0.0121** | +0.0065 |
| S1-NDCG | 0.7333 | 0.7371 | **0.7560** | **+0.0227** | +0.0189 |

## Batch Recommendation for HUMAN_SYNC

**Promote v0004** as a modest improvement over v0:
- AUC 0.8363 (+0.0015, 9W/3L — best W/L)
- AP 0.3951 (+0.0015 — best AP)
- VCAP@100 0.0205 (+0.0056, 10W/2L, p=0.039 — statistically significant)
- NDCG 0.7371 (+0.0038)

**Document v0006** as a research finding:
- Novel monotone constraint structure effect on ranking quality
- Potential value if business acts only on top-100 predictions

**Feature set ceiling reached**: 6 experiments, 3 independent levers, AUC range [0.832, 0.836]. Next improvement requires new data sources.

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
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed |
| hp-tune-134412 iter1 | HP tuning (v0003-HP) | H3 refuted |
| hp-tune-144146 iter1 | Interaction features (v0002) | H4 not supported |
| feat-eng-194243 iter1 | Training window 10-14 (v0003) | H5 inconclusive |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | H6 partially confirmed |
| feat-eng-060938 iter2 | Window 18 + importance (v0005) | H7 failed |
| **feat-eng-060938 iter3** | **Feature pruning 17-13 (v0006)** | **H8 tradeoff discovery** |
