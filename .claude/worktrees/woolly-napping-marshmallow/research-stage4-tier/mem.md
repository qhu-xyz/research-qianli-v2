# Stage 4: LTR Tier Ranking

## Two-Stage Evaluation Strategy

| Stage | Months | Time (LightGBM) | Purpose |
|-------|--------|------------------|---------|
| Screen | 12 representative (1/quarter, 2020-09 to 2023-05) | ~36s | Fast hypothesis testing |
| Full | 36 rolling (2020-06 to 2023-05) | ~108s | Comprehensive validation |

**Workflow:** Screen first on 12 months. If promising (passes Group A gates), run full 36 months. If not, move to next hypothesis.

## Backend Choice: LightGBM lambdarank

- LightGBM: ~3s/month (lambdarank, rank-transformed integer labels)
- XGBoost: ~71s/month (rank:pairwise, 22x slower)
- LightGBM chosen for speed. Both backends still supported via `cfg.backend`.

## Gate Metrics

Group A (blocking): VC@20, VC@100, Recall@20, Recall@50, Recall@100, NDCG
Group B (monitor): VC@10, VC@25, VC@50, VC@200, Recall@10, Spearman, Tier0-AP, Tier01-AP

Recall@50 promoted to Group A — ~50 constraints have SP > $3k (ground truth for high-value set).

## Recall@k is NOT monotone in k

Recall@k = |model_top_k intersect true_top_k| / k. Denominator grows with k, so Recall@20 > Recall@100 is normal when model excels at top-20 but struggles with mid-tier ranking.

## Current Versions

- v0: V6.2B baseline (hand-tuned formula, champion)
- v1: First LTR (XGBoost, V6.2B features only, 34 features zero-filled). Beats v0 on VC@20, VC@100, Recall@20, Recall@100. NDCG regresses due to zero-filled features.
- v2: Pending — LightGBM backend, same features as v1

## Task

- use spice 6.2b as baseline
- LTR ranking to beat V6.2B's hand-tuned formula
- 6-month train + 2-month val rolling window
- 40 features (34 stage3 + 6 V6.2B), monotone constraints
- semi-autonomous workflow: propose hypotheses, user approves, implement + run
