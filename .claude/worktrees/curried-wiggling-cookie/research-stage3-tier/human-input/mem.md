## purpose of this repo
- Stage 3: **tier classification** — replaces the two-stage (classifier + regressor) pipeline with a single multi-class XGBoost model predicting 5 shadow price tiers directly
- Ported from research-stage2-shadow, inheriting the 3-iter-per-batch autonomous pipeline structure (orchestrator plan → worker → dual review → orchestrator synthesize)
- Key innovation: `tier_ev_score = sum(P(tier=t) * midpoint[t])` as continuous ranking signal for capital allocation

## tier definitions
| Tier | Shadow Price Range | Midpoint |
|------|-------------------|----------|
| 0 | [3000, +inf) | $4000 |
| 1 | [1000, 3000) | $2000 |
| 2 | [100, 1000) | $550 |
| 3 | [0, 100) | $50 |
| 4 | (-inf, 0) | $0 |

Bins: `[-inf, 0, 100, 1000, 3000, inf]`, labels: `[4, 3, 2, 1, 0]`

## model architecture
- Single XGBoost `multi:softprob` with `num_class=5`
- Class weights handle imbalance: `{0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}`
- All TierConfig parameters are mutable (no frozen classifier)
- 34 features from stage 2's regressor feature set

## evaluation setup
- **train**: 6 months, **val**: 2 months, **test**: target month
- report on **target month only**
- 12 eval months: 2020-09, 2020-11, 2021-01, 2021-03, 2021-05, 2021-07, 2021-09, 2021-11, 2022-03, 2022-06, 2022-09, 2022-12
- class_type=onpeak, ptype=f0
- All metrics are higher-is-better (no direction inversions)

## metrics
**Blocking (Group A):** Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK
**Monitor (Group B):** Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1

## iteration efficiency protocol (inherited from stage 2)
- 2-hypothesis screening per iteration
- `benchmark.py --overrides '{"tier": {...}}' --eval-months M1 M2` for screening
- Winner gets full 12-month benchmark

## v0 baseline
- v0 uses default TierConfig: 34 features, default class weights, default XGB hyperparams
- Gate floors will be calibrated from v0 results using `populate_v0_gates.py`
