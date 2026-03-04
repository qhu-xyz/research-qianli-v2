## Status: ITER 3 PLANNED — feat-eng-3-20260304-121042
**Champion**: v0012 (34 features, n_estimators=600, lr=0.03)
**Batch started as**: Feature engineering / selection → relaxed to HP changes at iter 2

### Current Champion Metrics (v0012)
| Metric | Mean | Floor | Margin |
|--------|------|-------|--------|
| EV-VC@100 | 0.0758 | 0.0664 | +14.2% |
| EV-VC@500 | 0.2348 | 0.2179 | +7.8% |
| EV-NDCG | 0.7518 | 0.7137 | +5.3% |
| Spearman | 0.3940 | 0.3736 | +5.5% |

### Pipeline Health
- **No gate at limit** — all margins comfortable
- **All Group B gates pass** (first time: C-RMSE and C-MAE pass since v0011)
- **0 tail failures** on any Group A gate

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | v0011 | **PROMOTED** | EV-VC@100 +5.2%, EV-VC@500 -2.5% (prune dead features) |
| 2 | v0012 | **PROMOTED** | EV-VC@500 +3.5%, EV-VC@100 -5.3% (more trees, lower LR) |
| 3 | — | **PLANNED** | Target: EV-VC@100 recovery via mcw=15 or value_weighted=True |

### Iter 3 Plan
- **Objective**: Recover EV-VC@100 precision without surrendering EV-VC@500 gains
- **Hypothesis A (primary)**: min_child_weight 25→15 — sharper leaf predictions for top-100
- **Hypothesis B (alternative)**: value_weighted=True — emphasize high-$ constraints in loss
- **Screen months**: 2021-03 (weak EV-VC@100) + 2022-12 (strong, regressed -21.8% in iter 2)
- **Feature set**: frozen at 34; n_estimators=600, lr=0.03 settled
- **Final iteration**: If both hypotheses vetoed, accept v0012 as batch champion
