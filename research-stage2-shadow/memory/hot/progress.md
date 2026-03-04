## Status: ITER 2 COMPLETE — feat-eng-3-20260304-121042
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
| 3 | — | **PLANNED** | Target: EV-VC@100 recovery via mcw reduction or value_weighted |

### Iter 3 Plan
- **Objective**: Recover EV-VC@100 precision without surrendering EV-VC@500 gains
- **Primary lever**: min_child_weight reduction (25→15-20) for sharper leaf predictions
- **Alternative lever**: value_weighted=True to emphasize high-$ constraints in loss
- **Feature set**: frozen at 34; n_estimators=600, lr=0.03 settled
