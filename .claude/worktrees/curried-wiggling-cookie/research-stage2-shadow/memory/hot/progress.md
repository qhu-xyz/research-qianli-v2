## Status: BATCH CLOSED — feat-eng-3-20260304-121042
**Champion**: v0012 (34 features, n_estimators=600, lr=0.03, mcw=25)
**Batch outcome**: 2 promotions out of 3 iterations (iter 3 lost to worker failure)

### Current Champion Metrics (v0012)
| Metric | Mean | Floor | Margin |
|--------|------|-------|--------|
| EV-VC@100 | 0.0758 | 0.0664 | +14.2% |
| EV-VC@500 | 0.2348 | 0.2179 | +7.8% |
| EV-NDCG | 0.7518 | 0.7137 | +5.3% |
| Spearman | 0.3940 | 0.3736 | +5.5% |

### Pipeline Health
- **No gate at limit** — all margins comfortable
- **All Group B gates pass**
- **0 tail failures** on any Group A gate

### Batch Progress
| Iter | Version | Outcome | Key Delta |
|------|---------|---------|-----------|
| 1 | v0011 | **PROMOTED** | EV-VC@100 +5.2%, EV-VC@500 -2.5% (prune dead features) |
| 2 | v0012 | **PROMOTED** | EV-VC@500 +3.5%, EV-VC@100 -5.3% (more trees, lower LR) |
| 3 | v0013 | **FAILED** | Worker phantom completion — no artifacts produced |

### Net Batch Progress (v0009 → v0012)
| Metric | v0009 | v0012 | Delta | % |
|--------|-------|-------|-------|---|
| EV-VC@100 | 0.0762 | 0.0758 | -0.0004 | -0.5% (wash) |
| EV-VC@500 | 0.2329 | 0.2348 | +0.0019 | +0.8% |
| EV-NDCG | 0.7548 | 0.7518 | -0.0030 | -0.4% |
| Spearman | 0.3910 | 0.3940 | +0.0030 | +0.8% |

### Next Batch Priorities
1. **mcw=15** (untested from iter 3 — moderate risk, EV-VC@100 recovery)
2. **value_weighted=True** (untested from iter 3 — higher uncertainty, verify pipeline.py first)
3. **Gate recalibration** to v0012 recommended before next batch starts
