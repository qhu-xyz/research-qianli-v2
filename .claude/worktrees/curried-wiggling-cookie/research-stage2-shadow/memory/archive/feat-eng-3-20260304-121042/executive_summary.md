# Executive Summary — Batch feat-eng-3-20260304-121042

## Batch Overview
- **Objective**: Feature engineering and HP optimization for the 34-feature regressor
- **Iterations**: 3 (2 successful promotions, 1 worker failure)
- **Entry champion**: v0009 (39 nominal / 34 effective features, n_est=400, lr=0.05, mcw=25)
- **Exit champion**: v0012 (34 features, n_est=600, lr=0.03, mcw=25)
- **Duration**: 2026-03-04

## Iteration Summary

| Iter | Version | Hypothesis | Outcome | Key Change |
|------|---------|-----------|---------|------------|
| 1 | v0011 | Prune 5 dead features (39→34) | **PROMOTED** | EV-VC@100 +5.2%, EV-VC@500 -2.5% |
| 2 | v0012 | More trees + lower LR (600/0.03) | **PROMOTED** | EV-VC@500 +3.5%, EV-VC@100 -5.3% |
| 3 | v0013 | mcw=15 or value_weighted=True | **FAILED** | Worker phantom completion — no data |

## Net Results (v0009 → v0012)

| Metric | v0009 | v0012 | Delta | % |
|--------|-------|-------|-------|---|
| EV-VC@100 | 0.0762 | 0.0758 | -0.0004 | -0.5% |
| EV-VC@500 | 0.2329 | 0.2348 | +0.0019 | +0.8% |
| EV-NDCG | 0.7548 | 0.7518 | -0.0030 | -0.4% |
| Spearman | 0.3910 | 0.3940 | +0.0030 | +0.8% |
| C-RMSE | 2827.4 | 2855.3 | +27.9 | +1.0% |
| C-MAE | 1136.7 | 1135.0 | -1.7 | -0.1% |

**Assessment**: The batch was a **structural improvement** rather than a metric-level improvement. The net deltas are small (~1% on all metrics), but the batch:
1. Cleaned the feature set from 39 nominal (34 effective) to 34 actual — removing dead code and improving colsample efficiency
2. Optimized ensemble config (600 trees, lr=0.03) — eliminating the critical EV-VC@500 tail failure that was the binding constraint
3. Left all gate margins comfortable (no gate at limit) — pipeline is in its healthiest state

## Gate Health (v0012)

| Gate | Mean | Floor | Margin | Tail Fails | L3 Status |
|------|------|-------|--------|------------|-----------|
| EV-VC@100 | 0.0758 | 0.0664 | +14.2% | 0 | P (+0.0175) |
| EV-VC@500 | 0.2348 | 0.2179 | +7.8% | 0 | P (+0.0357) |
| EV-NDCG | 0.7518 | 0.7137 | +5.3% | 0 | P (+0.0231) |
| Spearman | 0.3940 | 0.3736 | +5.5% | 0 | P (+0.0218) |

## Key Learnings

1. **Pruning dead features sharpens top-100 but trades breadth** — colsample efficiency gains improve discrimination but reduce coverage
2. **More trees + lower LR recovers breadth** — 600/0.03 provides finer mid-tier discrimination, fixes tail failures
3. **These two changes partially cancel on EV-VC@100** — iter 1 gained +5.2%, iter 2 lost -5.3%, net wash
4. **flow_direction has no regression signal** — tested and rejected
5. **colsample=0.9 does not help breadth** — breadth is about ensemble rounds, not per-tree coverage
6. **Worker reliability remains a problem** — phantom completion on iter 3 wasted the final iteration slot

## Hypotheses Confirmed
- H3: Pruning dead features (CONFIRMED — v0011)
- H5: More trees + lower LR for breadth (CONFIRMED — v0012)

## Hypotheses Failed
- H4: flow_direction feature (FAILED — screen)
- H6: colsample=0.9 for breadth (FAILED — screen)

## Hypotheses Untested (carry forward)
- H7: min_child_weight 25→15 (planned for iter 3, worker failed)
- H8: value_weighted=True (planned for iter 3, worker failed)

## Accumulated Code Debt (unfixed)
1. **HIGH**: Gated regressor train-inference mismatch (pipeline.py:195-208)
2. **LOW**: Feature importance never populated
3. **LOW**: Temporal leakage in data_loader.py
4. **LOW**: R-REC@500 metric definition mismatch
5. **LOW**: Pipeline classifier override docstring mismatch

## Recommendations for Next Batch
1. **Priority 1**: Test mcw=15 (moderate risk, directly targets EV-VC@100 recovery)
2. **Priority 2**: Test value_weighted=True (higher uncertainty, verify pipeline.py implementation first)
3. **Gate recalibration**: Recalibrate gates to v0012 before next batch (tightens EV-VC@100 floor +8.4%, EV-VC@500 floor +2.4%)
4. **Scale-aware noise_tolerance**: 3 consecutive iterations of reviewer feedback — should be fixed before next batch
5. **Consider fixing gated-training semantics** (HIGH debt from Codex reviews, 3 iterations running)
