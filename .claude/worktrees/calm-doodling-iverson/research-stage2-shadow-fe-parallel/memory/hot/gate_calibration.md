## Gate Calibration

### Current State: Gates Set at v0 Exact Mean (DYSFUNCTIONAL)

Current `registry/gates.json` has Group A floors at v0's exact mean values. v0007 promoted by +0.0004 on Spearman — indistinguishable from noise.

| Gate | Floor | v0 Mean | v0007 Mean | Margin |
|------|-------|---------|------------|--------|
| EV-VC@100 | 0.069032 | 0.069032 | 0.0699 | +0.0009 |
| EV-VC@500 | 0.21595 | 0.21595 | 0.2294 | +0.0134 |
| EV-NDCG | 0.747213 | 0.74721 | 0.7513 | +0.0041 |
| Spearman | 0.392798 | 0.39280 | 0.3932 | +0.0004 |

**Impact**: Blocked 2 of 3 iterations in this batch. v0005 (EV-VC@100 +6.5%) was blocked by 0.0008 Spearman miss.

### Recommended Fix (HUMAN_SYNC — 3rd request)

Now that v0007 is champion, floors should be recalibrated to v0007's metrics with appropriate headroom:

| Gate | Recommended Floor | Basis | Headroom |
|------|------------------|-------|----------|
| EV-VC@100 | 0.0600 | ~0.86x v0007 mean | 14% |
| EV-VC@500 | 0.2000 | ~0.87x v0007 mean | 13% |
| EV-NDCG | 0.7000 | ~0.93x v0007 mean | 7% |
| Spearman | 0.3500 | ~0.89x v0007 mean | 11% |

### Noise Tolerance (L3) Issue

`noise_tolerance=0.02` is not scale-aware:
- EV-VC@100 (bot2 ~0.007): 0.02 makes L3 threshold negative → no protection
- C-RMSE (bot2 ~5300): 0.02 is meaningless → effectively requires exact match
- Spearman (bot2 ~0.27): 0.02 is reasonable

**Recommendation**: Use percentage-based tolerance (e.g., 5% of champion's bottom_2_mean).
