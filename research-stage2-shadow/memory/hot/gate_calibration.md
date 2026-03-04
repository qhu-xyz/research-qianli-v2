## Gate Calibration

### Current State: Gates v4, Recalibrated to v0007 Champion (0.95x mean, 0.90x worst)

Recalibrated 2026-03-04. Significant improvement over v0-exact floors.

| Gate | Floor | v0009 Mean | Margin | Assessment |
|------|-------|-----------|--------|------------|
| EV-VC@100 | 0.0664 | 0.0762 | +14.8% | Reasonable |
| EV-VC@500 | 0.2179 | 0.2329 | +6.9% | Tight but OK |
| EV-NDCG | 0.7137 | 0.7548 | +5.8% | Tight |
| Spearman | 0.3736 | 0.3910 | +4.7% | **Binding constraint** — could block borderline EV improvements |

### Tail Floor Assessment (v0009)

| Gate | Tail Floor | Worst Month | Margin |
|------|-----------|-------------|--------|
| EV-VC@100 | 0.000135 | 0.0012 (2021-05) | 8.9x headroom — tail_floor essentially zero |
| EV-VC@500 | 0.0536 | 0.0617 (2022-09) | +15% headroom |
| EV-NDCG | 0.5434 | 0.6069 (2022-06) | +11.7% headroom |
| Spearman | 0.2363 | 0.2674 (2021-11) | +13.2% headroom |

### L3 Noise Tolerance Issue (ONGOING)

`noise_tolerance=0.02` is not scale-aware:
- EV-VC@100 (bot2 ~0.007): threshold goes negative → L3 provides no protection
- C-RMSE (bot2 ~5300): 0.02 is negligible → effectively requires exact match
- Spearman (bot2 ~0.27): 0.02 is reasonable (~7% of value)

**Recommendation**: Use `max(abs_floor, pct * |champ_bottom_2_mean|)` with per-metric caps. Both reviewers flagged this.

### If v0009 Promoted, Should Gates Recalibrate?

If v0009 becomes champion, floors should be updated to 0.95x of v0009 means:
| Gate | New Floor | Current Floor | Change |
|------|-----------|---------------|--------|
| EV-VC@100 | 0.0724 | 0.0664 | +9.0% |
| EV-VC@500 | 0.2213 | 0.2179 | +1.6% |
| EV-NDCG | 0.7171 | 0.7137 | +0.5% |
| Spearman | 0.3715 | 0.3736 | -0.6% (would loosen) |

Note: Spearman floor would actually loosen slightly since v0009 mean is lower than v0007. This is acceptable — Spearman is already the binding constraint and the regression is within noise.
