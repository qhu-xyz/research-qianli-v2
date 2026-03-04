## Gate Calibration

### Current State: Gates v4, Calibrated to v0007 (0.95x mean, 0.90x worst)

Now evaluating against v0011 champion.

| Gate | Floor | v0011 Mean | Margin | Assessment |
|------|-------|-----------|--------|------------|
| EV-VC@100 | 0.0664 | 0.0801 | +20.6% | Loose — ample headroom |
| EV-VC@500 | 0.2179 | 0.2270 | +4.2% | **Tight** — one bad iteration could fail L1 |
| EV-NDCG | 0.7137 | 0.7499 | +5.1% | Moderate |
| Spearman | 0.3736 | 0.3925 | +5.1% | Was binding, now looser than EV-VC@500 |

### Tail Floor Assessment (v0011)

| Gate | Tail Floor | Worst Month | Fails | Max | Status |
|------|-----------|-------------|-------|-----|--------|
| EV-VC@100 | 0.000135 | 0.0034 (2021-05) | 0 | 1 | OK but tail_floor is non-protective |
| EV-VC@500 | 0.0536 | 0.0527 (2022-09) | **1** | 1 | **AT LIMIT** — next regression flips L2 |
| EV-NDCG | 0.5434 | 0.6060 (2022-06) | 0 | 1 | OK |
| Spearman | 0.2363 | 0.2616 (2021-11) | 0 | 1 | OK |

### L3 Margins (v0011 vs v0009 champion)

| Gate | v0011 bot2 | Threshold (champ bot2 - 0.02) | Margin |
|------|-----------|-------------------------------|--------|
| EV-VC@100 | 0.0111 | -0.0135 | +0.0246 (trivially passing, tolerance > metric scale) |
| EV-VC@500 | 0.0541 | 0.0518 | **+0.0023** (razor-thin) |
| EV-NDCG | 0.6403 | 0.6246 | +0.0157 (OK) |
| Spearman | 0.2678 | 0.2505 | +0.0173 (OK) |

### Binding Gate Constraint
**EV-VC@500 has replaced Spearman as the binding constraint.** With L1 margin at +4.2%, L2 at exact limit, and L3 margin at +0.0023, any further EV-VC@500 degradation will block promotion.

### L3 Noise Tolerance Issue (ONGOING, UNFIXED)

`noise_tolerance=0.02` is not scale-aware:
- EV-VC@100 (bot2 ~0.011): L3 threshold goes to -0.009 → provides zero protection
- EV-VC@500 (bot2 ~0.054): 0.02 is ~37% of value → very generous
- C-RMSE (bot2 ~5300): 0.02 is negligible → requires near-exact match
- Spearman (bot2 ~0.27): 0.02 is ~7% → reasonable

**Recommendation (repeated)**: `max(abs_floor, pct * |champ_bottom_2_mean|)` with per-metric tuning. Both reviewers flagged this in both batches.

### Should Gates Recalibrate to v0011?

If gates recalibrate to v0011 champion (0.95x mean):
| Gate | New Floor | Current Floor | Change |
|------|-----------|---------------|--------|
| EV-VC@100 | 0.0761 | 0.0664 | +14.6% (tighter) |
| EV-VC@500 | 0.2157 | 0.2179 | -1.0% (would loosen!) |
| EV-NDCG | 0.7124 | 0.7137 | -0.2% (would loosen!) |
| Spearman | 0.3729 | 0.3736 | -0.2% (would loosen!) |

Note: EV-VC@500, EV-NDCG, and Spearman floors would actually LOOSEN because v0011 means are lower than the v0007 champion used for calibration. This creates a ratchet-down risk. **Do NOT recalibrate gates downward.** Keep current floors.
