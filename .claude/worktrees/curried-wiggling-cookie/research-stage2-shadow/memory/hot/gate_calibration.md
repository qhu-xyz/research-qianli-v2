## Gate Calibration

### Current State: Gates v4, Calibrated to v0007 (0.95x mean, 0.90x worst)

Now evaluating against v0012 champion.

| Gate | Floor | v0012 Mean | Margin | Assessment |
|------|-------|-----------|--------|------------|
| EV-VC@100 | 0.0664 | 0.0758 | +14.2% | Moderate — narrowed from +20.6% (v0011) |
| EV-VC@500 | 0.2179 | 0.2348 | +7.8% | **Recovered** — was +4.2% (v0011), now healthy |
| EV-NDCG | 0.7137 | 0.7518 | +5.3% | Moderate, improved from +5.1% |
| Spearman | 0.3736 | 0.3940 | +5.5% | Moderate, improved from +5.1% |

### Tail Floor Assessment (v0012)

| Gate | Tail Floor | Worst Month | Fails | Max | Status |
|------|-----------|-------------|-------|-----|--------|
| EV-VC@100 | 0.000135 | 0.0006 (2021-05) | 0 | 1 | OK but tail_floor is non-protective |
| EV-VC@500 | 0.0536 | 0.0676 (2022-06) | **0** | 1 | **CLEAR** — was at limit (1 fail) with v0011 |
| EV-NDCG | 0.5434 | 0.6052 (2022-06) | 0 | 1 | OK |
| Spearman | 0.2363 | 0.2651 (2021-11) | 0 | 1 | OK |

### L3 Margins (v0012 vs v0011 champion)

| Gate | v0012 bot2 | Threshold (champ bot2 - 0.02) | Margin |
|------|-----------|-------------------------------|--------|
| EV-VC@100 | 0.0086 | -0.0089 | +0.0175 (trivially passing, tolerance > metric scale) |
| EV-VC@500 | 0.0698 | 0.0341 | **+0.0357** (massive improvement from +0.0023) |
| EV-NDCG | 0.6434 | 0.6203 | +0.0231 (OK) |
| Spearman | 0.2696 | 0.2478 | +0.0218 (OK) |

### Binding Gate Constraint

**No gate is tight.** For the first time in the batch, all gates have comfortable margins:
- EV-VC@100 L1 +14.2% is the tightest, but still not concerning
- EV-VC@500 fully recovered from critical state to +7.8% L1, 0 tail fails, +0.0357 L3

### Weakest Months (v0012)

| Month | Persistently Weak Metrics |
|-------|--------------------------|
| 2022-06 | EV-VC@100 (0.0166), EV-VC@500 (0.0676), EV-NDCG (0.6052), Spearman (0.2742) |
| 2021-05 | EV-VC@100 (0.0006), EV-VC@500 (0.0918) |
| 2022-09 | EV-VC@500 (0.0720), EV-NDCG (0.6816) |
| 2021-11 | Spearman (0.2651) |

2022-06 is the structural weak month across all metrics — likely a market regime issue (low summer congestion).

### L3 Noise Tolerance Issue (ONGOING, UNFIXED — 3rd iteration flagged)

`noise_tolerance=0.02` is not scale-aware:
- EV-VC@100 (bot2 ~0.009): threshold goes to -0.011 → provides zero protection
- EV-VC@500 (bot2 ~0.070): 0.02 is ~29% of value → very generous
- C-RMSE (bot2 ~5283): 0.02 is negligible → requires near-exact match
- Spearman (bot2 ~0.27): 0.02 is ~7% → reasonable

**Recommendation (repeated, 3rd time)**: `max(abs_floor, pct * |champ_bottom_2_mean|)` with per-metric tuning.

### Should Gates Recalibrate to v0012?

If gates recalibrate to v0012 champion (0.95x mean):
| Gate | New Floor | Current Floor | Change |
|------|-----------|---------------|--------|
| EV-VC@100 | 0.0720 | 0.0664 | +8.4% (tighter) |
| EV-VC@500 | 0.2231 | 0.2179 | +2.4% (tighter) |
| EV-NDCG | 0.7142 | 0.7137 | +0.1% (negligible) |
| Spearman | 0.3743 | 0.3736 | +0.2% (negligible) |

v0012 would tighten EV-VC@100 and EV-VC@500 floors modestly. Unlike v0011, this would NOT loosen any gate. Could consider recalibrating after batch concludes, but not mid-batch. **Keep current gates v4 for iter 3.**
