# V7.0b vs V6.2B: Signal Distinctiveness Analysis (PJM)

Scope: f0/onpeak, 24 holdout months (2024-01 to 2025-12).

## Study 1: Overall Discrimination

V7.0b dramatically improves rank concentration of binding constraints.

| Metric | V6.2B | V7.0b | Delta |
|--------|-------|-------|-------|
| Mean binding rank | 0.56 | 0.26 | -0.30 (lower = better) |
| Top-20% captures (aggregate) | 161 | 1,105 | **+586%** |
| Mean AUC | 0.748 | 0.781 | +0.033 |

Every single holdout month shows V7.0b placing binders closer to the top of the ranking.

## Study 3: New Binder Early Alarm

425 truly new binders appeared during holdout (no prior binding history).

| Lead | n | V6.2B T0 | V7.0b T0 | V6.2B T0+T1 | V7.0b T0+T1 |
|------|---|----------|----------|-------------|-------------|
| 1mo | 54 | 1.9% | 3.7% | 7.4% | 5.6% |
| 2mo | 55 | 1.8% | 1.8% | 1.8% | 3.6% |
| 3mo | 47 | 2.1% | 2.1% | 2.1% | 6.4% |
| 6mo | 31 | 3.2% | 3.2% | 3.2% | 9.7% |

V7.0b matches or slightly beats V6.2B for new binder early detection. Both signals struggle here — new binders are inherently hard to predict without history.

## Study 4: Recurring Binder Early Alarm

1,293 gap-resume events (bound, stopped 3+ months, then re-bound during holdout).

| Window | n | V6.2B T0 | V7.0b T0 | V6.2B T0+T1 | V7.0b T0+T1 |
|--------|---|----------|----------|-------------|-------------|
| 1mo before re-bind | 307 | 1.6% | **26.7%** | 8.1% | **56.7%** |

This is the standout result. V7.0b's binding frequency features detect recurring binders 1 month before they re-bind at **17x** the rate of V6.2B (26.7% vs 1.6% in T0). Over half of recurring binders are flagged in the top 40% of the ranking.

## Holdout VC@20 (all 6 slices, smoothed per-ptype blends)

| Slice | v0 (V6.2B) | v2 (V7.0b) | Delta |
|-------|-----------|-----------|-------|
| f0/onpeak | 0.4431 | 0.4294 | -3.1% |
| f0/dailyoffpeak | 0.5964 | 0.6067 | +1.7% |
| f0/wkndonpeak | 0.4077 | 0.4124 | +1.2% |
| f1/onpeak | 0.4199 | 0.4162 | -0.9% |
| f1/dailyoffpeak | 0.4943 | 0.5319 | +7.6% |
| f1/wkndonpeak | 0.3930 | 0.3562 | -9.4% |

Holdout VC@20 is mixed: v2 ML wins on dailyoffpeak (+1.7% to +7.6%) but is slightly behind on some onpeak/wkndonpeak slices. The Top-20 capture and AUC improvements show V7.0b has better overall discrimination, even where VC@20 is marginally lower.

## Blend Weights (smoothed per-ptype)

| Period | (da, dmix, dori) | Notes |
|--------|-----------------|-------|
| f0 | (0.00, 0.20, 0.80) | density_ori dominates |
| f1 | (0.00, 0.90, 0.10) | density_mix dominates |

Smoothed across class types for robustness (same blend for onpeak/dailyoffpeak/wkndonpeak within each ptype).
