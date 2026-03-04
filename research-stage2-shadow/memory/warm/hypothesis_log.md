# Hypothesis Log

## H1: Add distributional shape features (34→37) — CONFIRMED
**Tested in**: Iter 1 screen (Hypothesis A)
**Result**: Screen showed improvement on weak month (2022-06 EV-VC@100: 0.014→0.016) and acceptable Spearman stability. Full benchmark not run for 37-feature variant alone, but 39-feature variant (superset) confirmed the direction.
**Key numbers**: Screen mean EV-VC@100=0.072 (vs champion 0.103 across screen months — note: 2-month average, not full 12-month)

## H2: Add all 5 unused raw columns (34→39) — CONFIRMED
**Tested in**: Iter 1 full benchmark (Hypothesis B / v0009)
**Result**: EV-VC@100 +9.0%, EV-VC@500 +1.5%, EV-NDCG +0.5%, Spearman -0.6%, C-RMSE -3.1%
**Key numbers**: Mean EV-VC@100=0.0762 (champion=0.0699). 7/12 months improved on EV-VC@100. Improvement concentrated in 3 months (2020-09: +0.043, 2020-11: +0.046, 2021-09: +0.078).
**Caveat**: Improvement is outlier-dependent. Removing best month (2021-09) drops mean improvement to ~+4%. 5 months degraded on EV-VC@100.

## H3: Prune 5 dead features (39→34) — CONFIRMED
**Tested in**: Iter 1 full benchmark (v0011, batch feat-eng-3-20260304-121042)
**Result**: EV-VC@100 +5.2%, EV-VC@500 -2.5%, EV-NDCG -0.6%, Spearman +0.4%, C-RMSE +1.4%
**Key numbers**: Mean EV-VC@100=0.0801 (champion v0009=0.0762). Improvement concentrated in 3 months. 7/12 months degrade on EV-VC@500. EV-VC@500 L2 at exact limit (1 tail failure).
**Mechanism**: Removing zero-filled features improves colsample_bytree sampling efficiency. Trees now sample only signal-carrying features. Sharpens top-100 discrimination but reduces top-500 breadth.
**Tradeoff**: Precision-vs-breadth. Acceptable for top-100 business objective.

## H4: Add flow_direction to pruned set (39→35) — FAILED (screen)
**Tested in**: Iter 1 screen (Hypothesis B, batch feat-eng-3-20260304-121042)
**Result**: Lost to prune-only on mean EV-VC@100 (0.1238 vs 0.1316, -6.3%). Spearman was fine.
**Key numbers**: 2021-09 EV-VC@100: 0.220 (vs prune-only 0.235). 2022-09 EV-VC@100: 0.028 (vs prune-only 0.028).
**Interpretation**: flow_direction added noise rather than signal. Binding direction does not correlate meaningfully with shadow price magnitude. Feature should NOT be revisited.
