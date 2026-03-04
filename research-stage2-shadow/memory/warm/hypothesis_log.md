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

## H5: More trees + lower LR for EV-VC@500 breadth recovery (600t, lr=0.03) — CONFIRMED
**Tested in**: Iter 2 full benchmark (v0012, batch feat-eng-3-20260304-121042)
**Result**: EV-VC@500 +3.5%, EV-VC@100 -5.3%, EV-NDCG +0.2%, Spearman +0.4%, C-RMSE -0.4%, C-MAE -0.7%
**Key numbers**: Mean EV-VC@500=0.2348 (champion v0011=0.2270). 2022-09 tail failure eliminated: 0.0527→0.0720 (+36.7%). bot2_mean jumped 0.0541→0.0698 (+29%).
**Mechanism**: More ensemble rounds at lower step size provides finer-grained mid-tier discrimination. The extra trees help resolve value differences among constraints ranked 100-500, especially in low-signal months where fewer binding constraints exist. Boosting budget reduced 20→18 but more averaging offsets.
**Tradeoff**: EV-VC@100 -5.3% — the model's aggressive top-100 discrimination was slightly diluted by the additional granularity. Concentrated in 2022-12 (-21.8%) and 2022-03 (-15.9%).

## H6: Higher colsample_bytree for per-tree signal breadth (col=0.9, 500t, lr=0.04) — FAILED (screen)
**Tested in**: Iter 2 screen (Hypothesis B, batch feat-eng-3-20260304-121042)
**Result**: Mean EV-VC@500 identical to champion (0.1992). Failed Spearman veto on 2022-12 (-0.054).
**Key numbers**: 2022-09 EV-VC@500: 0.0544 (+3.2% vs champion), barely above tail_floor. 2022-12 Spearman: 0.3809 (-0.054 vs champion).
**Interpretation**: Increasing colsample from 0.8 to 0.9 with 34 features (31/34 per tree vs 27/34) did not materially improve breadth. The EV-VC@500 problem was about ensemble granularity (more rounds), not per-tree signal coverage. The Spearman collapse on 2022-12 suggests colsample=0.9 reduced beneficial tree diversity.
