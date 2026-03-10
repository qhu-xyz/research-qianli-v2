# Hypothesis Log

## Tested Hypotheses

### H-fe2-1a: Add 3 interaction features 34→37 (tier-fe-2 iter1) — CONFIRMED (marginal)
- **Hypothesis**: Adding overload_x_hist, prob110_x_recent_hist, tail_x_hist to existing 34 features should improve tier 0/1 discrimination and Tier-VC@100
- **Rationale**: Pre-computed products of top-importance features (recent_hist_da 21.1%, hist_da 13.3%) with physical flow signals create explicit compound severity signals
- **Result**: CONFIRMED — all metrics improved, no regressions. VC@100 +5.4% (0.0708→0.0746). Bottom_2_mean +64%.
- **But**: Insufficient to cross VC@100 floor (0.0746 vs 0.0750). Effect size small (Cohen's d~0.06).
- **Learning**: Interaction features provide consistent but modest improvement. XGBoost was already partially approximating these via multi-level splits.

### H-fe2-1b: Add 3 interactions + prune 5 low-importance 34→32 (tier-fe-2 iter1) — FAILED (screening)
- **Hypothesis**: Adding 3 interactions + pruning 5 lowest-importance features should improve sampling efficiency
- **Result**: Lost screening to H-fe2-1a. Mean VC@100 0.1307 vs A's 0.1372. Pruning hurt weak month (2022-06: 0.0112 vs A's 0.0255).
- **Learning**: Pruning low-importance features can hurt weak months where those features provide marginal signal. Additive approach is safer.

## Candidate Hypotheses (for iter2 and future)

### FE-eligible (current batch):
1. **Log transforms** (log1p_hist_da, log1p_expected_overload): Compress long-tailed distributions for better high-end discrimination — QUEUED for iter2
2. **More interactions** (overload_x_recent_hist, prob_range_high): Extend compound signal approach — QUEUED for iter2
3. **Aggressive pruning**: REJECTED based on H-fe2-1b result — pruning hurts weak months

### Blocked (need non-FE batch):
1. **Increase tier 1 class weight** (5→15-20): Should improve Tier-Recall@1 from 0.045
2. **Reduce to 4 classes**: Drop tier 4 (always 0 samples)
3. **Lower min_child_weight** (25→10): Finer splits for rare tier 0/1
4. **Increase n_estimators** (400→800): More capacity for rare classes
