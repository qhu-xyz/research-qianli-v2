# v003-xgb-20260222-001 — Expanded Features + Higher Capacity

## Hypothesis

The AUC gap (0.66 vs 0.80 floor) is caused by two factors working together:

1. **Too few features**: Only 9 step1 features, with 5 more available but commented out. The classifier lacks signal to discriminate binding from non-binding — more probability thresholds and distribution shape features should help.

2. **Underfitting**: Conservative hyperparameters (max_depth=4, min_child_weight=10, n_estimators=200) limit model capacity. With a 7% binding rate and ~800K training rows, the model has room for more complexity.

We also revert threshold_beta to 0.5 (from v001/v002's 2.0) since F2.0 hurts AUC by ~3 pp without sufficient benefit elsewhere.

## Changes

| Parameter | Baseline (v000) | This version | Rationale |
|-----------|-----------------|--------------|-----------|
| `threshold_beta` | 0.5 | **0.5** (same) | Revert from v001/v002 F2.0; AUC is priority |
| Step1 features | 9 features | **14 features** | +prob_exceed_85/80, +prob_below_100, +density_mean, +density_kurtosis |
| Step2 features | 13 features | **18 features** | Same 5 additions |
| `density_skewness` constraint | +1 | **0** | From v002: confirmed neutral, no downside |
| `max_depth` (all models) | 4 | **6** | Allow deeper trees for complex interactions |
| `n_estimators` (all models) | 200 | **400** | More boosting rounds for finer learning |
| `min_child_weight` (all models) | 10 | **5** | Allow splits on smaller leaf nodes |

**Total changes**: 4 categories (threshold, features, constraints, capacity) across 7 specific parameters.

## Expected Impact

- Features: +2-4 pp AUC from additional discriminative signals
- Capacity: +2-3 pp AUC from reduced underfitting
- Threshold revert: +3 pp AUC from not sacrificing discrimination for recall
- Combined estimate: AUC 0.66 → 0.72-0.76 (narrowing the gap significantly)

## Results

| Gate | Onpeak | Offpeak | Mean | Floor | vs v000 |
|------|-------:|--------:|-----:|------:|--------:|
| S1-AUC | 0.6581 | 0.6657 | 0.6619 | 0.80 | **-3.4 pp** |
| S1-REC | 0.2735 | 0.2734 | 0.2734 | 0.30 | +0.0 pp |
| S2-SPR | 0.3735 | 0.4256 | 0.3995 | 0.30 | -1.3 pp |
| C-VC@1000 | 0.7752 | 0.8028 | 0.7890 | 0.50 | **-6.2 pp** |
| C-RMSE | $1,488 | $1,429 | $1,459 | $2,000 | -$97 |

Promotable: **No** — S1-AUC fails floor (worse than baseline); S1-REC fails floor; C-VC@1000 regresses.

Wall time: 47 min (32 workers, concurrency=4, ~6 min/worker).

## Conclusion

**Negative result.** Adding 5 features and increasing model capacity made things worse, not better.

Key observations:
1. **AUC dropped 3.4 pp vs baseline** — the new features (prob_exceed_85/80, prob_below_100, density_mean, density_kurtosis) are adding noise, not signal. The classifier was already using the best features; the commented-out ones were commented out for a reason.
2. **Recall didn't change** — despite reverting to F0.5 threshold, recall stayed at 0.27 (same as baseline). The threshold revert didn't help because the F0.5 threshold was already the default.
3. **Value capture dropped 6.2 pp** — more features + deeper trees → overfitting. The model makes more confident but wrong predictions, hurting ranking.
4. **RMSE improved slightly** ($1,459 vs $1,556) — the only positive, likely from the regressor having more capacity.

**Lessons:**
- The AUC ceiling (~0.66-0.70) is not caused by underfitting or missing features. The available density-based features have a fundamental discrimination limit.
- Adding features without new information sources won't help. Need either (a) fundamentally different features (structural, temporal, network-based) or (b) better training strategy (sample weighting, curriculum learning).
- Feature selection was already working correctly — it was keeping the right features and dropping the weak ones. Forcing weak features in made things worse.
- The problem is likely **feature quality**, not model capacity.

**Next direction:** Instead of more of the same feature type, investigate whether the feature selection process itself is dropping useful information, or look at what the top-performing individual runs (by season/class) do differently.
