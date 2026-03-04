# Direction — Iteration 1 (feat-eng-3-20260304-102111)

## Batch Constraint
**Feature engineering / selection ONLY.** No hyperparameter changes. No training mode changes.
All regressor HPs must remain at v0007 defaults: n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25.

## Analysis

**Champion v0007** (34 features, mcw=25, reg_lambda=1.0):
- Mean: EV-VC@100=0.0699, EV-VC@500=0.2294, EV-NDCG=0.7513, Spearman=0.3932
- Weakest months: 2022-06 (EV-VC@100=0.014, EV-NDCG=0.604, Spearman=0.271), 2021-05 (EV-VC@100≈0)
- Strongest months: 2022-12 (EV-VC@100=0.192, EV-NDCG=0.815), 2022-03 (EV-NDCG=0.845)

**Key gap**: The classifier uses 14 features including `density_skewness`, `density_kurtosis`, `density_cv` — distributional shape features. The regressor uses 34 features but does NOT include these 3. This is likely an oversight from the original pipeline setup. Additionally, `season_hist_da_3` and `prob_below_85` are available from the data loader but unused.

**Rationale**: Distributional shape features (skewness, kurtosis, CV) capture non-linear flow distribution characteristics that directly relate to shadow price magnitude. A highly skewed or heavy-tailed flow distribution signals different congestion dynamics than a symmetric one. Since these features already proved useful for the classifier's binding prediction, they should help the regressor estimate dollar magnitude.

---

## Hypothesis A (primary): Add 3 distributional shape features (34 → 37)

**What**: Add `density_skewness`, `density_kurtosis`, `density_cv` to the regressor feature set. These are the 3 features used by the classifier but missing from the regressor.

**Why**: High-confidence addition — proven useful in classifier, raw columns from data loader (no FE code needed), captures shape of flow distribution which directly informs shadow price magnitude.

**Monotone constraints**: All 3 unconstrained (0) — skewness/kurtosis/CV have non-monotonic relationships with shadow price.

Hypothesis A overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]}}
```

---

## Hypothesis B (alternative): Add all 5 unused raw columns (34 → 39)

**What**: Add `density_skewness`, `density_kurtosis`, `density_cv`, `season_hist_da_3`, `prob_below_85` to the regressor.

**Why**: Tests the full set of unused raw columns. `season_hist_da_3` adds a third seasonal harmonic for historical DA prices, capturing more seasonal structure. `prob_below_85` adds a deeper below-threshold probability band. If B outperforms A, the additional 2 features carry signal; if A is better, they add noise.

**Monotone constraints**: density_skewness=0, density_kurtosis=0, density_cv=0, season_hist_da_3=+1, prob_below_85=-1.

Hypothesis B overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "hist_physical_interaction", "overload_exceedance_product", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "band_severity", "sf_exceed_interaction", "hist_seasonal_band", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1]}}
```

---

## Screen Months

| Month | Role | Rationale |
|-------|------|-----------|
| **2022-06** | Weak | Worst EV-VC@100 (0.014), worst EV-NDCG (0.604), 2nd worst Spearman (0.271). Distributional features should help most here — likely unusual flow distributions the regressor currently misses. |
| **2022-12** | Strong | Best EV-VC@100 (0.192), 2nd best EV-NDCG (0.815). Must confirm new features don't regress performance on months the model already handles well. |

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months
2. **Safety check**: Spearman must not drop > 0.03 on either screen month vs champion
3. **Tiebreaker**: Higher mean EV-NDCG across screen months
4. **If both degrade vs champion**: Pick the one with smallest EV-VC@100 degradation. If both are substantially worse (> 10% mean EV-VC@100 drop), STOP and escalate — the feature additions are harmful.

---

## Code Changes for Winner

After screening picks a winner, the worker should make these code changes:

1. **No code changes needed** — both hypotheses use raw columns already available in the data loader and already defined in `_ALL_REGRESSOR_FEATURES` in `ml/config.py`.

2. The winner's feature list and monotone_constraints become the new version's config. The worker should register the version with the winning overrides applied.

3. **Verification**: After the full 12-month benchmark, confirm the registered `config.json` has the correct feature count (37 for Hyp A winner, 39 for Hyp B winner) and matching monotone_constraints length.

---

## Expected Impact

| Gate | Expected Change | Reasoning |
|------|----------------|-----------|
| EV-VC@100 | +2-5% mean | Distributional shape → better magnitude estimates for extreme constraints |
| EV-VC@500 | +1-3% mean | Broader ranking benefit from richer feature set |
| EV-NDCG | +0.5-1.5% mean | Better ranking from more informative features |
| Spearman | ±0.5% | Feature additions should be neutral or slightly positive for rank correlation |
| C-RMSE | -1-3% | Better magnitude prediction from distributional features |

**Tail improvement**: Weak months (2022-06, 2021-05) should see largest gains — these months likely have unusual flow distributions that the 3 shape features can capture.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| New features add noise, degrading weak months | Low | Skewness/kurtosis proven in classifier; unconstrained monotone gives XGBoost freedom to use or ignore |
| Overfitting with 37-39 features (from 34) | Low | mcw=25 provides strong leaf regularization; 3-5 extra features is modest |
| Data loader missing these columns for some months | Very low | These are base columns from MisoDataLoader used by the classifier |
| Feature order mismatch in overrides vs data | Low | Worker must verify config.json feature count matches after benchmark |
