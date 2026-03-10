# v007 — Enriched Features + Tuned XGBoost

**Hypothesis**: v006 (17 density-only features, 10/2 split) produces marginal gains over v000
because effective feature dimensionality is low (~3 signals: tail probability, distribution
shape, historical price). Adding severity signals (expected overload magnitude), distribution
shape features (entropy, CV, tail concentration, probability bands), historical trend features,
tuned XGBoost (lower LR, more trees, regularization), and balanced F1 threshold should
meaningfully improve recall and regression quality.

**Changes vs v006**:
- **8 new features** from existing density + DA shadow price data (no new data sources):
  - `expected_overload`: ∫|x|·f(x)dx over overload region — severity signal
  - `tail_concentration`: prob_exceed_100 / prob_exceed_80 — exceedance curve shape
  - `prob_band_95_100`: P(95% < flow < 100%) — approaching-limit mass
  - `prob_band_100_105`: P(100% < flow < 105%) — just-barely-binding mass
  - `density_entropy`: -∫f(x)·log(f(x))dx — distribution uncertainty
  - `density_cv`: σ/|μ| — scale-free volatility
  - `hist_da_trend`: recent_hist_da / mean(season_hist_da) — congestion momentum
  - `hist_da_max_season`: max of 3 seasonal windows — worst historical year
- **Dropped 2 redundant features**: prob_below_85, prob_below_80 (= 1 - prob_exceed_*)
- **Step1 features**: 17 → 22; **Step2 features**: 21 → 27
- **XGBoost tuning**: lr 0.1→0.05, depth 4→5, n_estimators 200→400 (default) / 100→200 (branch),
  min_child_weight 10→5 (default clf), added subsample=0.8, colsample_bytree=0.8,
  reg_alpha=0.1, reg_lambda=1.0
- **Training split**: 10/2/0 → 9/3/0 (more validation for threshold optimization)
- **Threshold beta**: 0.5 → 1.0 (balanced F1 instead of precision-heavy)

## Results (32-run benchmark, 2026-02-23)

| Gate | v000 (champ) | v006 | v007 | Δ vs v000 | Δ vs v006 | Floor | Pass |
|------|-------------|------|------|-----------|-----------|-------|------|
| S1-AUC | 0.6954 | 0.6890 | 0.6815 | -0.014 | -0.008 | 0.65 | PASS |
| S1-REC | 0.2734 | 0.2902 | 0.3054 | **+0.032** | +0.015 | 0.25 | PASS |
| S2-SPR | 0.4120 | 0.4033 | 0.4011 | -0.011 | -0.002 | 0.30 | PASS |
| C-VC@1000 | 0.8508 | 0.8357 | 0.8194 | -0.031 | -0.016 | 0.50 | PASS |
| C-RMSE | 1555.79 | 1133.53 | 1057.36 | **-498** | -76 | 2000 | PASS |

**Gates passed**: 5/5
**Beats champion on**: 2/5 (S1-REC, C-RMSE)
**Promotable**: No (S2-SPR and C-VC@1000 do not beat champion)

### Per-class breakdown

| Gate | onpeak | offpeak |
|------|--------|---------|
| S1-AUC | 0.6777 | 0.6853 |
| S1-REC | 0.3043 | 0.3066 |
| S2-SPR | 0.3582 | 0.4440 |
| C-VC@1000 | 0.8081 | 0.8306 |
| C-RMSE | 1068.08 | 1046.64 |

### Feature selection results (smoke test 2020-07/onpeak/f0)

All 22 step1 features selected (22/22). New features show strong signals:

| Feature | AUC | Spearman | Keep? |
|---------|-----|----------|-------|
| expected_overload | 0.867 | 0.255 | Yes |
| tail_concentration | 0.845 | 0.239 | Yes |
| prob_band_95_100 | 0.873 | 0.256 | Yes |
| prob_band_100_105 | 0.870 | 0.256 | Yes |
| density_entropy | — | — | Yes (unconstrained) |
| density_cv | — | — | Yes (unconstrained) |
| hist_da_trend | 0.755 | 0.277 | Yes |

### Key observations

1. **S1-REC improved significantly** (+3.2pp vs v000, +1.5pp vs v006): Largest recall
   improvement in the series. The combination of threshold_beta=1.0 (balanced F1) and
   richer features lifts recall from 0.273 to 0.305. This exceeds the 0.25 floor
   comfortably for the first time.

2. **C-RMSE dramatically better** (-498 vs v000, -76 vs v006): At $1,057, RMSE is 32%
   better than v000 ($1,556) and 7% better than v006 ($1,134). The severity features
   (expected_overload, prob_bands) help the regressor predict shadow price magnitude.

3. **S1-AUC slightly declined** (-1.4pp vs v000): The balanced threshold trades precision
   for recall. AUC measures discrimination at all thresholds, and the additional features
   don't offset this trade-off. Still above 0.65 floor.

4. **S2-SPR essentially flat** (-1.1pp vs v000, -0.2pp vs v006): Despite adding severity
   features designed to improve regression ranking, Spearman correlation for true positives
   didn't improve. The new features have high AUC but are correlated with existing
   prob_exceed features, adding little orthogonal signal for regression.

5. **C-VC@1000 slightly declined** (-3.1pp vs v000): Value capture at K=1000 dropped from
   85% to 82%. The lower precision from balanced F1 threshold means some top-K slots go
   to non-binding constraints. Still well above 0.50 floor.

6. **New features are informative but not orthogonal**: expected_overload (AUC=0.867) and
   prob_bands (AUC~0.87) have excellent predictive power but are mechanically correlated
   with prob_exceed features. They help the classifier marginally but don't unlock a new
   dimension of signal for regression.

### Progression v000 → v006 → v007

| Gate | v000 → v006 | v006 → v007 | v000 → v007 |
|------|-------------|-------------|-------------|
| S1-AUC | -0.006 | -0.008 | -0.014 |
| S1-REC | +0.017 | +0.015 | **+0.032** |
| S2-SPR | -0.009 | -0.002 | -0.011 |
| C-VC@1000 | -0.015 | -0.016 | -0.031 |
| C-RMSE | -422 | -76 | **-498** |

**Conclusion**: v007 passes all 5 gates and achieves the strongest recall (0.305) and
lowest RMSE ($1,057) in the series. The enriched features and tuned XGBoost deliver
incremental gains over v006 on recall (+1.5pp) and RMSE (-76). However, the fundamental
problem remains: features extracted from the same density distributions are highly
correlated and don't provide orthogonal signal for regression ranking (S2-SPR) or
value capture (C-VC@1000). To beat v000 on those metrics, the model likely needs
genuinely new data sources (shift factors, constraint metadata, load forecasts) or
a fundamentally different model architecture (e.g., temporal features across outage dates).
