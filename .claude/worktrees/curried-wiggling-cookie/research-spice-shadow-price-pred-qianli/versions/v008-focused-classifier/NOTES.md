# v008 — Focused Classifier, Rich Regressor

**Hypothesis**: v007's classifier quality (S1-AUC, C-VC@1000) declined because it used
too many correlated features (22) with too little training data (9 months) and a
more complex model (max_depth=5, lr=0.05). Reverting the classifier to a simpler
architecture (fewer features, v000-like hyperparams with regularization) while
keeping v007's rich regressor should restore classifier metrics while preserving
regression gains.

**Changes vs v007**:
- **Step1 features: 22 → 14** — dropped 8 correlated features (prob_exceed_{85,80},
  density_{mean,variance,entropy}, tail_concentration, prob_band_{95_100,100_105}).
  Kept orthogonal signals: expected_overload, density_cv, hist_da_trend.
- **Step2 features: 27 → 27** — no change (rich regressor preserved)
- **Training split: 9/3 → 10/2** — restored 1 month of training data
- **Threshold beta: 1.0 → 0.7** — moderate balance between precision (0.5) and recall (1.0)
- **Classifier hyperparams**: reverted to v000 core + regularization:
  - default: n_estimators 400→200, max_depth 5→4, lr 0.05→0.1, min_child_weight 5→10
  - branch: n_estimators 200→100, max_depth 5→4, lr 0.05→0.1, min_child_weight 5→1
  - Kept subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- **Regressor hyperparams**: unchanged from v007

## Results (32-run benchmark, 2026-02-24)

| Gate | v000 (champ) | v007 | v008 | Δ vs v000 | Δ vs v007 | Floor | Pass |
|------|-------------|------|------|-----------|-----------|-------|------|
| S1-AUC | 0.6954 | 0.6815 | 0.6894 | -0.006 | **+0.008** | 0.65 | PASS |
| S1-REC | 0.2734 | 0.3054 | 0.2894 | **+0.016** | -0.016 | 0.25 | PASS |
| S2-SPR | 0.4120 | 0.4011 | 0.4117 | -0.000 | **+0.011** | 0.30 | PASS |
| C-VC@1000 | 0.8508 | 0.8194 | 0.8428 | -0.008 | **+0.023** | 0.50 | PASS |
| C-RMSE | 1555.79 | 1057.36 | 1108.91 | **-447** | +52 | 2000 | PASS |

**Gates passed**: 5/5
**Beats champion (within 2% tolerance)**: 5/5 (S1-REC genuine +0.016, C-RMSE genuine -447)
**Promotable**: Yes

### Per-class breakdown

| Gate | onpeak | offpeak |
|------|--------|---------|
| S1-AUC | 0.6849 | 0.6940 |
| S1-REC | 0.2861 | 0.2927 |
| S2-SPR | 0.3694 | 0.4539 |
| C-VC@1000 | 0.8234 | 0.8623 |
| C-RMSE | 1136.99 | 1080.83 |

### Key observations

1. **Classifier quality restored**: S1-AUC recovered +0.008 from v007 (0.6815 → 0.6894),
   closing 57% of the gap to v000 (0.6954). C-VC@1000 recovered +0.023 (0.8194 → 0.8428),
   closing 74% of the gap. The "focused classifier" strategy (fewer features + simpler
   model + more training data) significantly improved probability calibration.

2. **S2-SPR essentially matches v000**: At 0.4117 vs v000's 0.4120 (Δ=-0.0003), regression
   ranking is effectively tied. The improvement from v007 (+0.011) comes from two sources:
   (a) purer TP set from higher-precision classifier, (b) richer regressor features.

3. **S1-REC trades down from v007 but still beats v000**: At 0.2894, recall is +0.016 above
   v000 (0.2734) but -0.016 below v007 (0.3054). The beta=0.7 threshold selects a more
   precision-oriented operating point than beta=1.0. This is an acceptable trade-off since
   it enables the C-VC@1000 and S2-SPR improvements.

4. **C-RMSE remains strong**: At $1,109, RMSE is 29% better than v000 ($1,556), though +$52
   higher than v007 ($1,057). The slight regression vs v007 is expected from reduced recall
   (fewer TPs means a few more false negatives predicting $0). The rich regressor features
   keep it well below v000.

5. **Asymmetric classifier/regressor design works**: The core insight validated — classifiers
   benefit from simplicity and data volume (fewer features, simpler model, more training months),
   while regressors benefit from richer features and deeper models. Decoupling these configs
   allowed each stage to optimize independently.

6. **Onpeak S2-SPR is the weakest metric** (0.3694 vs offpeak 0.4539): Onpeak congestion
   patterns are more variable and harder to rank. This remains the bottleneck for further
   improvement.

### Progression v000 → v006 → v007 → v008

| Gate | v000→v006 | v006→v007 | v007→v008 | v000→v008 |
|------|-----------|-----------|-----------|-----------|
| S1-AUC | -0.006 | -0.008 | **+0.008** | -0.006 |
| S1-REC | +0.017 | +0.015 | -0.016 | **+0.016** |
| S2-SPR | -0.009 | -0.002 | **+0.011** | -0.000 |
| C-VC@1000 | -0.015 | -0.016 | **+0.023** | -0.008 |
| C-RMSE | -422 | -76 | +52 | **-447** |

v008 reverses the declining trend on S1-AUC, S2-SPR, and C-VC@1000 while maintaining
strong recall and RMSE improvements.

**Conclusion**: v008 is the first version to pass all 5 gates AND be promotable (within
2% noise tolerance of champion on all metrics, with genuine improvement on S1-REC and
C-RMSE). The "focused classifier, rich regressor" architecture provides the best overall
balance in the series. To surpass v000 outright on S1-AUC and C-VC@1000, the classifier
likely needs genuinely new data sources (shift factors, constraint metadata) rather than
more density-derived features.
