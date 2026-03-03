## Champion: v0 (baseline)
Gated regressor with 24 features, frozen stage-1 classifier (13 features, XGBoost 200 trees).
Regressor: XGBoost 400 trees, max_depth=5, trains on binding-only samples.
Target: log1p(max(0, shadow_price)), predictions via expm1.
EV scoring: P(binding) × predicted_shadow_price.

### v0 Baseline Metrics (12 months, f0, onpeak)

| Metric (Group A) | Mean | Std | Min | Max | Bottom-2 |
|---|---|---|---|---|---|
| EV-VC@100 | 0.0303 | 0.0222 | 0.0024 | 0.0714 | 0.0035 |
| EV-VC@500 | 0.1180 | 0.0603 | 0.0328 | 0.2107 | 0.0488 |
| EV-NDCG | 0.7400 | 0.0425 | 0.6547 | 0.8095 | 0.6735 |
| Spearman | 0.3921 | 0.0425 | 0.3189 | 0.4579 | 0.3296 |

| Metric (Group B) | Mean | Std | Min | Max | Bottom-2 |
|---|---|---|---|---|---|
| C-RMSE | 3400.4 | 1407.8 | 1601.1 | 6246.2 | 5967.6 |
| C-MAE | 1276.5 | 453.8 | 723.8 | 2145.5 | 2133.2 |
| EV-VC@1000 | 0.1864 | 0.0666 | 0.0893 | 0.2882 | 0.0990 |
| R-REC@500 | 0.0192 | 0.0042 | 0.0123 | 0.0252 | 0.0124 |

### Gate Floors (from v0)
- EV-VC@100: floor=0.0223, tail=0.0014
- EV-VC@500: floor=0.0880, tail=0.0228
- EV-NDCG: floor=0.6900, tail=0.6247
- Spearman: floor=0.3421, tail=0.2889
- C-RMSE: floor=3900.4, tail=6746.2
- C-MAE: floor=1476.5, tail=2345.5
- EV-VC@1000: floor=0.1364, tail=0.0693
- R-REC@500: floor=0.0142, tail=0.0093
