## Gate Calibration

Gates bootstrapped from v0 baseline (12 months, f0, onpeak).

### Group A (hard, blocking)
| Gate | Floor | Tail Floor | Bottom-2 Mean (champion) | Direction |
|------|-------|------------|-------------------------|-----------|
| EV-VC@100 | 0.069032 | 0.000114 | 0.006836 | higher |
| EV-VC@500 | 0.21595 | 0.040651 | 0.055834 | higher |
| EV-NDCG | 0.747213 | 0.604491 | 0.647623 | higher |
| Spearman | 0.392798 | 0.264900 | 0.268866 | higher |

### Group B (monitor)
| Gate | Floor | Tail Floor | Bottom-2 Mean (champion) | Direction |
|------|-------|------------|-------------------------|-----------|
| C-RMSE | 3133.34 | 5918.34 | 5328.98 | lower |
| C-MAE | 1158.39 | 2283.26 | 2101.97 | lower |
| EV-VC@1000 | 0.312266 | 0.090844 | 0.108581 | lower |
| R-REC@500 | 0.034261 | 0.013317 | 0.021013 | higher |

### Observations
- EV-VC@100 floor (0.069) is the v0 mean — very tight, any regression will fail Layer 1
- EV-VC@100 tail_floor (0.000114) is extremely permissive — v0's worst month was catastrophic
- Spearman floor (0.393) = v0 mean — also tight, value-weighting could threaten this
- Gates are calibrated from v0 itself, so the bar is "don't regress" rather than "improve"
- **No iteration data yet** — cannot assess calibration quality until we get our first successful run
