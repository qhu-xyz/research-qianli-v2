## Gate Calibration

### ⚠️ CRITICAL: Working tree gates.json is CONTAMINATED

The `registry/gates.json` in the working tree was overwritten by the iter-1 worker with gates calibrated from a WRONG v0 baseline (6/2 train/val, 34 features). The committed gates.json (from `git show HEAD:`) has the CORRECT values.

| Source | EV-VC@100 floor | EV-VC@500 floor | EV-NDCG floor | Spearman floor |
|--------|----------------|----------------|--------------|---------------|
| **Committed (CORRECT)** | 0.022266 | 0.087962 | 0.689979 | 0.342121 |
| Dirty working tree (WRONG) | 0.069032 | 0.215950 | 0.747213 | 0.392798 |
| Ratio (dirty/committed) | 3.1x | 2.5x | 1.08x | 1.15x |

**Action required**: `git checkout -- registry/gates.json registry/v0/` to restore correct gates before any gate evaluation.

### Committed Gates (CORRECT — calibrated from committed v0 with 10/2/24feat)

#### Group A (hard, blocking)
| Gate | Floor | Tail Floor | v0 Bottom-2 | Direction |
|------|-------|------------|-------------|-----------|
| EV-VC@100 | 0.022266 | 0.001383 | 0.003476 | higher |
| EV-VC@500 | 0.087962 | 0.022810 | 0.048828 | higher |
| EV-NDCG | 0.689979 | 0.624664 | 0.673529 | higher |
| Spearman | 0.342121 | 0.288853 | 0.329625 | higher |

#### Group B (monitor)
| Gate | Floor | Tail Floor | v0 Bottom-2 | Direction |
|------|-------|------------|-------------|-----------|
| C-RMSE | 3900.37 | 6746.20 | 5967.59 | lower |
| C-MAE | 1476.48 | 2345.51 | 2133.17 | lower |
| EV-VC@1000 | 0.136432 | 0.069288 | 0.098985 | lower |
| R-REC@500 | 0.014179 | 0.009272 | 0.012397 | higher |

### v0003 Gate Check (against committed gates) — PASSES ALL
| Gate | Mean | Floor | L1 | Months below tail | L2 | Bottom-2 | Champ B2 - 0.02 | L3 |
|------|------|-------|----|-------------------|----|----------|------------------|----|
| EV-VC@100 | 0.0337 | 0.0223 | ✅ | 0 | ✅ | 0.0048 | -0.017 | ✅ |
| EV-VC@500 | 0.1174 | 0.0880 | ✅ | 0 | ✅ | 0.0429 | 0.029 | ✅ |
| EV-NDCG | 0.7435 | 0.6900 | ✅ | 0 | ✅ | 0.6738 | 0.654 | ✅ |
| Spearman | 0.3921 | 0.3421 | ✅ | 0 | ✅ | 0.3299 | 0.310 | ✅ |

### v0004 Gate Check (against committed gates) — PASSES ALL
| Gate | Mean | Floor | L1 | Months below tail | L2 | Bottom-2 | Champ B2 - 0.02 | L3 |
|------|------|-------|----|-------------------|----|----------|------------------|----|
| EV-VC@100 | 0.0306 | 0.0223 | ✅ | 0 | ✅ | 0.0070 | -0.017 | ✅ |
| EV-VC@500 | 0.1110 | 0.0880 | ✅ | 0 | ✅ | 0.0603 | 0.029 | ✅ |
| EV-NDCG | 0.7420 | 0.6900 | ✅ | 0 | ✅ | 0.6741 | 0.654 | ✅ |
| Spearman | 0.3929 | 0.3421 | ✅ | 0 | ✅ | 0.3325 | 0.310 | ✅ |

### Calibration Observations
- Committed gates are well-calibrated — floor ≈ 0.7x v0 mean, giving ~30% headroom for variance
- EV-VC@100 has highest relative variance (std/mean ≈ 0.88) — floor at 0.022 vs mean 0.030 provides appropriate slack
- Both v0003 and v0004 pass comfortably — no gates are borderline
- Gates remain appropriate for the current pipeline configuration (10/2, 24 features)
- The noise_tolerance of 0.02 on Layer 3 is generous — provides safety for tail variation without being exploitable
