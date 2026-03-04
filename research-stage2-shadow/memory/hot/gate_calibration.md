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
| Gate | Floor | Tail Floor | Bottom-2 Mean (champion) | Direction |
|------|-------|------------|-------------------------|-----------|
| EV-VC@100 | 0.022266 | 0.001383 | 0.003476 | higher |
| EV-VC@500 | 0.087962 | 0.022810 | 0.048828 | higher |
| EV-NDCG | 0.689979 | 0.624664 | 0.673529 | higher |
| Spearman | 0.342121 | 0.288853 | 0.329625 | higher |

#### Group B (monitor)
| Gate | Floor | Tail Floor | Bottom-2 Mean (champion) | Direction |
|------|-------|------------|-------------------------|-----------|
| C-RMSE | 3900.37 | 6746.20 | 5967.59 | lower |
| C-MAE | 1476.48 | 2345.51 | 2133.17 | lower |
| EV-VC@1000 | 0.136432 | 0.069288 | 0.098985 | lower |
| R-REC@500 | 0.014179 | 0.009272 | 0.012397 | higher |

### v0003 Gate Check (against committed gates)
- **Layer 1**: All 4 Group A gates PASS (means above committed floors)
- **Layer 2**: All 4 Group A gates PASS (0 months below committed tail floors)
- **Layer 3**: All 4 Group A gates PASS (bottom-2 means within noise tolerance of committed v0)
- **Verdict**: v0003 is promotable against committed gates (blocked only by infra failure)

### Calibration Observations
- Committed gates are well-calibrated — floor ≈ 0.7x v0 mean, giving ~30% headroom for variance
- EV-VC@100 has highest relative variance (std/mean ≈ 0.88) — floor at 0.022 vs mean 0.030 provides appropriate slack
- Tail floors match champion.md numbers well
- Gates remain appropriate for the current pipeline configuration (10/2, 24 features)
