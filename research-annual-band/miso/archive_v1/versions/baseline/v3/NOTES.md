# v3

## What
Nodal f0 stitch: `path_mcp = sink_f0 - source_f0` from `get_mcp_df(market_month=PY-1)` col 0,
averaged over 3 delivery months. Node replacements via BFS on `MisoNodalReplacement`.
Coverage 88.8-100% with H fallback on missing nodes.

## Why
- f0 monthly forward prices directly reflect current congestion expectations
- Nodal stitching gives near-universal coverage (vs 36-45% for path-level f0)
- 3-month averaging smooths monthly noise while capturing structural congestion

## Results (corrected monthly units, 2026-03-14)

### vs v1 (H baseline)

| Quarter | v3 MAE | v1 MAE | Reduction | v3 Dir% | v1 Dir% | Gain |
|---------|-------:|-------:|----------:|--------:|--------:|-----:|
| aq1 | 307 | 380 | -19.3% | 80.9% | 67.6% | +13.3pp |
| aq2 | 293 | 390 | -24.8% | 82.1% | 68.5% | +13.6pp |
| aq3 | 257 | 316 | -18.7% | 83.8% | 69.0% | +14.8pp |
| aq4 | 201 | 390 | -48.5% | 84.8% | 64.3% | +20.5pp |

### Per-bin MAE

| Quarter | tiny (<50) | small (50-250) | med (250-1k) | large (1k+) |
|---------|----------:|---------------:|-------------:|------------:|
| aq1 | 79 | 155 | 341 | 1,123 |
| aq2 | 70 | 144 | 324 | 1,014 |
| aq3 | 59 | 122 | 288 | 868 |
| aq4 | 45 | 100 | 248 | 749 |

### Full baseline comparison (all available)

| Baseline | Avg MAE | Avg Dir% | Avg P95 | Coverage |
|----------|--------:|---------:|--------:|---------:|
| f1 path-level | 209 | 80.8% | 809 | 36.0% |
| f0 path-level | 217 | 81.1% | 845 | 45.1% |
| Prior year R3 MCP | 232 | 79.6% | 909 | 27.9% |
| Prior year R2 MCP | 236 | 79.6% | 924 | 25.9% |
| **Nodal f0 (v3)** | **264** | **82.9%** | **1,037** | **91.9%** |
| H (v1) | 369 | 67.4% | 1,292 | 100.0% |
| Prior year R1 MCP | 715 | 79.7% | 2,920 | 22.5% |

*Previous results (MAE 798/947/797/704) were inflated ~3x due to R1 mcp_mean unit bug.
Improvements vs H (14-21%) were underestimated — actual improvements are 19-49%.*

## Decision
Promoted as current best. Best MAE and Dir% across all 4 quarters among high-coverage
baselines. f0/f1 path-level have lower MAE but only cover 36-45% of paths.
