# v3

## What
Nodal f0 stitch: `path_mcp = sink_f0 - source_f0` from `get_mcp_df(market_month=PY-1)` col 0,
averaged over 3 delivery months. Node replacements via BFS on `MisoNodalReplacement`.
Coverage 98.8-100% with H fallback on missing nodes (~0-1.2%).

## Why
- f0 monthly forward prices directly reflect current congestion expectations
- Nodal stitching gives near-universal coverage (vs 45-55% for path-level f0)
- 3-month averaging smooths monthly noise while capturing structural congestion

## Results vs v1 (H baseline)
- aq1: MAE 798 vs 934 (-14.6%), Dir% 80.9 vs 67.7 (+13.2pp)
- aq2: MAE 947 vs 1070 (-11.5%), Dir% 82.1 vs 69.0 (+13.1pp)
- aq3: MAE 797 vs 920 (-13.4%), Dir% 83.8 vs 69.1 (+14.7pp)
- aq4: MAE 704 vs 893 (-21.2%), Dir% 84.8 vs 64.3 (+20.5pp)

## Decision
Promoted as current best. Best MAE and Dir% across all 4 quarters with near-complete
coverage. Zero learned parameters — purely derived from market forward prices.
