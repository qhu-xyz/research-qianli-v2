# v1

## What
Production reference baseline. H = 0.85 * DA congestion with counter-flow shrinkage.
This is what ftr23/v8 uses for R1 annual FTR auctions. 100% coverage.

## Why
Established production baseline. Serves as the reference point for all improvements.

## Results
- aq1: MAE 934, Dir% 67.7%
- aq2: MAE 1070, Dir% 69.0%
- aq3: MAE 920, Dir% 69.1%
- aq4: MAE 893, Dir% 64.3%

## Decision
Superseded by v3 (nodal f0). H has worst direction accuracy of all tested baselines.
