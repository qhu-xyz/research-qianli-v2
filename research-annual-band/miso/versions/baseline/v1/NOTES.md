# v1

## What
Production reference baseline. H = 0.85 * DA congestion with counter-flow shrinkage.
This is what ftr23/v8 uses for R1 annual FTR auctions. 100% coverage.

## Why
Established production baseline. Serves as the reference point for all improvements.

## Results (corrected monthly units, 2026-03-14)
- aq1: MAE 380, Dir% 67.6%, P95 1,376
- aq2: MAE 390, Dir% 68.5%, P95 1,365
- aq3: MAE 316, Dir% 69.0%, P95 1,059
- aq4: MAE 390, Dir% 64.3%, P95 1,367

*Previous results (MAE 934/1070/920/893) were inflated ~3x due to R1 mcp_mean unit bug.*

## Decision
Superseded by v3 (nodal f0). H has worst direction accuracy of all tested baselines.
