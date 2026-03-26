# v2

## What
Pure DA congestion without the 0.85 counter-flow shrinkage. Derived as `mtm_1st_mean / 0.85`.
Tests whether removing the production shrinkage factor improves predictions.

## Why
H (v1) applies 0.85 shrinkage which systematically underestimates congestion.
Removing it should reduce the persistent positive bias (+400-500).

## Results (corrected monthly units, 2026-03-14)
- aq1: MAE 432, Dir% 67.6%, P95 1,560
- aq2: MAE 445, Dir% 68.5%, P95 1,543
- aq3: MAE 362, Dir% 69.0%, P95 1,214
- aq4: MAE 449, Dir% 64.3%, P95 1,568

**Worse than v1.** Removing shrinkage increases MAE by 10-15%. The 0.85 factor
partially corrects for overestimation, and removing it amplifies errors.

*Previous results were inflated ~3x due to R1 mcp_mean unit bug.*

## Decision
Superseded by v3 (nodal f0). Pure DA without shrinkage is worse than with shrinkage.
