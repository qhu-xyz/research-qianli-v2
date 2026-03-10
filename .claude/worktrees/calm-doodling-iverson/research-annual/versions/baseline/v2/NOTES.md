# v2

## What
Pure DA congestion without the 0.85 counter-flow shrinkage. Derived as `mtm_1st_mean / 0.85`.
Tests whether removing the production shrinkage factor improves predictions.

## Why
H (v1) applies 0.85 shrinkage which systematically underestimates congestion.
Removing it should reduce the persistent positive bias (+400-500).

## Results vs v1
- Bias reduced (less underestimation) since shrinkage removed
- Coverage identical (100%) since derived from same source

## Decision
Superseded by v3 (nodal f0). Pure DA is an incremental improvement over H but
nodal f0 provides fundamentally better direction accuracy.
