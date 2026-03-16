# Research Annual Band — Multi-RTO

## Structure

```
miso/           # MISO annual band research (baseline + banding complete)
pjm/            # PJM annual band research (baseline complete, banding in progress)
```

## MISO Status
- V10 empirical asymmetric bands: dev + holdout validated
- Pre-port validation: 6/6 steps passed
- Remaining: production module not built, E2E test not committed as reusable script
- Baseline: nodal_f0 × 3 (R1), mtm_1st_mean × 3 (R2/R3). Quarterly scale.

## PJM Status
- Baseline research complete: `mtm_1st_mean` for all 4 rounds (exhaustive search, can't improve >3%)
- Preliminary V1 band calibration: metrics saved for R1-R4
- No scripts yet — using MISO functions directly from inline code
- Next: write PJM band script, holdout validation, comprehensive report

## Notes
- No common/shared module exists yet. Interface will emerge from PJM banding implementation.
- MISO scripts under miso/scripts/ have stale docstring paths (execution still works via relative ROOT).
