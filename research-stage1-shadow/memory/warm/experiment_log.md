# Experiment Log

> Previous experiments archived in memory/archive/feat-eng-20260303-060938/
> See memory/hot/learning.md for distilled learnings from 6 real-data experiments.

## E7: v0007 — Shift Factor + Constraint Metadata Features (feat-eng-2-20260303-092848, iter1)

**Hypothesis**: H9 — Adding 6 new features from entirely new signal categories (network topology via shift factors + constraint structural metadata) will break the AUC ceiling at ~0.836.

**Changes**: 13→19 features. Added sf_max_abs(+1), sf_mean_abs(+1), sf_std(0), sf_nonzero_frac(0), is_interface(0), constraint_limit(0). All other settings unchanged (14mo window, v0 HPs, beta=0.7).

**Results (vs v0 baseline)**:

| Metric | v0007 | v0 | Delta | W/L | p (sign) |
|--------|-------|-----|-------|-----|----------|
| S1-AUC | 0.8485 | 0.8348 | **+0.0137** | **12W/0L** | 0.0002 |
| S1-AP | 0.4391 | 0.3936 | **+0.0455** | **11W/1L** | 0.006 |
| S1-VCAP@100 | 0.0247 | 0.0149 | +0.0098 | 9W/3L | 0.073 |
| S1-NDCG | 0.7333 | 0.7333 | +0.0000 | 5W/7L | 0.387 |
| S1-BRIER | 0.1395 | 0.1503 | **-0.0108** | — | — |

**Bottom-2 Mean**:

| Metric | v0007 bot2 | v0 bot2 | Delta |
|--------|-----------|---------|-------|
| S1-AUC | 0.8188 | 0.8105 | +0.0083 |
| S1-AP | 0.3685 | 0.3322 | **+0.0363** |
| S1-VCAP@100 | 0.0094 | 0.0014 | +0.0080 |
| S1-NDCG | 0.6562 | 0.6716 | **-0.0154** |

**Feature Importance (new features)**: sf_max_abs 1.20%(#11), sf_std 1.05%(#14), constraint_limit 0.98%(#15), sf_mean_abs 0.60%(#17), sf_nonzero_frac 0.54%(#18), is_interface 0.29%(#19). Combined: 4.66%.

**Gate Status**: All Group A pass all 3 layers. NDCG bot2 margin to L3 tolerance: only 0.0046. CAP@100/500 (Group B) very close to floors (0.002 and 0.004 headroom).

**Outcome**: **PROMOTED to champion.** H9 strongly confirmed. First version to satisfy all human-input success criteria (AUC>0.840, AP>0.400, 8+/12 wins). Largest single-experiment improvement in pipeline history.
