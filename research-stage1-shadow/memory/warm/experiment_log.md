# Experiment Log

> Previous experiments archived in memory/archive/feat-eng-20260303-060938/
> See memory/hot/learning.md for distilled learnings from 6 real-data experiments.

## E8: v0008 — Distribution Shape + Near-Boundary Band + Seasonal Historical Features (feat-eng-3-20260303-104101, iter1)

**Hypothesis**: H10 — Adding 7 features from distribution shape (density_mean, density_variance, density_entropy), near-boundary bands (tail_concentration, prob_band_95_100, prob_band_100_105), and seasonal historical signal (hist_da_max_season) will improve NDCG ranking quality while maintaining AUC/AP gains.

**Changes**: 19→26 features. All 7 from source loader. No HP, threshold, or window changes.

**Results (vs v0007 champion)**:

| Metric | v0008 | v0007 | Delta | W/L |
|--------|-------|-------|-------|-----|
| S1-AUC | 0.8498 | 0.8485 | +0.0013 | 8W/4L |
| S1-AP | 0.4418 | 0.4391 | +0.0027 | 9W/3L |
| S1-VCAP@100 | 0.0240 | 0.0247 | -0.0007 | 4W/8L |
| S1-NDCG | 0.7346 | 0.7333 | +0.0013 | 8W/4L |
| S1-BRIER | 0.1383 | 0.1395 | -0.0012 (better) | — |

**Bottom-2 Mean**:

| Metric | v0008 bot2 | v0007 bot2 | Delta | L3 Floor | Margin |
|--------|-----------|-----------|-------|----------|--------|
| S1-AUC | 0.8199 | 0.8188 | +0.0011 | 0.7988 | +0.0211 |
| S1-AP | 0.3726 | 0.3685 | +0.0041 | 0.3485 | +0.0241 |
| S1-VCAP@100 | 0.0061 | 0.0094 | -0.0033 | -0.0106 | +0.0167 |
| S1-NDCG | 0.6663 | 0.6562 | **+0.0101** | 0.6362 | **+0.0301** |

**Feature Importance (new features)**: prob_band_95_100 3.82%(#5), hist_da_max_season 2.60%(#7), prob_band_100_105 1.07%(#10), density_variance 0.91%(#12), density_mean 0.81%(#13), density_entropy 0.72%(#14), tail_concentration 0.37%(#16). Combined: 10.3%.

**Gate Status**: All Group A pass all 3 layers. NDCG bot2 margin expanded from 0.0046 to 0.0301 — tightest constraint relieved. CAP@100/500 (Group B) now below floors (0.7142 vs 0.7325 and 0.7175 vs 0.7240).

**Outcome**: **PROMOTED to champion.** H10 confirmed — distribution/band features improve NDCG ranking quality, specifically lifting the worst months.

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
