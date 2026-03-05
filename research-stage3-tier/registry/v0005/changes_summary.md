# v0005 Changes Summary

## Hypothesis Screening

### Winner: Hypothesis A (add 3 interaction features, 34 -> 37)

Hypothesis A adds 3 interaction features without pruning. Hypothesis B adds the same 3 but also prunes 5 low-importance features (34 -> 32).

### Screen Results (2 months: 2022-06, 2021-09)

| Metric | Screen A (37 feat) | Screen B (32 feat) |
|--------|--------------------|--------------------|
| **2022-06** | | |
| Tier-VC@100 | **0.0255** | 0.0112 |
| QWK | 0.270 | 0.274 |
| Tier0-AP | 0.106 | 0.100 |
| **2021-09** | | |
| Tier-VC@100 | 0.249 | **0.250** |
| QWK | 0.452 | 0.455 |
| Tier0-AP | 0.331 | 0.342 |
| **Mean VC@100** | **0.1372** | 0.1307 |

**Decision**: A wins on primary criterion (higher mean VC@100). Both pass QWK safety check (no drop > 0.05). A's advantage is driven by the weak month (2022-06) where it captured 2.3x more value.

## Code Changes

1. **`ml/features.py`** — Added 3 interaction feature computations to `compute_interaction_features()`:
   - `overload_x_hist = expected_overload * hist_da`
   - `prob110_x_recent_hist = prob_exceed_110 * recent_hist_da`
   - `tail_x_hist = tail_concentration * hist_da`

2. **`ml/config.py`** — Appended 3 features to `_ALL_TIER_FEATURES` and 3 monotone constraints (+1 each) to `_ALL_TIER_MONOTONE`. Feature count: 34 -> 37.

3. **`ml/tests/`** — Updated feature count assertions from 34 to 37 in test_config.py, test_data_loader.py, test_features.py.

## Full 12-Month Results (v0005)

| Month | Tier-VC@100 | Tier-VC@500 | Tier0-AP | Tier01-AP | Tier-NDCG | QWK |
|-------|------------|------------|---------|----------|----------|-----|
| 2020-09 | 0.1071 | 0.2992 | 0.6048 | 0.3693 | 0.8586 | 0.4413 |
| 2020-11 | 0.1284 | 0.3288 | 0.4499 | 0.3264 | 0.8238 | 0.4330 |
| 2021-01 | 0.0615 | 0.1797 | 0.2503 | 0.4025 | 0.7637 | 0.3730 |
| 2021-03 | 0.0492 | 0.2524 | 0.4850 | 0.4029 | 0.7928 | 0.4595 |
| 2021-05 | 0.0314 | 0.2165 | 0.2498 | 0.3107 | 0.7730 | 0.3323 |
| 2021-07 | 0.0451 | 0.2924 | 0.3613 | 0.3466 | 0.7392 | 0.3351 |
| 2021-09 | 0.2489 | 0.3757 | 0.3310 | 0.3472 | 0.8520 | 0.4520 |
| 2021-11 | 0.0082 | 0.1790 | 0.2829 | 0.2929 | 0.7692 | 0.2519 |
| 2022-03 | 0.1202 | 0.3401 | 0.3293 | 0.3334 | 0.8511 | 0.3903 |
| 2022-06 | 0.0255 | 0.0484 | 0.1059 | 0.1949 | 0.6361 | 0.2703 |
| 2022-09 | 0.0288 | 0.0597 | 0.1169 | 0.2284 | 0.7168 | 0.3840 |
| 2022-12 | 0.0407 | 0.2233 | 0.1838 | 0.2027 | 0.7245 | 0.3242 |

### Aggregate Comparison (v0005 vs v0 champion)

| Metric | v0005 Mean | v0 Mean | Delta |
|--------|-----------|---------|-------|
| Tier-VC@100 | 0.0746 | 0.0708 | +0.0038 (+5.4%) |
| Tier-VC@500 | 0.2330 | 0.2296 | +0.0034 (+1.5%) |
| Tier0-AP | 0.3126 | 0.3062 | +0.0064 (+2.1%) |
| Tier01-AP | 0.3131 | 0.3110 | +0.0021 (+0.7%) |
| Tier-NDCG | 0.7751 | 0.7711 | +0.0040 (+0.5%) |
| QWK | 0.3706 | 0.3698 | +0.0008 (+0.2%) |

### Key Observations

- VC@100 improved from 0.0708 to 0.0746 (+5.4%), moving toward the 0.075 floor
- Tier0-AP improved from 0.3062 to 0.3126 (+2.1%)
- All metrics improved or held steady — no regressions
- The interaction features help most on months with moderate binding (2020-09, 2020-11, 2022-03)
- Worst months (2021-11, 2022-06) remain weak — these likely need class weight or hyperparameter changes beyond FE scope
