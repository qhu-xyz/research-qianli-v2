# v0002 Changes Summary — Interaction Features (iter1, hp-tune-20260302-144146)

## Hypothesis

**H4: Interaction features provide new discriminative signal for ranking quality.**

Previous iteration (v0003) proved HP tuning alone cannot improve ranking metrics — deeper trees degraded AUC in 11/12 months. The 14 independent features have reached their informational ceiling. Pre-computed interaction features should provide new signal that requires fewer tree splits to exploit.

## Changes Made

### 1. Reverted HyperparamConfig to v0 defaults (ml/config.py)
| Param | v0003 (reverted from) | v0 (reverted to) |
|-------|----------------------|-------------------|
| n_estimators | 400 | **200** |
| max_depth | 6 | **4** |
| learning_rate | 0.05 | **0.1** |
| min_child_weight | 5 | **10** |

This isolates the feature effect from HP changes.

### 2. Added 3 interaction features (ml/config.py → FeatureConfig)
Total features: 14 → 17. All 3 new features have monotone constraint +1.

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `exceed_severity_ratio` | prob_exceed_110 / (prob_exceed_90 + 1e-6) | Tail concentration of exceedance |
| `hist_physical_interaction` | hist_da × prob_exceed_100 | Historical × physical confirmation |
| `overload_exceedance_product` | expected_overload × prob_exceed_105 | Severity-weighted likelihood |

### 3. Computed interactions in prepare_features (ml/features.py)
Added `df.with_columns(...)` block before `df.select(cols)` to compute the 3 interaction features from base columns. No changes to data_loader.py — interactions computed at feature-prep time.

### 4. Updated tests (ml/tests/)
- `conftest.py`: synthetic_features fixture 14→17 columns
- `test_config.py`: feature count 14→17, monotone constraints string, feature names list, HP defaults
- `test_features.py`: shape assertions 14→17, use base features for DataFrame creation

## Results (12 months, f0, onpeak)

### Aggregate Comparison (v0002 vs v0)
| Metric | v0 | v0002 | Delta | Direction |
|--------|-----|-------|-------|-----------|
| S1-AUC | 0.8348 | 0.8348 | +0.0000 | Neutral |
| S1-AP | 0.3936 | 0.3946 | +0.0010 | Slight improvement |
| S1-NDCG | 0.7333 | 0.7349 | +0.0016 | Slight improvement |
| S1-VCAP@100 | 0.0149 | 0.0158 | +0.0009 | Slight improvement |
| S1-BRIER | 0.1503 | 0.1505 | +0.0002 | Neutral (within noise) |

### Per-Month Win/Loss (v0002 vs v0)
| Metric | Wins | Losses | Ties |
|--------|------|--------|------|
| S1-AUC | 5 | 6 | 1 |
| S1-AP | 7 | 5 | 0 |
| S1-NDCG | 8 | 4 | 0 |

### Per-Month AUC Detail
| Month | v0 | v0002 | Delta |
|-------|-----|-------|-------|
| 2020-09 | 0.8434 | 0.8441 | +0.0007 |
| 2020-11 | 0.8300 | 0.8300 | 0.0000 |
| 2021-01 | 0.8555 | 0.8561 | +0.0006 |
| 2021-04 | 0.8353 | 0.8349 | -0.0004 |
| 2021-06 | 0.8246 | 0.8239 | -0.0007 |
| 2021-08 | 0.8532 | 0.8535 | +0.0003 |
| 2021-10 | 0.8507 | 0.8502 | -0.0005 |
| 2021-12 | 0.8123 | 0.8117 | -0.0006 |
| 2022-03 | 0.8446 | 0.8442 | -0.0004 |
| 2022-06 | 0.8258 | 0.8257 | -0.0001 |
| 2022-09 | 0.8334 | 0.8338 | +0.0004 |
| 2022-12 | 0.8088 | 0.8093 | +0.0005 |

### Per-Month AP Detail
| Month | v0 | v0002 | Delta |
|-------|-----|-------|-------|
| 2020-09 | 0.3866 | 0.3922 | +0.0056 |
| 2020-11 | 0.4330 | 0.4344 | +0.0014 |
| 2021-01 | 0.4442 | 0.4502 | +0.0060 |
| 2021-04 | 0.4198 | 0.4247 | +0.0049 |
| 2021-06 | 0.3494 | 0.3466 | -0.0028 |
| 2021-08 | 0.3959 | 0.3956 | -0.0003 |
| 2021-10 | 0.4439 | 0.4441 | +0.0002 |
| 2021-12 | 0.4253 | 0.4239 | -0.0014 |
| 2022-03 | 0.3625 | 0.3577 | -0.0048 |
| 2022-06 | 0.3850 | 0.3861 | +0.0011 |
| 2022-09 | 0.3150 | 0.3143 | -0.0007 |
| 2022-12 | 0.3623 | 0.3654 | +0.0031 |

### Tail Safety
- Bottom-2 AUC: v0=0.8105, v0002=0.8105 (identical)
- Bottom-2 AP: v0=0.3322, v0002=0.3305 (slight decrease, still well above floor)

### Gate Status
All Group A and Group B gates pass all 3 layers (mean, tail, tail regression). No gate failures.

## Interpretation

The interaction features produced **marginal, mixed results**:

1. **AUC (5W/6L/1T)**: Essentially neutral. The interaction features did not provide meaningful new discriminative signal for overall separation. The model's AUC is bounded by the information content of the physical flow features, and interactions of these features don't substantially increase that information.

2. **AP (7W/5L)**: Slightly positive. The interaction features may provide a small benefit for precision-recall ranking in the positive class, particularly in earlier months (2020-09 through 2021-04, all wins). The benefit fades in later months (2022-03, 2022-09 losses).

3. **NDCG (8W/4L)**: Most positive signal. The interaction features improve ranking quality at top positions, with the largest improvement at 2021-01 (NDCG 0.764→0.806, +0.042). This suggests `hist_physical_interaction` and `overload_exceedance_product` help rank high-confidence predictions.

4. **VCAP@100 (+0.0009)**: Marginal improvement in value capture at top-100, consistent with NDCG gains.

5. **BRIER (+0.0002)**: Neutral as expected — interaction features affect ranking, not calibration.

## Key Learnings

- **The model is strongly feature-limited** — even physically-motivated interaction features don't substantially move AUC. The 14 base features from probabilistic power flow capture most of the available signal.
- **XGBoost with depth=4 can already discover most useful interactions** — the pre-computed features save tree depth but don't unlock fundamentally new patterns.
- **Interaction features help ranking (NDCG) more than discrimination (AUC)** — this is consistent with them improving the ordering of already-positive predictions rather than improving the binary separation boundary.
- **The late-2022 weakness persists** (2022-09 AP=0.314, 2022-12 AUC=0.809) — neither HP tuning nor interaction features help. This is likely a true distribution shift requiring temporal/seasonal features or expanded training windows.
