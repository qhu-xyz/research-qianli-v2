# Direction — Iteration 2 (feat-eng-2-20260303-092848, v0007 as champion)

## Hypothesis H10: Distribution Shape + Probability Band Features with NDCG-Targeted Monotone Tuning

**Hypothesis**: Adding 6 distribution/band features (density_mean, density_variance, density_entropy, tail_concentration, prob_band_95_100, prob_band_100_105) AND tuning monotone constraints on v0007's unconstrained features will achieve AUC ≥ 0.845 while recovering NDCG bot2 above 0.670. The distribution features capture WHERE in the flow distribution mass sits (central tendency, spread, uncertainty, tail shape) — complementary to the exceedance probabilities that only measure cumulative tail mass. The monotone tuning targets the NDCG regression observed in v0007.

**Evidence supporting this hypothesis**:
1. v0007 confirmed the model is feature-starved: +0.0137 AUC from 6 features with only 4.66% gain. More features from different signal classes should continue to help.
2. direction_iter1.md pre-planned this exact feature set for iter 2 if H9 succeeded (AUC > 0.837).
3. v0006 showed that removing all unconstrained features (monotone=0) improved NDCG +0.0227 — monotone constraint structure directly affects NDCG. v0007 added 4 new unconstrained features and NDCG went flat. Selectively constraining some of these may recover NDCG.
4. density_skewness, density_kurtosis, density_cv were pruned in v0006 as noise (1.3% combined gain). The new distribution features (density_mean, density_variance, density_entropy) measure different properties and may contain more signal.
5. Probability band features (prob_band_95_100, prob_band_100_105) capture localized flow density that existing exceedance features don't.

## Specific Changes

### 1. Add 6 new features to `ml/config.py` — FeatureConfig

Add to `step1_features` (19 → 25 total):

```python
# --- NEW: Distribution shape features ---
("density_mean", 0),          # expected flow point (unconstrained — direction unclear)
("density_variance", 0),      # flow uncertainty (unconstrained — high variance could mean more OR less binding)
("density_entropy", 0),       # information content (unconstrained)
# --- NEW: Probability band + tail features ---
("tail_concentration", 1),    # prob_exceed_100 / prob_exceed_80 — higher = more peaked tail = more likely to bind
("prob_band_95_100", 1),      # near-binding band probability — higher = more flow near limit = more likely to bind
("prob_band_100_105", 1),     # mild-overload band — higher = flow already exceeding = more likely to bind
```

**Monotone constraint design (NDCG-targeted)**:
- `density_mean`, `density_variance`, `density_entropy`: monotone=0 (direction genuinely unclear)
- `tail_concentration`: monotone=1 (more concentrated tail → more binding)
- `prob_band_95_100`: monotone=1 (more flow near limit → more binding)
- `prob_band_100_105`: monotone=1 (flow exceeding limit → more binding)

**Why 3 constrained + 3 unconstrained**: v0006 showed full monotone enforcement improves NDCG. v0007 has 8 unconstrained features out of 19 (sf_std, sf_nonzero_frac, is_interface, constraint_limit, hist_physical_interaction, overload_exceedance_product, plus the 2 interaction features). Adding 3 more unconstrained and 3 constrained keeps the ratio manageable while testing whether band features with monotone=1 help NDCG.

### 2. ALSO: Tune existing unconstrained features

**Change `sf_nonzero_frac` from monotone=0 to monotone=1.** Rationale: A constraint that affects more nodes (higher sf_nonzero_frac) has broader network reach and is more likely to bind. The v0007 result shows sf_nonzero_frac at 0.54% importance — it's weak, but enforcing monotone=1 may help NDCG without hurting AUC (v0006 showed that monotone enforcement helps ranking consistency).

**Change `constraint_limit` from monotone=0 to monotone=1.** Rationale: Larger constraints (higher MW limits) are typically high-voltage transmission lines that carry more flow and are more likely to bind under stress. Both Claude and Codex suggested testing this in their reviews.

This reduces unconstrained features from 8/25 to 9/25 (adding 3 new unconstrained, constraining 2 existing). Net effect: slightly more monotone enforcement.

### 3. Update `ml/data_loader.py` — _load_smoke()

Add synthetic values for the 6 new features:
```python
# Distribution shape features
for feat in ["density_mean", "density_variance", "density_entropy"]:
    if feat == "density_mean":
        data[feat] = rng.uniform(0.5, 1.2, n).tolist()  # flow as fraction of limit
    elif feat == "density_variance":
        data[feat] = np.abs(rng.randn(n) * 0.1).tolist()  # small positive variance
    else:  # entropy
        data[feat] = rng.uniform(1.0, 5.0, n).tolist()  # entropy units

# Probability band + tail features
for feat in ["tail_concentration", "prob_band_95_100", "prob_band_100_105"]:
    data[feat] = rng.uniform(0, 0.3, n).tolist()  # probabilities bounded [0,1], typically small
```

### 4. Update tests

In `ml/tests/test_config.py`:
- Update feature count: 19 → 25
- Update monotone constraint string
- Update expected feature names list
- sf_nonzero_frac constraint: 0 → 1
- constraint_limit constraint: 0 → 1

### 5. Do NOT change

- `train_months`: keep at 14 (HARD MAX)
- `threshold_beta`: keep at 0.7
- HPs: keep v0 defaults
- `evaluate.py`: do NOT modify
- `gates.json`: do NOT modify
- Existing 19 features: keep all (do NOT remove any)

## Expected Impact

| Metric | Expected Direction | Reasoning |
|--------|-------------------|-----------|
| S1-AUC | **maintain ≥ 0.845** | More features should maintain or slightly improve AUC. v0007's 0.849 may not be exceeded but should be preserved. |
| S1-AP | **maintain ≥ 0.430** | Additional distribution features add positive-class discrimination signal. |
| S1-VCAP@100 | **+0.002 to +0.010** | Band features may help rank top constraints more precisely. |
| S1-NDCG | **+0.005 to +0.020** | KEY TARGET. Monotone constraint tuning (sf_nonzero_frac, constraint_limit → monotone=1) combined with 3 new constrained features should recover NDCG. |
| S1-BRIER | **neutral** | More features may slightly worsen calibration, but v0007 showed topology features help BRIER. |

## Success Criteria

- **Strong success**: AUC ≥ 0.845 AND NDCG bot2 ≥ 0.670 (recovers above v0 level) AND AP ≥ 0.430
- **Moderate success**: AUC ≥ 0.840 AND NDCG ≥ 0.735 (slight improvement) AND AP ≥ 0.420
- **Neutral**: AUC/AP maintained, NDCG unchanged — features added noise but didn't help or hurt
- **Failure**: AUC < 0.840 OR NDCG bot2 < 0.650 (breaches L3 concern zone) OR AP < 0.400

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Distribution features are noise (density_skewness/kurtosis/cv were pruned as noise in v0006) | MEDIUM | New features (mean, variance, entropy) measure different properties. Even if weak, monotone=0 won't hurt AUC. |
| Monotone constraint changes on sf_nonzero_frac/constraint_limit hurt AUC | LOW | Both features have <1% importance. Constraint direction is physically plausible. |
| 25 features is too many (overfitting) | LOW | XGBoost with colsample_bytree=0.8 and subsample=0.8 handles feature count well. v0007 showed 19 features improved over 13. |
| NDCG doesn't recover despite monotone tuning | MEDIUM | If NDCG stays flat, iter 3 would try removing all unconstrained features except the ones with clear signal, or test LambdaRank objective. |
| New features not in source loader output | LOW | All 6 are verified computed in source loader (shadow_price_prediction/data_loader.py lines 392-475). |

## Layer 3 Constraints (v0007 as champion)

Future version must satisfy:
- AUC bot2 ≥ 0.7988 (margin: +0.020 from v0007)
- AP bot2 ≥ 0.3485 (margin: +0.020 from v0007)
- VCAP@100 bot2 ≥ -0.0106 (margin: +0.020 from v0007)
- **NDCG bot2 ≥ 0.6362** (margin: only 0.0200 from v0007's 0.6562 — this is the binding constraint)

The NDCG L3 threshold at 0.6362 means iter 2 cannot regress NDCG bot2 below this. v0's NDCG bot2 was 0.6716, so there is a 0.0354 gap between v0 level and the L3 floor. The goal is to push NDCG bot2 back toward or above 0.670.

## Connection to Iteration 3

- **If H10 succeeds (NDCG recovers, AUC maintained)**: Iter 3 tries HP tuning on the 25-feature base (max_depth=5 or n_estimators=300) to exploit the richer feature space
- **If H10 is partial (AUC maintained, NDCG unchanged)**: Iter 3 tries aggressive monotone enforcement (set ALL features to monotone ≠ 0) to replicate v0006's NDCG success on the larger feature set
- **If H10 fails (AUC regresses below 0.840)**: Iter 3 reverts to v0007 features and tries different approach (LambdaRank objective or selective feature removal)
