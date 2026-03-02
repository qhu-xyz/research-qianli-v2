# Learning

## From Infrastructure Validation (smoke-v6, smoke-v7 — synthetic n=20 data)

### Pipeline
- Pipeline is fully deterministic: seed=42 for data + XGBoost produces bit-for-bit identical metrics across runs
- Version registry structure works: config.json, metrics.json, meta.json, changes_summary.md, comparison.md, model/classifier.ubj.gz

### F-beta Formula (CRITICAL — got this wrong once)
- **beta < 1 → weights PRECISION more** (higher threshold, fewer positives)
- **beta > 1 → weights RECALL more** (lower threshold, more positives)
- **beta = 1 → standard F1** (equal weighting)
- F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
- Business objective is precision > recall, so keep beta <= 1.0

### Code Quality
- `apply_threshold` uses strict `>` while `precision_recall_curve` thresholds are inclusive `>=` — known mismatch, minor impact at real data scale (270K rows) vs significant at n=20

### Reviewer Dynamics
- Codex found deeper structural bugs; Claude had better practical gate analysis
- Reading reviews independently is valuable — they catch different things

## From v0 Real-Data Baseline (12 months, f0, onpeak)

### Key Numbers
- AUC=0.835 (std=0.015) — strong, stable ranking
- AP=0.394 (std=0.041) — moderate, room for improvement
- VCAP@100=0.015 (std=0.012) — very low value capture at top-100
- Precision=0.442 — when predicting bind, right 44% of the time
- Threshold=0.834 — conservative, precision-favoring (beta=0.7)

### Weakest Months
- 2022-09: AP=0.315, AUC=0.833 — worst AP
- 2022-12: AUC=0.809, AP=0.362 — worst AUC
- 2022-06: REC=0.313 — worst recall
- Late 2022 consistently weaker — possible distribution shift

### Gate Headroom
- Group A gates all pass with ~0.05 headroom from mean to floor
- S1-BRIER (Group B) tightest: only 0.02 headroom (floor=0.170 vs mean=0.150)
- S1-VCAP@100 has negative floor (-0.035) — effectively non-binding

## From v0003 — HP Tuning Experiment (iter1, real data)

### Model Complexity vs Feature Informativeness
- **v0 HP defaults are near-optimal** — standard XGBoost tuning (depth 4→6, lr 0.1→0.05, trees 200→400, min_child_weight 10→5) produced zero improvement on Group A ranking metrics
- **Model is feature-limited, not complexity-limited** — deeper trees cannot extract more discriminative signal from these 14 features
- AUC degraded in 11/12 months (p≈0.003) — a real, systematic effect, not noise
- BRIER improved in 12/12 months (p≈0.0002) — deeper/slower trees improve calibration but hurt discrimination
- This means: the probability estimates become better-calibrated but the ranking order gets marginally worse

### Calibration vs Discrimination Tradeoff
- Deeper trees + lower learning rate → better calibrated probabilities (lower Brier) but slightly worse separation (lower AUC)
- For our business objective (precision at high threshold), ranking quality (AUC/AP) matters more than calibration (BRIER)
- This tradeoff means HP tuning alone cannot simultaneously improve both — need new signal (features)

### Late-2022 Distribution Shift
- Weakest months (2022-09, 2022-12) remain equally weak in v0003 — tree complexity doesn't help
- The shift may require temporal features (season, trend) or expanded training window to address

### Statistical Testing Insight
- Month-level win/loss counts (0W/11L for AUC) are more informative than mean deltas (Δ=-0.0025)
- A delta within noise tolerance can still be statistically significant when directionally consistent across months
- Future iterations should track win/loss counts alongside mean deltas

## From v0002 — Interaction Features (iter1, hp-tune-20260302-144146)

### Feature Interactions Don't Break the AUC Ceiling
- 3 physically-motivated interaction features (exceed_severity_ratio, hist_physical_interaction, overload_exceedance_product) produced AUC +0.0000 (5W/6L/1T)
- XGBoost depth-4 already discovers most useful 2-feature interactions — pre-computing saves 1 split but adds no new information
- AP +0.0010 (7W/5L) and NDCG +0.0016 (8W/4L) are marginal and noise-level
- The AUC ceiling at ~0.835 is now **confirmed across two independent levers** (HP tuning AND interactions)

### Interaction Features Help Top-K But Hurt Broader Ranking
- VCAP@100 +0.0009 (slight improvement at top-100)
- VCAP@500 -0.0043 and VCAP@1000 -0.0031 (regression at broader K)
- Interactions act as "confidence boosters" for extreme cases but add noise to borderline predictions
- Bottom-2 regressed on 3/4 Group A metrics — gains concentrated in distribution middle, not tails

### Temporal Pattern in Feature Effectiveness
- Early months (2020-09 through 2021-04) consistently benefit from interactions
- Late months (2022-03, 2022-09, 2022-12) neutral or negative
- This confirms the **late-2022 distribution shift** as the dominant remaining problem
- Feature engineering within the current feature set cannot address temporal non-stationarity

### Cumulative Evidence — What Works and What Doesn't
| Lever | Result | Conclusion |
|-------|--------|------------|
| HP tuning (v0003) | AUC -0.0025, 0W/11L | Wrong lever — model not complexity-limited |
| Interaction features (v0002) | AUC +0.0000, 5W/6L/1T | Wrong lever — information ceiling reached |
| **Next**: Longer training window | Untested | Addresses distribution shift directly |

### Outlier Sensitivity
- Always check for single-month outliers driving mean improvements
- 2021-01 NDCG outlier (+0.042) accounts for most of v0002 NDCG mean gain; without it, delta is ~-0.002
- Report "mean excluding best month" alongside raw mean for honest assessment
