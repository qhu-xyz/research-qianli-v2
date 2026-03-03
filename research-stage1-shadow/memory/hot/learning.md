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
- Codex finds deeper structural bugs; Claude has better statistical/practical gate analysis
- Reading reviews independently is valuable — they catch different things

## From v0 Real-Data Baseline (12 months, f0, onpeak)

### Key Numbers
- AUC=0.835 (std=0.015) — strong, stable ranking
- AP=0.394 (std=0.041) — moderate, room for improvement
- VCAP@100=0.015 (std=0.012) — very low value capture at top-100
- Precision=0.442 — when predicting bind, right 44% of the time
- Threshold=0.834 — conservative, precision-favoring (beta=0.7)

### Weakest Months
- 2022-09: AP=0.315, AUC=0.833 — worst AP, lowest binding rate (6.63%)
- 2022-12: AUC=0.809, AP=0.362 — worst AUC
- 2022-06: REC=0.313 — worst recall
- Late 2022 consistently weaker — possible distribution shift

### Gate Headroom
- Group A gates all pass with ~0.05 headroom from mean to floor
- S1-BRIER (Group B) tightest: only 0.02 headroom (floor=0.170 vs mean=0.150)
- S1-VCAP@100 has negative floor (-0.035) — effectively non-binding

## From 6 Prior Real-Data Experiments (v0002-v0006)

### Model Is Feature-Limited, Not Complexity-Limited
- HP tuning produced zero Group A improvement (v0003-HP: 0W/11L AUC)
- Interaction features provided no AUC lift (v0002: 5W/6L/1T)
- Window expansion 10→14 gave +0.0013 AUC; 14→18 gave nothing
- Feature pruning was a tradeoff (NDCG up, AP down)
- AUC operating range across all 6 experiments: [0.832, 0.836] — 0.004 span

### Feature Importance Structure
- **79% historical**: hist_da_trend (54%), hist_physical_interaction (14%), hist_da (11%)
- **18% physical flow**: prob_below_90, prob_exceed_90/95/100
- **1.3% distribution shape**: skewness, kurtosis, cv — noise candidates

### Monotone Constraints Are Structural
- Full monotone enforcement sharpens NDCG/VCAP but degrades AP
- This is a model behavior change, not noise removal
- Removing unconstrained features changes the optimization landscape

### AP Bot2 Was Monotonically Declining
- 6-experiment trend: -0.0017 → -0.0094 vs v0
- Every modification worsened AP in the weakest months
- v0 feature set was coincidentally well-calibrated for worst-month AP

### 2022-09 Was Structurally Broken
- 5 independent interventions failed to improve AP (stayed at ~0.298-0.315)
- Lowest binding rate (6.63%) makes class separation fundamentally harder

## From v0007 — Shift Factor + Constraint Metadata Features (NEW CHAMPION)

### Network Topology Breaks the Ceiling (CRITICAL — Biggest Finding)
- **AUC: 0.8348 → 0.8485 (+0.0137, 12W/0L, p≈0.0002)** — first 12/12 result ever
- **AP: 0.3936 → 0.4391 (+0.0455, 11W/1L, p≈0.006)** — 3x largest AP delta
- 6 new features from entirely new signal classes (shift factors + constraint metadata) broke the AUC ceiling that held across 6 prior experiments
- Confirms the feature-starvation hypothesis beyond expectations

### Low Importance ≠ Low Value
- 6 new features contribute only 4.66% combined gain (training-loss metric)
- Yet AUC improved by +0.0137 — largest single-experiment improvement
- These act as "auxiliary discriminators" — useful at the margins where existing features are ambiguous
- **Do not prune based on training-loss importance alone** — generalization value can far exceed it

### AP Bot2 Trend Reversed
- From 6-experiment monotonic decline (ending at -0.0094 vs v0) to +0.0363
- The new features help the model maintain ranking quality in the weakest months
- The prior decline was not structural — it was a symptom of insufficient feature diversity

### 2022-09 Finally Improved
- AUC: 0.833 → 0.853 (+0.019), AP: 0.315 → 0.347 (+0.032)
- The constraint's network position (shift factors) helps discriminate when flow-based features are ambiguous in low-binding-rate periods
- After 5 failed interventions, fundamentally new signal was the answer

### BRIER Improved Unexpectedly
- 0.1503 → 0.1395 (-0.0108), reversing 6-experiment narrowing trend
- Topology features improve both discrimination AND calibration — a rare combination

### NDCG Was the Constraint — Now Addressed by v0008
- v0007: Mean 0.7333, 5W/7L, bot2 margin only 0.0046
- v0008: Mean 0.7346 (+0.0013), 8W/4L, bot2 margin expanded to **0.0301**
- Near-boundary band features (prob_band_95_100, prob_band_100_105) discriminate binding intensity → NDCG-targeted approach worked
- Spring transition months (2021-04, 2022-03) remain structurally weak but improved
- **VCAP@100 is now the closest L3 risk** (margin +0.0167, 4W/8L)

### CAP@100/500 Degraded
- CAP@100: 0.7825 → 0.7342 (-0.0483), headroom 0.002 from Group B floor
- CAP@500: 0.7740 → 0.7280 (-0.0460), headroom 0.004 from Group B floor
- Higher threshold (0.851 vs 0.834) reduces predicted positive count, hurting CAP
- Model profile shifted from threshold-dependent capture to ranking quality — an acceptable trade for business objective

## From v0008 — Distribution Shape + Near-Boundary Band + Seasonal Historical Features (NEW CHAMPION)

### NDCG-Targeted Feature Design Works
- 7 new features chosen specifically to improve ranking quality (NDCG bot2)
- NDCG bot2: 0.6562 → 0.6663 (+0.0101) — lifted both worst months simultaneously
- prob_band_95_100 (#5 importance, 3.82%) — near-binding mass is a powerful NDCG discriminator
- Bot2 improvement NOT driven by mean — it's driven by lifting the tails

### Additive Feature Engineering: Diminishing Returns
- v0007: 6 features, 4.66% importance → AUC +0.0137 (12W/0L)
- v0008: 7 features, 10.3% importance → AUC +0.0013 (8W/4L)
- **Higher feature importance does NOT mean proportionally higher metric lift**
- Each marginal feature category adds less new information
- Future gains likely require: (a) interactions between existing features, (b) regularization tuning, (c) fundamentally new signal sources

### VCAP@100 Dilution Risk from Feature Expansion
- VCAP@100: 4W/8L vs champion, bot2 -0.0033
- More features → importance spread → weaker concentration at the very top of rankings
- colsample_bytree=0.8 with 26 features means only ~21 features per tree — critical features for top-100 ranking may be randomly excluded
- **Consider colsample_bytree increase for feature-rich models**

### Spring Transition Months Are Structurally Harder
- 2021-04 (NDCG 0.651) and 2022-03 (NDCG 0.682) consistently worst across all versions
- These are spring transition periods — heating/cooling load profile shifts
- v0008 improved 2022-03 by +0.018 but 2021-04 only by +0.003
- May need season-specific features or temporal conditioning

### Cumulative Evidence (8 experiments)

| Lever | AUC Δ vs v0 | AUC W/L | Promoted |
|-------|-------------|---------|----------|
| HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | No |
| Combined (v0004) | +0.0015 | 9W/3L | No |
| Window 14→18 (v0005) | +0.0013 | 7W/5L | No |
| Feature pruning (v0006) | +0.0006 | 5W/7L | No |
| **SF + metadata (v0007)** | **+0.0137** | **12W/0L** | **YES** |
| **Distrib + band (v0008)** | **+0.0150** | **8W/4L** | **YES** |
