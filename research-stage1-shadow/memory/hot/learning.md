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

## From v0003-HP — HP Tuning Experiment (hp-tune batch, real data)

### Model Complexity vs Feature Informativeness
- **v0 HP defaults are near-optimal** — standard XGBoost tuning produced zero Group A improvement
- **Model is feature-limited, not complexity-limited** — deeper trees cannot extract more signal from these 14 features
- AUC degraded in 11/12 months (p≈0.003) — a real, systematic effect
- BRIER improved in 12/12 months (p≈0.0002) — deeper/slower trees improve calibration but hurt discrimination

### Calibration vs Discrimination Tradeoff
- Deeper trees + lower learning rate → better calibrated probabilities (lower Brier) but slightly worse separation (lower AUC)
- For our business objective (precision at high threshold), ranking quality (AUC/AP) matters more than calibration (BRIER)

## From v0002 — Interaction Features (hp-tune batch, real data)

### Feature Interactions Don't Break the AUC Ceiling
- 3 interaction features produced AUC +0.0000 (5W/6L/1T)
- NDCG 8W/4L and AP 7W/5L: marginal positive ranking signal
- XGBoost depth-4 already discovers most useful interactions
- Interaction features help top-K but hurt broader ranking (VCAP@500 -0.0043)
- Bottom-2 regressed on 3/4 Group A metrics

### Temporal Pattern in Feature Effectiveness
- Early months (2020–2021H1) benefit from changes more than late months (2022)
- Confirms late-2022 distribution shift as the dominant remaining problem

## From v0003 — Training Window Expansion (feat-eng batch, real data)

### Window Expansion Provides Small, Real Signal
- 14-month window is the **first lever to produce positive AUC wins** (7W/4L/1T, up from 5W interactions and 0W HP tuning)
- Effect is small (+0.0013 AUC, not statistically significant at p≈0.27) but distributed (not single-outlier driven)
- VCAP@100 improved most (9W/3L, +0.0034, p≈0.07) — closest to significance

### 2022-12 vs 2022-09 Divergence
- 2022-12 (weakest AUC month) improved substantially: AUC +0.0098, AP +0.0142
- 2022-09 (weakest AP month) unchanged on AUC, regressed on AP (-0.0091)
- These two months have different failure modes: 2022-12 benefits from seasonal diversity, 2022-09 has a feature-target mismatch (lowest binding rate at 6.63%)
- 3 independent levers all failed to improve 2022-09 — may need fundamentally new features

### Broader vs Top-K Ranking Tradeoff
- VCAP@100 improved while VCAP@500 (-0.0063) and CAP@100/500 degraded
- Pattern seen in both v0002 and v0003: top-100 improves at expense of broader ranking
- Consistent with business objective (top-of-stack precision matters most)

## From v0004 — Combined Window + Interactions (feat-eng-060938, real data)

### Additivity of Levers
- **VCAP@100 is super-additive**: combined (+0.0056) > sum of parts (+0.0043). Interaction features are particularly effective at re-ranking the top of the score distribution when training data is more diverse.
- **NDCG is roughly additive**: combined (+0.0038) ≈ sum (+0.0035).
- **AUC is mostly window-driven**: combined (+0.0015) ≈ window alone (+0.0013). Interactions add negligible AUC signal.
- **AP is sub-additive**: combined (+0.0015) < sum (+0.0022). Interactions reduce AP consistency — they may add noise to the positive-class ranking that offsets their re-ranking benefit.

### Statistical Significance Milestone
- **VCAP@100 10W/2L**: two-sided sign test p=0.039. This is the first statistically significant metric improvement in 4 real-data experiments. The model genuinely captures more shadow price value at the top 100 predictions.
- AUC 9W/3L: sign test p=0.073 — approaching but not significant.
- AP 6W/6L: no signal at all.

### Feature Set Ceiling Confirmed
- After 4 experiments spanning HP tuning, interaction features, window expansion, and combination:
  - AUC range: [0.8323, 0.8363] — total span of 0.004
  - AP range: [0.3921, 0.3951] — total span of 0.003
  - NDCG range: [0.7323, 0.7371] — total span of 0.005
- The 14 base features + 3 interactions define a hard AUC ceiling at ~0.836
- Breaking through requires fundamentally different information, not different model configurations

### VCAP@500 Systematic Regression
- 3 consecutive experiments show VCAP@500 decline: v0002(-0.0043), v0003(-0.0063), v0004(-0.0065)
- v0004 bot2=0.0387 is within 0.0021 of floor (0.0408)
- This appears inherent to improving top-100 precision: the model concentrates probability mass more tightly at the top, at the expense of the 100-500 range
- Group B (non-blocking) so not fatal, but the trend needs monitoring

### 2022-09 Is Structural
- 4 independent interventions failed to improve 2022-09 AP (stays at ~0.307-0.315)
- Lowest binding rate (6.63%) makes class separation inherently harder
- AUC actually improved slightly (+0.0011) but AP didn't follow — the model ranks constraints slightly better overall but cannot identify the sparse positives in this month
- Likely requires either new feature sources or time-series features capturing temporal regime shifts

## From v0005 — Extended Training Window 18 months (feat-eng-060938 iter2, real data)

### Window Expansion Is Definitively Exhausted
- 14→18 months provides zero marginal benefit: AUC -0.0002, AP -0.0023, VCAP@100 -0.0012, NDCG -0.0007 (all vs v0004)
- All deltas vs v0004 are pure noise (W/L all within 5-7 range out of 12)
- Optimal training window is 14 months. Older data (2018-2019) adds noise, not signal.
- The productive window expansion range was 10→14 months. Beyond that, diminishing returns dominate.

### Feature Importance Reveals Model Architecture (CRITICAL — First Empirical Data)
- **The model is 79% a historical trend predictor**: hist_da_trend (54%), hist_physical_interaction (14%), hist_da (11%)
- **Physical flow features provide 18% of signal**: prob_below_90 (5.1%), prob_exceed_90 (3.1%), prob_exceed_95 (2.1%), plus minor physical features
- **Distribution shape features are noise**: density_skewness (0.31%), density_cv (0.40%), density_kurtosis (0.58%) = 1.3% collectively
- **exceed_severity_ratio (0.38%) does not earn its keep** — weakest interaction feature, prune candidate
- **hist_physical_interaction validates iter 1** — #2 feature at 14% gain
- Feature importance is remarkably stable across 12 months (hist_da_trend CV=3.5%) — pruning decisions are reliable

### AP Bot2 Trend Is the New Risk
- Monotonically worsening: v0002(-0.0017) → v0003(-0.0045) → v0004(-0.0040) → v0005(-0.0075)
- Each window expansion degrades the AP tail — more training data may spread the model's attention across more patterns, making it slightly worse at the hardest months
- Still within 0.02 tolerance (margin 0.0125) but the trend is clear
- Reverting to 14-month window in iter 3 may partially reverse this

### 2022-09 Is Definitively Structural
- AP at 0.2986 — all-time worst across 5 experiments
- 5 independent interventions have failed to improve it
- Binding rate 6.63% (lowest in eval set) makes class separation fundamentally harder
- This month likely requires entirely new feature sources (e.g., economic indicators, seasonal forecasts, grid topology changes)

### VCAP@500 Bot2 Is Not Structural After All
- v0005 bot2=0.0449 recovered from v0004's 0.0387 (which was approaching floor 0.0408)
- The 18-month window stabilized the tail even while mean ranking quality was flat
- This suggests VCAP@500 bot2 variability is more about training data composition than a systematic tradeoff

## Cumulative Evidence Summary (5 real-data experiments)

| Lever | AUC Δ | AUC W/L | Key Learning |
|-------|-------|---------|-------------|
| HP tuning (v0003-HP) | -0.0025 | 0W/11L | Model not complexity-limited |
| Interactions (v0002) | +0.0000 | 5W/6L/1T | Information ceiling within features |
| Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | Small signal, best single lever |
| **Combined (v0004)** | **+0.0015** | **9W/3L** | **Partially additive; VCAP@100 super-additive** |
| Window 14→18 (v0005) | -0.0002 | 7W/5L (vs v0) | **Exhausted — zero marginal benefit** |
| **Feature pruning (v0006)** | **+0.0006** | **5W/7L** | **Tradeoff: NDCG +0.023, AP -0.004. Not promotable.** |

### Statistical Testing
- Month-level win/loss counts more informative than mean deltas at n=12
- Effect of +0.001 AUC requires ~200+ months to reach significance at these std levels
- Practical decision making must rely on consistency (W/L ratio) and direction, not p-values
- Always check for outlier-driven means: exclude best month and recalculate
- Sign test is more appropriate than z-test for W/L data — doesn't assume normality

## From v0006 — Feature Pruning 17→13 (feat-eng-060938 iter3, real data)

### Monotone Constraint Structure Matters (CRITICAL — Novel Finding)
- Removing all 3 unconstrained features (monotone=0: density_skewness, density_kurtosis, density_cv) made the model fully monotone-constrained
- Full monotone enforcement acts as structural regularization: sharpens ranking consistency (NDCG, VCAP) but degrades positive-class breadth ranking (AP)
- This is not noise removal — it's a fundamental change to model behavior
- NDCG +0.0227 and VCAP@100 +0.0121 are both statistically significant (p=0.039, 10W/2L)
- AP -0.0044 is broadly distributed (3W/9L, not outlier-driven) — a genuine tradeoff

### Feature Pruning Is Not Simply Noise Removal
- Features contributing <1% gain still serve a purpose: the unconstrained direction (monotone=0) provides implicit regularization
- Removing them didn't reduce noise — it changed the model's optimization landscape
- hist_da doubled from 11.3% to 24.1%, compensating for pruned features
- The model became more balanced between level (24%) and trend (44%) of historical shadow prices

### AP Bot2 Monotonic Decline Is a Systemic Risk
- 6-experiment trend: 0.3322 → 0.3305 → 0.3277 → 0.3282 → 0.3247 → 0.3228
- Margin to Layer 3 failure (0.02 tolerance) is only 0.0106
- Every model modification we've tested has worsened AP in the weakest months
- This suggests the v0 feature set happens to be well-calibrated for worst-month AP, and any change — even beneficial on average — destabilizes the tail

### BRIER Does Not Simply Respond to Model Simplification
- Expected: fewer features → simpler model → better calibration (as seen in v0003-HP)
- Actual: v0006 BRIER worsened (+0.0037 vs v0), 6th consecutive narrowing
- The key difference: v0003-HP simplified via regularization (deeper trees, slower learning), v0006 simplified via information loss (feature removal)
- Regularization helps calibration; feature removal doesn't

### Two Distinct Model Profiles Exist Within the Same Feature Set
- **v0004 profile** (17 features, 14mo): AUC-optimized, balanced across all Group A metrics
- **v0006 profile** (13 features, 14mo): NDCG/VCAP-optimized, best ranking consistency but AP degraded
- The operational question for HUMAN_SYNC: which profile better serves the business?
  - If acting on top-100 predictions: v0006 captures more value (VCAP@100=0.0270 vs 0.0205)
  - If using the full positive-class prediction: v0004 ranks positives better (AP=0.3951 vs 0.3892)

### Levers Exhausted vs Available (FINAL — Batch Complete)
- **Exhausted**: HP tuning, window expansion (10-14-18), interaction features, feature pruning
- **Available (next batch)**: New data sources (fuel prices, weather, load forecasts, outage data, transmission topology), time-series regime features, ranking-focused objectives (LambdaRank/LambdaMART), monotone constraint optimization (selective enforcement), alternative models
- **Best configuration found**: v0004 (17 features, 14-month window, v0 HPs) for balanced improvement; v0006 (13 features, 14-month window) if top-K ranking is the only objective
