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

### Cumulative Evidence After 3 Real-Data Experiments
| Lever | AUC Δ | AUC W/L | Key Learning |
|-------|-------|---------|-------------|
| HP tuning (v0003-HP) | -0.0025 | 0W/11L | Model not complexity-limited |
| Interactions (v0002) | +0.0000 | 5W/6L/1T | Information ceiling within features |
| Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | Small signal, best lever so far |
| **Next: Combined** | **TBD** | **TBD** | **Tests additivity of interactions + window** |

### Statistical Testing
- Month-level win/loss counts more informative than mean deltas at n=12
- Effect of +0.001 AUC requires ~200+ months to reach significance at these std levels
- Practical decision making must rely on consistency (W/L ratio) and direction, not p-values
- Always check for outlier-driven means: exclude best month and recalculate
