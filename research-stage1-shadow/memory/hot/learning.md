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
