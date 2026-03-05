# Experiment Log

## v0 — Baseline (2026-03-04)
- **Config**: n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- **Class weights**: {0:10, 1:5, 2:2, 3:1, 4:0.5}
- **Features**: 34 (11 flow prob + 7 distribution shape + 1 overload + 5 historical + 10 engineered)
- **Results** (12-month mean):
  - Tier-VC@100=0.0708, Tier-VC@500=0.2296, Tier0-AP=0.3062, Tier01-AP=0.3110
  - Tier-NDCG=0.7711, QWK=0.3698, Macro-F1=0.3560
  - Tier-Recall@0=0.4389, Tier-Recall@1=0.0467
- **Key finding**: Tier 4 has 0 samples in all months. Tier-Recall@1 catastrophically low.

## Prior Batch (tier-fe-1) — ALL 3 ITERATIONS FAILED
- Root cause: Uncommitted changes to HUMAN-WRITE-ONLY files caused pre-merge guard to reject worker output
- No data collected. All hypotheses untested. Fixed in commit a2a38c5.

## Batch tier-fe-2 Iter 1 (previous attempt) — WORKER FAILED
- Same root cause as tier-fe-1. Fixed before current batch.

## Batch tier-fe-2-20260305-001606, Iter 1 — v0005 (2026-03-05)
- **Hypothesis**: A (add 3 interaction features 34→37) won screening over B (add 3 + prune 5 → 32)
- **Screen results** (2 months: 2022-06 weak, 2021-09 strong):
  - A: mean VC@100=0.1372, B: mean VC@100=0.1307 — A wins on primary criterion
  - A advantage driven by 2022-06 (0.0255 vs 0.0112, 2.3x)
- **Features added**: overload_x_hist, prob110_x_recent_hist, tail_x_hist (37 total)
- **Code changes**: features.py (3 interactions), config.py (3 features + monotone), tests (34→37)
- **Results** (12-month mean, delta vs v0):
  - Tier-VC@100=0.0746 (+5.4%) — **FAILS L1** (floor 0.0750, gap 0.0004)
  - Tier-VC@500=0.2329 (+1.5%) — PASS all layers
  - Tier0-AP=0.3126 (+2.1%) — PASS all layers
  - Tier01-AP=0.3132 (+0.7%) — PASS all layers
  - Tier-NDCG=0.7751 (+0.5%) — PASS
  - QWK=0.3706 (+0.2%) — PASS
  - Macro-F1=0.3560 (unchanged) — FAIL L1 (structural)
  - Tier-Recall@0=0.4403 (+0.3%) — PASS
  - Tier-Recall@1=0.0447 (-4.3%) — FAIL L1 (structural)
- **Gate assessment**: NOT PROMOTED. Tier-VC@100 is the sole blocking gate (0.0004 below floor).
- **Per-month highlights**:
  - Biggest VC@100 gain: 2022-12 (+0.023), 2020-11 (+0.009), 2022-03 (+0.008)
  - Worst months unchanged: 2021-11 (0.0082), 2022-06 (0.0255)
  - Bottom_2_mean improved 64%: 0.0103 → 0.0169 — interactions help weak months most
  - VC@100 improved in 8/12 months (sign test p~0.19, Cohen's d~0.06 — not significant)
- **Key learnings**: Interaction features provide modest but consistent improvement. The 0.0004 gap is within noise but gate is binary. FE ceiling reached for VC@100 without hyperparameter/class weight changes.
