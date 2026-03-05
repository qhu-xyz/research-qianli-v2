# Experiment Log

## v0 — Baseline (2026-03-04)
- **Config**: n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- **Class weights**: {0:10, 1:5, 2:2, 3:1, 4:0.5}
- **Features**: 34 (11 flow prob + 7 distribution shape + 1 overload + 5 historical + 10 engineered)
- **Results** (12-month mean):
  - Tier-VC@100=0.075, Tier-VC@500=0.217, Tier-NDCG=0.767
  - QWK=0.359, Macro-F1=0.369
  - Tier-Recall@0=0.374, Tier-Recall@1=0.098
- **Key finding**: Tier 4 has 0 samples in all months. Tier-Recall@1 catastrophically low.

## Iter 1 — WORKER FAILED (batch: tier-fe-1-20260304-182037, 2026-03-05)
- **Planned**: Feature swap — remove 3 low-importance features, add 3 interaction features (Hyp A) vs aggressive pruning 34→28 (Hyp B)
- **Status**: FAILED — worker wrote handoff claiming "done" but produced NO artifacts
  - `registry/v0001/` does not exist (no metrics, no config, no changes_summary)
  - No reports generated in `reports/tier-fe-1-20260304-182037/iter1/`
  - No reviews generated
  - Version counter advanced to 2 (leaked side effect)
- **Root cause**: Worker exited prematurely after writing handoff signal but before running benchmark or producing any registry artifacts. Likely a session timeout or crash during screening phase.
- **Result**: No data collected. Hypotheses A and B remain UNTESTED.
- **Recovery**: Retry with simplified direction in iter2 — same hypotheses but streamline worker instructions to reduce execution risk.

## Iter 2 — WORKER FAILED (batch: tier-fe-1-20260304-182037, 2026-03-05)
- **Planned**: Add 3 interaction features (hist_physical_interaction, overload_exceedance_product, hist_seasonal_band) → 37 features (Hyp A) vs prune 6 + add 3 → 31 features (Hyp B). Screen on 2021-11 (weak) and 2021-09 (strong).
- **Status**: FAILED — identical failure mode to iter1
  - Worker wrote handoff `"status": "done"` with `artifact_path: registry/v0002/changes_summary.md`
  - `registry/v0002/` does not exist (no metrics, no config, no changes_summary)
  - No reports in `reports/tier-fe-1-20260304-182037/iter2/`
  - No reviews generated
  - Version counter leaked again: 2→3
- **Root cause**: Systematic worker failure. Two consecutive identical failures — worker writes handoff claiming completion but never runs benchmark or produces artifacts. Not a random timeout; likely a bug in the worker's execution flow where it writes the handoff signal before completing actual work.
- **Result**: No data collected. All hypotheses remain UNTESTED after 2 iterations.
- **Recovery**: Iter3 is the LAST iteration. Must simplify to absolute minimum: single hypothesis, explicit commands, no screening phase.
