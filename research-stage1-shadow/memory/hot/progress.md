# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | hp-tune-20260302-144146 |
| Iteration | 1 of 3 (synthesis complete, planning iter2) |
| State | ORCHESTRATOR_SYNTHESIZING → next: iter2 |
| Champion | None (v0 baseline) |
| Last Hypothesis | H4: Interaction features — **NOT SUPPORTED** (AUC +0.000, 5W/6L) |
| Next Hypothesis | H5: Longer training window (10→14 months) + keep interaction features |

## Iteration 1 Result Summary (v0002)

- **Objective**: Improve ranking via 3 interaction features + HP revert to v0
- **AUC**: 0.8348 → 0.8348 (+0.0000, 5W/6L/1T) — zero discrimination improvement
- **AP**: 0.3936 → 0.3946 (+0.0010, 7W/5L) — marginal, noise-level
- **NDCG**: 0.7333 → 0.7349 (+0.0016, 8W/4L) — marginal, driven by 2021-01 outlier
- **Gates**: All pass all 3 layers (mean, tail, regression)
- **Promoted**: No — no meaningful improvement
- **Key Learning**: AUC ceiling at ~0.835 confirmed across HP tuning AND interaction features. Distribution shift is the dominant remaining problem.

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| hp-tune-20260302-144146 iter2 | Training window expansion | Planned — train_months 10→14 + keep interactions |
