# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | hp-tune-20260302-144146 |
| Iteration | 1 of 3 (planning complete) |
| State | ORCHESTRATOR_PLANNING → WORKER |
| Champion | None (v0 baseline) |
| Current Hypothesis | H4: Interaction features (3 new cross-feature interactions + HP revert to v0) |
| Previous Hypothesis | H3: HP tuning — **REFUTED** (AUC -0.0025, 0W/11L) |

## Iteration 1 Plan Summary

- **Objective**: Improve ranking quality via feature engineering (3 interaction features)
- **Changes**: (1) Revert HPs to v0 defaults, (2) Add 3 interaction features in `ml/features.py`, (3) Update `ml/config.py` FeatureConfig
- **New features**: exceed_severity_ratio, hist_physical_interaction, overload_exceedance_product
- **Expected**: AUC +0.005–0.015, AP +0.010–0.025
- **Risk**: Low — additive features on v0 baseline, no architectural changes
- **Direction file**: `memory/direction_iter1.md`

## Prior Batch Results (hp-tune-20260302-134412)

- **Version**: v0003
- **Changes**: max_depth 4→6, n_estimators 200→400, lr 0.1→0.05, min_child_weight 10→5
- **Outcome**: All Group A metrics regressed. AUC -0.0025 (0W/11L), AP -0.0015 (4W/8L), NDCG -0.0010 (4W/8L). BRIER improved -0.004 (12W/0L, Group B only).
- **Promoted**: No
- **Key Insight**: Model is feature-limited, not complexity-limited. HP tuning is the wrong lever.

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003) | H3 refuted — no Group A improvement, AUC degraded |
| hp-tune-20260302-144146 iter1 | Interaction features | Planned — 3 new features + HP revert |
