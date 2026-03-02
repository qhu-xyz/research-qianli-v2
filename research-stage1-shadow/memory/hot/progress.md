# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | hp-tune-20260302-134412 |
| Iteration | 1 of 3 (synthesis complete) |
| State | ORCHESTRATOR_SYNTHESIZING → next: iter2 |
| Champion | None (v0 baseline) |
| Last Hypothesis | H3: HP tuning — **REFUTED** (AUC -0.0025, 0W/11L) |
| Next Hypothesis | H4: Interaction features (revert HPs + add cross-feature interactions) |

## Iteration 1 Results Summary

- **Version**: v0003
- **Changes**: max_depth 4→6, n_estimators 200→400, lr 0.1→0.05, min_child_weight 10→5
- **Outcome**: All Group A metrics regressed. AUC -0.0025 (0W/11L), AP -0.0015 (4W/8L), NDCG -0.0010 (4W/8L). BRIER improved -0.004 (12W/0L, Group B only).
- **Promoted**: No
- **Key Insight**: Model is feature-limited, not complexity-limited. HP tuning is the wrong lever.

## Iteration 2 Plan Summary

- **Objective**: Improve ranking quality via feature engineering
- **Changes**: Revert HPs to v0 defaults + add 3 interaction features in `ml/features.py` + update `ml/config.py`
- **Expected**: AUC +0.005–0.010, AP +0.01–0.02
- **Risk**: Low — additive features on v0 baseline, no architectural changes
- **Direction file**: `memory/direction_iter2.md`

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune iter1 | HP tuning (v0003) | H3 refuted — no Group A improvement, AUC degraded |
| hp-tune iter2 | Feature engineering | Planned — interaction features + HP revert |
