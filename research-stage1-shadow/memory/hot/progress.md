# Progress

## Current State

| Field | Value |
|-------|-------|
| Batch | feat-eng-20260303-060938 |
| Iteration | 3 of 3 (synthesis complete, planning next) |
| State | ORCHESTRATOR_SYNTHESIZING → next: ORCHESTRATOR_PLANNING (iter 3) |
| Champion | None (v0 baseline) |
| Last Hypothesis | H7: train_months=18 + feature importance — FAILED (diminishing returns) |
| Next Hypothesis | H8: Feature pruning (17→13 features) + revert to 14-month window |

## This Batch Strategy

From human input: test additivity of the two positive-signal levers, then refine or pivot.
- **Iter 1**: H6 — 14-month window + 3 interaction features → **DONE** (AUC +0.0015, 9W/3L, VCAP@100 +0.0056, 10W/2L. Not promoted.)
- **Iter 2**: H7 — train_months=18 with 17 features + feature importance → **DONE** (Diminishing returns confirmed. AUC -0.0002 vs v0004. Feature importance collected. Not promoted.)
- **Iter 3**: H8 — Prune bottom 4 features (17→13), revert to 14-month window. Test if noise reduction improves AP tail.

## v0005 Key Results (iter 2)

| Metric | v0 | v0004 (iter 1) | v0005 (iter 2) | Δ vs v0 | Δ vs v0004 |
|--------|-----|----------------|----------------|---------|------------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | +0.0013 | -0.0002 |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | -0.0007 | -0.0023 |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | +0.0044 | -0.0012 |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | +0.0032 | -0.0007 |

All gates pass. Not promoted (strictly worse than v0004 on all Group A means). Feature importance data successfully collected.

## Feature Importance Summary (from v0005)

| Tier | Features | % Gain |
|------|----------|--------|
| Dominant | hist_da_trend | 53.9% |
| Strong | hist_physical_interaction, hist_da | 25.5% |
| Moderate | prob_below_90, prob_exceed_90, prob_exceed_95 | 10.3% |
| Weak | 7 features (prob_below_95 through expected_overload) | 8.9% |
| **Prune** | **density_kurtosis, density_cv, exceed_severity_ratio, density_skewness** | **1.4%** |

## Iter 3 Strategy (H8)

- **Primary**: Remove 4 near-zero features (17→13): density_skewness, exceed_severity_ratio, density_cv, density_kurtosis
- **Secondary**: Revert train_months from 18 to 14 (v0004 config was strictly better)
- **Expected**: Small positive or neutral on mean metrics. Primary hope is improving AP bot2 and BRIER calibration.
- **Endgame awareness**: This is the final iteration. If pruning produces modest improvement, consider promoting the best version (likely v0004). If pruning is neutral/negative, declare the ceiling reached and summarize findings for HUMAN_SYNC.

## Cumulative Evidence (all real-data experiments)

| Batch | Lever | AUC Δ | AUC W/L | Promoted |
|-------|-------|-------|---------|----------|
| hp-tune-134412 | HP tuning (v0003-HP) | -0.0025 | 0W/11L | No |
| hp-tune-144146 | Interactions (v0002) | +0.0000 | 5W/6L/1T | No |
| feat-eng-194243 iter1 | Window 10→14 (v0003) | +0.0013 | 7W/4L/1T | No |
| feat-eng-060938 iter1 | Window + Interactions (v0004) | +0.0015 | 9W/3L | No |
| feat-eng-060938 iter2 | Window 18 + importance (v0005) | -0.0002 vs v0004 | 7W/5L vs v0 | No |
| feat-eng-060938 iter3 | Feature pruning (H8) | ? | ? | ? |

## History

| Batch | Type | Result |
|-------|------|--------|
| smoke-v6 | Infrastructure validation | PASS (determinism confirmed) |
| smoke-v7 | Bug fixes + beta experiment | Fixes merged, H2 failed (beta direction inverted) |
| hp-tune-20260302-134412 iter1 | HP tuning (v0003-HP) | H3 refuted — AUC -0.0025, 0W/11L |
| hp-tune-20260302-144146 iter1 | Interaction features (v0002) | H4 not supported — AUC +0.000, 5W/6L |
| feat-eng-20260302-194243 iter1 | Training window 10→14 (v0003) | H5 inconclusive — AUC +0.0013, 7W/4L/1T |
| feat-eng-20260303-060938 iter1 | Window + Interactions (v0004) | H6 partially confirmed — AUC +0.0015, 9W/3L |
| **feat-eng-20260303-060938 iter2** | **Window 18 + importance (v0005)** | **H7 failed — diminishing returns, AUC -0.0002 vs v0004** |
