# Stage 5 Repo Audit

## Findings

### 1. `Tier0-AP` / `Tier01-AP` are degenerate, so the gate system built from them is invalid
Files: `ml/evaluate.py:60-72`, `scripts/run_v0_formula_baseline.py:66-105`, `registry/gates.json`, `audit.md:55-62`

`tier_ap()` converts the top `top_frac` of rows into positives by thresholding on the `k`th-largest `actual` value. In this dataset, every default eval month has far fewer positives than `20%` or `40%` of rows: I verified 50-85 binding rows per month versus 448-780 total rows. That makes the threshold `0.0` for both fractions in all 12 default eval months, so `y_true = (actual >= 0)` becomes all ones. As a result:

- `Tier0-AP` and `Tier01-AP` are always `1.0` (or floating-point-near-1.0) in the registry and holdout artifacts.
- `registry/gates.json` calibrates floors/tails from those meaningless values.
- comparison reports can mark versions as failing purely because `0.9999999999999998 < 1.0`, not because model quality changed.

This directly contradicts `audit.md`, which states the evaluation metrics are correct.

### 2. `Recall@100` is tie-contaminated and not a valid gate for this problem setup
Files: `ml/evaluate.py:26-31`, `scripts/run_v0_formula_baseline.py:71-75`, `registry/gates.json`

`recall_at_k()` defines the truth set as the top `k` rows by `actual`. That only works cleanly when at least `k` rows have meaningful positive relevance. Here, every default eval month has fewer than 100 binding constraints, so the "true top 100" always includes 15-50 zero-valued rows selected only by arbitrary tie order. That makes `Recall@100` partly a function of numpy tie-breaking over non-binding rows rather than model skill.

Impact:

- `Recall@100` should not be used as a production gate in this dataset.
- month-to-month and model-to-model differences in that metric are noisier than they appear in the registry.

### 3. `FEATURES_V3` is not actually implemented; a future v3 run would silently zero-fill the ml_pred features
Files: `ml/config.py:83-85`, `ml/data_loader.py:68-94`, `ml/features.py:30-45`, `ml/mlpred_loader.py:26-67`, `docs/plans/2026-03-08-stage5-ml-pipeline-design.md:77-82`

The config exposes `FEATURES_V3 = FEATURES_V1B + _MLPRED_FEATURES`, and `mlpred_loader.py` exists, but `data_loader.py` never calls `load_mlpred()` or joins the ml_pred columns into the monthly frame. `prepare_features()` then treats missing configured columns as normal and silently substitutes zero vectors.

I verified that `load_v62b_month('2022-06')` is missing all three advertised ml_pred columns:

- `predicted_shadow_price`
- `binding_probability`
- `binding_probability_scaled`

So a future v3 benchmark would appear to run successfully while not using the intended features at all. The design doc currently claims optional ml_pred enrichment exists, which is inaccurate.

### 4. The repo’s governance artifacts are stale relative to the later experiment results
Files: `registry/champion.json`, `registry/comparisons/stage5_iter1.md`, `registry/v5/metrics.json`, `registry/v6b/metrics.json`, `holdout/comparison.json`

The only comparison report in `registry/comparisons/` is `stage5_iter1`, which stops at `v0/v1/v1b`, and `registry/champion.json` still points to `v0`. Meanwhile the repo contains later `v5/v6*` dev results and 2024-2025 holdout results.

Even if `v0` remains the right champion, the current registry metadata does not demonstrate that decision against the later variants. Combined with Finding 1, the present gate/champion state should be treated as non-authoritative.

## Open Questions / Assumptions

- I did not re-run the full 12-month, 24-month, and holdout suites from scratch. I spot-checked reproducibility on `2022-06` for `v0` and `v6b`, and both matched the stored registry metrics exactly.
- I assumed the intended label universe is "constraints present in V6.2B, with missing realized DA interpreted as non-binding". The code and docs are aligned on that point.

## Result Check

What looks correct:

- The main stage-5 fix is real: training/evaluation uses `realized_sp`, not `shadow_price_da`.
- The current stored metrics appear reproducible for sampled months.
- The test suite passes in the project venv: `25 passed`.

What does not hold up:

- The repo’s claim of "No leakage, no bugs, no wrong targets" in `audit.md` is too strong.
- The AP-based monitoring/gating metrics are invalid in practice.
- `Recall@100` is not a reliable gate for these label densities.
- The advertised v3/ml_pred path is incomplete.

## Bottom Line

The core implementation is directionally correct and the published month-level results are mostly believable, but the evaluation/governance layer is not. I would trust the stored `VC@k`, `NDCG`, and `Spearman` numbers more than the repo’s gate/champion conclusions. I would not treat `Tier0-AP`, `Tier01-AP`, `Recall@100`, `registry/gates.json`, or `registry/champion.json` as production-ready evidence without fixing the issues above.
