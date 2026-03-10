# Repo Audit

## Verdict

The repository is internally consistent and the published dev metrics are reproducible, but I do **not** consider the implementation or all reported results fully valid yet.

What checked out:

- The project docs are detailed enough to reconstruct the intended pipeline.
- The stored V6.1 formula is real: I verified `rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value` exactly on cached `2024-06/aq1`, with the documented quintile tier split.
- I reran `v0`, `v1`, and `v5` into `codex-review/tmp-registry/`; their aggregate metrics matched the checked-in registry outputs exactly.

What does **not** check out is the target-label construction and part of the evaluation/reporting.

## Findings

### 1. High: realized DA labels are built with an unsafe many-to-many `constraint_id -> branch_name` join

Relevant code:

- `ml/ground_truth.py:37-50`
- `ml/ground_truth.py:87-107`
- `ml/ground_truth.py:149-153`

`_load_cid_to_branch()` intentionally loads **all** rows from `MISO_SPICE_CONSTRAINT_INFO` and keeps unique `(constraint_id, branch_name)` pairs across every auction/month/period. `fetch_realized_da_quarter()` then aggregates DA shadow by `constraint_id` and joins that table to the global mapping.

That assumption is false in the source data. In my audit run against `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet`:

- 19,644 unique `constraint_id`s existed.
- 17,559 of them mapped to more than one `branch_name`.
- The worst `constraint_id` mapped to 24 different `branch_name`s.

So the current code can fan a single realized DA constraint out across multiple branches before the branch-level aggregation. This directly contaminates `realized_shadow_price`, which is the training label and the evaluation target.

I replayed `2024-06/aq1` with live DA data from `get_da_shadow_by_peaktype()`:

- Current repo logic produced 969 realized branches and total branch value `2,009,759.71`.
- Restricting the bridge table to the target annual partition (`auction_type='annual'`, `auction_month='2024-06'`, `period_type='aq1'`, `class_type='onpeak'`) produced 898 realized branches and total branch value `1,871,222.49`.
- Even inside the V6.1 universe, 2 branches changed: `WEMPLETN CT_P` went from `702.92` to `0.0`, and `DRIVER_SANSSU8 1` dropped from `289.08` to `94.68`.

The exact impact across all quarters still needs a full rebuild, but the label-generation path is not trustworthy as implemented.

### 2. High: the checked-in 2025 holdout includes an incomplete future quarter

Relevant code and artifacts:

- `ml/config.py:87-133`
- `runbook.md:47-50`
- `registry/v1_holdout/metrics.json:3-9`
- `registry/v1_holdout/metrics.json:63-78`
- `ml/ground_truth.py:68-79`

`aq4` is defined as March-May of the following calendar year. For `2025-06/aq4`, `get_market_months()` resolves to:

- March 2026
- April 2026
- May 2026

Today is **March 8, 2026**. That means the target period for `2025-06/aq4` is not complete yet; April 2026 and May 2026 are still in the future, and even March 2026 is only partially observed.

Despite that, the runbook explicitly tells users to evaluate holdout on all four 2025 quarters, and `registry/v1_holdout/metrics.json` includes `2025-06/aq4` as if it were final. Those holdout numbers are therefore not valid final results.

At minimum, `aq4` should be excluded until after June 1, 2026, or the code should refuse to score incomplete target windows.

### 3. Medium: `Recall@100` is order-dependent when fewer than 100 constraints bind

Relevant code and docs:

- `ml/evaluate.py:26-31`
- `runbook.md:39`

`recall_at_k()` defines the true set as `np.argsort(actual)[::-1][:k]`. When fewer than `k` rows have positive target value, the remainder of that set is filled by arbitrary zero-valued ties, so the metric depends on row order rather than only on data.

This is not hypothetical. In the cached data:

- `2022-06/aq3` has 92 positive rows.
- `2025-06/aq4` has 76 positive rows.

I permuted the row order while keeping the same `(actual, score)` pairs:

- `2022-06/aq3` baseline `Recall@100` was `0.51`, but the same data moved between `0.44` and `0.49` under permutation.
- `2025-06/aq4` baseline `Recall@100` was `0.37`, but the same data moved between `0.39` and `0.44` under permutation.

Because the runbook says gate failures are driven by the `Recall@100` tail check, some pass/fail decisions are currently influenced by row ordering instead of model quality.

### 4. Low: the XGBoost fallback path is dead

Relevant code:

- `ml/train.py:140-167`
- `ml/config.py:136-151`

`_train_xgboost()` reads `cfg.max_depth`, but `LTRConfig` does not define that field. Running the path raises:

`AttributeError: 'LTRConfig' object has no attribute 'max_depth'`

This does not affect the checked-in LightGBM experiments, but it means the advertised fallback backend is currently unusable.

## Result Validity Summary

### Dev results (`registry/v0`, `registry/v1`, `registry/v5`)

Reproducibility: **PASS**

- I reran them locally from cache and reproduced the stored metrics exactly.

Validity: **PARTIAL**

- The saved numbers are consistent with the current code.
- But the target-label mapping bug means those results are built on labels that can be wrong.

### Holdout results (`registry/v1_holdout`)

Reproducibility: not rerun in full.

Validity: **FAIL**

- `2025-06/aq4` is an incomplete future target window as of March 8, 2026.
- Any aggregate that includes that quarter is not a valid final holdout number.

## Bottom Line

The repo has good documentation and reproducible experiment artifacts, but the existing `audit.md` conclusion of "no critical issues found" is too optimistic. The highest-priority fix is the realized-DA mapping in `ml/ground_truth.py`; after that, the holdout window and `Recall@100` definition should be corrected before treating the reported results as decision-grade.
