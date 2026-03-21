# 2026-03-14 Code Audit Review

## Findings

### 1. HIGH: The official evaluator is still hardcoded to the old `@50/@100` policy, so the repo cannot reproduce the current `(@150,@300)` / `(@200,@400)` champion decisions.

The core evaluation path still computes extra K metrics only for `[20, 50, 100]` in [ml/evaluate.py:162](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/evaluate.py#L162), and the blocking metric definitions in [ml/config.py:169](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/config.py#L169) and [ml/config.py:175](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/config.py#L175) are still `VC/Recall/Abs_SP` at `50/100`. Current `150/300/200/400` comparisons are therefore happening in ad hoc scripts instead of the library contract and registry format. This creates a reproducibility gap: future readers cannot derive the new “champion” from the official saved metrics path.

### 2. HIGH: `ambiguous_sp` diagnostics are silently wrong for ground truth and any caller that does not pass a column named `realized_sp`.

[ml/bridge.py:96](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/bridge.py#L96) only sums ambiguous value on a column literally named `realized_sp`. But ground truth passes quarter-combined DA under `total_sp` in [ml/ground_truth.py:82](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/ground_truth.py#L82). As a result, `annual_ambiguous_sp` in [ml/ground_truth.py:197](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/ground_truth.py#L197) will read `0.0` even when ambiguous cids carried nonzero SP. That under-reports lost value in coverage diagnostics and makes ambiguity impact look smaller than it is.

### 3. HIGH: The two-track runners’ `R=0` rows are not true “solo baseline” runs and should not be interpreted as such.

In [scripts/run_two_track_experiment.py:141](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_two_track_experiment.py#L141), Track A contains only `established`, while Track B contains `history_dormant + history_zero`. In [scripts/run_phase4a_experiment.py:218](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_phase4a_experiment.py#L218), Track A contains only `established`, and Track B contains only `history_dormant`, with zero-history appended afterward. `merge_tracks()` then allocates `k - r_actual` slots to Track A and `r_actual` to Track B in [ml/merge.py:58](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/merge.py#L58). When `r=0`, dormant branches remain in the evaluation universe but are categorically barred from top-K, which is not how `v0c` or `v3a` behave as single global rankers. Any table that labels these rows as “`v0c solo`” or “`v3a solo`” is biased.

### 4. MED: Two-track `NDCG` and `Spearman` are not describing the deployed shortlist policy.

`evaluate_group()` always computes `NDCG` and `Spearman` from the raw `score` column in [ml/evaluate.py:173](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/evaluate.py#L173), even when `top_k_override` was used to define the actual shortlist. The two-track runners then retain these global metrics from the `m50` evaluation in [scripts/run_two_track_experiment.py:167](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_two_track_experiment.py#L167), [scripts/run_phase4a_experiment.py:238](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_phase4a_experiment.py#L238), and [scripts/run_phase4b_regression.py:184](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_phase4b_regression.py#L184). Because Track A and Track B are scored on different scales and the final shortlist is created by reserved-slot merge rather than pure score order, these metrics do not reflect the actual deployed ranking behavior.

### 5. MED: The main ML runner computes gate results but never saves them, which is why later registry entries have no gate audit trail.

[scripts/run_ml_experiment.py:234](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_ml_experiment.py#L234) computes `gate_results`, but [scripts/run_ml_experiment.py:260](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_ml_experiment.py#L260) calls `save_experiment(args.version, config, metrics)` without passing them. This is why newer models like `v3a`, `v3e_nb`, and `v3f_nbonly` have metrics but no `gate_results.json`, making later champion claims harder to verify cleanly from the registry alone.

### 6. MED: `scripts/run_two_track_experiment.py` still advertises `--track-a v3a`, but that code path is unimplemented and raises at runtime.

The CLI exposes `choices=["v0c", "v3a"]` at [scripts/run_two_track_experiment.py:239](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_two_track_experiment.py#L239), but [scripts/run_two_track_experiment.py:147](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_two_track_experiment.py#L147) raises `NotImplementedError` for `v3a`. That is a stale handoff trap and directly conflicts with later analysis that informally compares `v3a + NB`.

### 7. MED: Ground-truth tiering has no explicit edge-case policy for very small positive sets, so labels can become degenerate silently.

Ground truth assigns tertiles using `third = n // 3` in [ml/ground_truth.py:169](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/ground_truth.py#L169). For `n < 3`, `third` becomes `0`, so all positive branches will be assigned tier `3` by [ml/ground_truth.py:175](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/ground_truth.py#L175). That may be acceptable in practice if every quarter has many positives, but the policy is implicit rather than frozen. If sparse groups ever appear, the labels will collapse without a warning or a testable rule.

### 8. LOW: The threshold calibration artifact is branch-level and annual-bridge-only, but the rest of the pipeline wording still makes it easy to over-read it as a general DA coverage threshold.

[scripts/calibrate_threshold.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/calibrate_threshold.py) now documents the branch-level contract clearly, which is good. But the threshold is still chosen against annual-bridge-mapped branch SP only, without monthly fallback. That is fine if treated as a pragmatic calibration artifact, but it should not be casually described elsewhere as “95% of all DA SP” coverage. This is more of a contract-discipline issue than a code bug.

## Additional Notes

- The data-processing core is in better shape than the reporting layer. The important shared contracts now look consistent in:
  - [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/data_loader.py)
  - [ml/ground_truth.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/ground_truth.py)
  - [ml/history_features.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/history_features.py)
  - [ml/realized_da.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/realized_da.py)

- The biggest remaining reliability problem is no longer “feature engineering bugs.” It is that the official evaluation and registry contract has not kept up with the new shortlist policy and two-track analysis workflow.

## Recommended Fix Order

1. Move `@150/@300/@200/@400` and dangerous-branch metrics into [ml/evaluate.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/evaluate.py) and update [ml/config.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/config.py) gate definitions accordingly.
2. Fix [ml/bridge.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/bridge.py) so ambiguity-value diagnostics accept a caller-specified payload column or infer from `realized_sp` / `total_sp`.
3. Stop labeling two-track `R=0` runs as solo baselines; compare to actual saved solo model outputs instead.
4. Save gate artifacts in [scripts/run_ml_experiment.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_ml_experiment.py) for all future ML registry entries.
5. Remove or implement the `v3a` Track A option in [scripts/run_two_track_experiment.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/run_two_track_experiment.py).
