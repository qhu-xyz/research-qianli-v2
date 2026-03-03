# Direction — Iteration 2 (feat-eng-20260303-060938)

## Hypothesis H7: Extended Training Window (18 months) + Feature Importance Diagnostic

**Core question**: Does further window expansion (14→18 months) continue the positive AUC trend, and which of the 17 features actually drive predictions?

**Rationale**: Window expansion has been the only productive lever across 4 real-data experiments. The 10→14 expansion gave AUC +0.0013 (7W/4L). If the marginal benefit of additional training diversity hasn't saturated, train_months=18 could push AUC toward 0.837+. This iteration also collects the first empirical feature importance data — after 4 experiments, we still don't know which features drive the model. This data is essential for iter 3 strategy (prune noise vs add new features).

## Specific Changes (Priority Order)

### Change 1 (PRIMARY): Expand training window from 14 to 18 months

**File**: `ml/config.py`, line 95
**What**: Change `train_months: int = 14` → `train_months: int = 18` in `PipelineConfig`

This gives the model 80% more training data than v0's original 10-month window and 29% more than v0004's 14-month window. The increased seasonal diversity should help late-2022 months especially.

**Feasibility**: Earliest eval month is 2020-09. With train_months=18 + val_months=2 = 20 months lookback, the model needs data from ~2019-01. MISO data should go back further than this. If data is unavailable for early months, reduce to train_months=16 as fallback and document.

### Change 2 (SECONDARY): Export feature importance from each trained model

**File**: `ml/benchmark.py`

Capture XGBoost's gain-based feature importance during the per-month evaluation loop, then save to a separate JSON file. Implementation:

**Step A** — In `_eval_single_month()`, capture importance before cleanup. Between the `evaluate_classifier` call (line 81) and the cleanup `del` block (line 84), add:

```python
# Feature importance (gain-based)
importance = dict(zip(feature_config.features, model.feature_importances_.tolist()))
metrics["_feature_importance"] = importance
```

Note the underscore prefix `_feature_importance` to signal it's metadata, not a gate metric.

**Step B** — In `run_benchmark()`, extract importance from per_month dicts BEFORE calling `aggregate_months()`. After the eval loop (after `per_month[month] = metrics` on line 178) and before `agg = aggregate_months(per_month)` on line 188, add:

```python
# Extract feature importance before aggregation (not a numeric metric)
importance_per_month = {}
for month in list(per_month.keys()):
    imp = per_month[month].pop("_feature_importance", None)
    if imp:
        importance_per_month[month] = imp
```

**Step C** — After writing metrics.json (around line 213), save feature importance to a separate file:

```python
if importance_per_month:
    import statistics
    all_features = list(next(iter(importance_per_month.values())).keys())
    mean_imp = {}
    std_imp = {}
    for feat in all_features:
        vals = [importance_per_month[m].get(feat, 0.0) for m in importance_per_month]
        mean_imp[feat] = statistics.mean(vals)
        std_imp[feat] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    fi_data = {
        "importance_type": "gain",
        "n_months": len(importance_per_month),
        "per_month": importance_per_month,
        "aggregate": {
            "mean": mean_imp,
            "std": std_imp,
        },
        "ranked": sorted(mean_imp.items(), key=lambda x: x[1], reverse=True),
    }
    with open(version_dir / "feature_importance.json", "w") as f:
        json.dump(fi_data, f, indent=2)
    print(f"[benchmark] Wrote feature importance to {version_dir / 'feature_importance.json'}")
```

**Critical**: The `_feature_importance` key MUST be popped from per_month dicts before `aggregate_months()` is called. The aggregation function expects only numeric values and will crash or produce garbage if it encounters a dict.

**Priority note**: If implementing feature importance proves complex (e.g., breaks tests, requires significant refactoring), the worker MAY skip it and note in the handoff. The window expansion experiment is the primary deliverable. Feature importance is a valuable diagnostic but not blocking.

### Change 3: No feature changes

Keep all 17 features (14 base + 3 interactions) in `FeatureConfig`. Do NOT modify.

### Change 4: No hyperparameter changes

Keep all HPs at v0 defaults in `HyperparamConfig`. Do NOT modify.

### Change 5: Update tests

- Update any tests that assert `train_months == 14` to expect `18`
- If adding feature importance to benchmark, ensure existing test assertions on metrics.json structure still pass (importance should be in a separate file, not in metrics.json)

## Expected Impact

| Metric | v0 Baseline | v0004 (iter 1) | Expected v0005 (trend continues) | Expected (diminishing returns) |
|--------|-------------|----------------|----------------------------------|-------------------------------|
| S1-AUC | 0.8348 | 0.8363 | 0.837–0.839 | 0.836–0.837 |
| S1-AP | 0.3936 | 0.3951 | 0.396–0.400 | ~0.395 (flat) |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.022–0.028 | ~0.020 |
| S1-NDCG | 0.7333 | 0.7371 | 0.738–0.742 | ~0.737 |
| AUC W/L vs v0 | — | 9W/3L | ≥9/12 | 7–8/12 |

**Honest assessment**: Diminishing returns is the more likely outcome. The 10→14 expansion was only +0.0013 AUC, and older training data (2018-2019) may be less informative than the recent months already in the window. The primary value of this iteration is the feature importance diagnostic data.

## Risk Assessment

1. **Diminishing returns (MEDIUM probability)**: 14→18 may give AUC +0.0005 or less. Mitigation: even zero AUC gain is acceptable if we get feature importance data to guide iter 3.

2. **VCAP@500 may breach Group B floor (LOW-MEDIUM)**: v0004 bot2=0.0387 vs floor=0.0408 (margin 0.0021). Three consecutive experiments show VCAP@500 regression. Mitigation: Group B, non-blocking. Document if breached.

3. **Data availability for early eval months (LOW)**: 2020-09 needs ~2019-01 training data. Should be available. If not, reduce to train_months=16 as fallback.

4. **BRIER headroom narrowing (LOW concern)**: Currently 0.0187. More data slightly degrades calibration. Group B, non-blocking.

5. **2022-09 still stuck (HIGH likelihood, LOW impact)**: This month's AP=0.307 has resisted 4 interventions. Expected failure, document.

## Success Criteria

| Outcome | Criteria | Iter 3 Action |
|---------|----------|---------------|
| **Promotion-worthy** | AUC > 0.837, ≥8/12 wins, AP > 0.396 | Promote, end batch |
| **Encouraging** | AUC ≥ 0.836, 7+/12 wins, feature importance collected | Use importance to guide iter 3 (prune or add features) |
| **Diminishing returns** | AUC ≤ v0004 (0.8363), or W/L ≤ 6/6 | Window expansion exhausted. Iter 3: feature importance-guided pruning or ratio features |
| **Regression** | AUC < v0 (0.8348) | Revert to 14-month window. Iter 3: feature pruning |

## Worker Checklist

1. Read VERSION_ID from `${PROJECT_DIR}/state.json` (NOT the worktree copy)
2. Change `train_months` from 14 to 18 in `ml/config.py` line 95
3. Add feature importance extraction to `ml/benchmark.py` (per Step A/B/C above)
4. Update tests if needed: `python -m pytest ml/tests/ -v`
5. Run benchmark: `python ml/benchmark.py --version-id ${VERSION_ID} --ptype f0 --class-type onpeak`
6. Verify `registry/${VERSION_ID}/metrics.json` has 12 months (none skipped)
7. Verify `registry/${VERSION_ID}/feature_importance.json` was created (if implemented)
8. Run `python ml/validate.py --version-id ${VERSION_ID}` to confirm gate compliance
9. Run `python ml/compare.py --version-id ${VERSION_ID} --baseline v0` to generate comparison
10. Commit all changes, then write handoff JSON

## What NOT To Do

- Do NOT change hyperparameters (confirmed dead end — 4 experiments)
- Do NOT change threshold_beta (keep 0.7)
- Do NOT change val_months (keep 2)
- Do NOT modify gates.json or evaluate.py
- Do NOT add or remove features (keep all 17)
- Do NOT touch registry/v0/ or any other registry/v*/ except the assigned VERSION_ID
- Do NOT invest >30 minutes on feature importance extraction if it proves complex — it's diagnostic, not critical
