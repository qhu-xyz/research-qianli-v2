# Direction — Iteration 3 (FINAL) (batch: tier-fe-1-20260304-182037)

## CRITICAL: Two previous iterations FAILED

Iterations 1 and 2 both failed identically — worker wrote handoff claiming "done" but produced NO artifacts. **This is the LAST iteration.** The worker MUST produce artifacts or the batch ends with zero data.

**Failure pattern to avoid**: Do NOT write the handoff signal until ALL artifacts exist in `registry/v0003/`. Verify files exist before writing handoff.

## Batch Constraint

Feature engineering ONLY. No hyperparameter changes, no class weight changes, no bin/midpoint changes.

## Single Hypothesis: Add 3 interaction features (34 → 37 features)

Add these 3 features from `_DEAD_FEATURES` in `ml/config.py`:
- `hist_physical_interaction` = hist_da × prob_exceed_100
- `overload_exceedance_product` = expected_overload × prob_exceed_105
- `hist_seasonal_band` = hist_da_max_season × prob_band_100_105

These are already computed by `compute_interaction_features()` in `ml/features.py`. They exist as DataFrame columns but are excluded by `_DEAD_FEATURES`. No new feature computation code is needed.

**Monotone constraints for the 3 new features**: all `0` (interactions don't have guaranteed monotonicity).

## Exact Code Changes Required

### Step 1: Edit `ml/config.py`

Remove `hist_physical_interaction`, `overload_exceedance_product`, and `hist_seasonal_band` from `_DEAD_FEATURES` (lines 17-23). The set should become:
```python
_DEAD_FEATURES: set[str] = {
    "band_severity",
    "sf_exceed_interaction",
}
```

That's it. The filtering logic at lines 72-79 will automatically include these 3 features in `_V1_CLF_FOR_TIER`, and they will flow into `_ALL_TIER_FEATURES` (which builds on `_V1_CLF_FOR_TIER`). The monotone constraints for these features are already defined at line 62 as `0, 0` (for hist_physical_interaction, overload_exceedance_product) and line 69 as `0` (for hist_seasonal_band).

**Result**: Feature count goes from 34 → 37. No other code changes needed.

### Step 2: Update tests

If any test in `ml/tests/` asserts feature count == 34, update to 37.

### Step 3: Run tests

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python -m pytest ml/tests/ -v
```

### Step 4: Run full 12-month benchmark

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python ml/benchmark.py --version-id v0003 2>&1 | tee registry/v0003/benchmark.log
```

Note: version_id is v0003 because v0001 and v0002 were consumed by failed iterations. The benchmark.py will create `registry/v0003/` and populate it.

### Step 5: Run comparison

```bash
PYTHONPATH=. python ml/compare.py --candidate v0003 --champion v0
```

### Step 6: Write changes_summary.md

Create `registry/v0003/changes_summary.md` documenting the feature addition.

### Step 7: VERIFY artifacts exist before handoff

```bash
ls registry/v0003/metrics.json registry/v0003/config.json registry/v0003/changes_summary.md
```

All 3 files MUST exist before writing the handoff signal.

## NO Screening Phase

Skip screening. Go directly to the full 12-month benchmark. This is the last iteration — we need all months anyway, and screening adds execution complexity that contributed to prior failures.

## Expected Impact

| Gate | Current (v0 mean) | Floor | Expected Direction |
|------|-------------------|-------|--------------------|
| Tier-VC@100 | 0.071 | 0.075 | ↑ Interaction features provide compound severity signal |
| Tier-VC@500 | 0.230 | 0.217 | → Neutral to slight improvement |
| Tier-NDCG | 0.771 | 0.767 | → Neutral |
| QWK | 0.370 | 0.359 | ↑ Better tier 0/1 discrimination |

The 3 interaction features combine price history (model's strongest signal, 34.4% combined gain) with flow exceedance (physical cause of binding). This should help XGBoost detect compound severity patterns that distinguish tier 0/1 from tier 2/3.
