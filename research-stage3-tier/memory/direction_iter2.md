# Direction — Iteration 2 (batch: tier-fe-1-20260304-182037)

## Recovery Context

Iter 1 FAILED — worker produced no artifacts despite claiming "done". The planned hypotheses (interaction feature swap vs aggressive pruning) were never tested. This iteration retries with a **single, simplified hypothesis** to reduce execution risk.

## Situation Analysis

**Champion**: v0 (34 features, baseline)
**Batch constraint**: Feature engineering / selection ONLY (no hyperparams, no class weights, no bins/midpoints)
**Key v0 problems** (unchanged):
1. Tier-Recall@1 = 0.047 — model almost never predicts tier 1 ([1000, 3000))
2. Tier-VC@100 = 0.071 — top-100 ranking quality very poor (below floor 0.075)
3. High monthly variance: 2021-11 is a disaster month

**Feature importance** (v0 12-month mean gain, bottom 5):
- density_skewness: 0.0109
- prob_exceed_90: 0.0113
- density_cv: 0.0113
- density_variance: 0.0117
- prob_below_90: 0.0119

**Key insight**: `compute_interaction_features()` in `features.py` already computes 5 interaction features on every pipeline run — they exist as DataFrame columns but are excluded from the model by `_DEAD_FEATURES` in `config.py`. Reintroducing them requires ONLY config changes (no features.py edits needed). Additionally, 3 new interaction features from `human_input.md` candidates can be added to `compute_interaction_features()`.

---

## Single Hypothesis: Prune 3 + Reintroduce 2 existing interactions + Add 3 new interactions

**Rationale**: The model needs compound severity signals to distinguish tier 0/1 from tier 2/3. The most promising existing interactions are:
- `hist_physical_interaction` (hist_da × prob_exceed_100) — combines the #1 and #2 importance features with flow exceedance
- `overload_exceedance_product` (expected_overload × prob_exceed_105) — compound overload signal

The most promising NEW interactions (from human_input.md candidates):
- `overload_x_hist` (expected_overload × hist_da) — historical binding severity × current overload
- `prob110_x_hist` (prob_exceed_110 × recent_hist_da) — extreme flow × recent price
- `tail_x_hist` (tail_concentration × hist_da) — upper-tail density × price signal

Pruning the 3 lowest-importance features removes noise and keeps feature count manageable.

**Changes**:
- **Remove** from features: `density_skewness` (0.0109), `prob_exceed_90` (0.0113), `density_cv` (0.0113)
- **Reintroduce** from _DEAD_FEATURES: `hist_physical_interaction`, `overload_exceedance_product`
- **Add NEW** to compute_interaction_features: `overload_x_hist`, `prob110_x_hist`, `tail_x_hist`
- **Net**: 34 - 3 pruned + 2 reintroduced + 3 new = 36 features

---

## Step-by-Step Worker Instructions

### Step 1: Edit `ml/config.py` — Remove 3 features from `_ALL_TIER_FEATURES`

Remove these 3 entries from `_ALL_TIER_FEATURES` and their corresponding monotone values from `_ALL_TIER_MONOTONE`:
- `density_skewness` (monotone: 0)
- `density_cv` (monotone: 0)
- `prob_exceed_90` (monotone: 1)

### Step 2: Edit `ml/config.py` — Reintroduce 2 from `_DEAD_FEATURES`

Remove `hist_physical_interaction` and `overload_exceedance_product` from `_DEAD_FEATURES` set.

Add to end of `_ALL_TIER_FEATURES`:
```python
"hist_physical_interaction",
"overload_exceedance_product",
```
Add to end of `_ALL_TIER_MONOTONE`:
```python
1, 1,
```

### Step 3: Edit `ml/features.py` — Add 3 new interaction features

Add these 3 new columns to `compute_interaction_features()`:
```python
(pl.col("expected_overload") * pl.col("hist_da"))
    .alias("overload_x_hist"),
(pl.col("prob_exceed_110") * pl.col("recent_hist_da"))
    .alias("prob110_x_hist"),
(pl.col("tail_concentration") * pl.col("hist_da"))
    .alias("tail_x_hist"),
```

### Step 4: Edit `ml/config.py` — Add 3 new features to config

Add to end of `_ALL_TIER_FEATURES`:
```python
"overload_x_hist",
"prob110_x_hist",
"tail_x_hist",
```
Add to end of `_ALL_TIER_MONOTONE`:
```python
1, 1, 1,
```

### Step 5: Update tests

Update any test in `ml/tests/` that asserts feature count = 34 to assert 36.

### Step 6: Run tests

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python -m pytest ml/tests/ -v
```

All tests must pass before proceeding.

### Step 7: Run full 12-month benchmark

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python ml/benchmark.py --version-id v0002 2>&1 | tee registry/v0002/benchmark.log
```

### Step 8: Run validate + compare

```bash
PYTHONPATH=. python ml/validate.py --version-id v0002
PYTHONPATH=. python ml/compare.py --new v0002 --champion v0 --batch-id tier-fe-1-20260304-182037 --iter 2
```

### Step 9: Write changes_summary.md

Write `registry/v0002/changes_summary.md` describing what changed.

---

## Verification Checklist

Before writing handoff:
- [ ] `_ALL_TIER_FEATURES` has exactly 36 entries
- [ ] `_ALL_TIER_MONOTONE` has exactly 36 entries
- [ ] `_DEAD_FEATURES` has 3 entries (was 5, removed 2)
- [ ] `compute_interaction_features()` computes 8 features (was 5, added 3)
- [ ] Tests pass
- [ ] `registry/v0002/metrics.json` exists
- [ ] `reports/tier-fe-1-20260304-182037/iter2/comparison.md` exists

---

## Expected Impact

| Gate | Current (v0 mean) | Floor | Expected Direction |
|------|-------------------|-------|--------------------|
| Tier-VC@100 | 0.071 | 0.075 | ↑ Interaction features should improve top-ranking for tier 0/1 |
| Tier-VC@500 | 0.230 | 0.217 | ↑ Better tier discrimination from compound signals |
| Tier-NDCG | 0.771 | 0.767 | → Neutral to slight improvement |
| QWK | 0.370 | 0.359 | ↑ Better tier 1 detection improves ordinal consistency |

**Most impactful**: Tier-VC@100 is currently below floor (0.071 vs 0.075). The interaction features provide pre-computed compound severity — constraints with BOTH high prices AND flow violations — which should rank higher in tier_ev scoring.

---

## Risk Assessment

1. **Interaction features were dead in stage-2 regression**: But multi-class classification uses different splits. Tier boundary discrimination may benefit from pre-computed interactions that regression did not.
2. **3 new features untested**: They are simple products of high-importance base features. At worst they add mild noise; at best they provide strong tier 0/1 signal.
3. **Feature count increase (34→36)**: Modest. With colsample_bytree=0.8 and min_child_weight=25, unlikely to cause overfitting.
