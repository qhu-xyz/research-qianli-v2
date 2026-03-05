# Direction — Iteration 2 (batch: tier-fe-2)

## CRITICAL: READ THIS BEFORE DOING ANYTHING

**3 consecutive worker iterations have FAILED** because the worker wrote the handoff signal without doing any actual work. DO NOT write the handoff signal until ALL steps below are complete and verified. The handoff is the LAST thing you do, not the first.

## Single Hypothesis: Add 3 Interaction Features (34 → 37)

No screening. No A/B comparison. One hypothesis, full benchmark.

**What**: Add overload_x_hist, prob110_x_recent_hist, tail_x_hist to the existing 34 features.

**Why**: Top features (recent_hist_da 21.1%, hist_da 13.3%) are used independently. Pre-computing products with physical flow signals should help tier 0/1 discrimination and improve Tier-VC@100 (currently 0.071, needs ≥0.075).

## Step-by-Step Instructions

Execute these steps IN ORDER. Do NOT skip any step. Do NOT write the handoff until step 6.

### Step 1: Modify `ml/features.py`

Add 3 new interaction columns to `compute_interaction_features()`. The function currently computes 5 dead features. Add 3 more columns after them:

```python
def compute_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Existing 5 dead features (keep as-is)
        (pl.col("hist_da") * pl.col("prob_exceed_100"))
            .alias("hist_physical_interaction"),
        (pl.col("expected_overload") * pl.col("prob_exceed_105"))
            .alias("overload_exceedance_product"),
        (pl.col("prob_band_95_100") * pl.col("expected_overload"))
            .alias("band_severity"),
        (pl.col("sf_max_abs") * pl.col("prob_exceed_100"))
            .alias("sf_exceed_interaction"),
        (pl.col("hist_da_max_season") * pl.col("prob_band_100_105"))
            .alias("hist_seasonal_band"),
        # NEW: 3 interaction features
        (pl.col("expected_overload") * pl.col("hist_da"))
            .alias("overload_x_hist"),
        (pl.col("prob_exceed_110") * pl.col("recent_hist_da"))
            .alias("prob110_x_recent_hist"),
        (pl.col("tail_concentration") * pl.col("hist_da"))
            .alias("tail_x_hist"),
    ])
```

### Step 2: Modify `ml/config.py`

Add 3 features to `_ALL_TIER_FEATURES` and 3 monotone constraints to `_ALL_TIER_MONOTONE`.

Change the end of `_ALL_TIER_FEATURES` (line 82-93) from:
```python
_ALL_TIER_FEATURES: list[str] = _V1_CLF_FOR_TIER + [
    "prob_exceed_85",
    "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
    "density_skewness",
    "density_kurtosis",
    "density_cv",
    "season_hist_da_3",
    "prob_below_85",
]
```
to:
```python
_ALL_TIER_FEATURES: list[str] = _V1_CLF_FOR_TIER + [
    "prob_exceed_85",
    "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1",
    "season_hist_da_2",
    "density_skewness",
    "density_kurtosis",
    "density_cv",
    "season_hist_da_3",
    "prob_below_85",
    "overload_x_hist",
    "prob110_x_recent_hist",
    "tail_x_hist",
]
```

Change the end of `_ALL_TIER_MONOTONE` (line 95-102) from:
```python
_ALL_TIER_MONOTONE: list[int] = _V1_CLF_MONO_FOR_TIER + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
    0, 0, 0,  # density_skewness, density_kurtosis, density_cv
    1,        # season_hist_da_3
    -1,       # prob_below_85
]
```
to:
```python
_ALL_TIER_MONOTONE: list[int] = _V1_CLF_MONO_FOR_TIER + [
    1, 1,     # prob_exceed_85, prob_exceed_80
    1,        # recent_hist_da
    1, 1,     # season_hist_da_1, season_hist_da_2
    0, 0, 0,  # density_skewness, density_kurtosis, density_cv
    1,        # season_hist_da_3
    -1,       # prob_below_85
    1, 1, 1,  # overload_x_hist, prob110_x_recent_hist, tail_x_hist
]
```

### Step 3: Update tests

Find the test that checks feature count (34) and update it to 37. Look in `ml/tests/` for assertions like `== 34` or `len(features) == 34`.

### Step 4: Run tests

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python -m pytest ml/tests/ -v
```

All tests must pass before proceeding.

### Step 5: Run full 12-month benchmark

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python ml/benchmark.py --version-id v0003 2>&1 | tee registry/v0003/benchmark.log
```

Wait for this to complete. It will take several minutes. Do NOT proceed until it finishes.

### Step 6: Verify artifacts exist

Before writing the handoff, verify:
```bash
ls registry/v0003/metrics.json registry/v0003/config.json registry/v0003/changes_summary.md
```

All 3 files must exist. If any are missing, the benchmark did not complete — investigate and fix before continuing.

### Step 7: Run comparison

```bash
PYTHONPATH=. python ml/compare.py --candidate v0003 --champion v0 2>&1 | tee reports/tier-fe-2-20260304-225923/iter2/comparison.md
```

### Step 8: Write changes_summary.md

Write `registry/v0003/changes_summary.md` describing:
- What changed: added 3 interaction features (overload_x_hist, prob110_x_recent_hist, tail_x_hist)
- Feature count: 34 → 37
- Why: compound severity signals for tier 0/1 discrimination

### Step 9: Write handoff signal (ONLY after steps 1-8 are complete)

Only now write the handoff JSON.

## Allowed File Modifications

- `ml/features.py` — `compute_interaction_features()` only
- `ml/config.py` — `_ALL_TIER_FEATURES` and `_ALL_TIER_MONOTONE` lists only
- `ml/tests/` — feature count assertions only
- `registry/v0003/` — version artifacts (created by benchmark)

## FORBIDDEN Changes

- Any TierConfig field except features/monotone_constraints
- ml/train.py, ml/pipeline.py, ml/evaluate.py, ml/benchmark.py
- registry/gates.json
- Any hyperparameter, class weight, bin, or midpoint
