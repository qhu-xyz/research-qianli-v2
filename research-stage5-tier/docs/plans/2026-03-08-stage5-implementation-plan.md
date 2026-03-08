# Stage 5 ML Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ML-based constraint ranking pipeline that uses realized DA shadow prices as ground truth (fixing stage 4's circular evaluation), and run v0/v1/v1b experiments.

**Architecture:** Copy audited modules from stage 4 (`research-stage4-tier/ml/`), add realized DA caching layer, swap ground truth from `shadow_price_da` to `realized_sp`, run LightGBM lambdarank experiments with versioned feature sets.

**Tech Stack:** Python 3.11, polars, LightGBM (lambdarank), numpy, scipy, Ray (for realized DA fetch only)

**Stage 4 source:** `/home/xyz/workspace/research-qianli-v2/research-stage4-tier/`
**Stage 5 target:** `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/`
**Python venv:** `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`
**Design doc:** `docs/plans/2026-03-08-stage5-ml-pipeline-design.md`

---

### Task 1: Project scaffolding

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/tests/__init__.py`

**Step 1:** Create directories and init files.

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
mkdir -p ml/tests scripts data/realized_da registry
touch ml/__init__.py ml/tests/__init__.py
```

**Step 2:** Commit.

```bash
git add ml/ scripts/ data/ registry/
git commit -m "task 1: project scaffolding for stage5 ML pipeline"
```

---

### Task 2: Copy and audit evaluate.py

**Files:**
- Copy: `ml/evaluate.py` from stage4
- Copy: `ml/tests/test_evaluate.py` from stage4

**Step 1:** Copy evaluate.py from stage4.

```bash
cp ../research-stage4-tier/ml/evaluate.py ml/evaluate.py
cp ../research-stage4-tier/ml/tests/test_evaluate.py ml/tests/test_evaluate.py
```

**Step 2:** Audit evaluate.py line-by-line.

Checklist:
- [x] No reference to `shadow_price_da` -- PASS (takes generic `actual_shadow_price` numpy array)
- [x] No feature leakage -- PASS (pure metric computation)
- [x] Higher score = better convention -- PASS (all metrics expect higher=better)
- [x] No constraint_id handling -- PASS (operates on numpy arrays only)

This module is a pure metric library. No changes needed.

**Step 3:** Run existing tests to verify.

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
python -m pytest ml/tests/test_evaluate.py -v
```

Expected: all tests pass.

**Step 4:** Commit.

```bash
git add ml/evaluate.py ml/tests/test_evaluate.py
git commit -m "task 2: copy and audit evaluate.py (pure metrics, no changes needed)"
```

---

### Task 3: Copy and audit train.py

**Files:**
- Copy: `ml/train.py` from stage4
- Copy: `ml/tests/test_train.py` from stage4

**Step 1:** Copy.

```bash
cp ../research-stage4-tier/ml/train.py ml/train.py
cp ../research-stage4-tier/ml/tests/test_train.py ml/tests/test_train.py
```

**Step 2:** Audit train.py line-by-line.

Checklist:
- [x] Takes generic `y_train` numpy array -- PASS (doesn't know what column it came from)
- [x] `_rank_transform_labels` operates on passed-in y values -- PASS
- [x] No reference to `shadow_price_da` -- PASS
- [x] No data loading -- PASS (pure training logic)
- [x] LightGBM lambdarank label_gain computed from actual labels -- PASS
- [x] Monotone constraints passed from LTRConfig -- PASS

No changes needed. The circularity is in what gets passed as y_train, not in this module.

**Step 3:** Run tests.

```bash
python -m pytest ml/tests/test_train.py -v
```

Expected: all tests pass. (Tests may need config.py first -- if so, defer to after task 5.)

**Step 4:** Commit.

```bash
git add ml/train.py ml/tests/test_train.py
git commit -m "task 3: copy and audit train.py (pure training, no changes needed)"
```

---

### Task 4: Copy and audit features.py, v62b_formula.py, spice6_loader.py

**Files:**
- Copy: `ml/features.py`, `ml/v62b_formula.py`, `ml/spice6_loader.py` from stage4
- Copy: `ml/tests/test_features.py`, `ml/tests/test_v62b_formula.py` from stage4

**Step 1:** Copy all.

```bash
cp ../research-stage4-tier/ml/features.py ml/features.py
cp ../research-stage4-tier/ml/v62b_formula.py ml/v62b_formula.py
cp ../research-stage4-tier/ml/spice6_loader.py ml/spice6_loader.py
cp ../research-stage4-tier/ml/tests/test_features.py ml/tests/test_features.py
cp ../research-stage4-tier/ml/tests/test_v62b_formula.py ml/tests/test_v62b_formula.py
```

**Step 2:** Audit features.py.

- [x] `prepare_features` extracts columns listed in `cfg.features` -- PASS
- [x] `compute_query_groups` counts rows per query_month -- PASS
- [x] No reference to shadow_price_da -- PASS
- [x] No data loading -- PASS

**Step 3:** Audit v62b_formula.py.

- [x] `v62b_score()` computes: `0.6*da_rank + 0.3*dmix + 0.1*dori` -- PASS
- [x] `dense_rank_normalized()` is pure math -- PASS
- [x] No reference to ground truth -- PASS

**Step 4:** Audit spice6_loader.py.

- [x] Reads `score_df.parquet` columns `"110"`, `"100"`, `"90"`, `"85"`, `"80"` -- VERIFIED on disk
- [x] Renames to `prob_exceed_*` via `.alias()` -- PASS
- [x] Aggregates across outage dates by mean -- PASS
- [x] Joins constraint_limit from `limit.parquet` -- PASS
- [x] Returns (constraint_id, flow_direction, prob_exceed_*, constraint_limit) -- PASS
- [x] No reference to shadow_price_da or realized DA -- PASS

**Step 5:** Run tests.

```bash
python -m pytest ml/tests/test_features.py ml/tests/test_v62b_formula.py -v
```

**Step 6:** Commit.

```bash
git add ml/features.py ml/v62b_formula.py ml/spice6_loader.py ml/tests/
git commit -m "task 4: copy and audit features, v62b_formula, spice6_loader"
```

---

### Task 5: Write config.py (modified from stage4)

**Files:**
- Create: `ml/config.py`
- Copy + modify: from stage4 `ml/config.py`
- Copy: `ml/tests/test_config.py` from stage4

**Step 1:** Copy stage4 config.py and test.

```bash
cp ../research-stage4-tier/ml/config.py ml/config.py
cp ../research-stage4-tier/ml/tests/test_config.py ml/tests/test_config.py
```

**Step 2:** Apply modifications to `ml/config.py`:

1. **Fix misleading comment** (lines 16-18):
   ```python
   # OLD:
   # The 60% da_rank_value component is pure leakage.

   # NEW:
   # da_rank_value is a historical 60-month lookback, legitimate as a feature.
   # What IS leaky: rank, rank_ori, tier (derived formula outputs).
   ```

2. **Add Group C (historical DA) and Group D (ML predictions)**:
   ```python
   # Group C: Historical DA signal
   _HIST_DA_FEATURES: list[str] = ["da_rank_value"]
   _HIST_DA_MONOTONE: list[int] = [-1]  # lower rank_value = more binding

   # Group D: ML predictions (from ml_pred/final_results.parquet)
   _MLPRED_FEATURES: list[str] = [
       "predicted_shadow_price",
       "binding_probability",
       "binding_probability_scaled",
   ]
   _MLPRED_MONOTONE: list[int] = [1, 1, 1]
   ```

3. **Replace composed feature lists**:
   ```python
   # Remove _ENGINEERED_FEATURES, _ENGINEERED_MONOTONE, _FULL_FEATURES, _FULL_MONOTONE

   # v1: Groups A+B (11 features) -- pure forecasts, no historical DA
   FEATURES_V1: list[str] = _V62B_FEATURES + _SPICE6_FEATURES
   MONOTONE_V1: list[int] = _V62B_MONOTONE + _SPICE6_MONOTONE

   # v1b: Groups A+B+C (12 features) -- add historical DA signal
   FEATURES_V1B: list[str] = _V62B_FEATURES + _SPICE6_FEATURES + _HIST_DA_FEATURES
   MONOTONE_V1B: list[int] = _V62B_MONOTONE + _SPICE6_MONOTONE + _HIST_DA_MONOTONE

   # v3: Groups A+B+C+D (15 features) -- add ML predictions
   FEATURES_V3: list[str] = FEATURES_V1B + _MLPRED_FEATURES
   MONOTONE_V3: list[int] = MONOTONE_V1B + _MLPRED_MONOTONE
   ```

4. **Change PipelineConfig defaults**:
   ```python
   @dataclass
   class PipelineConfig:
       ltr: LTRConfig = field(default_factory=LTRConfig)
       train_months: int = 8   # was 6
       val_months: int = 0     # was 2
   ```

5. **Add data paths for new sources**:
   ```python
   SPICE6_MLPRED_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/ml_pred"
   REALIZED_DA_CACHE = "data/realized_da"
   ```

**Step 3:** Update test_config.py if needed (adjust for new defaults).

**Step 4:** Run tests.

```bash
python -m pytest ml/tests/test_config.py -v
```

**Step 5:** Commit.

```bash
git add ml/config.py ml/tests/test_config.py
git commit -m "task 5: config with correct feature groups, fixed comment, 8/0 defaults"
```

---

### Task 6: Write realized_da.py (new module)

**Files:**
- Create: `ml/realized_da.py`
- Create: `ml/tests/test_realized_da.py`

**Step 1:** Write test for `load_realized_da()` (reads cached parquet).

```python
# ml/tests/test_realized_da.py
import polars as pl
import pytest
from pathlib import Path

def test_load_realized_da_returns_correct_schema(tmp_path):
    """Cached parquet should have constraint_id (str) and realized_sp (f64)."""
    from ml.realized_da import load_realized_da

    # Create a fake cached parquet
    df = pl.DataFrame({
        "constraint_id": ["1000", "2000", "3000"],
        "realized_sp": [150.0, 0.0, 75.5],
    })
    month_path = tmp_path / "2022-06.parquet"
    df.write_parquet(month_path)

    result = load_realized_da("2022-06", cache_dir=str(tmp_path))
    assert result.columns == ["constraint_id", "realized_sp"]
    assert result["constraint_id"].dtype == pl.String
    assert result["realized_sp"].dtype == pl.Float64
    assert len(result) == 3

def test_load_realized_da_missing_month_raises(tmp_path):
    from ml.realized_da import load_realized_da
    with pytest.raises(FileNotFoundError):
        load_realized_da("9999-01", cache_dir=str(tmp_path))
```

**Step 2:** Run test -- expect FAIL.

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

**Step 3:** Write `ml/realized_da.py`.

```python
"""Realized DA shadow price fetching and caching.

Ground truth for the LTR ranking pipeline: actual DA shadow prices
for a delivery month, fetched from MISO market data via Ray.

Aggregation: abs(sum(shadow_price)) per constraint_id.
This nets directional shadow prices within the month, then takes magnitude.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import REALIZED_DA_CACHE


def load_realized_da(
    month: str,
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Load cached realized DA for a single month.

    Returns DataFrame with columns [constraint_id (str), realized_sp (f64)].
    """
    path = Path(cache_dir) / f"{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cached realized DA not found: {path}")
    return pl.read_parquet(path)


def fetch_and_cache_month(month: str, cache_dir: str = REALIZED_DA_CACHE) -> Path:
    """Fetch realized DA for one month via Ray and cache to parquet.

    Requires Ray to be initialized before calling.
    """
    import pandas as pd
    from pbase.analysis.tools.all_positions import MisoApTools

    st = pd.Timestamp(f"{month}-01")
    et = st + pd.DateOffset(months=1)

    aptools = MisoApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st, et_ex=et, peak_type="onpeak"
    )

    # Aggregate: abs(sum(shadow_price)) per constraint_id
    realized = (
        da_shadow.groupby("constraint_id")["shadow_price"]
        .sum()
        .reset_index()
    )
    realized.columns = ["constraint_id", "realized_sp"]
    realized["realized_sp"] = realized["realized_sp"].abs()

    # Convert to polars and ensure types
    result = pl.from_pandas(realized)
    result = result.with_columns([
        pl.col("constraint_id").cast(pl.String),
        pl.col("realized_sp").cast(pl.Float64),
    ])

    out_path = Path(cache_dir) / f"{month}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)
    return out_path
```

**Step 4:** Run tests -- expect PASS.

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

**Step 5:** Commit.

```bash
git add ml/realized_da.py ml/tests/test_realized_da.py
git commit -m "task 6: realized DA fetch/cache module with tests"
```

---

### Task 7: Write cache_realized_da.py script

**Files:**
- Create: `scripts/cache_realized_da.py`

**Step 1:** Write the caching script.

```python
"""One-time script: fetch and cache realized DA for all needed months.

Requires Ray. Run once, then pipeline uses cached parquets without Ray.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
    python scripts/cache_realized_da.py
"""
import os
import sys
import resource
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def main():
    import pandas as pd

    # All months needed: training lookback + eval months
    # Eval: 2020-06 to 2023-05 (36 full months)
    # Training: 8 months before earliest eval = 2019-10
    # Buffer: start from 2019-06 to be safe
    all_months = [
        f"{y:04d}-{m:02d}"
        for y in range(2019, 2024)
        for m in range(1, 13)
        if (y, m) >= (2019, 6) and (y, m) <= (2023, 5)
    ]

    cache_dir = "data/realized_da"
    existing = set(p.stem for p in Path(cache_dir).glob("*.parquet"))
    to_fetch = [m for m in all_months if m not in existing]

    if not to_fetch:
        print(f"All {len(all_months)} months already cached. Nothing to do.")
        return

    print(f"Need to fetch {len(to_fetch)} months (skipping {len(existing)} cached)")
    print(f"Months: {to_fetch[0]} to {to_fetch[-1]}")

    # Initialize Ray
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    from ml.realized_da import fetch_and_cache_month

    for i, month in enumerate(to_fetch):
        print(f"[{i+1}/{len(to_fetch)}] Fetching {month} ... (mem={mem_mb():.0f} MB)")
        try:
            path = fetch_and_cache_month(month, cache_dir=cache_dir)
            import polars as pl
            df = pl.read_parquet(path)
            n_binding = len(df.filter(pl.col("realized_sp") > 0))
            print(f"  -> {len(df)} constraints, {n_binding} binding, saved to {path}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    import ray
    ray.shutdown()
    print(f"\nDone. Cached {len(to_fetch)} months to {cache_dir}/")


if __name__ == "__main__":
    main()
```

**Step 2:** Run the script (one-time, requires Ray).

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
python scripts/cache_realized_da.py
```

Expected: ~47 months fetched, each taking ~5-10s. Total ~5 min.

**Step 3:** Verify a few cached files.

```bash
python -c "
import polars as pl
for m in ['2022-06', '2020-12', '2023-03']:
    df = pl.read_parquet(f'data/realized_da/{m}.parquet')
    n_bind = len(df.filter(pl.col('realized_sp') > 0))
    print(f'{m}: {len(df)} constraints, {n_bind} binding')
"
```

**Step 4:** Commit (script only, not cached data).

```bash
echo "data/realized_da/*.parquet" >> .gitignore
git add scripts/cache_realized_da.py .gitignore
git commit -m "task 7: cache_realized_da script (one-time Ray fetch)"
```

---

### Task 8: Write mlpred_loader.py (new module)

**Files:**
- Create: `ml/mlpred_loader.py`
- Create: `ml/tests/test_mlpred_loader.py`

**Step 1:** Write test.

```python
# ml/tests/test_mlpred_loader.py
import polars as pl
import pytest

def test_load_mlpred_returns_correct_columns():
    """Smoke test: load real ml_pred data for 2022-06."""
    from ml.mlpred_loader import load_mlpred_month
    df = load_mlpred_month("2022-06")
    assert "constraint_id" in df.columns
    assert "flow_direction" in df.columns
    assert "predicted_shadow_price" in df.columns
    assert "binding_probability" in df.columns
    assert "binding_probability_scaled" in df.columns
    # Should NOT include leaky columns
    assert "actual_shadow_price" not in df.columns
    assert "actual_binding" not in df.columns
    assert "error" not in df.columns
    assert "abs_error" not in df.columns
    assert len(df) > 0

def test_load_mlpred_missing_month_returns_empty():
    from ml.mlpred_loader import load_mlpred_month
    df = load_mlpred_month("1999-01")
    assert len(df) == 0
```

**Step 2:** Run test -- expect FAIL.

**Step 3:** Write `ml/mlpred_loader.py`.

```python
"""Load spice6 ML prediction features for a single auction month.

Source: ml_pred/auction_month={m}/market_month={m}/class_type=onpeak/final_results.parquet

Extracts: predicted_shadow_price, binding_probability, binding_probability_scaled.
Drops leaky columns (actual_shadow_price, actual_binding, error, abs_error).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_MLPRED_BASE

_MLPRED_KEEP_COLS = [
    "constraint_id",
    "flow_direction",
    "predicted_shadow_price",
    "binding_probability",
    "binding_probability_scaled",
]


def load_mlpred_month(
    auction_month: str,
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load ML prediction features for one month.

    Returns DataFrame with columns:
    [constraint_id, flow_direction, predicted_shadow_price,
     binding_probability, binding_probability_scaled]

    Returns empty DataFrame if data not available.
    """
    path = (
        Path(SPICE6_MLPRED_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={auction_month}"
        / f"class_type={class_type}"
        / "final_results.parquet"
    )

    if not path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(path)
    available = [c for c in _MLPRED_KEEP_COLS if c in df.columns]
    return df.select(available)
```

**Step 4:** Run tests -- expect PASS.

```bash
python -m pytest ml/tests/test_mlpred_loader.py -v
```

**Step 5:** Commit.

```bash
git add ml/mlpred_loader.py ml/tests/test_mlpred_loader.py
git commit -m "task 8: mlpred_loader module with tests"
```

---

### Task 9: Write data_loader.py (rewritten from stage4)

**Files:**
- Create: `ml/data_loader.py`
- Create: `ml/tests/test_data_loader.py`

This is the most critical module -- where the circularity fix lives.

**Step 1:** Write test that verifies realized_sp is the ground truth column.

```python
# ml/tests/test_data_loader.py
import polars as pl
import pytest

def test_load_v62b_month_has_realized_sp():
    """V6.2B month data MUST include realized_sp column (not shadow_price_da as label)."""
    from ml.data_loader import load_v62b_month
    df = load_v62b_month("2022-06")
    assert "realized_sp" in df.columns, "Missing realized_sp -- ground truth not joined!"
    assert df["realized_sp"].dtype == pl.Float64

def test_load_v62b_month_has_spice6_features():
    from ml.data_loader import load_v62b_month
    df = load_v62b_month("2022-06")
    for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
        assert col in df.columns, f"Missing spice6 feature: {col}"

def test_load_v62b_month_no_engineered_features():
    """Engineered features were proven useless -- must NOT be present."""
    from ml.data_loader import load_v62b_month
    df = load_v62b_month("2022-06")
    banned = ["flow_utilization", "mix_utilization", "branch_utilization",
              "prob_exceed_max", "ori_mix_ratio", "da_x_dmix"]
    for col in banned:
        assert col not in df.columns, f"Banned engineered feature present: {col}"

def test_train_val_test_splits_have_realized_sp():
    """Every split must have realized_sp as the ground truth column."""
    from ml.data_loader import load_train_val_test
    train_df, val_df, test_df = load_train_val_test(
        "2022-06", train_months=2, val_months=0  # small for speed
    )
    assert "realized_sp" in train_df.columns
    assert "realized_sp" in test_df.columns
    # With val_months=0, val_df should be None
    assert val_df is None

def test_each_training_month_has_own_labels():
    """Each training month must get realized DA for THAT month, not the eval month."""
    from ml.data_loader import load_train_val_test
    train_df, _, _ = load_train_val_test("2022-06", train_months=2, val_months=0)
    # Train months should be 2022-04 and 2022-05
    months = train_df["query_month"].unique().sort().to_list()
    assert len(months) == 2
    # Each month should have some non-zero realized_sp (most binding months have some)
    for m in months:
        month_df = train_df.filter(pl.col("query_month") == m)
        assert "realized_sp" in month_df.columns
```

**Step 2:** Run test -- expect FAIL.

**Step 3:** Write `ml/data_loader.py`.

```python
"""Data loading for LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 density features,
and joins realized DA ground truth.

CRITICAL: Ground truth is realized_sp (actual binding from MISO market data),
NOT shadow_price_da (which is a historical lookback FEATURE, not a target).
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.realized_da import load_realized_da
from ml.spice6_loader import load_spice6_density


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density and realized DA.

    Returns DataFrame with:
    - V6.2B columns (features + metadata)
    - spice6 density features (prob_exceed_*, constraint_limit)
    - realized_sp column (ground truth: abs(sum(DA shadow prices)))
    """
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(
            spice6,
            on=["constraint_id", "flow_direction"],
            how="left",
        )
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6 enrichment: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth
    try:
        realized = load_realized_da(auction_month)
        df = df.join(realized, on="constraint_id", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
        n_binding = len(df.filter(pl.col("realized_sp") > 0))
        print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding")
    except FileNotFoundError:
        print(f"[data_loader] WARNING: no cached realized DA for {auction_month}")
        df = df.with_columns(pl.lit(0.0).alias("realized_sp"))

    return df


def load_train_val_test(
    eval_month: str,
    train_months: int = 8,
    val_months: int = 0,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    """Load train/val/test splits for a single evaluation month.

    Each training month gets its OWN realized DA labels (not the eval month's).

    Returns (train_df, val_df, test_df) with query_month and realized_sp columns.
    """
    import pandas as pd

    eval_ts = pd.Timestamp(eval_month)
    total_lookback = train_months + val_months

    train_month_strs = []
    for i in range(total_lookback, val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    val_month_strs = []
    for i in range(val_months, 0, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        val_month_strs.append(m)

    print(f"[data_loader] eval={eval_month} train={train_month_strs} val={val_month_strs}")
    print(f"[data_loader] mem: {mem_mb():.0f} MB")

    def _load_months(month_strs: list[str]) -> pl.DataFrame:
        dfs = []
        for m in month_strs:
            try:
                df = load_v62b_month(m, period_type, class_type)
                df = df.with_columns(pl.lit(m).alias("query_month"))
                dfs.append(df)
            except FileNotFoundError:
                print(f"[data_loader] WARNING: skipping {m} (not found)")
        if not dfs:
            raise ValueError(f"No data found for months: {month_strs}")
        return pl.concat(dfs)

    train_df = _load_months(train_month_strs)
    val_df = _load_months(val_month_strs) if val_month_strs else None
    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    val_len = len(val_df) if val_df is not None else 0
    print(f"[data_loader] train={len(train_df)} val={val_len} test={len(test_df)} "
          f"mem: {mem_mb():.0f} MB")

    return train_df, val_df, test_df
```

**CRITICAL AUDIT NOTE:** Compare to stage4 data_loader.py:
- REMOVED: `_add_engineered_features()` call (line 70 in stage4)
- REMOVED: entire `_add_engineered_features()` function (lines 75-157 in stage4)
- ADDED: `load_realized_da()` join in `load_v62b_month()` -- ground truth
- CHANGED: `load_train_val_test()` defaults from `train_months=6, val_months=2` to `8, 0`
- UNCHANGED: spice6 enrichment logic (verified correct)
- NOT in this module: `y = df["realized_sp"]` -- that happens in pipeline.py

**Step 4:** Run tests -- expect PASS.

```bash
python -m pytest ml/tests/test_data_loader.py -v
```

**Step 5:** Commit.

```bash
git add ml/data_loader.py ml/tests/test_data_loader.py
git commit -m "task 9: data_loader with realized DA ground truth (fixes circularity)"
```

---

### Task 10: Write pipeline.py (modified from stage4)

**Files:**
- Create: `ml/pipeline.py`
- Copy: `ml/tests/test_pipeline.py` from stage4 (then modify)

**Step 1:** Copy stage4 pipeline.py and modify the 3 critical lines.

```bash
cp ../research-stage4-tier/ml/pipeline.py ml/pipeline.py
```

**Step 2:** Apply the ground truth swap. Change these 3 lines:

```python
# Line 52 (was): y_train = train_df["shadow_price_da"].to_numpy().astype(np.float64)
# Change to:
y_train = train_df["realized_sp"].to_numpy().astype(np.float64)

# Line 59 (was): y_val = val_df["shadow_price_da"].to_numpy().astype(np.float64)
# Change to:
y_val = val_df["realized_sp"].to_numpy().astype(np.float64)

# Line 84 (was): actual_sp = test_df["shadow_price_da"].to_numpy().astype(np.float64)
# Change to:
actual_sp = test_df["realized_sp"].to_numpy().astype(np.float64)
```

**Step 3:** Verify no other references to shadow_price_da in the file.

```bash
grep -n "shadow_price_da" ml/pipeline.py
```

Expected: zero matches. If any found, fix them.

**Step 4:** Copy and adapt test.

```bash
cp ../research-stage4-tier/ml/tests/test_pipeline.py ml/tests/test_pipeline.py
```

Modify test to use realized_sp instead of shadow_price_da if referenced.

**Step 5:** Run tests.

```bash
python -m pytest ml/tests/test_pipeline.py -v
```

**Step 6:** Commit.

```bash
git add ml/pipeline.py ml/tests/test_pipeline.py
git commit -m "task 10: pipeline with realized_sp ground truth (3-line swap)"
```

---

### Task 11: Copy and audit compare.py and benchmark.py

**Files:**
- Copy: `ml/compare.py`, `ml/benchmark.py` from stage4

**Step 1:** Copy.

```bash
cp ../research-stage4-tier/ml/compare.py ml/compare.py
cp ../research-stage4-tier/ml/benchmark.py ml/benchmark.py
```

**Step 2:** Audit compare.py.

- [x] Operates on metrics.json files (doesn't touch raw data) -- PASS
- [x] No reference to shadow_price_da -- PASS
- [x] Gate checking is generic (metric name + value) -- PASS

**Step 3:** Audit benchmark.py.

- [x] Calls `run_pipeline()` which now uses realized_sp -- PASS (inherits fix)
- [x] Saves metrics to registry -- PASS
- [x] No direct data access -- PASS

**Step 4:** Commit.

```bash
git add ml/compare.py ml/benchmark.py
git commit -m "task 11: copy and audit compare.py, benchmark.py"
```

---

### Task 12: Write run_v0_formula_baseline.py (rewritten)

**Files:**
- Create: `scripts/run_v0_formula_baseline.py`

**Step 1:** Write the v0 script that evaluates against realized DA (NOT shadow_price_da).

```python
"""v0: V6.2B formula baseline evaluated against REALIZED DA.

CRITICAL: This evaluates against realized_sp (cached parquet), NOT shadow_price_da.
Stage 4's version evaluated against shadow_price_da which was circular.

Formula: score = -(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)
Negated because lower rank_value = more binding, but evaluation expects higher = better.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.v62b_formula import v62b_score


def evaluate_formula_month(auction_month: str) -> dict:
    """Evaluate V6.2B formula ranking against REALIZED DA for one month."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))

    # Load realized DA ground truth (NOT shadow_price_da)
    realized = load_realized_da(auction_month)
    df = df.join(realized, on="constraint_id", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    actual = df["realized_sp"].to_numpy().astype(np.float64)
    n_binding = int((actual > 0).sum())

    # Formula score (negated: lower rank_value = more binding -> higher score = better)
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )

    metrics = evaluate_ltr(actual, scores)
    metrics["n_binding"] = n_binding
    return metrics


def main():
    eval_months = list(_DEFAULT_EVAL_MONTHS)
    per_month = {}
    for m in eval_months:
        print(f"Evaluating V6.2B formula on {m} (vs realized DA)...")
        per_month[m] = evaluate_formula_month(m)
        n_b = per_month[m].get("n_binding", "?")
        vc20 = per_month[m].get("VC@20", 0)
        sp = per_month[m].get("Spearman", 0)
        print(f"  n_binding={n_b} VC@20={vc20:.4f} Spearman={sp:.4f}")

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "model": "v62b_formula",
            "ground_truth": "realized_da (NOT shadow_price_da)",
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "n_months_requested": len(eval_months),
        "skipped_months": [],
    }

    # Save to registry
    v0_dir = Path("registry/v0")
    v0_dir.mkdir(parents=True, exist_ok=True)
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(v0_dir / "config.json", "w") as f:
        json.dump({
            "model": "v62b_formula",
            "ground_truth": "realized_da",
            "formula": "score = -(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)",
        }, f, indent=2)
    with open(v0_dir / "meta.json", "w") as f:
        json.dump({"version_id": "v0", "model": "v62b_formula"}, f, indent=2)

    # Recalibrate gates
    gates = {}
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    group_b = ["VC@10", "VC@25", "VC@50", "VC@200",
               "Recall@10", "Spearman", "Tier0-AP", "Tier01-AP"]

    for metric in group_a + group_b:
        group = "A" if metric in group_a else "B"
        mean_val = agg["mean"].get(metric, 0)
        min_val = agg["min"].get(metric, 0)
        gates[metric] = {
            "floor": round(0.9 * mean_val, 6),
            "tail_floor": round(min_val, 6),
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "version": 2,
        "note": "Calibrated from V6.2B formula vs REALIZED DA (not shadow_price_da)",
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": eval_months},
        "gates": gates,
    }

    registry_dir = Path("registry")
    with open(registry_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)
    with open(registry_dir / "champion.json", "w") as f:
        json.dump({"version": "v0"}, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("V6.2B FORMULA BASELINE (v0) vs REALIZED DA")
    print("=" * 60)
    for metric in group_a:
        mean = agg["mean"].get(metric, 0)
        mn = agg["min"].get(metric, 0)
        mx = agg["max"].get(metric, 0)
        print(f"  {metric:>12}: mean={mean:.4f}  min={mn:.4f}  max={mx:.4f}")
    print()
    for metric in group_b:
        mean = agg["mean"].get(metric, 0)
        print(f"  {metric:>12}: mean={mean:.4f}")
    print()
    print("Gates recalibrated. Champion set to v0.")


if __name__ == "__main__":
    main()
```

**CRITICAL AUDIT:** Compare to stage4 version:
- Line 32 stage4: `actual = df["shadow_price_da"]` -- **FIXED** to `df["realized_sp"]`
- Added realized DA loading via `load_realized_da()`
- Added `ground_truth` field to eval_config
- Everything else is structurally identical

**Step 2:** Run the v0 baseline.

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
python scripts/run_v0_formula_baseline.py
```

**Step 3:** Verify v0 numbers match experiment-setup.md Section 8.

Expected (within rounding):
- VC@20 mean ~0.2817
- VC@100 mean ~0.6008
- Spearman mean ~0.2045

If numbers differ significantly (>0.01), STOP and investigate.

**Step 4:** Commit.

```bash
git add scripts/run_v0_formula_baseline.py registry/
git commit -m "task 12: v0 formula baseline against realized DA, gates recalibrated"
```

---

### Task 13: Run v1 experiment (Groups A+B, 11 features)

**Files:**
- Create: `scripts/run_v1_experiment.py`

**Step 1:** Write experiment script.

```python
"""v1: LTR with Groups A+B (11 features, no historical DA signal).

Features: mean_branch_max, ori_mean, mix_mean, density_mix_rank_value,
          density_ori_rank_value, prob_exceed_110/100/90/85/80, constraint_limit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    FEATURES_V1, MONOTONE_V1,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark


def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(FEATURES_V1),
            monotone_constraints=list(MONOTONE_V1),
            backend="lightgbm",
        ),
        train_months=8,
        val_months=0,
    )

    # Screen first (4 months)
    print("=" * 60)
    print("v1 SCREEN (4 months)")
    print("=" * 60)
    result = run_benchmark(
        version_id="v1_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    # Check if promising (VC@20 mean > v0's 0.28)
    vc20_mean = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    print(f"\nv1 screen VC@20 mean: {vc20_mean:.4f} (v0 baseline: 0.2817)")

    if vc20_mean > 0.20:  # reasonable threshold
        print("\nPromising! Running full 12-month eval...")
        run_benchmark(
            version_id="v1",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )
    else:
        print("\nNot promising. Skipping full eval.")


if __name__ == "__main__":
    main()
```

**Step 2:** Run.

```bash
python scripts/run_v1_experiment.py
```

**Step 3:** Review results and commit.

```bash
git add scripts/run_v1_experiment.py registry/v1*/
git commit -m "task 13: v1 experiment (Groups A+B, 11 features)"
```

---

### Task 14: Run v1b experiment (Groups A+B+C, 12 features)

**Files:**
- Create: `scripts/run_v1b_experiment.py`

**Step 1:** Write experiment script (same structure as v1, with da_rank_value added).

```python
"""v1b: LTR with Groups A+B+C (12 features, adds da_rank_value).

Same as v1 + da_rank_value (historical 60-month DA signal).
Purpose: measure the value of the historical DA signal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    LTRConfig, PipelineConfig,
    FEATURES_V1B, MONOTONE_V1B,
    _SCREEN_EVAL_MONTHS, _DEFAULT_EVAL_MONTHS,
)
from ml.benchmark import run_benchmark


def main():
    config = PipelineConfig(
        ltr=LTRConfig(
            features=list(FEATURES_V1B),
            monotone_constraints=list(MONOTONE_V1B),
            backend="lightgbm",
        ),
        train_months=8,
        val_months=0,
    )

    # Screen first
    print("=" * 60)
    print("v1b SCREEN (4 months)")
    print("=" * 60)
    result = run_benchmark(
        version_id="v1b_screen",
        eval_months=_SCREEN_EVAL_MONTHS,
        config=config,
        mode="screen",
    )

    vc20_mean = result.get("aggregate", {}).get("mean", {}).get("VC@20", 0)
    print(f"\nv1b screen VC@20 mean: {vc20_mean:.4f} (v0: 0.2817, v1: check registry)")

    if vc20_mean > 0.20:
        print("\nPromising! Running full 12-month eval...")
        run_benchmark(
            version_id="v1b",
            eval_months=_DEFAULT_EVAL_MONTHS,
            config=config,
            mode="eval",
        )
    else:
        print("\nNot promising. Skipping full eval.")


if __name__ == "__main__":
    main()
```

**Step 2:** Run.

```bash
python scripts/run_v1b_experiment.py
```

**Step 3:** Run comparison against v0.

```bash
python -m ml.compare --batch-id stage5 --iteration 1 --output registry/comparisons/stage5_v0_v1_v1b.md
```

**Step 4:** Commit.

```bash
git add scripts/run_v1b_experiment.py registry/
git commit -m "task 14: v1b experiment (Groups A+B+C, +da_rank_value) and comparison"
```

---

### Task 15: Update mem.md with results

**Files:**
- Modify: `mem.md`

**Step 1:** Update mem.md with:
- Corrected CHECK 9 (constraint_id is String, not i64)
- v0 verified results
- v1 and v1b actual results
- Any new learnings

**Step 2:** Commit.

```bash
git add mem.md
git commit -m "task 15: update mem.md with stage5 results and corrections"
```

---
