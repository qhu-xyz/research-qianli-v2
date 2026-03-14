# LTR Tier Ranking Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a learning-to-rank pipeline that ranks MISO constraints by predicted shadow price, using V6.2B's constraint universe and XGBoost `rank:pairwise`, with full registry/gates/promotion infrastructure.

**Architecture:** Load V6.2B signal data (~500 constraints/month) as the universe, compute features from raw spice6 density/SF parquet, train XGBoost LTR model with 6-month rolling window, evaluate with VC@k, Recall@k, NDCG metrics. Full model registry with 3-layer gate checks ported from stage 3.

**Tech Stack:** XGBoost (rank:pairwise), polars, pyarrow, numpy, scikit-learn (for AP/Spearman), pytest

**Working directory:** `/home/xyz/workspace/research-qianli-v2/research-stage4-tier`

**Venv:** `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`

**Reference code:** `/home/xyz/workspace/research-qianli-v2/research-stage3-tier/ml/` (stage 3 tier classifier)

---

## Task 1: Project scaffolding and config

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/config.py`
- Create: `ml/tests/__init__.py`
- Create: `ml/tests/test_config.py`
- Create: `.gitignore`

**Step 1: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.logs/
```

**Step 2: Write the config test**

```python
# ml/tests/test_config.py
"""Tests for LTR pipeline configuration."""
import json
import pytest
from ml.config import LTRConfig, PipelineConfig, GateConfig

def test_ltr_config_defaults():
    cfg = LTRConfig()
    assert cfg.objective == "rank:pairwise"
    assert cfg.n_estimators == 400
    assert cfg.max_depth == 5
    assert cfg.early_stopping_rounds == 50
    assert len(cfg.features) == 40
    assert len(cfg.monotone_constraints) == len(cfg.features)

def test_ltr_config_roundtrip():
    cfg = LTRConfig()
    d = cfg.to_dict()
    cfg2 = LTRConfig.from_dict(d)
    assert cfg2.features == cfg.features
    assert cfg2.objective == cfg.objective
    assert cfg2.n_estimators == cfg.n_estimators

def test_pipeline_config_roundtrip():
    cfg = PipelineConfig()
    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.train_months == 6
    assert cfg2.val_months == 2
    assert cfg2.ltr.objective == "rank:pairwise"

def test_gate_config_missing_file(tmp_path):
    cfg = GateConfig.from_json(tmp_path / "nonexistent.json")
    assert cfg.gates == {}

def test_gate_config_loads(tmp_path):
    data = {"gates": {"VC@100": {"floor": 0.8, "direction": "higher", "group": "A"}},
            "noise_tolerance": 0.02, "tail_max_failures": 1}
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(data))
    cfg = GateConfig.from_json(path)
    assert "VC@100" in cfg.gates
    assert cfg.noise_tolerance == 0.02
```

**Step 3: Run test to verify it fails**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage4-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python -m pytest ml/tests/test_config.py -v
```

Expected: FAIL — `ml.config` does not exist

**Step 4: Write config.py**

```python
# ml/config.py
"""LTR pipeline configuration.

LTRConfig      -- XGBoost rank:pairwise model config.
PipelineConfig -- composition of LTR config + pipeline params.
GateConfig     -- quality gates loaded from JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── V6.2B columns available as features ──
_V62B_FEATURES: list[str] = [
    "mean_branch_max", "ori_mean", "mix_mean",
    "density_mix_rank_value", "density_ori_rank_value", "da_rank_value",
]

# ── Stage 3 features (34 proven) ──
_STAGE3_FEATURES: list[str] = [
    "prob_exceed_110", "prob_exceed_105", "prob_exceed_100",
    "prob_exceed_95", "prob_exceed_90",
    "prob_below_100", "prob_below_95", "prob_below_90",
    "expected_overload",
    "hist_da", "hist_da_trend",
    "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac",
    "is_interface", "constraint_limit",
    "density_mean", "density_variance", "density_entropy",
    "tail_concentration",
    "prob_band_95_100", "prob_band_100_105",
    "hist_da_max_season",
    "prob_exceed_85", "prob_exceed_80",
    "recent_hist_da",
    "season_hist_da_1", "season_hist_da_2",
    "density_skewness", "density_kurtosis", "density_cv",
    "season_hist_da_3",
    "prob_below_85",
]

_STAGE3_MONOTONE: list[int] = [
    1, 1, 1, 1, 1,    # prob_exceed_110..90
    -1, -1, -1,        # prob_below_100..90
    1,                  # expected_overload
    1, 1,              # hist_da, hist_da_trend
    1, 1, 0, 0,        # sf_*
    0, 0,              # is_interface, constraint_limit
    0, 0, 0,           # density_mean, variance, entropy
    1,                  # tail_concentration
    0, 0,              # prob_band_*
    1,                  # hist_da_max_season
    1, 1,              # prob_exceed_85, 80
    1,                  # recent_hist_da
    1, 1,              # season_hist_da_1, 2
    0, 0, 0,           # density_skewness, kurtosis, cv
    1,                  # season_hist_da_3
    -1,                # prob_below_85
]

_ALL_FEATURES: list[str] = _STAGE3_FEATURES + _V62B_FEATURES
_ALL_MONOTONE: list[int] = _STAGE3_MONOTONE + [0] * len(_V62B_FEATURES)

# ── Eval months: 12 months spread across 2020-06 to 2023-05 ──
_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"
SPICE6_SF_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/sf"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info"


@dataclass
class LTRConfig:
    """XGBoost learning-to-rank configuration."""

    features: list[str] = field(default_factory=lambda: list(_ALL_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(_ALL_MONOTONE))

    # XGBoost hyperparams
    objective: str = "rank:pairwise"
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    min_child_weight: int = 25
    early_stopping_rounds: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "monotone_constraints": list(self.monotone_constraints),
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    ltr: LTRConfig = field(default_factory=LTRConfig)
    train_months: int = 6
    val_months: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "ltr": self.ltr.to_dict(),
            "train_months": self.train_months,
            "val_months": self.val_months,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(
            ltr=LTRConfig.from_dict(d["ltr"]),
            train_months=d["train_months"],
            val_months=d["val_months"],
        )


@dataclass
class GateConfig:
    """Quality gates loaded from JSON."""

    gates: dict[str, Any] = field(default_factory=dict)
    eval_months: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(
                gates=data.get("gates", {}),
                eval_months=data.get("eval_months", {}),
                noise_tolerance=data.get("noise_tolerance", 0.0),
                tail_max_failures=data.get("tail_max_failures", 0),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()
```

**Step 5: Run tests**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_config.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add ml/ .gitignore
git commit -m "task 1: project scaffolding and LTR config"
```

---

## Task 2: Evaluation module (VC@k, Recall@k, NDCG, Spearman, AP)

**Files:**
- Create: `ml/evaluate.py`
- Create: `ml/tests/test_evaluate.py`

**Step 1: Write the evaluation tests**

```python
# ml/tests/test_evaluate.py
"""Tests for LTR evaluation metrics."""
import numpy as np
import pytest
from ml.evaluate import (
    value_capture_at_k, recall_at_k, ndcg, spearman_corr,
    tier_ap, evaluate_ltr, aggregate_months,
)

def test_vc_at_k_perfect():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # perfect ranking
    assert value_capture_at_k(actual, scores, 2) == pytest.approx(150 / 166, abs=1e-4)

def test_vc_at_k_worst():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # reverse ranking
    assert value_capture_at_k(actual, scores, 2) == pytest.approx(6 / 166, abs=1e-4)

def test_recall_at_k_perfect():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert recall_at_k(actual, scores, 2) == 1.0

def test_recall_at_k_partial():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([1.0, 5.0, 3.0, 4.0, 2.0])  # top-2 by score: idx 1,3
    # true top-2: idx 0,1. overlap = {1}
    assert recall_at_k(actual, scores, 2) == 0.5

def test_ndcg_perfect():
    actual = np.array([10.0, 5.0, 1.0])
    scores = np.array([3.0, 2.0, 1.0])
    assert ndcg(actual, scores) == pytest.approx(1.0, abs=1e-4)

def test_spearman_perfect():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_corr(actual, scores) == pytest.approx(1.0, abs=1e-4)

def test_tier_ap():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    # top-20% = 1 constraint. Perfect score should give AP=1
    ap = tier_ap(actual, scores, top_frac=0.2)
    assert ap == pytest.approx(1.0, abs=1e-4)

def test_evaluate_ltr_returns_all_metrics():
    actual = np.array([100.0, 50.0, 10.0, 5.0, 1.0] * 20)
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0] * 20)
    metrics = evaluate_ltr(actual, scores)
    expected_keys = {"VC@10", "VC@20", "VC@25", "VC@50", "VC@100", "VC@200",
                     "Recall@10", "Recall@20", "Recall@50", "Recall@100",
                     "NDCG", "Spearman", "Tier0-AP", "Tier01-AP",
                     "n_samples"}
    assert expected_keys.issubset(set(metrics.keys()))

def test_aggregate_months():
    pm = {
        "2021-01": {"VC@100": 0.8, "NDCG": 0.9},
        "2021-02": {"VC@100": 0.9, "NDCG": 0.95},
    }
    agg = aggregate_months(pm)
    assert agg["mean"]["VC@100"] == pytest.approx(0.85, abs=1e-4)
    assert "bottom_2_mean" in agg
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_evaluate.py -v
```

**Step 3: Write evaluate.py**

```python
# ml/evaluate.py
"""Evaluation harness for LTR ranking pipeline.

Group A (blocking): VC@20, VC@100, Recall@20, Recall@100, NDCG.
Group B (monitor): VC@10, VC@25, VC@50, VC@200, Recall@10, Recall@50,
                   Spearman, Tier0-AP, Tier01-AP.

All metrics are higher-is-better.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def value_capture_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of total actual value captured by top-k scored items."""
    total = actual.sum()
    if total <= 0:
        return 0.0
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(actual[top_k_idx].sum() / total)


def recall_at_k(actual: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the true top-k by actual value, how many are in model's top-k."""
    k = min(k, len(scores))
    true_top_k = set(np.argsort(actual)[::-1][:k].tolist())
    pred_top_k = set(np.argsort(scores)[::-1][:k].tolist())
    return len(true_top_k & pred_top_k) / k if k > 0 else 0.0


def ndcg(actual: np.ndarray, scores: np.ndarray) -> float:
    """NDCG using actual shadow price as relevance, ranked by scores."""
    n = len(actual)
    if n == 0:
        return 0.0
    discounts = np.log2(np.arange(2, n + 2))

    ranked_idx = np.argsort(scores)[::-1]
    dcg = float((actual[ranked_idx] / discounts).sum())

    ideal_idx = np.argsort(actual)[::-1]
    ideal_dcg = float((actual[ideal_idx] / discounts).sum())

    if ideal_dcg <= 0:
        return 0.0
    return dcg / ideal_dcg


def spearman_corr(actual: np.ndarray, scores: np.ndarray) -> float:
    """Spearman rank correlation between actual and predicted scores."""
    if len(actual) < 3:
        return 0.0
    corr, _ = spearmanr(actual, scores)
    return float(corr) if not np.isnan(corr) else 0.0


def tier_ap(
    actual: np.ndarray,
    scores: np.ndarray,
    top_frac: float = 0.2,
) -> float:
    """Average Precision for top-frac% of constraints by actual value."""
    n = len(actual)
    k = max(1, int(n * top_frac))
    threshold = np.sort(actual)[::-1][min(k - 1, n - 1)]
    y_true = (actual >= threshold).astype(int)
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, scores))


def evaluate_ltr(
    actual_shadow_price: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Compute all LTR metrics.

    Parameters
    ----------
    actual_shadow_price : np.ndarray
        Ground-truth shadow prices.
    scores : np.ndarray
        Model's ranking scores (higher = more binding).

    Returns
    -------
    dict
        All metrics: Group A + Group B + monitoring.
    """
    n = len(actual_shadow_price)

    return {
        # Group A (blocking)
        "VC@20": value_capture_at_k(actual_shadow_price, scores, 20),
        "VC@100": value_capture_at_k(actual_shadow_price, scores, 100),
        "Recall@20": recall_at_k(actual_shadow_price, scores, 20),
        "Recall@100": recall_at_k(actual_shadow_price, scores, 100),
        "NDCG": ndcg(actual_shadow_price, scores),
        # Group B (monitor)
        "VC@10": value_capture_at_k(actual_shadow_price, scores, 10),
        "VC@25": value_capture_at_k(actual_shadow_price, scores, 25),
        "VC@50": value_capture_at_k(actual_shadow_price, scores, 50),
        "VC@200": value_capture_at_k(actual_shadow_price, scores, 200),
        "Recall@10": recall_at_k(actual_shadow_price, scores, 10),
        "Recall@50": recall_at_k(actual_shadow_price, scores, 50),
        "Spearman": spearman_corr(actual_shadow_price, scores),
        "Tier0-AP": tier_ap(actual_shadow_price, scores, top_frac=0.2),
        "Tier01-AP": tier_ap(actual_shadow_price, scores, top_frac=0.4),
        # Monitoring
        "n_samples": n,
    }


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into summary statistics."""
    months = list(per_month.keys())
    if not months:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    metric_names = list(per_month[months[0]].keys())
    result: dict[str, dict] = {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    for metric in metric_names:
        values = [per_month[m][metric] for m in months]
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        arr = np.array(values)
        result["mean"][metric] = float(np.mean(arr))
        result["std"][metric] = float(np.std(arr, ddof=0))
        result["min"][metric] = float(np.min(arr))
        result["max"][metric] = float(np.max(arr))
        # All metrics higher-is-better: worst = lowest
        sorted_vals = np.sort(arr)
        worst_2 = sorted_vals[:2] if len(sorted_vals) >= 2 else sorted_vals
        result["bottom_2_mean"][metric] = float(np.mean(worst_2))

    return result
```

**Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_evaluate.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add ml/evaluate.py ml/tests/test_evaluate.py
git commit -m "task 2: evaluation module — VC@k, Recall@k, NDCG, Spearman, AP"
```

---

## Task 3: Data loader (V6.2B universe + raw spice6 features)

**Files:**
- Create: `ml/data_loader.py`
- Create: `ml/tests/test_data_loader.py`

This is the most complex module. It loads V6.2B signal data for the constraint universe and ground truth, then joins features from raw spice6 density/SF parquet.

**Step 1: Write data loader tests**

```python
# ml/tests/test_data_loader.py
"""Tests for LTR data loader."""
import os
import numpy as np
import polars as pl
import pytest
from ml.data_loader import load_v62b_month, load_train_val_test

def test_load_v62b_month_returns_dataframe():
    """Integration test: requires V6.2B data on disk."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    df = load_v62b_month("2021-07", period_type="f0", class_type="onpeak")
    assert len(df) > 400
    assert "constraint_id" in df.columns
    assert "shadow_price_da" in df.columns
    assert "rank" in df.columns
    # V6.2B columns should be present
    assert "mean_branch_max" in df.columns
    assert "da_rank_value" in df.columns

def test_load_v62b_month_no_nulls():
    """shadow_price_da should have no nulls."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    df = load_v62b_month("2021-07", period_type="f0", class_type="onpeak")
    assert df["shadow_price_da"].null_count() == 0
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_data_loader.py -v
```

**Step 3: Write data_loader.py**

The data loader loads V6.2B signal data. Feature computation from raw spice6 is deferred to Task 4 (features.py). Initially we use V6.2B's own columns as features.

```python
# ml/data_loader.py
"""Data loading for LTR ranking pipeline.

Loads V6.2B signal data for constraint universe and ground truth.
Feature enrichment from raw spice6 is handled by features.py.
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load V6.2B signal data for a single month.

    Parameters
    ----------
    auction_month : str
        Auction month in YYYY-MM format.
    period_type : str
        Period type (f0, f1, etc.).
    class_type : str
        onpeak or offpeak.

    Returns
    -------
    pl.DataFrame
        V6.2B data with constraint_id, flow_direction, rank, tier,
        shadow_price_da, and all V6.2B feature columns.
    """
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    # Drop index column if present
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")
    return df


def load_train_val_test(
    eval_month: str,
    train_months: int = 6,
    val_months: int = 2,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/val/test splits for a single evaluation month.

    For eval_month M with train=6, val=2:
    - Train: months M-8 through M-3 (6 months)
    - Val: months M-2, M-1 (2 months)
    - Test: month M (the target month)

    Each month's data comes from V6.2B with an added 'query_month' column
    for XGBoost query groups.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, val_df, test_df) with query_month column added.
    """
    import pandas as pd

    eval_ts = pd.Timestamp(eval_month)
    total_lookback = train_months + val_months

    # Generate month strings for train and val
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
    val_df = _load_months(val_month_strs)
    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    print(f"[data_loader] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"mem: {mem_mb():.0f} MB")

    return train_df, val_df, test_df
```

**Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_data_loader.py -v
```

Expected: PASS (or SKIP if data not on disk)

**Step 5: Commit**

```bash
git add ml/data_loader.py ml/tests/test_data_loader.py
git commit -m "task 3: data loader for V6.2B universe"
```

---

## Task 4: Feature preparation

**Files:**
- Create: `ml/features.py`
- Create: `ml/tests/test_features.py`

Initially uses V6.2B's own columns. Raw spice6 feature computation is a future iteration.

**Step 1: Write feature tests**

```python
# ml/tests/test_features.py
"""Tests for LTR feature preparation."""
import numpy as np
import polars as pl
import pytest
from ml.config import LTRConfig
from ml.features import prepare_features, compute_query_groups

def test_prepare_features_shape():
    cfg = LTRConfig(features=["a", "b", "c"], monotone_constraints=[1, 0, -1])
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, None], "c": [5.0, 6.0]})
    X, mono = prepare_features(df, cfg)
    assert X.shape == (2, 3)
    assert mono == [1, 0, -1]
    assert X[1, 1] == 0.0  # null filled

def test_compute_query_groups():
    df = pl.DataFrame({
        "query_month": ["2021-01", "2021-01", "2021-02", "2021-02", "2021-02"],
    })
    groups = compute_query_groups(df)
    assert groups.tolist() == [2, 3]  # 2 in first group, 3 in second
```

**Step 2: Write features.py**

```python
# ml/features.py
"""Feature preparation for LTR pipeline."""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import LTRConfig


def prepare_features(
    df: pl.DataFrame,
    cfg: LTRConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from df, fill nulls with 0."""
    cols = list(cfg.features)
    # Only use columns that exist in the dataframe
    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[features] WARNING: {len(missing)} features missing, filling with 0: {missing}")

    X_parts = []
    for c in cols:
        if c in df.columns:
            X_parts.append(df[c].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            X_parts.append(np.zeros(len(df), dtype=np.float64))

    X = np.column_stack(X_parts) if X_parts else np.zeros((len(df), 0))
    return X, list(cfg.monotone_constraints)


def compute_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Compute XGBoost query group sizes from query_month column.

    Data must be sorted by query_month before calling this.
    Returns array of group sizes (one per unique query_month).
    """
    months = df["query_month"].to_list()
    groups = []
    if not months:
        return np.array([], dtype=np.int32)

    current = months[0]
    count = 0
    for m in months:
        if m == current:
            count += 1
        else:
            groups.append(count)
            current = m
            count = 1
    groups.append(count)
    return np.array(groups, dtype=np.int32)
```

**Step 3: Run tests, commit**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_features.py -v
git add ml/features.py ml/tests/test_features.py
git commit -m "task 4: feature preparation with query groups"
```

---

## Task 5: Training module (XGBoost rank:pairwise)

**Files:**
- Create: `ml/train.py`
- Create: `ml/tests/test_train.py`

**Step 1: Write training tests**

```python
# ml/tests/test_train.py
"""Tests for LTR training."""
import numpy as np
import pytest
from ml.config import LTRConfig
from ml.train import train_ltr_model, predict_scores

def test_train_ltr_model_returns_booster():
    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = rng.rand(n) * 1000  # relevance labels
    groups = np.array([100, 100])  # 2 query groups
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        n_estimators=10,
    )
    model = train_ltr_model(X, y, groups, cfg)
    assert model is not None

def test_predict_scores_shape():
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 5)
    y_train = rng.rand(200) * 1000
    groups = np.array([100, 100])
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        n_estimators=10,
    )
    model = train_ltr_model(X_train, y_train, groups, cfg)
    X_test = rng.randn(50, 5)
    scores = predict_scores(model, X_test)
    assert scores.shape == (50,)

def test_train_with_early_stopping():
    rng = np.random.RandomState(42)
    X_train = rng.randn(200, 5)
    y_train = rng.rand(200) * 1000
    groups_train = np.array([100, 100])
    X_val = rng.randn(100, 5)
    y_val = rng.rand(100) * 1000
    groups_val = np.array([100])
    cfg = LTRConfig(
        features=[f"f{i}" for i in range(5)],
        monotone_constraints=[0] * 5,
        n_estimators=50,
        early_stopping_rounds=5,
    )
    model = train_ltr_model(
        X_train, y_train, groups_train, cfg,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
    )
    assert model is not None
```

**Step 2: Write train.py**

```python
# ml/train.py
"""XGBoost learning-to-rank training for tier ranking pipeline."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from ml.config import LTRConfig


def train_ltr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cfg: LTRConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
) -> xgb.XGBRanker:
    """Train XGBoost ranker with pairwise objective.

    Parameters
    ----------
    X_train, y_train : arrays
        Training features and relevance labels (shadow prices).
    groups_train : array
        Query group sizes for training data.
    cfg : LTRConfig
        Model configuration.
    X_val, y_val, groups_val : arrays, optional
        Validation data for early stopping.

    Returns
    -------
    xgb.XGBRanker
        Trained ranker model.
    """
    monotone = tuple(cfg.monotone_constraints)
    use_early_stopping = X_val is not None and y_val is not None

    model = xgb.XGBRanker(
        objective=cfg.objective,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        monotone_constraints=monotone,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=cfg.early_stopping_rounds if use_early_stopping else None,
    )

    fit_kwargs: dict = {"group": groups_train}
    if use_early_stopping and groups_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [groups_val]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)

    if use_early_stopping and hasattr(model, "best_iteration"):
        print(f"[train] early stopping: best_iteration={model.best_iteration} "
              f"of {cfg.n_estimators}")

    return model


def predict_scores(model: xgb.XGBRanker, X: np.ndarray) -> np.ndarray:
    """Predict ranking scores. Higher = more binding."""
    return model.predict(X)
```

**Step 3: Run tests, commit**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_train.py -v
git add ml/train.py ml/tests/test_train.py
git commit -m "task 5: XGBoost rank:pairwise training module"
```

---

## Task 6: Pipeline orchestration

**Files:**
- Create: `ml/pipeline.py`
- Create: `ml/tests/test_pipeline.py`

**Step 1: Write pipeline test**

```python
# ml/tests/test_pipeline.py
"""Tests for LTR pipeline orchestration."""
import os
import pytest
from ml.config import PipelineConfig
from ml.pipeline import run_pipeline

def test_pipeline_smoke():
    """Smoke test with real V6.2B data (requires data on disk)."""
    if not os.path.exists("/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/2021-07"):
        pytest.skip("V6.2B data not available")
    cfg = PipelineConfig()
    # Use small n_estimators for speed
    cfg.ltr.n_estimators = 10
    cfg.ltr.early_stopping_rounds = 5
    result = run_pipeline(cfg, "test", "2021-07")
    metrics = result["metrics"]
    assert "VC@100" in metrics
    assert "Recall@100" in metrics
    assert "NDCG" in metrics
    assert metrics["n_samples"] > 400
```

**Step 2: Write pipeline.py**

```python
# ml/pipeline.py
"""LTR pipeline: load -> features -> train -> predict -> evaluate.

5-phase pipeline with memory tracking.
"""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig
from ml.data_loader import load_train_val_test
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> dict[str, Any]:
    """Run the LTR pipeline for a single evaluation month.

    Returns dict with "metrics" key.
    """
    print(f"[pipeline] version={version_id} eval_month={eval_month}")

    # Phase 1: Load data
    print(f"[phase 1] Loading data ... (mem={mem_mb():.0f} MB)")
    train_df, val_df, test_df = load_train_val_test(
        eval_month, config.train_months, config.val_months,
        period_type, class_type,
    )

    # Phase 2: Prepare features
    print(f"[phase 2] Preparing features ... (mem={mem_mb():.0f} MB)")

    # Sort by query_month for proper group computation
    train_df = train_df.sort("query_month")
    val_df = val_df.sort("query_month")

    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["shadow_price_da"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    X_val, _ = prepare_features(val_df, config.ltr)
    y_val = val_df["shadow_price_da"].to_numpy().astype(np.float64)
    groups_val = compute_query_groups(val_df)

    print(f"[phase 2] train={X_train.shape} val={X_val.shape} "
          f"groups_train={groups_train} groups_val={groups_val} "
          f"(mem={mem_mb():.0f} MB)")

    del train_df, val_df
    gc.collect()

    # Phase 3: Train
    print(f"[phase 3] Training LTR model ... (mem={mem_mb():.0f} MB)")
    model = train_ltr_model(
        X_train, y_train, groups_train, config.ltr,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
    )
    del X_train, y_train, groups_train, X_val, y_val, groups_val
    gc.collect()

    # Phase 4: Predict on test
    print(f"[phase 4] Predicting on test ... (mem={mem_mb():.0f} MB)")
    X_test, _ = prepare_features(test_df, config.ltr)
    scores = predict_scores(model, X_test)
    actual_sp = test_df["shadow_price_da"].to_numpy().astype(np.float64)

    # Phase 5: Evaluate
    print(f"[phase 5] Evaluating ... (mem={mem_mb():.0f} MB)")
    metrics = evaluate_ltr(actual_sp, scores)

    # Feature importance
    importance = model.feature_importances_
    feat_names = config.ltr.features
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(
            zip(feat_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    del X_test, scores, actual_sp, test_df
    gc.collect()

    print(f"[pipeline] complete (mem={mem_mb():.0f} MB)")
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    return {"metrics": metrics}
```

**Step 3: Run tests, commit**

```bash
PYTHONPATH=. python -m pytest ml/tests/test_pipeline.py -v
git add ml/pipeline.py ml/tests/test_pipeline.py
git commit -m "task 6: LTR pipeline orchestration"
```

---

## Task 7: Benchmark runner

**Files:**
- Create: `ml/benchmark.py`

Port from stage 3 with metric name changes. This module doesn't need its own unit test — it orchestrates pipeline.py across months.

**Step 1: Write benchmark.py**

Adapt stage 3's `ml/benchmark.py` replacing:
- `TierConfig` → `LTRConfig`
- `PipelineConfig.tier` → `PipelineConfig.ltr`
- metric names: `Tier-VC@100` → `VC@100`, etc.
- Remove Ray init (no longer needed — V6.2B data is local parquet)
- Keep: registry output, feature importance aggregation, eval_months from gates.json

The structure is nearly identical to `research-stage3-tier/ml/benchmark.py`. Key change: no Ray dependency, no MisoDataLoader. The `run_pipeline` function handles everything.

**Step 2: Commit**

```bash
git add ml/benchmark.py
git commit -m "task 7: benchmark runner (no Ray needed)"
```

---

## Task 8: Compare and registry modules

**Files:**
- Create: `ml/compare.py` — copy from stage 3, update metric names
- Create: `ml/registry.py` — copy from stage 3 unchanged

**Step 1: Copy and adapt compare.py**

Copy `research-stage3-tier/ml/compare.py` to `research-stage4-tier/ml/compare.py`. Change the docstring metric references from `Tier-VC@100, Tier-VC@500, Tier0-AP, Tier01-AP` to `VC@20, VC@100, Recall@20, Recall@100, NDCG`. The actual gate logic is metric-name-agnostic (reads from gates.json), so no code changes needed beyond docs.

**Step 2: Copy registry.py**

Copy `research-stage3-tier/ml/registry.py` unchanged.

**Step 3: Commit**

```bash
git add ml/compare.py ml/registry.py
git commit -m "task 8: compare and registry modules"
```

---

## Task 9: V6.2B baseline evaluation and gate calibration

**Files:**
- Create: `ml/baseline_v62b.py` — script to evaluate V6.2B's own ranking
- Create: `registry/v0/metrics.json` — V6.2B baseline metrics
- Create: `registry/gates.json` — calibrated from V6.2B baseline
- Create: `registry/champion.json`
- Create: `registry/version_counter.json`

**Step 1: Write V6.2B baseline evaluation script**

```python
# ml/baseline_v62b.py
"""Evaluate V6.2B's own ranking as the v0 baseline.

Uses V6.2B's rank column (inverted: 1-rank = score) to compute
all LTR metrics. Saves results to registry/v0/.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from ml.config import V62B_SIGNAL_BASE, _DEFAULT_EVAL_MONTHS
from ml.evaluate import aggregate_months, evaluate_ltr


def evaluate_v62b_month(auction_month: str) -> dict:
    """Evaluate V6.2B ranking for one month."""
    path = Path(V62B_SIGNAL_BASE) / auction_month / "f0" / "onpeak"
    df = pl.read_parquet(str(path))
    actual = df["shadow_price_da"].to_numpy().astype(np.float64)
    scores = 1.0 - df["rank"].to_numpy().astype(np.float64)  # invert: higher = better
    return evaluate_ltr(actual, scores)


def main():
    eval_months = _DEFAULT_EVAL_MONTHS
    per_month = {}
    for m in eval_months:
        print(f"Evaluating V6.2B on {m}...")
        per_month[m] = evaluate_v62b_month(m)

    agg = aggregate_months(per_month)
    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": "onpeak",
            "period_type": "f0",
            "model": "v62b_baseline",
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
        json.dump({"model": "v62b_baseline", "note": "V6.2B formula ranking, no ML"}, f, indent=2)
    with open(v0_dir / "meta.json", "w") as f:
        json.dump({"version_id": "v0", "model": "v62b_baseline"}, f, indent=2)

    # Calibrate gates
    gates = {}
    group_a_metrics = ["VC@20", "VC@100", "Recall@20", "Recall@100", "NDCG"]
    group_b_metrics = ["VC@10", "VC@25", "VC@50", "VC@200",
                       "Recall@10", "Recall@50", "Spearman", "Tier0-AP", "Tier01-AP"]

    for metric in group_a_metrics + group_b_metrics:
        group = "A" if metric in group_a_metrics else "B"
        mean_val = agg["mean"].get(metric, 0)
        min_val = agg["min"].get(metric, 0)
        gates[metric] = {
            "floor": round(0.9 * mean_val, 6),
            "tail_floor": round(min_val, 6),
            "direction": "higher",
            "group": group,
        }

    gates_data = {
        "version": 1,
        "note": "Calibrated from V6.2B baseline",
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": eval_months},
        "gates": gates,
    }

    registry_dir = Path("registry")
    registry_dir.mkdir(exist_ok=True)
    with open(registry_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)
    with open(registry_dir / "champion.json", "w") as f:
        json.dump({"version": "v0"}, f, indent=2)
    with open(registry_dir / "version_counter.json", "w") as f:
        json.dump({"next_id": 1}, f, indent=2)

    # Print summary
    print("\n=== V6.2B Baseline (v0) ===")
    for metric in group_a_metrics:
        mean = agg["mean"].get(metric, 0)
        mn = agg["min"].get(metric, 0)
        mx = agg["max"].get(metric, 0)
        print(f"  {metric}: mean={mean:.4f} min={mn:.4f} max={mx:.4f}")
    print("\nGates calibrated. Champion set to v0.")


if __name__ == "__main__":
    main()
```

**Step 2: Run the baseline evaluation**

```bash
PYTHONPATH=. python ml/baseline_v62b.py
```

**Step 3: Verify outputs**

```bash
cat registry/v0/metrics.json | python -m json.tool | head -20
cat registry/gates.json | python -m json.tool | head -30
cat registry/champion.json
```

**Step 4: Commit**

```bash
git add ml/baseline_v62b.py registry/
git commit -m "task 9: V6.2B baseline evaluation and gate calibration"
```

---

## Task 10: First LTR model run (v1)

**Files:**
- No new files — uses existing pipeline

**Step 1: Run the first LTR benchmark using V6.2B features only**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage4-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python ml/benchmark.py --version-id v1 --ptype f0 --class-type onpeak
```

**Step 2: Compare against v0 baseline**

```bash
PYTHONPATH=. python ml/compare.py --batch-id initial --iteration 1 --output reports/initial_comparison.md
```

**Step 3: Review results**

Check if v1 (ML model) beats v0 (V6.2B formula) on any Group A metrics. This is the first signal of whether the LTR approach adds value.

**Step 4: Commit results**

```bash
git add registry/v1/ reports/
git commit -m "task 10: first LTR model run (v1) — V6.2B features only"
```

---

## Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Project scaffolding + config | None |
| 2 | Evaluation module | None |
| 3 | Data loader | Task 1 |
| 4 | Feature preparation | Task 1 |
| 5 | Training module | Task 1 |
| 6 | Pipeline orchestration | Tasks 2-5 |
| 7 | Benchmark runner | Task 6 |
| 8 | Compare + registry | None |
| 9 | V6.2B baseline + gates | Tasks 2, 7, 8 |
| 10 | First LTR run | Task 9 |

Tasks 1-5 and 8 can be parallelized. Tasks 6-10 are sequential.
