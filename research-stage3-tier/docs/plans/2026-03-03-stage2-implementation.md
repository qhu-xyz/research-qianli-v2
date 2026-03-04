# Stage 2 Shadow Price Regression — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully autonomous agentic ML research pipeline for shadow price regression, porting infrastructure from stage 1 and building fresh ML code from the original shadow price prediction repo.

**Architecture:** Hybrid approach — freeze stage 1's classifier config, iterate only on the regressor. EV-based threshold-independent metrics. Selective port of stage 1 infrastructure (state machine, handoffs, watchdog) with fresh ML pipeline code.

**Tech Stack:** Python 3.10+, XGBoost, polars, Ray, bash/tmux, Claude CLI, Codex CLI

**Reference repos:**
- Stage 1: `/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/`
- Original: `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/`

---

### Task 1: Project Scaffold

**Files:**
- Create: `.gitignore`
- Create: `runbook.md` (placeholder, filled in Task 13)
- Create directory structure

**Step 1: Create directory structure**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
mkdir -p agents/prompts ml/tests registry/v0 memory/hot memory/warm memory/archive \
  reports reviews handoff .logs/sessions docs/plans human-input
touch handoff/.gitkeep .logs/sessions/.gitkeep
```

**Step 2: Create .gitignore**

```
state.json
state.lock
.logs/
!.logs/sessions/.gitkeep
handoff/
!handoff/.gitkeep
.claude/worktrees/
ml/**/*.parquet
__pycache__/
*.pyc
.pytest_cache/
```

**Step 3: Commit**

```bash
git add -A && git commit -m "scaffold: project directory structure"
```

---

### Task 2: Port Infrastructure Scripts (Verbatim)

These scripts are generic and need only path changes.

**Files:**
- Create: `agents/config.sh` (adapted paths from stage 1's `agents/config.sh`)
- Copy: `agents/state_utils.sh` (verbatim from stage 1)
- Copy: `agents/watchdog.sh` (verbatim from stage 1)

**Step 1: Write `agents/config.sh`**

Adapt from stage 1's config.sh. Change:
- `PROJECT_DIR` → `/home/xyz/workspace/research-qianli-v2/research-stage2-shadow`
- Keep all other variables identical (RAY_ADDRESS, DATA_ROOT, VENV_ACTIVATE, timeouts)
- Increase `TIMEOUT_WORKER` to `3600` (60 min) since regression pipeline is slower

**Step 2: Copy state_utils.sh verbatim**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/agents/state_utils.sh agents/state_utils.sh
```

**Step 3: Copy watchdog.sh verbatim**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/agents/watchdog.sh agents/watchdog.sh
```

**Step 4: Commit**

```bash
git add agents/ && git commit -m "infra: port state_utils, watchdog, config from stage 1"
```

---

### Task 3: Port Launcher Scripts

These need path adaptation but logic is identical.

**Files:**
- Create: `agents/launch_orchestrator.sh` (adapted from stage 1)
- Create: `agents/launch_worker.sh` (adapted from stage 1)
- Create: `agents/launch_reviewer_claude.sh` (adapted from stage 1)
- Create: `agents/launch_reviewer_codex.sh` (adapted from stage 1)

**Step 1: Copy all launcher scripts from stage 1**

```bash
for f in launch_orchestrator.sh launch_worker.sh launch_reviewer_claude.sh launch_reviewer_codex.sh; do
  cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/agents/$f agents/$f
done
```

**Step 2: Update paths in each script**

In each file, the only change needed is sourcing `config.sh` which already has the correct `PROJECT_DIR`. Verify that all scripts source `config.sh` relative to their own location (they should use `SCRIPT_DIR`). If any hardcode stage 1 paths, fix them.

**Step 3: Commit**

```bash
git add agents/ && git commit -m "infra: port launcher scripts from stage 1"
```

---

### Task 4: Port Pipeline Controller Scripts

These need adaptation for regression-specific differences.

**Files:**
- Create: `agents/run_pipeline.sh` (adapted from stage 1)
- Create: `agents/run_single_iter.sh` (adapted from stage 1)

**Step 1: Copy both scripts from stage 1**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/agents/run_pipeline.sh agents/run_pipeline.sh
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/agents/run_single_iter.sh agents/run_single_iter.sh
```

**Step 2: Adapt run_pipeline.sh**

Changes needed:
- Update the v0 baseline check path to `registry/v0/metrics.json`
- No other changes — logic is generic

**Step 3: Adapt run_single_iter.sh**

Changes needed:
- The compare.py invocation (line ~145 in stage 1): update to use stage 2's compare.py
- The champion.md update logic (line ~238-262): update to read stage 2 metrics format
- Everything else (state transitions, handoff polling, merge logic) stays the same

**Step 4: Commit**

```bash
git add agents/ && git commit -m "infra: port pipeline controller scripts from stage 1"
```

---

### Task 5: ML Config (`ml/config.py`)

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/config.py`
- Test: `ml/tests/__init__.py`, `ml/tests/test_config.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_config.py
from ml.config import ClassifierConfig, RegressorConfig, PipelineConfig, GateConfig

def test_classifier_config_frozen_defaults():
    cfg = ClassifierConfig()
    assert len(cfg.step1_features) == 13
    assert cfg.n_estimators == 200
    assert cfg.max_depth == 4
    assert cfg.threshold_beta == 0.7

def test_regressor_config_defaults():
    cfg = RegressorConfig()
    assert len(cfg.step2_features) == 24
    assert cfg.n_estimators == 400
    assert cfg.max_depth == 5
    assert cfg.unified_regressor is False

def test_pipeline_config_composition():
    cfg = PipelineConfig()
    assert isinstance(cfg.classifier, ClassifierConfig)
    assert isinstance(cfg.regressor, RegressorConfig)
    assert cfg.train_months == 10
    assert cfg.val_months == 2
    assert cfg.ev_scoring is True

def test_gate_config_loads_json(tmp_path):
    import json
    gates_file = tmp_path / "gates.json"
    gates_file.write_text(json.dumps({
        "version": 2,
        "noise_tolerance": 0.02,
        "gates": {"EV-VC@100": {"floor": 0.1, "direction": "higher", "group": "A"}},
        "eval_months": {"primary": ["2021-07"]},
        "cascade_stages": [],
    }))
    cfg = GateConfig(str(gates_file))
    assert "EV-VC@100" in cfg.gates
```

**Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow && python -m pytest ml/tests/test_config.py -v`
Expected: FAIL (imports don't exist yet)

**Step 3: Write `ml/config.py`**

```python
"""Configuration dataclasses for the shadow price regression pipeline.

ClassifierConfig is FROZEN from stage 1's champion.
RegressorConfig is MUTABLE — this is what the agentic loop iterates on.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

# Stage 1 champion features (v0006): 13 features with monotone constraints
_STEP1_FEATURES: list[tuple[str, int]] = [
    ("prob_exceed_110", 1),
    ("prob_exceed_105", 1),
    ("prob_exceed_100", 1),
    ("prob_exceed_95", 1),
    ("prob_exceed_90", 1),
    ("prob_below_100", -1),
    ("prob_below_95", -1),
    ("prob_below_90", -1),
    ("expected_overload", 1),
    ("density_skewness", 0),
    ("density_kurtosis", 0),
    ("hist_da", 1),
    ("hist_da_trend", 1),
]

# Stage 2 regressor features: all step1 + 11 additional
_STEP2_FEATURES: list[tuple[str, int]] = [
    *_STEP1_FEATURES,
    ("prob_exceed_85", 1),
    ("prob_exceed_80", 1),
    ("tail_concentration", 1),
    ("prob_band_95_100", 0),
    ("prob_band_100_105", 0),
    ("density_mean", 0),
    ("density_variance", 0),
    ("density_entropy", 0),
    ("recent_hist_da", 1),
    ("season_hist_da_1", 1),
    ("season_hist_da_2", 1),
]


@dataclass
class ClassifierConfig:
    """Frozen from stage 1 champion. Updated at HUMAN_SYNC only."""
    step1_features: list[tuple[str, int]] = field(default_factory=lambda: list(_STEP1_FEATURES))
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10
    threshold_beta: float = 0.7


@dataclass
class RegressorConfig:
    """Mutable — this is what the agentic loop iterates on."""
    step2_features: list[tuple[str, int]] = field(default_factory=lambda: list(_STEP2_FEATURES))
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10
    unified_regressor: bool = False
    value_weighted: bool = False


@dataclass
class PipelineConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    train_months: int = 10
    val_months: int = 2
    ev_scoring: bool = True

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        return cls(
            classifier=ClassifierConfig(**d.get("classifier", {})),
            regressor=RegressorConfig(**d.get("regressor", {})),
            train_months=d.get("train_months", 10),
            val_months=d.get("val_months", 2),
            ev_scoring=d.get("ev_scoring", True),
        )


@dataclass
class GateConfig:
    """Loads gate definitions from registry/gates.json."""
    gates: dict = field(default_factory=dict)
    eval_months: dict = field(default_factory=dict)
    noise_tolerance: float = 0.02
    tail_max_failures: int = 1
    cascade_stages: list = field(default_factory=list)

    def __init__(self, gates_path: str | None = None):
        if gates_path and Path(gates_path).exists():
            data = json.loads(Path(gates_path).read_text())
            self.gates = data.get("gates", {})
            self.eval_months = data.get("eval_months", {})
            self.noise_tolerance = data.get("noise_tolerance", 0.02)
            self.tail_max_failures = data.get("tail_max_failures", 1)
            self.cascade_stages = data.get("cascade_stages", [])
        else:
            self.gates = {}
            self.eval_months = {}
            self.noise_tolerance = 0.02
            self.tail_max_failures = 1
            self.cascade_stages = []
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest ml/tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: ml config with frozen ClassifierConfig + mutable RegressorConfig"
```

---

### Task 6: ML Data Loader (`ml/data_loader.py`)

**Files:**
- Create: `ml/data_loader.py`
- Test: `ml/tests/test_data_loader.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_data_loader.py
import os
import polars as pl
from ml.data_loader import load_data
from ml.config import PipelineConfig

def test_load_smoke_data():
    os.environ["SMOKE_TEST"] = "true"
    cfg = PipelineConfig()
    train_df, val_df = load_data(cfg, auction_month="2021-07", class_type="onpeak", period_type="f0")
    assert isinstance(train_df, pl.DataFrame)
    assert isinstance(val_df, pl.DataFrame)
    assert len(train_df) > 0
    assert len(val_df) > 0
    # Check all step2 features exist (superset of step1)
    feature_names = [f[0] for f in cfg.regressor.step2_features]
    for feat in feature_names:
        assert feat in train_df.columns, f"Missing feature: {feat}"
    # Check target column exists
    assert "actual_shadow_price" in train_df.columns
```

**Step 2: Run test to verify it fails**

Expected: FAIL (import error)

**Step 3: Write `ml/data_loader.py`**

Adapt from two sources:
- Stage 1's `ml/data_loader.py` for structure (smoke mode, mem_mb tracking, polars)
- Original repo's `src/shadow_price_prediction/data_loader.py` for real data loading logic

Key differences from stage 1:
- Returns `actual_shadow_price` as continuous target (not just binary label)
- Includes all 24 step2 features (not just 13 step1)
- Smoke mode generates continuous shadow prices (not just 0/1)

```python
"""Data loader for shadow price regression pipeline.

Supports two modes:
- SMOKE_TEST=true: synthetic data for fast validation
- Real: loads from Ray/pbase (density files + shadow prices)
"""
from __future__ import annotations
import os
import resource
import numpy as np
import polars as pl
from ml.config import PipelineConfig


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_data(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str = "onpeak",
    period_type: str = "f0",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if os.environ.get("SMOKE_TEST", "").lower() == "true":
        return _load_smoke(cfg)
    return _load_real(cfg, auction_month, class_type, period_type)


def _load_smoke(cfg: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic data for testing."""
    rng = np.random.default_rng(42)
    n = 100
    feature_names = [f[0] for f in cfg.regressor.step2_features]

    data = {feat: rng.random(n).tolist() for feat in feature_names}
    # Generate shadow prices: ~90% zero, ~10% positive
    binding = rng.random(n) > 0.9
    shadow_prices = np.where(binding, rng.exponential(200, n), 0.0)
    data["actual_shadow_price"] = shadow_prices.tolist()
    data["constraint_id"] = [f"c{i}" for i in range(n)]
    data["branch_name"] = [f"branch_{i % 10}" for i in range(n)]

    df = pl.DataFrame(data)
    train_df = df.head(80)
    val_df = df.tail(20)
    return train_df, val_df


def _load_real(
    cfg: PipelineConfig,
    auction_month: str,
    class_type: str,
    period_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load real data via Ray + pbase."""
    import ray
    from pbase.config.ray import init_ray
    import pmodel

    if not ray.is_initialized():
        os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
        init_ray(extra_modules=[pmodel])

    print(f"[data_loader] Loading data for {auction_month}, {class_type}, {period_type}")
    print(f"[data_loader] Memory before load: {mem_mb():.0f} MB")

    # TODO: Implement real data loading from density files + shadow prices
    # Reference: research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py
    # This will be implemented when we run the first real benchmark
    raise NotImplementedError(
        "Real data loading not yet implemented. "
        "Use SMOKE_TEST=true for synthetic data."
    )
```

**Step 4: Run test to verify it passes**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_data_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: data loader with smoke test mode"
```

---

### Task 7: ML Features (`ml/features.py`)

**Files:**
- Create: `ml/features.py`
- Test: `ml/tests/test_features.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_features.py
import numpy as np
import polars as pl
from ml.features import prepare_clf_features, prepare_reg_features, compute_binary_labels
from ml.config import ClassifierConfig, RegressorConfig

def test_prepare_clf_features():
    cfg = ClassifierConfig()
    feature_names = [f[0] for f in cfg.step1_features]
    data = {feat: [0.5, 0.3, 0.7] for feat in feature_names}
    data["actual_shadow_price"] = [0.0, 100.0, 50.0]
    df = pl.DataFrame(data)
    X, monotone = prepare_clf_features(df, cfg)
    assert X.shape == (3, 13)
    assert len(monotone) == 13

def test_prepare_reg_features():
    cfg = RegressorConfig()
    feature_names = [f[0] for f in cfg.step2_features]
    data = {feat: [0.5, 0.3, 0.7] for feat in feature_names}
    data["actual_shadow_price"] = [0.0, 100.0, 50.0]
    df = pl.DataFrame(data)
    X, monotone = prepare_reg_features(df, cfg)
    assert X.shape == (3, 24)
    assert len(monotone) == 24

def test_compute_binary_labels():
    df = pl.DataFrame({"actual_shadow_price": [0.0, 0.5, 100.0, 0.0, 50.0]})
    labels = compute_binary_labels(df, threshold=0.0)
    assert labels.sum() == 3  # 0.5, 100.0, 50.0 are > 0

def test_compute_regression_target():
    from ml.features import compute_regression_target
    df = pl.DataFrame({"actual_shadow_price": [0.0, 100.0, 50.0]})
    target = compute_regression_target(df)
    assert len(target) == 3
    assert target[0] == 0.0  # log1p(0) = 0
    assert abs(target[1] - np.log1p(100.0)) < 1e-6
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write `ml/features.py`**

```python
"""Feature preparation for classifier and regressor stages."""
from __future__ import annotations
import numpy as np
import polars as pl
from ml.config import ClassifierConfig, RegressorConfig


def prepare_clf_features(
    df: pl.DataFrame, cfg: ClassifierConfig
) -> tuple[np.ndarray, list[int]]:
    feature_names = [f[0] for f in cfg.step1_features]
    monotone_constraints = [f[1] for f in cfg.step1_features]
    X = df.select(feature_names).fill_null(0.0).to_numpy()
    return X, monotone_constraints


def prepare_reg_features(
    df: pl.DataFrame, cfg: RegressorConfig
) -> tuple[np.ndarray, list[int]]:
    feature_names = [f[0] for f in cfg.step2_features]
    monotone_constraints = [f[1] for f in cfg.step2_features]
    X = df.select(feature_names).fill_null(0.0).to_numpy()
    return X, monotone_constraints


def compute_binary_labels(df: pl.DataFrame, threshold: float = 0.0) -> np.ndarray:
    return (df["actual_shadow_price"].to_numpy() > threshold).astype(int)


def compute_regression_target(df: pl.DataFrame) -> np.ndarray:
    prices = df["actual_shadow_price"].to_numpy()
    return np.log1p(np.maximum(prices, 0.0))


def compute_scale_pos_weight(labels: np.ndarray) -> float:
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: feature preparation for classifier and regressor"
```

---

### Task 8: ML Train (`ml/train.py`)

**Files:**
- Create: `ml/train.py`
- Test: `ml/tests/test_train.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_train.py
import os
os.environ["SMOKE_TEST"] = "true"

import numpy as np
from ml.train import train_classifier, train_regressor, predict_proba, predict_shadow_price
from ml.config import ClassifierConfig, RegressorConfig

def test_train_classifier():
    rng = np.random.default_rng(42)
    X = rng.random((100, 13))
    y = (rng.random(100) > 0.9).astype(int)
    cfg = ClassifierConfig()
    model, threshold = train_classifier(X, y, cfg, X_val=X[:20], y_val=y[:20])
    assert model is not None
    assert 0.0 <= threshold <= 1.0

def test_predict_proba():
    rng = np.random.default_rng(42)
    X = rng.random((100, 13))
    y = (rng.random(100) > 0.9).astype(int)
    cfg = ClassifierConfig()
    model, _ = train_classifier(X, y, cfg, X_val=X[:20], y_val=y[:20])
    proba = predict_proba(model, X[:10])
    assert proba.shape == (10,)
    assert all(0 <= p <= 1 for p in proba)

def test_train_regressor():
    rng = np.random.default_rng(42)
    X = rng.random((50, 24))
    y = rng.exponential(2.0, 50)  # log1p(shadow_price) targets
    cfg = RegressorConfig()
    model = train_regressor(X, y, cfg)
    assert model is not None

def test_predict_shadow_price():
    rng = np.random.default_rng(42)
    X = rng.random((50, 24))
    y = rng.exponential(2.0, 50)
    cfg = RegressorConfig()
    model = train_regressor(X, y, cfg)
    preds = predict_shadow_price(model, X[:10])
    assert preds.shape == (10,)
    assert all(p >= 0 for p in preds)  # expm1(log1p(x)) >= 0
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write `ml/train.py`**

```python
"""Model training for classifier (frozen config) and regressor (mutable)."""
from __future__ import annotations
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from ml.config import ClassifierConfig, RegressorConfig
from ml.features import compute_scale_pos_weight
from ml.threshold import find_optimal_threshold


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ClassifierConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[XGBClassifier, float]:
    monotone_constraints = tuple(f[1] for f in cfg.step1_features)
    scale_pos = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        scale_pos_weight=scale_pos,
        monotone_constraints=monotone_constraints,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    threshold = 0.5
    if X_val is not None and y_val is not None:
        proba_val = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, proba_val, beta=cfg.threshold_beta)

    return model, threshold


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RegressorConfig,
    sample_weight: np.ndarray | None = None,
) -> XGBRegressor:
    monotone_constraints = tuple(f[1] for f in cfg.step2_features)

    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        monotone_constraints=monotone_constraints,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_shadow_price(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    log_preds = model.predict(X)
    return np.expm1(np.maximum(log_preds, 0.0))
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: train classifier (frozen) + regressor (mutable)"
```

---

### Task 9: ML Threshold (`ml/threshold.py`)

**Files:**
- Create: `ml/threshold.py`
- Test: `ml/tests/test_threshold.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_threshold.py
import numpy as np
from ml.threshold import find_optimal_threshold, apply_threshold

def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.35, 0.25, 0.9])
    threshold = find_optimal_threshold(y_true, y_proba, beta=0.7)
    assert 0.0 < threshold < 1.0

def test_apply_threshold():
    proba = np.array([0.3, 0.5, 0.7, 0.9])
    preds = apply_threshold(proba, threshold=0.5)
    np.testing.assert_array_equal(preds, [0, 0, 1, 1])
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write `ml/threshold.py`**

Port directly from stage 1's `ml/threshold.py` — identical logic.

```python
"""Threshold optimization using F-beta score."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import fbeta_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 0.7,
    n_thresholds: int = 200,
) -> float:
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if preds.sum() == 0:
            continue
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold)


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: F-beta threshold optimization"
```

---

### Task 10: ML Evaluate (`ml/evaluate.py`) — HUMAN-WRITE-ONLY

**Files:**
- Create: `ml/evaluate.py` (HUMAN-WRITE-ONLY — agents cannot modify this)
- Test: `ml/tests/test_evaluate.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_evaluate.py
import numpy as np
from ml.evaluate import evaluate_pipeline, aggregate_months

def test_evaluate_pipeline_basic():
    n = 200
    rng = np.random.default_rng(42)
    actual_shadow = np.where(rng.random(n) > 0.9, rng.exponential(100, n), 0.0)
    pred_proba = rng.random(n)
    pred_shadow = np.where(pred_proba > 0.5, rng.exponential(80, n), 0.0)
    ev_scores = pred_proba * pred_shadow

    metrics = evaluate_pipeline(
        actual_shadow_price=actual_shadow,
        pred_proba=pred_proba,
        pred_shadow_price=pred_shadow,
        ev_scores=ev_scores,
    )
    # Check all gate metrics exist
    for key in ["EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman",
                "C-RMSE", "C-MAE", "EV-VC@1000", "R-REC@500"]:
        assert key in metrics, f"Missing metric: {key}"

def test_aggregate_months():
    months = {
        "2021-07": {"EV-VC@100": 0.15, "EV-VC@500": 0.45},
        "2021-08": {"EV-VC@100": 0.20, "EV-VC@500": 0.50},
        "2021-09": {"EV-VC@100": 0.10, "EV-VC@500": 0.40},
    }
    agg = aggregate_months(months)
    assert "mean" in agg
    assert "bottom_2_mean" in agg
    assert abs(agg["mean"]["EV-VC@100"] - 0.15) < 1e-6
    # bottom_2 = mean of lowest 2 = (0.10 + 0.15) / 2 = 0.125
    assert abs(agg["bottom_2_mean"]["EV-VC@100"] - 0.125) < 1e-6
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write `ml/evaluate.py`**

This is the EV-based evaluation harness. Key metrics:
- **EV-VC@K**: Sort by EV score descending, compute sum(actual_shadow_price for top-K) / sum(all actual_shadow_price)
- **EV-NDCG**: NDCG using actual shadow price as relevance, ranked by EV score
- **Spearman**: Rank correlation between predicted and actual shadow price (binding-only)
- **C-RMSE/C-MAE**: Regression calibration on binding-only samples
- **R-REC@K**: Fraction of true binding constraints in top-K by EV score

```python
"""Evaluation harness for shadow price regression pipeline.

HUMAN-WRITE-ONLY: Agents cannot modify this file.
All gate metrics are threshold-independent (EV-based).
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def _value_capture_at_k(
    actual: np.ndarray, scores: np.ndarray, k: int
) -> float:
    total_value = actual.sum()
    if total_value <= 0:
        return 0.0
    order = np.argsort(-scores)
    top_k = order[:min(k, len(order))]
    return float(actual[top_k].sum() / total_value)


def _recall_at_k(
    actual: np.ndarray, scores: np.ndarray, k: int, threshold: float = 0.0
) -> float:
    binding_mask = actual > threshold
    n_binding = binding_mask.sum()
    if n_binding == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k = order[:min(k, len(order))]
    return float(binding_mask[top_k].sum() / n_binding)


def _ndcg(actual: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    sorted_actual = actual[order]
    # DCG
    discounts = np.log2(np.arange(2, len(sorted_actual) + 2))
    dcg = (sorted_actual / discounts).sum()
    # Ideal DCG
    ideal_order = np.argsort(-actual)
    ideal_sorted = actual[ideal_order]
    idcg = (ideal_sorted / discounts).sum()
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def evaluate_pipeline(
    actual_shadow_price: np.ndarray,
    pred_proba: np.ndarray,
    pred_shadow_price: np.ndarray,
    ev_scores: np.ndarray,
) -> dict[str, float]:
    metrics = {}

    # EV-based ranking metrics (threshold-independent)
    metrics["EV-VC@100"] = _value_capture_at_k(actual_shadow_price, ev_scores, 100)
    metrics["EV-VC@500"] = _value_capture_at_k(actual_shadow_price, ev_scores, 500)
    metrics["EV-VC@1000"] = _value_capture_at_k(actual_shadow_price, ev_scores, 1000)
    metrics["EV-NDCG"] = _ndcg(actual_shadow_price, ev_scores)
    metrics["R-REC@500"] = _recall_at_k(actual_shadow_price, ev_scores, 500)

    # Regression calibration on binding-only samples
    binding_mask = actual_shadow_price > 0
    if binding_mask.sum() > 1:
        actual_binding = actual_shadow_price[binding_mask]
        pred_binding = pred_shadow_price[binding_mask]
        metrics["C-RMSE"] = float(np.sqrt(np.mean((actual_binding - pred_binding) ** 2)))
        metrics["C-MAE"] = float(np.mean(np.abs(actual_binding - pred_binding)))
        spearman_r, _ = stats.spearmanr(actual_binding, pred_binding)
        metrics["Spearman"] = float(spearman_r) if not np.isnan(spearman_r) else 0.0
    else:
        metrics["C-RMSE"] = 0.0
        metrics["C-MAE"] = 0.0
        metrics["Spearman"] = 0.0

    # Monitoring metrics
    metrics["binding_rate"] = float(binding_mask.mean())
    metrics["n_samples"] = int(len(actual_shadow_price))
    metrics["n_binding"] = int(binding_mask.sum())

    return metrics


def aggregate_months(per_month: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    gate_keys = ["EV-VC@100", "EV-VC@500", "EV-VC@1000", "EV-NDCG",
                 "Spearman", "C-RMSE", "C-MAE", "R-REC@500"]
    months = list(per_month.keys())
    n = len(months)

    agg: dict[str, dict[str, float]] = {
        "mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {},
    }

    for key in gate_keys:
        values = [per_month[m].get(key, 0.0) for m in months]
        arr = np.array(values)
        agg["mean"][key] = float(arr.mean())
        agg["std"][key] = float(arr.std())
        agg["min"][key] = float(arr.min())
        agg["max"][key] = float(arr.max())

        # bottom_2_mean: for "lower is better" metrics (RMSE, MAE), take highest 2
        if key in ("C-RMSE", "C-MAE"):
            sorted_vals = np.sort(arr)[-2:] if n >= 2 else arr
        else:
            sorted_vals = np.sort(arr)[:2] if n >= 2 else arr
        agg["bottom_2_mean"][key] = float(sorted_vals.mean())

    return agg
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: EV-based evaluation harness (HUMAN-WRITE-ONLY)"
```

---

### Task 11: ML Pipeline (`ml/pipeline.py`)

**Files:**
- Create: `ml/pipeline.py`
- Test: `ml/tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# ml/tests/test_pipeline.py
import os
os.environ["SMOKE_TEST"] = "true"

from ml.pipeline import run_pipeline
from ml.config import PipelineConfig

def test_run_pipeline_smoke():
    cfg = PipelineConfig()
    result = run_pipeline(
        config=cfg,
        version_id="v_test",
        auction_month="2021-07",
        class_type="onpeak",
        period_type="f0",
    )
    assert "metrics" in result
    metrics = result["metrics"]
    assert "EV-VC@100" in metrics
    assert "EV-VC@500" in metrics
    assert "EV-NDCG" in metrics
    assert "Spearman" in metrics
    assert "C-RMSE" in metrics
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Write `ml/pipeline.py`**

```python
"""Main pipeline: load → clf → reg → EV-score → evaluate.

Usage:
  python ml/pipeline.py --version-id v0001 --auction-month 2021-07 --class-type onpeak --period-type f0
"""
from __future__ import annotations
import argparse
import gc
import json
import resource
from pathlib import Path

import numpy as np

from ml.config import PipelineConfig
from ml.data_loader import load_data, mem_mb
from ml.features import (
    prepare_clf_features,
    prepare_reg_features,
    compute_binary_labels,
    compute_regression_target,
)
from ml.train import train_classifier, train_regressor, predict_proba, predict_shadow_price
from ml.evaluate import evaluate_pipeline


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    auction_month: str,
    class_type: str = "onpeak",
    period_type: str = "f0",
    from_phase: int = 1,
) -> dict:
    print(f"[pipeline] Starting pipeline for {version_id}, {auction_month}")
    print(f"[pipeline] Memory at start: {mem_mb():.0f} MB")

    # Phase 1: Load data
    if from_phase <= 1:
        print("[pipeline] Phase 1: Loading data...")
        train_df, val_df = load_data(config, auction_month, class_type, period_type)
        print(f"[pipeline] Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        print(f"[pipeline] Memory after load: {mem_mb():.0f} MB")

    # Phase 2: Prepare features
    if from_phase <= 2:
        print("[pipeline] Phase 2: Preparing features...")
        X_train_clf, mono_clf = prepare_clf_features(train_df, config.classifier)
        X_val_clf, _ = prepare_clf_features(val_df, config.classifier)
        y_train_binary = compute_binary_labels(train_df)
        y_val_binary = compute_binary_labels(val_df)

        X_train_reg, mono_reg = prepare_reg_features(train_df, config.regressor)
        y_train_reg = compute_regression_target(train_df)

    # Phase 3: Train classifier (frozen config)
    if from_phase <= 3:
        print("[pipeline] Phase 3: Training classifier (frozen config)...")
        clf_model, threshold = train_classifier(
            X_train_clf, y_train_binary, config.classifier,
            X_val=X_val_clf, y_val=y_val_binary,
        )
        print(f"[pipeline] Threshold: {threshold:.3f}")

    # Phase 4: Train regressor
    if from_phase <= 4:
        print("[pipeline] Phase 4: Training regressor...")
        if config.regressor.unified_regressor:
            # Unified: train on all samples
            reg_model = train_regressor(X_train_reg, y_train_reg, config.regressor)
        else:
            # Gated: train on binding samples only
            binding_mask = y_train_binary == 1
            if binding_mask.sum() < 5:
                print("[pipeline] WARNING: <5 binding samples, using all samples")
                reg_model = train_regressor(X_train_reg, y_train_reg, config.regressor)
            else:
                reg_model = train_regressor(
                    X_train_reg[binding_mask],
                    y_train_reg[binding_mask],
                    config.regressor,
                )
        print(f"[pipeline] Memory after train: {mem_mb():.0f} MB")

    # Phase 5: Evaluate on validation set
    if from_phase <= 5:
        print("[pipeline] Phase 5: Evaluating...")
        X_val_reg, _ = prepare_reg_features(val_df, config.regressor)
        val_proba = predict_proba(clf_model, X_val_clf)

        if config.regressor.unified_regressor:
            val_shadow = predict_shadow_price(reg_model, X_val_reg)
        else:
            # Gated: only predict for samples above threshold
            val_shadow = np.zeros(len(val_df))
            above_threshold = val_proba >= threshold
            if above_threshold.sum() > 0:
                val_shadow[above_threshold] = predict_shadow_price(
                    reg_model, X_val_reg[above_threshold]
                )

        # EV scoring
        if config.ev_scoring:
            ev_scores = val_proba * val_shadow
        else:
            ev_scores = val_shadow

        actual = val_df["actual_shadow_price"].to_numpy()
        metrics = evaluate_pipeline(
            actual_shadow_price=actual,
            pred_proba=val_proba,
            pred_shadow_price=val_shadow,
            ev_scores=ev_scores,
        )
        metrics["threshold"] = threshold
        print(f"[pipeline] Metrics: { {k: f'{v:.4f}' for k, v in metrics.items() if isinstance(v, float)} }")

    # Cleanup
    del train_df, val_df
    gc.collect()

    return {"metrics": metrics, "threshold": threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--auction-month", required=True)
    parser.add_argument("--class-type", default="onpeak")
    parser.add_argument("--period-type", default="f0")
    parser.add_argument("--from-phase", type=int, default=1)
    parser.add_argument("--config-override", type=str, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.config_override:
        override = json.loads(args.config_override)
        cfg = PipelineConfig.from_dict({**cfg.to_dict(), **override})

    result = run_pipeline(
        config=cfg,
        version_id=args.version_id,
        auction_month=args.auction_month,
        class_type=args.class_type,
        period_type=args.period_type,
        from_phase=args.from_phase,
    )
    print(f"\n[pipeline] Done. Metrics saved for {args.version_id}")
```

**Step 4: Run test to verify it passes**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: main pipeline (load → clf → reg → EV → evaluate)"
```

---

### Task 12: ML Registry, Compare, Benchmark, Populate Gates

These are largely ported from stage 1 with metric name changes.

**Files:**
- Create: `ml/registry.py` (port from stage 1)
- Create: `ml/compare.py` (port from stage 1, change metric names)
- Create: `ml/benchmark.py` (port from stage 1, call regression pipeline)
- Create: `ml/populate_v0_gates.py` (port from stage 1, change metric names)

**Step 1: Port `ml/registry.py` from stage 1**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/registry.py ml/registry.py
```

This is 99% generic. The only change: update the default `registry_dir` path to point at `research-stage2-shadow/registry`.

**Step 2: Port `ml/compare.py` from stage 1**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/compare.py ml/compare.py
```

Changes needed:
- Update gate metric names from `S1-*` to `EV-*`, `C-*`, `Spearman`, `R-REC@*`
- Update "lower is better" detection to include `C-RMSE` and `C-MAE`
- Everything else (3-layer gate checks, markdown table generation, comparison JSON) stays the same

**Step 3: Port `ml/benchmark.py` from stage 1**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py ml/benchmark.py
```

Changes needed:
- Import from `ml.pipeline` instead of stage 1's pipeline
- The `_eval_single_month()` function calls `run_pipeline()` instead of stage 1's classifier pipeline
- Feature importance: extract from regressor model (not classifier)
- Everything else (multi-month loop, Ray init, aggregation, registry write) stays the same

**Step 4: Port `ml/populate_v0_gates.py` from stage 1**

```bash
cp /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/populate_v0_gates.py ml/populate_v0_gates.py
```

Changes needed:
- Update gate metric names
- Update "lower is better" list to `["C-RMSE", "C-MAE"]`
- Update Group A/B assignments

**Step 5: Commit**

```bash
git add ml/ && git commit -m "feat: registry, compare, benchmark, populate_v0_gates (ported from stage 1)"
```

---

### Task 13: Agent Prompts

**Files:**
- Create: `agents/prompts/orchestrator_plan.md`
- Create: `agents/prompts/orchestrator_synthesize.md`
- Create: `agents/prompts/worker.md`
- Create: `agents/prompts/reviewer_claude.md`
- Create: `agents/prompts/reviewer_codex.md`

**Step 1: Write orchestrator_plan.md**

Adapt from stage 1's `agents/prompts/orchestrator_plan.md`:
- Change identity: "shadow price **regression** ML research pipeline"
- Replace all `S1-*` metric references with `EV-*`, `C-*`, `Spearman`, `R-REC@*`
- Replace gate groups: Group A = `EV-VC@100, EV-VC@500, EV-NDCG, Spearman`; Group B = `C-RMSE, C-MAE, EV-VC@1000, R-REC@500`
- Add constraint: "Do NOT modify ClassifierConfig — only RegressorConfig is mutable"
- Replace "precision > recall" business objective with: "Maximize expected value ranking quality. All gates are threshold-independent."
- Add context about frozen classifier from stage 1

**Step 2: Write orchestrator_synthesize.md**

Same adaptations as plan prompt, plus:
- Update promotion criteria to use EV-based gates
- Update memory update instructions with regression-specific content

**Step 3: Write worker.md**

Adapt from stage 1's `agents/prompts/worker.md`:
- Change pipeline invocation from `ml/pipeline.py` to stage 2's pipeline
- Add constraint: "NEVER modify `ClassifierConfig` in `ml/config.py`"
- Update test command
- Benchmark invocation stays similar

**Step 4: Write reviewer_claude.md and reviewer_codex.md**

Same metric name changes. Add focus on:
- Regression quality (RMSE, MAE, Spearman)
- EV ranking quality (VC@K, NDCG)
- Regressor-specific analysis (feature importance, training mode)

**Step 5: Commit**

```bash
git add agents/prompts/ && git commit -m "feat: regression-specific agent prompts"
```

---

### Task 14: Memory System & Runbook

**Files:**
- Create: `memory/hot/progress.md`
- Create: `memory/hot/champion.md`
- Create: `memory/hot/learning.md`
- Create: `memory/hot/gate_calibration.md`
- Create: `memory/hot/critique_summary.md`
- Create: `memory/hot/runbook.md`
- Create: `memory/warm/experiment_log.md`
- Create: `memory/warm/hypothesis_log.md`
- Create: `memory/warm/decision_log.md`
- Create: `memory/archive/index.md`
- Create: `memory/human_input.md`
- Create: `runbook.md`

**Step 1: Initialize memory stubs**

Create all hot/warm/archive files with minimal stub content (same pattern as stage 1):

```markdown
# memory/hot/progress.md
## Status: IDLE
No batch in progress.

# memory/hot/champion.md
## Champion: v0 (baseline)
No experiments run yet.

# memory/hot/learning.md
## Accumulated Learning
(Empty — will be populated after first batch)

# memory/hot/gate_calibration.md
## Gate Calibration
(Empty — will be populated after v0 gates are bootstrapped)

# memory/hot/critique_summary.md
## Critique Summary
(Empty — will be populated after first review)

# memory/hot/runbook.md
## Runbook — Safety Rules for Workers
[Same as stage 1 runbook, adapted for regression constraints]
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and `handoff/`
- NEVER touch `registry/v0/` (baseline is immutable)
- NEVER modify `registry/gates.json` or `ml/evaluate.py` (HUMAN-WRITE-ONLY)
- NEVER modify `ClassifierConfig` in `ml/config.py` (frozen from stage 1)
- Only modify `RegressorConfig` parameters
```

**Step 2: Initialize warm memory stubs**

```markdown
# memory/warm/experiment_log.md
## Experiment Log
| Batch | Iter | Version | Hypothesis | Result | Key Metrics |
|-------|------|---------|-----------|--------|-------------|

# memory/warm/hypothesis_log.md
## Hypothesis Log
(No hypotheses tested yet)

# memory/warm/decision_log.md
## Decision Log
(No decisions recorded yet)
```

**Step 3: Initialize archive**

```markdown
# memory/archive/index.md
## Archived Batches
(No batches archived yet)
```

**Step 4: Write runbook.md**

Adapt from stage 1's runbook.md with regression-specific content:
- Change all metric names
- Change pipeline commands
- Add note about ClassifierConfig being frozen

**Step 5: Write business_context.md for stage 2**

Create `human-input/business_context.md` that describes the regression problem, EV-based metrics, and the relationship with stage 1.

**Step 6: Commit**

```bash
git add memory/ runbook.md human-input/ && git commit -m "feat: memory system, runbook, and business context"
```

---

### Task 15: Registry Bootstrap & v0 Baseline

**Files:**
- Create: `registry/v0/config.json`
- Create: `registry/v0/meta.json`
- Create: `registry/gates.json` (skeleton with `pending_v0: true`)
- Create: `registry/champion.json`
- Create: `registry/version_counter.json`

**Step 1: Create v0 config.json**

```json
{
  "classifier": {
    "step1_features": [["prob_exceed_110", 1], ...],
    "n_estimators": 200,
    "max_depth": 4,
    ...
  },
  "regressor": {
    "step2_features": [["prob_exceed_110", 1], ...],
    "n_estimators": 400,
    "max_depth": 5,
    ...
  },
  "train_months": 10,
  "val_months": 2,
  "ev_scoring": true
}
```

Generate from `PipelineConfig().to_dict()`.

**Step 2: Create meta.json**

```json
{
  "version_id": "v0",
  "created_at": "2026-03-03T00:00:00Z",
  "description": "Baseline: gated regressor with 24 features, frozen stage 1 classifier"
}
```

**Step 3: Create gates.json skeleton**

```json
{
  "version": 2,
  "pending_v0": true,
  "noise_tolerance": 0.02,
  "tail_max_failures": 1,
  "eval_months": {
    "primary": ["2020-09", "2020-11", "2021-01", "2021-03", "2021-05",
                 "2021-07", "2021-09", "2021-11", "2022-03", "2022-06",
                 "2022-09", "2022-12"]
  },
  "cascade_stages": [
    {"stage": 1, "ptype": "f0", "blocking": true},
    {"stage": 2, "ptype": "f1", "blocking": true},
    {"stage": 3, "ptype": "f2p", "blocking": false}
  ],
  "gates": {
    "EV-VC@100": {"floor": null, "tail_floor": null, "direction": "higher", "group": "A"},
    "EV-VC@500": {"floor": null, "tail_floor": null, "direction": "higher", "group": "A"},
    "EV-NDCG": {"floor": null, "tail_floor": null, "direction": "higher", "group": "A"},
    "Spearman": {"floor": null, "tail_floor": null, "direction": "higher", "group": "A"},
    "C-RMSE": {"floor": null, "tail_floor": null, "direction": "lower", "group": "B"},
    "C-MAE": {"floor": null, "tail_floor": null, "direction": "lower", "group": "B"},
    "EV-VC@1000": {"floor": null, "tail_floor": null, "direction": "higher", "group": "B"},
    "R-REC@500": {"floor": null, "tail_floor": null, "direction": "higher", "group": "B"}
  }
}
```

**Step 4: Create champion.json and version_counter.json**

```json
// champion.json
{"version": "v0", "promoted_at": "2026-03-03T00:00:00Z"}

// version_counter.json
{"next_id": 1}
```

**Step 5: Commit**

```bash
git add registry/ && git commit -m "feat: registry bootstrap with v0 skeleton and gates"
```

---

### Task 16: Implement Real Data Loader

**Files:**
- Modify: `ml/data_loader.py` (implement `_load_real()`)

**Step 1: Implement `_load_real()` in `ml/data_loader.py`**

Port from the original repo's `src/shadow_price_prediction/data_loader.py`:
- Load density features via pbase data loaders
- Load historical shadow prices via `MisoDaLmpMonthlyAgg`
- Compute all 24 step2 features
- Apply rolling window (10 months train + 2 months val)
- Return continuous `actual_shadow_price` (not binary)

Key references:
- Original repo's `MisoDataLoader._load_training_data()` for density file loading
- Original repo's `BaseDataLoader._compute_historical_features()` for hist_da features
- Stage 1's `ml/data_loader.py._load_real()` for the polars/Ray integration pattern

This is the most complex single task — it bridges pbase data loaders with the stage 2 feature set.

**Step 2: Test with single real month**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
python -c "
from ml.data_loader import load_data
from ml.config import PipelineConfig
train_df, val_df = load_data(PipelineConfig(), '2021-07', 'onpeak', 'f0')
print(f'Train: {len(train_df)}, Val: {len(val_df)}')
print(f'Columns: {train_df.columns}')
print(f'Binding rate: {(train_df[\"actual_shadow_price\"] > 0).mean():.3f}')
"
```

**Step 3: Commit**

```bash
git add ml/ && git commit -m "feat: real data loader via Ray/pbase"
```

---

### Task 17: Run v0 Benchmark & Bootstrap Gates

**Step 1: Run v0 benchmark**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak
```

This runs the 12-month rolling benchmark and writes `registry/v0/metrics.json`.

**Step 2: Bootstrap gates from v0**

```bash
python ml/populate_v0_gates.py
```

This fills in the null floors in `gates.json` based on v0 baseline results.

**Step 3: Verify gates**

```bash
cat registry/gates.json | python -m json.tool
cat registry/v0/metrics.json | python -m json.tool
```

**Step 4: Update memory**

Update `memory/hot/champion.md` with v0 baseline metrics.

**Step 5: Commit**

```bash
git add registry/ memory/ && git commit -m "feat: v0 baseline benchmark + gate bootstrap"
```

---

### Task 18: End-to-End Smoke Test

**Step 1: Verify smoke test pipeline**

```bash
SMOKE_TEST=true python ml/pipeline.py --version-id v_smoke --auction-month 2021-07 --class-type onpeak --period-type f0
```

**Step 2: Verify full pipeline (single real month)**

```bash
python ml/pipeline.py --version-id v_test --auction-month 2021-07 --class-type onpeak --period-type f0
```

**Step 3: Verify compare works**

```bash
python ml/compare.py --batch-id smoke-test --iteration 1
```

**Step 4: Verify agents can launch (dry run)**

```bash
bash agents/run_pipeline.sh --batch-name smoke-test --max-iter 1 --foreground --dry-run
```

**Step 5: Run first real autonomous batch**

```bash
SMOKE_TEST=true bash agents/run_pipeline.sh --batch-name smoke-v1 --max-iter 1
```

**Step 6: Commit any fixes**

```bash
git add -A && git commit -m "fix: smoke test fixes from end-to-end verification"
```

---

### Task 19: Run First Autonomous Batch (Real Data)

**Step 1: Write human guidance**

```bash
echo "Focus on establishing a v0 baseline regressor. Use gated mode (binding-only training) with default hyperparameters. The first hypothesis to test will be determined by the orchestrator." > memory/human_input.md
```

**Step 2: Launch 3-iteration batch**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
bash agents/run_pipeline.sh --batch-name reg-baseline-$(date +%Y%m%d-%H%M%S)
```

**Step 3: Monitor**

```bash
# Attach to tmux
tmux attach -t pipeline-*

# Check state
jq . state.json

# Check logs
tail -f .logs/sessions/pipeline-*.log
```

---

## Summary: Critical Path

```
Task 1: Scaffold ──→ Task 2-4: Infrastructure ──→ Task 5-11: ML Code ──→ Task 12: Registry/Compare
                                                                              │
Task 13: Prompts ──→ Task 14: Memory ──→ Task 15: Registry Bootstrap ─────────┘
                                                        │
                                              Task 16: Real Data Loader
                                                        │
                                              Task 17: v0 Benchmark
                                                        │
                                              Task 18: Smoke Test
                                                        │
                                              Task 19: First Batch
```

Tasks 1-15 can be done with smoke data only. Task 16 requires Ray access. Task 17+ requires real data.
