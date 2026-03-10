# Tier Classification Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port stage-2 regression pipeline into a stage-3 tier classification pipeline with new metrics, configs, and agent prompts.

**Architecture:** Clone research-stage2-shadow, replace two-stage clf+reg with single multi-class XGBoost, replace regression metrics with tier classification metrics (Tier-VC@K, QWK, Macro-F1), adapt all 5 agent prompts.

**Tech Stack:** Python, XGBoost (multi:softprob), numpy, scipy, sklearn

---

### Task 1: Clone directory and scaffold

**Files:**
- Create: `research-stage3-tier/` (full directory tree)

**Step 1: Clone stage 2**

```bash
cd /home/xyz/workspace/research-qianli-v2
cp -r research-stage2-shadow research-stage3-tier
```

**Step 2: Clean clone**

```bash
cd research-stage3-tier
# Remove old registry versions (keep structure)
rm -rf registry/v0???* registry/v1*
# Reset state
echo '{"state":"IDLE","iter":0,"version_id":null,"batch_id":null}' > state.json
# Reset version counter
echo '{"next_id": 1}' > registry/version_counter.json
# Clean handoff and memory
rm -rf handoff/* memory/warm/*.md memory/hot/*.md memory/archive/*
# Create stub memory files
for f in memory/hot/progress.md memory/hot/champion.md memory/hot/learning.md memory/hot/runbook.md memory/hot/gate_calibration.md memory/hot/critique_summary.md; do
  echo "# $(basename $f .md)" > "$f"
done
for f in memory/warm/experiment_log.md memory/warm/hypothesis_log.md memory/warm/decision_log.md; do
  echo "# $(basename $f .md)" > "$f"
done
echo "# Archive Index" > memory/archive/index.md
```

**Step 3: Patch config.sh**

Update `PROJECT_DIR` to point to `research-stage3-tier`:
```bash
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage3-tier"
```

**Step 4: Verify structure**

```bash
ls -la research-stage3-tier/ml/
ls -la research-stage3-tier/agents/prompts/
ls -la research-stage3-tier/registry/
```

**Step 5: Commit**

```bash
git add research-stage3-tier/
git commit -m "scaffold: clone stage-2 as research-stage3-tier for tier classification"
```

---

### Task 2: Implement TierConfig (config.py)

**Files:**
- Modify: `research-stage3-tier/ml/config.py`

**Step 1: Write tests**

Create `research-stage3-tier/ml/tests/test_config.py`:
```python
import numpy as np
from ml.config import TierConfig, PipelineConfig

def test_tier_config_defaults():
    cfg = TierConfig()
    assert len(cfg.features) == 34
    assert cfg.bins == [float('-inf'), 0, 100, 1000, 3000, float('inf')]
    assert cfg.tier_midpoints == [4000, 2000, 550, 50, 0]
    assert cfg.num_class == 5
    assert len(cfg.class_weights) == 5

def test_tier_config_roundtrip():
    cfg = TierConfig()
    d = cfg.to_dict()
    cfg2 = TierConfig.from_dict(d)
    assert cfg2.features == cfg.features
    assert cfg2.bins == cfg.bins

def test_pipeline_config():
    cfg = PipelineConfig()
    assert cfg.train_months == 6
    assert cfg.val_months == 2
    d = cfg.to_dict()
    cfg2 = PipelineConfig.from_dict(d)
    assert cfg2.tier.features == cfg.tier.features
```

**Step 2: Run test to verify it fails**

```bash
cd research-stage3-tier && SMOKE_TEST=true python -m pytest ml/tests/test_config.py -v
```
Expected: FAIL (TierConfig doesn't exist yet)

**Step 3: Implement config.py**

Replace `ClassifierConfig` + `RegressorConfig` with:

```python
@dataclass
class TierConfig:
    """Tier classification configuration. Single model, all mutable."""
    features: list[str] = field(default_factory=lambda: list(_ALL_REGRESSOR_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(_ALL_REGRESSOR_MONOTONE))
    bins: list[float] = field(default_factory=lambda: [float('-inf'), 0, 100, 1000, 3000, float('inf')])
    tier_midpoints: list[float] = field(default_factory=lambda: [4000, 2000, 550, 50, 0])
    num_class: int = 5
    class_weights: dict[int, float] = field(default_factory=lambda: {0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5})

    # XGBoost hyperparams
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    min_child_weight: int = 25

@dataclass
class PipelineConfig:
    tier: TierConfig = field(default_factory=TierConfig)
    train_months: int = 6
    val_months: int = 2
```

Keep `GateConfig` unchanged. Remove `ClassifierConfig`, `RegressorConfig`.

**Step 4: Run test**

```bash
SMOKE_TEST=true python -m pytest ml/tests/test_config.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add ml/config.py ml/tests/test_config.py
git commit -m "feat: replace clf+reg configs with TierConfig for multi-class classification"
```

---

### Task 3: Implement tier labels and features (features.py)

**Files:**
- Modify: `research-stage3-tier/ml/features.py`

**Step 1: Write tests**

Add to `research-stage3-tier/ml/tests/test_features.py`:
```python
import numpy as np
from ml.features import compute_tier_labels
from ml.config import TierConfig

def test_compute_tier_labels():
    cfg = TierConfig()
    actual = np.array([5000, 2000, 500, 50, -10, 0, 100, 1000, 3000])
    labels = compute_tier_labels(actual, cfg)
    # 5000 -> tier 0, 2000 -> tier 1, 500 -> tier 2, 50 -> tier 3, -10 -> tier 4
    # 0 -> tier 3 (0 is in [0, 100)), 100 -> tier 2 (100 is in [100, 1000))
    # 1000 -> tier 1 (1000 is in [1000, 3000)), 3000 -> tier 0
    expected = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
    np.testing.assert_array_equal(labels, expected)

def test_tier_labels_all_nonbinding():
    cfg = TierConfig()
    actual = np.array([-5, -10, -100])
    labels = compute_tier_labels(actual, cfg)
    np.testing.assert_array_equal(labels, np.array([4, 4, 4]))
```

**Step 2: Implement**

Add `compute_tier_labels()`:
```python
def compute_tier_labels(actual_shadow_price: np.ndarray, config: TierConfig) -> np.ndarray:
    """Bin actual shadow prices into tier labels using config bins."""
    bins = np.array(config.bins)
    labels = np.array([4, 3, 2, 1, 0])  # tier 4 = (-inf,0), tier 0 = (3000,inf)
    indices = np.digitize(actual_shadow_price, bins[1:-1], right=False)
    return labels[indices]
```

Remove `compute_binary_labels()` and `compute_regression_target()`.
Keep `prepare_clf_features()` renamed to `prepare_features()` (single feature set).
Remove `prepare_reg_features()`.

**Step 3: Run tests, commit**

---

### Task 4: Implement multi-class training (train.py)

**Files:**
- Modify: `research-stage3-tier/ml/train.py`

**Step 1: Write tests**

```python
import numpy as np
from ml.train import train_tier_classifier, predict_tier_probabilities
from ml.config import TierConfig

def test_train_tier_classifier():
    np.random.seed(42)
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 5, 200)
    cfg = TierConfig()
    cfg.features = [f"f{i}" for i in range(5)]
    cfg.monotone_constraints = [0] * 5
    cfg.n_estimators = 10
    model = train_tier_classifier(X, y, cfg)
    assert model is not None

def test_predict_tier_probabilities():
    np.random.seed(42)
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 5, 200)
    cfg = TierConfig()
    cfg.features = [f"f{i}" for i in range(5)]
    cfg.monotone_constraints = [0] * 5
    cfg.n_estimators = 10
    model = train_tier_classifier(X, y, cfg)
    proba = predict_tier_probabilities(model, X)
    assert proba.shape == (200, 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
```

**Step 2: Implement**

Replace `train_classifier()`, `train_regressor()`, `predict_proba()`, `predict_shadow_price()` with:

```python
def train_tier_classifier(X, y, config: TierConfig, sample_weight=None):
    """Train XGBoost multi-class classifier for tier prediction."""
    if sample_weight is None:
        sample_weight = np.array([config.class_weights[int(label)] for label in y])
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=config.num_class,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        min_child_weight=config.min_child_weight,
        tree_method='hist',
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model

def predict_tier_probabilities(model, X) -> np.ndarray:
    """Return (n_samples, 5) matrix of tier probabilities."""
    return model.predict_proba(X)

def predict_tier(model, X) -> np.ndarray:
    """Return predicted tier labels (argmax of probabilities)."""
    return model.predict(X).astype(int)

def compute_tier_ev_score(proba: np.ndarray, midpoints: list[float]) -> np.ndarray:
    """Probability-weighted expected shadow price from tier probabilities."""
    return proba @ np.array(midpoints)
```

**Step 3: Run tests, commit**

---

### Task 5: Implement tier evaluation metrics (evaluate.py)

**Files:**
- Modify: `research-stage3-tier/ml/evaluate.py`

This is the HUMAN-WRITE-ONLY file. Workers cannot modify it.

**Step 1: Write tests**

```python
import numpy as np
from ml.evaluate import evaluate_tier_pipeline, _quadratic_weighted_kappa

def test_qwk_perfect():
    actual = np.array([0, 1, 2, 3, 4])
    pred = np.array([0, 1, 2, 3, 4])
    assert _quadratic_weighted_kappa(actual, pred, 5) == 1.0

def test_qwk_random():
    actual = np.array([0, 1, 2, 3, 4])
    pred = np.array([4, 3, 2, 1, 0])  # worst possible
    qwk = _quadratic_weighted_kappa(actual, pred, 5)
    assert qwk < 0  # negative for anti-correlated

def test_tier_vc_at_k():
    actual_sp = np.array([5000, 1000, 100, 10, 0])
    tier_ev = np.array([4000, 2000, 550, 50, 0])  # perfect ranking
    from ml.evaluate import _value_capture_at_k
    vc = _value_capture_at_k(actual_sp, tier_ev, 2)
    assert vc > 0.9  # top 2 should capture 5000+1000 out of 6110

def test_evaluate_tier_pipeline():
    np.random.seed(42)
    n = 100
    actual_sp = np.abs(np.random.randn(n)) * 500
    actual_tier = np.array([0]*5 + [1]*10 + [2]*20 + [3]*30 + [4]*35)
    proba = np.eye(5)[actual_tier] * 0.8 + 0.04  # mostly correct
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred_tier = actual_tier.copy()
    tier_ev = proba @ np.array([4000, 2000, 550, 50, 0])
    metrics = evaluate_tier_pipeline(actual_sp, actual_tier, pred_tier, proba, tier_ev)
    assert "Tier-VC@100" in metrics
    assert "QWK" in metrics
    assert "Macro-F1" in metrics
```

**Step 2: Implement evaluate.py**

```python
"""Evaluation harness for tier classification pipeline.

Tier metrics: ranking quality (Tier-VC@K, Tier-NDCG) + ordinal quality (QWK, Macro-F1).
All ranking metrics use tier_ev_score for ranking and actual_shadow_price as relevance.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

_LOWER_IS_BETTER: set[str] = set()  # No lower-is-better metrics in tier pipeline


def _value_capture_at_k(actual, ev_scores, k):
    # Same as stage 2 — reuse exactly
    ...

def _ndcg(actual, ev_scores):
    # Same as stage 2 — reuse exactly
    ...

def _quadratic_weighted_kappa(actual_tier, pred_tier, num_classes=5):
    """Cohen's Quadratic Weighted Kappa for ordinal classification."""
    ...

def evaluate_tier_pipeline(
    actual_shadow_price, actual_tier, pred_tier, tier_proba, tier_ev_score
) -> dict:
    return {
        # Group A (blocking)
        "Tier-VC@100": _value_capture_at_k(actual_shadow_price, tier_ev_score, 100),
        "Tier-VC@500": _value_capture_at_k(actual_shadow_price, tier_ev_score, 500),
        "Tier-NDCG": _ndcg(actual_shadow_price, tier_ev_score),
        "QWK": _quadratic_weighted_kappa(actual_tier, pred_tier),
        # Group B (monitor)
        "Macro-F1": float(f1_score(actual_tier, pred_tier, average='macro', zero_division=0)),
        "Tier-Accuracy": float(np.mean(actual_tier == pred_tier)),
        "Adjacent-Accuracy": float(np.mean(np.abs(actual_tier - pred_tier) <= 1)),
        "Tier-Recall@0": ...,  # recall for tier 0
        "Tier-Recall@1": ...,  # recall for tier 1
        # Monitoring
        "n_samples": len(actual_tier),
        "tier_distribution": {int(t): int((actual_tier == t).sum()) for t in range(5)},
    }

def aggregate_months(per_month):
    # Same as stage 2 — reuse exactly
    ...
```

**Step 3: Run tests, commit**

---

### Task 6: Implement tier pipeline (pipeline.py)

**Files:**
- Modify: `research-stage3-tier/ml/pipeline.py`

**Step 1: Write tests**

Add smoke test that runs 1 month with tiny data:
```python
def test_pipeline_smoke(monkeypatch):
    """Smoke test: pipeline runs end-to-end with mock data."""
    # Mock data_loader to return small DataFrames
    ...
```

**Step 2: Implement 6-phase pipeline**

Replace the 7-phase clf+reg pipeline with:

```python
def run_pipeline(config, version_id, auction_month, class_type, period_type, from_phase=1):
    # Phase 1: Load train/val data (identical)
    # Phase 2: Prepare features + tier labels (compute_tier_labels instead of binary+regression)
    # Phase 3: Train multi-class XGBoost (single model)
    # Phase 4: Load test data (identical)
    # Phase 5: Evaluate (predict_tier_probabilities → compute_tier_ev_score → evaluate_tier_pipeline)
    # Phase 6: Return results
```

Key differences:
- No separate classifier/regressor training
- `compute_tier_labels(train_df, config.tier)` produces tier labels
- `train_tier_classifier(X_train, y_train_tier, config.tier)` trains single model
- At test time: `proba = predict_tier_probabilities(model, X_test)`, `tier_ev = compute_tier_ev_score(proba, config.tier.tier_midpoints)`, `pred_tier = proba.argmax(axis=1)`

**Step 3: Run tests, commit**

---

### Task 7: Adapt benchmark.py and compare.py

**Files:**
- Modify: `research-stage3-tier/ml/benchmark.py`
- Modify: `research-stage3-tier/ml/compare.py`

Minimal changes:
- `benchmark.py`: Update config imports, remove `_feature_importance` extraction for classifier (replace with single model importance)
- `compare.py`: Update metric names in display. 3-layer gate logic is metric-agnostic and works as-is.

**Commit after tests pass.**

---

### Task 8: Write gates.json with pending baseline

**Files:**
- Modify: `research-stage3-tier/registry/gates.json`

```json
{
  "version": 1,
  "note": "Initial gates — pending v0 baseline calibration",
  "noise_tolerance": 0.02,
  "tail_max_failures": 1,
  "eval_months": {
    "primary": ["2020-09","2020-11","2021-01","2021-03","2021-05","2021-07","2021-09","2021-11","2022-03","2022-06","2022-09","2022-12"]
  },
  "gates": {
    "Tier-VC@100": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "A", "pending_baseline": true},
    "Tier-VC@500": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "A", "pending_baseline": true},
    "Tier-NDCG": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "A", "pending_baseline": true},
    "QWK": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "A", "pending_baseline": true},
    "Macro-F1": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "B", "pending_baseline": true},
    "Tier-Accuracy": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "B", "pending_baseline": true},
    "Adjacent-Accuracy": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "B", "pending_baseline": true},
    "Tier-Recall@0": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "B", "pending_baseline": true},
    "Tier-Recall@1": {"floor": 0.0, "tail_floor": 0.0, "direction": "higher", "group": "B", "pending_baseline": true}
  }
}
```

**Commit.**

---

### Task 9: Adapt agent prompts — orchestrator_plan.md

**Files:**
- Modify: `research-stage3-tier/agents/prompts/orchestrator_plan.md`

Key changes:
- Identity: "Planning Orchestrator for **tier classification** pipeline"
- Remove all "frozen classifier" / "RegressorConfig" references
- Replace with "TierConfig — all parameters mutable (features, bins, class_weights, hyperparams)"
- Update metric names: EV-VC → Tier-VC, Spearman → QWK, drop C-RMSE/C-MAE
- Add hypothesis types: "bin edge adjustment", "class weight tuning", "tier midpoint calibration"
- Winner criteria: "Tier-VC@100 (primary), QWK (secondary)"
- Overrides target `tier` key (not `regressor`): `--overrides '{"tier": {"n_estimators": 500}}'`

**Commit.**

---

### Task 10: Adapt agent prompts — worker.md, reviewers, orchestrator_synthesize.md

**Files:**
- Modify: `research-stage3-tier/agents/prompts/worker.md`
- Modify: `research-stage3-tier/agents/prompts/reviewer_claude.md`
- Modify: `research-stage3-tier/agents/prompts/reviewer_codex.md`
- Modify: `research-stage3-tier/agents/prompts/orchestrator_synthesize.md`

**Worker changes:**
- Remove two-stage pipeline references
- Single model: "Train multi-class XGBoost (`objective='multi:softprob'`)"
- Remove "ClassifierConfig is FROZEN" constraint
- File constraints: NEVER modify `ml/evaluate.py` or `registry/gates.json` (same)
- Allowed to modify all `ml/*.py` except evaluate.py

**Reviewer changes (both):**
- Replace regression dimensions (C-RMSE, C-MAE, Spearman analysis) with:
  - Tier confusion matrix analysis (adjacent vs distant errors)
  - Per-tier recall analysis (are rare tiers 0/1 being caught?)
  - Class imbalance handling assessment
  - QWK ordinal consistency
- Keep: code quality, hypothesis validation, gate analysis (3 layers), statistical rigor

**Orchestrator synthesize changes:**
- Same promotion logic (3-layer gate check)
- Updated metric names in analysis
- Add tier-distribution stability check across months

**Commit.**

---

### Task 11: Write business_context.md and mem.md

**Files:**
- Modify: `research-stage3-tier/human-input/business_context.md`
- Modify: `research-stage3-tier/human-input/mem.md`

**business_context.md** — adapted from stage 2:
- Goal: predict tier (0-4) for each constraint
- Tier-VC@100 is still "the money metric" — ranked by tier_ev_score
- Catastrophic tier errors: predicting tier 3 for a tier 0 constraint directly costs money
- Adjacent tier errors (off by 1) are tolerable; distant errors (off by 2+) are not
- Feature set: same 34 candidates as stage 2

**mem.md** — initial notes:
- Ported from stage-2 regression pipeline
- Tier bins: [-inf, 0, 100, 1000, 3000, inf], labels [4,3,2,1,0]
- Same eval months, same 6+2 lookback, same data loader
- 2-hypothesis screening protocol (inherited from stage 2)
- All TierConfig params mutable

**Commit.**

---

### Task 12: Run v0 baseline and calibrate gates

**Step 1: Run v0 baseline benchmark**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak
```

**Step 2: Validate**

```bash
python ml/compare.py
```

**Step 3: Calibrate gates from v0**

Run `populate_v0_gates.py` (adapt from stage 2):
- `floor = 0.95 * v0_mean`
- `tail_floor = 0.90 * v0_worst_month`

**Step 4: Commit calibrated gates**

```bash
git add registry/
git commit -m "calibrate: set gates from v0 baseline"
```

---

### Task 13: Launch autonomous batch

**Step 1: Write human prompt in mem.md**

Add iteration focus: "Iterate on feature selection, class weights, and hyperparameters. Do NOT change bins in first batch."

**Step 2: Launch pipeline**

```bash
cd research-stage3-tier/agents
bash run_pipeline.sh
```

This runs 3 autonomous iterations with 2-hypothesis screening each.
