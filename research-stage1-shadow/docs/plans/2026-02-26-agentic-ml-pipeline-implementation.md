# Agentic ML Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full agentic ML research pipeline for shadow price classification — all greenfield files from the v10 design doc.

**Architecture:** File-based state machine with 4 agent roles (orchestrator, worker, Claude reviewer, Codex reviewer) coordinated by shell scripts. ML code ported from source repo (Stage 1 classifier only). 3-iteration autonomous loop with human sync.

**Tech Stack:** Python (polars, XGBoost, scikit-learn), Bash (tmux, flock, jq), Claude CLI, Codex CLI

**Design doc:** `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v10)
**Verification plan:** `docs/plans/2026-02-26-verification-plan.md`
**Source repo:** `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/`

---

## Phase A: Foundation (Tasks 1–4)

### Task 1: Create CLAUDE.md (agent sandbox rules)

**Files:**
- Create: `CLAUDE.md`

**Step 1: Write CLAUDE.md**

Content must include all sections below (design doc §5.2, §9, §15):

```markdown
# CLAUDE.md — research-stage1-shadow (Agent Rules)

Inherits: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## For ALL agents (interactive session, orchestrator, worker, reviewers)

### Sandbox Constraints
- Only operate inside this repository directory
- NEVER run `rm -rf`
- NEVER delete any `registry/v*/` directory
- NEVER modify `registry/v0/` (baseline is immutable)
- NEVER suppress exit codes from correctness-critical subprocesses (cleanup/idempotent operations may use `|| true` but must log intent)
- All launch scripts support `--dry-run` — this flag must be preserved if modifying launchers

### HUMAN-WRITE-ONLY Files (agents must NEVER modify)
- `registry/gates.json` — promotion gate definitions (created during bootstrap; immutable to runtime agents thereafter)
- `ml/evaluate.py` — standardized evaluation harness (created during bootstrap; immutable to runtime agents thereafter)

Note: The implementation session creates these files during bootstrap (Tasks 2 and 10). The NEVER-modify constraint applies to pipeline runtime agents only (orchestrator, worker, reviewers).

### Memory Safety (from parent CLAUDE.md)
- Use polars over pandas
- Use `pl.scan_parquet().filter().collect()` (lazy scan)
- Print `mem_mb()` at each pipeline stage (including training)
- Free intermediates: `del df; gc.collect()`
- `ray.shutdown()` after data loading completes
- Gzip model files on write: `gzip registry/${VERSION_ID}/model/*.ubj`

### Virtual Environment
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

### Ray Init (required for real data loading, NOT needed in SMOKE_TEST mode)
```python
from pbase.config.ray import init_ray
import pmodel
import ml as shadow_ml
init_ray(address='ray://10.8.0.36:10001', extra_modules=[pmodel, shadow_ml])
```

### Per-Agent Context Slices (what memory each agent MUST read)
| Agent | Required reads |
|-------|---------------|
| Orchestrator (plan) | memory/hot/ (all) + memory/warm/ (all) + memory/archive/index.md + registry/gates.json + champion metrics (if champion is null, read registry/v0/metrics.json instead) |
| Worker | memory/direction_iter{N}.md + memory/hot/champion.md + memory/hot/learning.md + memory/hot/runbook.md |
| Claude Reviewer | direction + changes_summary + comparison table + warm/experiment_log + hot/gate_calibration + warm/decision_log + gates.json + ml/ codebase |
| Codex Reviewer | Same as Claude reviewer — does NOT see Claude's review |
| Orchestrator (synth) | Both raw reviews (read independently) + comparison table + warm/ (all) |

### Artifact Naming
- All handoff/review files include `{batch_id}` and `iter{N}` to prevent stale artifact reads across batches
- Handoff JSON `artifact_path` must use relative paths (not absolute)

## Worker-Specific Rules
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and `${PROJECT_DIR}/handoff/{batch_id}/iter{N}/worker_done.json` (absolute path — handoff/ is gitignored and does not exist in the worktree)
- NEVER touch other `registry/v*/` directories
- ALWAYS commit changes before writing handoff JSON
- If tests fail 3x: write failed handoff with error summary, do NOT commit
- Read VERSION_ID from the PROJECT_DIR copy of state.json (NOT the worktree copy, which is stale): `VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")`

## Reviewer-Specific Rules
- Only write to `reviews/` and `handoff/` directories
- Do NOT read the other reviewer's output — independence is mandatory
- You MAY critique gates as stale or miscalibrated
- Gate changes require human approval at HUMAN_SYNC

## Orchestrator-Specific Rules
- Do NOT modify any ML code or registry/ files
- Do NOT run training
```

**Step 2: Verify**

Run: `cat CLAUDE.md | head -5`
Expected: header visible

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "foundation: add CLAUDE.md agent sandbox rules"
```

---

### Task 2: Create directory structure and initial JSON/state files

**Files:**
- Create: `.gitignore`
- Create: `registry/gates.json`
- Create: `registry/version_counter.json`
- Create: `registry/champion.json`
- Create: `registry/comparisons/.gitkeep`
- Create: `state.json`
- Create: `reports/.gitkeep`
- Create: `reviews/.gitkeep`
- Create: `handoff/.gitkeep`

**Step 0: Write .gitignore**

```gitignore
state.json
state.lock
.logs/
!.logs/sessions/.gitkeep
handoff/
!handoff/.gitkeep
.claude/worktrees/
ml/**/*.parquet
```

**Step 1: Write registry/gates.json**

Exact content from design doc §7.2 (lines 664–682):
```json
{
  "version": 1,
  "effective_since": "2026-02-26",
  "noise_tolerance": 0.02,
  "gates": {
    "S1-AUC":       {"floor": 0.65,  "direction": "higher", "pending_v0": false},
    "S1-AP":        {"floor": 0.12,  "direction": "higher", "pending_v0": false},
    "S1-VCAP@100":  {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05},
    "S1-VCAP@500":  {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05},
    "S1-VCAP@1000": {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05},
    "S1-NDCG":      {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05},
    "S1-BRIER":     {"floor": null,  "direction": "lower",  "pending_v0": true,  "v0_offset": 0.02},
    "S1-REC":       {"floor": 0.40,  "direction": "higher", "pending_v0": false, "group": "B"},
    "S1-CAP@100":   {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05, "group": "B"},
    "S1-CAP@500":   {"floor": null,  "direction": "higher", "pending_v0": true,  "v0_offset": 0.05, "group": "B"}
  }
}
```

**Step 2: Write registry/version_counter.json**
```json
{"next_id": 1}
```

**Step 3: Write registry/champion.json**
```json
{"version": null, "promoted_at": null}
```

**Step 4: Write state.json** (runtime-only — gitignored, NOT committed; created during bootstrap but never tracked)
```json
{
  "state": "IDLE",
  "batch_id": null,
  "iteration": 0,
  "version_id": null,
  "entered_at": null,
  "max_seconds": null,
  "orchestrator_tmux": null,
  "worker_tmux": null,
  "claude_reviewer_tmux": null,
  "codex_reviewer_tmux": null,
  "history": [],
  "human_input": null,
  "error": null
}
```

**Step 5: Verify**

Run: `jq '.gates | keys | length' registry/gates.json`
Expected: `10`

Run: `jq '.state' state.json`
Expected: `"IDLE"`

**Step 6: Commit**

```bash
git add .gitignore registry/ reports/ reviews/ handoff/
git commit -m "foundation: add .gitignore, registry JSON files, and directory scaffolding"
```

---

### Task 3: Create agents/config.sh, memory stubs, and .logs

**Files:**
- Create: `agents/config.sh`
- Create: `memory/hot/progress.md`, `memory/hot/champion.md`, `memory/hot/critique_summary.md`, `memory/hot/gate_calibration.md`, `memory/hot/learning.md`, `memory/hot/runbook.md`
- Create: `memory/warm/experiment_log.md`, `memory/warm/hypothesis_log.md`, `memory/warm/decision_log.md`
- Create: `memory/archive/index.md`
- Create: `memory/human_input.md` (stub — orchestrator reads this on iter 1 if it exists)
- Create: `.logs/audit.jsonl` (empty — watchdog appends here)
- Create: `.logs/sessions/.gitkeep`

**Note:** `human-input/` directory (mem.md, requirement.md, reference.md) already exists from the design phase — do NOT recreate. These are read-only user files.

**Step 1: Write agents/config.sh**

From design doc §12.1:
```bash
#!/usr/bin/env bash
# All environment-specific settings. Source this at top of every script.
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage1-shadow"
RAY_ADDRESS="ray://10.8.0.36:10001"
DATA_ROOT="/opt/temp/tmp/pw_data/spice6"
VENV_ACTIVATE="/home/xyz/workspace/pmodel/.venv/bin/activate"
SMOKE_TEST="${SMOKE_TEST:-false}"   # Uses bash default substitution so env override works (intentional deviation from §12.1 which hardcodes false)
REGISTRY_DISK_LIMIT_MB=10240
CODEX_MODEL="gpt-5.3-codex"
STATE_FILE="${PROJECT_DIR}/state.json"
```

**Step 2: Write memory stubs**

Each file gets a minimal markdown stub (e.g., `# Progress\n\nNo batch running.`).

`memory/hot/runbook.md` must contain the full static safety rules (not just a pointer to CLAUDE.md, since workers in worktrees may not inherit it):
- Sandbox constraints verbatim from design §5.2 (only modify ml/, registry/${VERSION_ID}/, handoff/)
- NEVER touch registry/v0/, gates.json, evaluate.py, other registry/v*/
- NEVER run rm -rf or delete registry directories
- Commit before writing handoff JSON
- On 3x test failure: write failed handoff, do NOT commit
- Read VERSION_ID from state.json, not from env vars

**Step 3: Verify**

Run: `source agents/config.sh && echo $PROJECT_DIR`
Expected: `/home/xyz/workspace/research-qianli-v2/research-stage1-shadow`

Run: `find memory -name "*.md" | wc -l`
Expected: `11` (10 tiered + human_input.md)

**Step 4: Commit**

```bash
git add agents/config.sh memory/ .logs/
git commit -m "foundation: add config.sh, memory stubs, .logs directory"
```

---

### Task 4: Create ml/__init__.py

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/tests/__init__.py`

**Step 1: Write files**

`ml/__init__.py`:
```python
"""Shadow price classification ML pipeline."""
```

`ml/tests/__init__.py`: empty file

**Step 2: Commit** (batched with Task 5)

---

## Phase B: ML Core (Tasks 5–9)

### Task 5: Create ml/config.py + tests

**Files:**
- Create: `ml/config.py`
- Create: `ml/tests/conftest.py`
- Create: `ml/tests/test_config.py`

**Step 1: Write failing test (ml/tests/test_config.py)**

```python
from ml.config import FeatureConfig, HyperparamConfig, PipelineConfig

def test_feature_config_has_14_features():
    fc = FeatureConfig()
    assert len(fc.features) == 14

def test_monotone_constraints_length():
    fc = FeatureConfig()
    mc = fc.get_monotone_constraints_str()
    # String format: "(1,1,1,...)" — count commas + 1 == 14
    values = mc.strip("()").split(",")
    assert len(values) == 14

def test_hyperparam_defaults():
    hc = HyperparamConfig()
    assert hc.n_estimators == 200
    assert hc.max_depth == 4
    assert hc.learning_rate == 0.1

def test_pipeline_config_defaults():
    pc = PipelineConfig()
    assert pc.threshold_beta == 0.7
    assert pc.train_months == 10
    assert pc.val_months == 2
```

**Step 2: Run to verify it fails**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow && python -m pytest ml/tests/test_config.py -v`
Expected: FAIL (ImportError)

**Step 3: Write ml/config.py**

Port from source `config.py`. Key elements:
- `FeatureConfig` dataclass: 14 step1 features as `list[tuple[str, int]]` — (name, monotone_constraint)
  - 5 exceedance probs (+1): prob_exceed_110, 105, 100, 95, 90
  - 3 below-threshold (-1): prob_below_100, 95, 90
  - 1 severity (+1): expected_overload
  - 3 distribution (0): density_skewness, density_kurtosis, density_cv
  - 2 historical DA (+1): hist_da, hist_da_trend
- `get_monotone_constraints_str()` -> `"(1,1,1,1,1,-1,-1,-1,1,0,0,0,1,1)"`
- `HyperparamConfig` dataclass: XGBoost defaults from source (n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10, random_state=42)
- `PipelineConfig` dataclass: auction_month, class_type, period_type, version_id, train_months=10, val_months=2, threshold_beta=0.7, threshold_scaling_factor=1.0, scale_pos_weight_auto=True, registry_dir="registry"
- `GateConfig` class: loads from `registry/gates.json`, method to check if all floors populated

Write `ml/tests/conftest.py`:
- `synthetic_features()` fixture -> numpy array (100, 14) random floats
- `synthetic_labels()` fixture -> numpy array 100 with ~7% positive rate
- `mock_pipeline_config()` fixture
- `tmp_registry(tmp_path)` fixture -> creates registry/v0/metrics.json stub

All fixtures must use `np.random.RandomState(42)` for deterministic tests.

**Step 4: Run tests to verify pass**

Run: `python -m pytest ml/tests/test_config.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add ml/
git commit -m "ml: add config.py (FeatureConfig, HyperparamConfig, PipelineConfig) + tests"
```

---

### Task 6: Create ml/data_loader.py

**Files:**
- Create: `ml/data_loader.py`

**Step 1: Write ml/data_loader.py**

Two-mode loader. SMOKE_TEST detected via `os.environ.get("SMOKE_TEST", "false").lower() == "true"` (env var, NOT config.sh import — Python reads env directly).

Key functions:
- `load_data(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]`
- SMOKE_TEST branch: returns synthetic polars DataFrames (100 rows, fixed seed `rng = np.random.RandomState(42)` for reproducibility). No Ray needed. Required columns: 14 feature columns (from FeatureConfig) + `actual_shadow_price` (continuous: `np.where(binding, rng.lognormal(3, 1.5), 0.0)` where binding ~ 7% positive rate) + `constraint_id` (str) + `auction_month` (str). Note: `compute_binary_labels()` uses `actual_shadow_price > threshold` — the DataFrame must include this column, not a pre-computed binary label.
- Real branch: polars lazy scan, Ray init with `extra_modules=[pmodel, ml]` (local ml package must be serialized to cluster), pbase data access, `mem_mb()` at each stage, `ray.shutdown()` after load

Reference: source `data_loader.py` for real data path structure, but simplify for Stage 1 only.

**Step 2: Verify smoke mode works**

Run: `SMOKE_TEST=true python -c "from ml.data_loader import load_data; from ml.config import PipelineConfig; t, v = load_data(PipelineConfig()); print(t.shape, v.shape)"`
Expected: shapes printed, no errors

**Step 3: Commit** (batched with Task 7)

---

### Task 7: Create ml/features.py

**Files:**
- Create: `ml/features.py`

**Step 1: Write ml/features.py**

Functions:
- `prepare_features(df: pl.DataFrame, config: FeatureConfig) -> tuple[np.ndarray, list[str]]` — selects 14 feature columns, fills missing with 0, returns numpy array + column names
- `compute_binary_labels(df: pl.DataFrame, threshold: float = 0.0) -> np.ndarray` — binary from actual_shadow_price > threshold
- `compute_scale_pos_weight(labels: np.ndarray) -> float` — n_neg / n_pos

Must use polars only. Print `mem_mb()`.

**Step 2: Verify**

Run: `SMOKE_TEST=true python -c "from ml.data_loader import load_data; from ml.features import prepare_features; from ml.config import PipelineConfig, FeatureConfig; t, v = load_data(PipelineConfig()); X, cols = prepare_features(t, FeatureConfig()); print(X.shape, len(cols))"`
Expected: `(100, 14) 14` (or similar)

**Step 2a: Write ml/tests/test_data_loader.py and ml/tests/test_features.py**
- test_data_loader: SMOKE_TEST returns correct shape (100 rows), correct 14 feature columns, correct dtypes, no NaNs
- test_features: prepare_features output shape, compute_binary_labels output range {0,1}, compute_scale_pos_weight correctness on known input

**Step 3: Commit**

```bash
git add ml/data_loader.py ml/features.py ml/tests/test_data_loader.py ml/tests/test_features.py
git commit -m "ml: add data_loader.py (SMOKE_TEST mode) and features.py + tests"
```

---

### Task 8: Create ml/train.py + tests

**Files:**
- Create: `ml/train.py`
- Create: `ml/tests/test_train.py`

**Step 1: Write failing test**

```python
import numpy as np
from ml.train import train_classifier, predict_proba
from ml.config import HyperparamConfig, FeatureConfig

def test_train_returns_xgb_classifier(synthetic_features, synthetic_labels):
    from xgboost import XGBClassifier
    model = train_classifier(synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig())
    assert isinstance(model, XGBClassifier)

def test_predict_proba_shape(synthetic_features, synthetic_labels):
    model = train_classifier(synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig())
    proba = predict_proba(model, synthetic_features)
    assert proba.shape == (synthetic_features.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))
```

**Step 2: Run to verify fail**

Run: `python -m pytest ml/tests/test_train.py -v`
Expected: FAIL

**Step 3: Write ml/train.py**

Functions:
- `train_classifier(X_train, y_train, config: HyperparamConfig, feature_config: FeatureConfig) -> XGBClassifier`
  - Sets `monotone_constraints` from `feature_config.get_monotone_constraints_str()`
  - Sets `scale_pos_weight` from `compute_scale_pos_weight(y_train)`
  - Prints `mem_mb()` before and after `model.fit()` (training is memory-intensive)
  - Fits and returns model
- `predict_proba(model, X) -> np.ndarray` — returns P(binding) column

**Step 4: Run tests, verify pass**

Run: `python -m pytest ml/tests/test_train.py -v`
Expected: all PASS

**Step 5: Commit** (batched with Task 9)

---

### Task 9: Create ml/threshold.py

**Files:**
- Create: `ml/threshold.py`

**Step 1: Write ml/threshold.py**

Port `find_optimal_threshold()` from source `models.py`:
- `find_optimal_threshold(y_true, y_proba, beta=0.7, scaling_factor=1.0) -> tuple[float, float]`
  - Uses `sklearn.metrics.precision_recall_curve`
  - Computes F-beta score at each threshold
  - `np.nan_to_num(fbeta, nan=0.0)` before argmax
  - Returns (optimal_threshold, max_fbeta)
- `apply_threshold(y_proba, threshold) -> np.ndarray` — binary predictions

**Step 2: Add tests to test_train.py**

```python
from ml.threshold import find_optimal_threshold, apply_threshold

def test_threshold_in_valid_range(synthetic_features, synthetic_labels):
    model = train_classifier(synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig())
    proba = predict_proba(model, synthetic_features)
    threshold, fbeta = find_optimal_threshold(synthetic_labels, proba)
    assert 0 < threshold < 1
    assert 0 <= fbeta <= 1

def test_apply_threshold():
    proba = np.array([0.1, 0.5, 0.9])
    result = apply_threshold(proba, 0.5)
    assert np.array_equal(result, np.array([0, 0, 1]))  # 0.5 exactly is below threshold
```

**Step 3: Run full Phase B tests**

Run: `python -m pytest ml/tests/ -v`
Expected: all PASS

**Step 4: Commit**

```bash
git add ml/train.py ml/threshold.py ml/tests/test_train.py
git commit -m "ml: add train.py (XGBoost + monotone constraints) and threshold.py (F-beta)"
```

---

## Phase C: ML Eval + Pipeline + Registry (Tasks 10–15)

### Task 10: Create ml/evaluate.py + tests

**Files:**
- Create: `ml/evaluate.py`
- Create: `ml/tests/test_evaluate.py`

**Step 1: Write failing test**

```python
from ml.evaluate import evaluate_classifier

def test_evaluate_returns_all_gate_metrics(synthetic_features, synthetic_labels):
    # Train a quick model and get predictions
    from ml.train import train_classifier, predict_proba
    from ml.threshold import find_optimal_threshold, apply_threshold
    from ml.config import HyperparamConfig, FeatureConfig
    model = train_classifier(synthetic_features, synthetic_labels, HyperparamConfig(), FeatureConfig())
    proba = predict_proba(model, synthetic_features)
    threshold, _ = find_optimal_threshold(synthetic_labels, proba)
    y_pred = apply_threshold(proba, threshold)
    rng = np.random.RandomState(42)
    fake_sp = np.where(synthetic_labels == 1,
                       rng.lognormal(mean=3, sigma=1.5, size=synthetic_labels.shape),
                       0.0)  # continuous shadow prices — avoids degenerate ranking metrics
    metrics = evaluate_classifier(synthetic_labels, proba, y_pred, fake_sp, threshold)
    required_keys = ["S1-AUC", "S1-AP", "S1-VCAP@100", "S1-VCAP@500", "S1-VCAP@1000",
                     "S1-NDCG", "S1-BRIER", "S1-REC", "S1-CAP@100", "S1-CAP@500"]
    for key in required_keys:
        assert key in metrics, f"Missing gate metric: {key}"
```

**Step 2: Run to verify fail, then implement**

`evaluate_classifier()` computes all 10 gate metrics + monitoring metrics.
Port from source `evaluation.py`: AUC-ROC, Average Precision, Brier Score, Precision, Recall, F1, Value Capture@K, NDCG, Capture@K.

**HUMAN-WRITE-ONLY**: Add comment at top: `# HUMAN-WRITE-ONLY — agents must NEVER modify this file`

**Step 3: Run tests, verify pass**

Run: `python -m pytest ml/tests/test_evaluate.py -v`

**Step 4: Commit**

```bash
git add ml/evaluate.py ml/tests/test_evaluate.py
git commit -m "ml: add evaluate.py (HUMAN-WRITE-ONLY standardized eval harness) + tests"
```

---

### Task 11: Create ml/compare.py

**Files:**
- Create: `ml/compare.py`

**Step 1: Write ml/compare.py**

From design doc §5.3. Functions:
- `load_all_versions(registry_dir)` — reads all `registry/v*/metrics.json`
- `check_gates(metrics, gates, champion_metrics, noise_tolerance)` — per-gate pass/fail (§7.4)
- `build_comparison_table(versions, gates, champion_metrics, noise_tolerance)` — Markdown table
- `run_comparison(batch_id, iteration, registry_dir, gates_path, champion_path, output_path)` — loads all data, builds table, writes `reports/` markdown + `registry/comparisons/` JSON. Prints `mem_mb()`.

CLI: `python ml/compare.py --batch-id X --iteration N --output path`
(reads `registry/gates.json` and `registry/champion.json` from default paths, or accepts `--gates-path` / `--champion-path`)

**Step 1a: Write ml/tests/test_compare.py**
- test higher-direction gate pass/fail
- test lower-direction gate (Brier) pass/fail
- test noise_tolerance edge case (delta exactly at boundary)
- test missing v0 metrics handling
- test Markdown output has no broken `|` in cells

**Step 2: Commit** (batched with Task 12)

---

### Task 12: Create ml/registry.py + tests

**Files:**
- Create: `ml/registry.py`
- Create: `ml/tests/test_registry.py`

**Step 1: Write failing test**

```python
from ml.registry import allocate_version_id, register_version, get_champion

def test_allocate_increments(tmp_path):
    counter = tmp_path / "version_counter.json"
    counter.write_text('{"next_id": 1}')
    v1 = allocate_version_id(counter)
    v2 = allocate_version_id(counter)
    assert v1 == "v0001"
    assert v2 == "v0002"

def test_register_creates_directory(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    register_version(reg, "v0001", {"param": 1}, {"S1-AUC": 0.7}, {"created": "now"})
    assert (reg / "v0001" / "metrics.json").exists()
```

**Step 2: Implement ml/registry.py**

- `allocate_version_id(counter_path)` — flock-based atomic increment
- `register_version(registry_dir, version_id, config, metrics, meta, model_path=None)` — `exist_ok=False`
  - Creates `registry/{version_id}/` with: `config.json`, `metrics.json`, `meta.json`
  - If model_path: copies model files to `registry/{version_id}/model/`, gzips `.ubj` files (§12.5)
  - `changes_summary.md` is written by the worker, not by registry.py
- `promote_version(registry_dir, version_id, champion_path)`
- `get_champion(champion_path) -> str | None`

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git add ml/compare.py ml/registry.py ml/tests/test_registry.py ml/tests/test_compare.py
git commit -m "ml: add compare.py and registry.py (version management) + tests"
```

---

### Task 13: Create ml/pipeline.py + tests

**Files:**
- Create: `ml/pipeline.py`
- Create: `ml/tests/test_pipeline.py`

**Step 1: Write failing test**

```python
import pytest

@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")

def test_pipeline_smoke_creates_registry(tmp_path):
    from ml.pipeline import run_pipeline
    from ml.config import PipelineConfig
    config = PipelineConfig(version_id="v0001", registry_dir=str(tmp_path / "registry"))
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "version_counter.json").write_text('{"next_id": 1}')
    metrics = run_pipeline(config)
    assert (tmp_path / "registry" / "v0001" / "metrics.json").exists()
    assert "S1-AUC" in metrics
```

**Step 2: Implement ml/pipeline.py**

End-to-end: load → features → train → threshold → evaluate → register → save model. Note: `pipeline.py` does NOT call `compare.py` — comparison runs separately as a controller step (step 8 in run_single_iter.sh).
- `run_pipeline(config: PipelineConfig) -> dict` with phases
- `--from-phase N` for crash recovery
- `--overrides` JSON string for config overrides (used by worker per §10.2). Accepts a flat JSON dict. Keys are matched against HyperparamConfig fields first, then PipelineConfig fields (threshold_beta, scale_pos_weight_auto). Unknown keys cause a ValueError. The resolved config (base + overrides) is written to `registry/${VERSION_ID}/config.json` for reproducibility.
- Saves intermediates to parquet between phases
- `mem_mb()` at each stage
- CLI: `python ml/pipeline.py --version-id v0 --auction-month 2021-07 --class-type onpeak --period-type f0 [--overrides '{"n_estimators": 300}']`

**Step 3: Run test**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_pipeline.py -v`

**Step 4: Commit**

```bash
git add ml/pipeline.py ml/tests/test_pipeline.py
git commit -m "ml: add pipeline.py (end-to-end load->train->eval->register) + tests"
```

---

### Task 14: Create ml/populate_v0_gates.py

**Files:**
- Create: `ml/populate_v0_gates.py`

**Step 1: Implement**

From design doc §7.3:
1. Load `registry/v0/metrics.json` — abort if missing
2. Load `registry/gates.json`
3. For each gate with `pending_v0: true`:
   - higher direction: floor = v0_metric − v0_offset
   - lower direction (BRIER): floor = v0_metric + v0_offset
4. Set `pending_v0: false`, write back
5. Idempotent (no-op if no pending entries)

**Step 2: Add tests to test_registry.py**

```python
def test_populate_v0_gates_brier_direction(tmp_path):
    """BRIER is lower-direction: floor = v0 + offset, NOT v0 - offset."""
    # Setup v0 metrics with S1-BRIER = 0.089
    # Setup gates.json with S1-BRIER pending_v0=true, v0_offset=0.02
    # Run populate
    # Assert floor == 0.109 (0.089 + 0.02), NOT 0.069
```

**Step 3: Run tests**

Run: `python -m pytest ml/tests/test_registry.py -v`

**Step 4: Commit**

```bash
git add ml/populate_v0_gates.py ml/tests/test_registry.py
git commit -m "ml: add populate_v0_gates.py (auto-populate v0-relative gate floors)"
```

---

### Task 15: Phase C checkpoint — full ML test pass

Run: `python -m pytest ml/tests/ -v`
Expected: all pass

---

## Phase D: Shell Infrastructure (Tasks 16–22)

### Task 16: Create agents/state_utils.sh

**Files:**
- Create: `agents/state_utils.sh`

**Step 1: Implement from design doc §8 (lines 763–804)**

Functions:
- `cas_transition(expected_state, new_state, new_fields_json)` — CAS on state.json. Temp file MUST be in the same directory as STATE_FILE (use `${STATE_FILE}.tmp.$$`, NOT `mktemp`) to guarantee atomic `mv` on the same filesystem.
- `get_expected_artifact(state)` — 6-state case statement (ORCHESTRATOR_PLANNING, WORKER_RUNNING, REVIEW_CLAUDE, REVIEW_CODEX, ORCHESTRATOR_SYNTHESIZING, HUMAN_SYNC) reading from STATE_FILE dynamically
- `verify_handoff(handoff_file, state)` — failed bypasses path/sha; success requires match
- `poll_for_handoff(handoff_dir, filename, timeout_s, interval_s)` — poll loop. Must check for BOTH the normal handoff file AND timeout artifact (timeout_${state}.json). Return 0 for normal handoff, return 1 for timeout.
- `STATE_TO_HANDOFF` associative array (6 entries: ORCHESTRATOR_PLANNING, WORKER_RUNNING, REVIEW_CLAUDE, REVIEW_CODEX, ORCHESTRATOR_SYNTHESIZING, HUMAN_SYNC)
- Self-test mode: `bash agents/state_utils.sh test` runs verification plan tests 3.1–3.3

**Step 2: Run self-tests**

Run: `bash agents/state_utils.sh test`
Expected: all pass (6 verify_handoff cases, 6 get_expected_artifact cases, 6 STATE_TO_HANDOFF checks)

**Step 3: Commit**

```bash
git add agents/state_utils.sh
git commit -m "agents: add state_utils.sh (CAS, verify_handoff, get_expected_artifact) + self-tests"
```

---

### Task 17: Create agents/run_single_iter.sh

**Files:**
- Create: `agents/run_single_iter.sh`

**Step 1: Implement all 18 steps from design doc §6.2**

Key requirements:
- PIPELINE_LOCKED guard at top
- `set -euo pipefail` at top, BUT wrap each `poll_for_handoff` call with explicit error handling (HP-1 fix):
  ```bash
  set +e
  poll_for_handoff "${HANDOFF_DIR}" "orchestrator_plan_done.json" "${MAX_SECONDS}" 30
  POLL_RC=$?
  set -e
  if (( POLL_RC != 0 )); then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    cas_transition CURRENT_STATE IDLE '{"error":"agent timeout/crash at step N"}'
    echo "FATAL: poll timeout at iter ${N} step N" >&2
    exit 1
  fi
  ```
  Apply this pattern to ALL poll calls: steps 3/4 (orchestrator plan), 10/11 (Claude reviewer), 13/14 (Codex reviewer — already has timeout handling but should still reset on failure), 16/17 (synthesis orchestrator). On poll failure: kill the tmux session, reset state to IDLE with error field, exit. This ensures the pipeline never wedges on agent crashes.
- Step 0: export BATCH_ID N VERSION_ID PROJECT_DIR + export WORKER_FAILED=0
- Step 2: mkdir -p handoff dir
- Step 5: flock-based version allocation
- Step 5a: **commit orchestrator outputs before creating worktree** (CB-1 fix):
  ```bash
  git -C "${PROJECT_DIR}" add memory/
  if ! git -C "${PROJECT_DIR}" diff --cached --quiet; then
    git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator plan"
  fi
  ```
  Guard against "nothing to commit" (HP-2 fix) — if orchestrator wrote nothing, skip commit gracefully. Without this step, the worktree (created from HEAD) won't have the direction file or updated memory.
- Step 6: worker launches in worktree. **Worker handoff must use absolute path** (CB-2 fix):
  The worker prompt must write `${PROJECT_DIR}/handoff/{batch_id}/iter{N}/worker_done.json` (absolute), not the relative path, because `handoff/` is gitignored and doesn't exist in the worktree. Worker prompt must clearly distinguish:
  - **Relative paths** (worktree, committed via git): `ml/`, `registry/${VERSION_ID}/`
  - **Absolute paths** (PROJECT_DIR, shared state): `${PROJECT_DIR}/handoff/...`
- Step 7a: WORKER_FAILED=1 on failure, jump to step 15 (skip 7b-7d merge, step 8 compare, steps 9-14 reviews)
- Step 7b–7d: verify commit, sha256, pre-merge guard (verify `ml/evaluate.py` and `registry/gates.json` unchanged on worker branch), merge
- Steps 8–14: compare, then REVIEW_CLAUDE + REVIEW_CODEX as separate states (design doc §6.2)
- Steps 15–17: synthesis (controller calls `promote_version()` if synthesis recommends promotion)
- Step 17a: **commit synthesis outputs before next iteration's worktree** (O-3 fix):
  ```bash
  git -C "${PROJECT_DIR}" add memory/
  if ! git -C "${PROJECT_DIR}" diff --cached --quiet; then
    git -C "${PROJECT_DIR}" commit -m "iter${N}: orchestrator synthesis"
  fi
  ```
  Same HP-2 guard. Same root cause as CB-1: synthesis writes to memory/hot/ and memory/warm/ (and memory/direction_iter{N+1}.md for N<3). These must be committed before the next iteration creates a worktree.
- Step 18: state transition — branch on iteration count:

```bash
# Step 18: state transition (branch on iteration count)
if (( N == 3 )); then
  cas_transition ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC '{}'
else
  cas_transition ORCHESTRATOR_SYNTHESIZING IDLE '{}'
fi
```
Note: Design doc §6.2 line 582 lists IDLE as one option; §13.2 line 1306 expects ORCHESTRATOR_PLANNING. We use IDLE because step 1 always expects IDLE — the brief IDLE window between iterations is invisible (flock held by run_pipeline.sh, watchdog exits immediately for IDLE). The VP Tier 2 assertion should be updated to expect IDLE.

Commented code skeleton showing all 18 steps (matches design doc §6.2 numbering exactly):
```bash
# Step 0:  export env vars (BATCH_ID, N, VERSION_ID, PROJECT_DIR, WORKER_FAILED=0)
# Step 1:  cas_transition IDLE → ORCHESTRATOR_PLANNING
# Step 2:  mkdir -p handoff/${BATCH_ID}/iter${N}
# Step 3:  launch orchestrator --phase plan, poll for handoff
# Step 4:  cas_transition ORCHESTRATOR_PLANNING → WORKER_RUNNING
# Step 5:  flock version allocation (allocate_version_id)
# Step 5a: git add memory/ && git commit (so worktree inherits orchestrator outputs)
# Step 6:  launch worker (worktree), poll for handoff (worker writes to ${PROJECT_DIR}/handoff/)
# Step 7a: check worker handoff — if failed, WORKER_FAILED=1, jump to step 15
#          (skip 7b-7d merge, step 8 compare, steps 9-14 reviews)
# Step 7b: verify worker committed (git log)
# Step 7c: sha256 verify artifacts
# Step 7d: verify HUMAN-WRITE-ONLY files unchanged, then merge worker worktree
# Step 8:  run compare.py: `python ml/compare.py --batch-id ${BATCH_ID} --iteration ${N} --output reports/${BATCH_ID}/iter${N}/comparison.md`
#          Reviewers read from: reports/${BATCH_ID}/iter${N}/comparison.md
# Step 9:  cas_transition WORKER_RUNNING → REVIEW_CLAUDE, max_s=1200
# Step 10: launch Claude reviewer, poll for handoff
# Step 11: verify Claude review sha256
# Step 12: cas_transition REVIEW_CLAUDE → REVIEW_CODEX, max_s=1200
# Step 13: launch Codex reviewer, poll for handoff (or timeout)
# Step 14: (poll complete or timeout detected)
# Step 15: cas_transition REVIEW_CODEX → ORCHESTRATOR_SYNTHESIZING, max_s=600
#          (on WORKER_FAILED path: cas_transition WORKER_RUNNING → ORCHESTRATOR_SYNTHESIZING)
# Step 16: launch orchestrator --phase synthesize (inject WORKER_FAILED), poll for handoff
# Step 17: read synthesis handoff; if decisions.promote_version is non-null, controller calls promote_version()
# Step 17a: git add memory/ && git commit (so next iteration's worktree inherits synthesis outputs)
# Step 18: cas_transition → IDLE (N<3) or HUMAN_SYNC (N==3)
```

**Step 2: Commit** (batched with Task 18)

---

### Task 18: Create agents/run_pipeline.sh

**Files:**
- Create: `agents/run_pipeline.sh`

**Step 1: Implement from design doc §12.3 (lines 1191–1235)**

Fix v10 review finding #1 — proper `--batch-name` arg parsing:
```bash
BATCH_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-name)   BATCH_NAME="$2"; shift 2 ;;
    --batch-name=*) BATCH_NAME="${1#--batch-name=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
# Validate batch name (tmux session names cannot contain ., :, or whitespace)
if [[ -n "$BATCH_NAME" ]] && ! [[ "$BATCH_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "ERROR: --batch-name must be alphanumeric, hyphens, or underscores only (got: '$BATCH_NAME')" >&2
  exit 1
fi
# If user provided a name, use it as prefix; otherwise auto-generate
if [[ -n "$BATCH_NAME" ]]; then
  BATCH_ID="${BATCH_NAME}-$(date +%Y%m%d-%H%M%S)"
else
  BATCH_ID="batch-$(date +%Y%m%d-%H%M%S)"
fi
```

Key: flock, PIPELINE_LOCKED, v0 guard, populate_v0_gates.py, `export N` inside the for loop before calling run_single_iter.sh, 3-iteration loop.

**Runtime fix RT-1**: Add `--max-iter N` parameter (default 3) for partial runs (e.g. `--max-iter 1` for testing). Also `cd "$PROJECT_DIR"`, `export PYTHONPATH="${PROJECT_DIR}"`, and `source "$VENV_ACTIVATE"` BEFORE the iteration loop — `run_single_iter.sh` calls `python -c "from ml.registry import ..."` (step 5) and `python ml/compare.py` (step 8) which require PYTHONPATH and the venv active. Without this, the pipeline fails on version allocation.

**Runtime fix RT-2**: Launch scripts must set full environment inside tmux commands (tmux sessions don't inherit parent env): `cd`, `export PYTHONPATH`, `source venv`, `export PROJECT_DIR` (worker), `export SMOKE_TEST` (worker). Also track EXIT_CODE in log files.

**Runtime fix RT-3**: Add `rm -f "${PROJECT_DIR}/memory/direction_iter"*.md` before the loop (per design doc §12.3 line 1218) to clean up stale direction files from prior crashed batches.

**Runtime fix RT-4**: `verify_handoff` for the worker must run with cwd set to the worktree (`pushd "$WT_DIR"`), not PROJECT_DIR. The worker's `artifact_path` in the handoff JSON is relative (e.g. `registry/v0001/changes_summary.md`), and this file exists in the worktree — not the main tree until after merge. Without this fix, `sha256sum` fails (file not found), verify_handoff returns 1, and `set -e` kills the pipeline.

**Runtime fix RT-5**: Before `git merge`, run `git checkout -- . 2>/dev/null || true` to clean the working tree. Stale files from prior crashed/simulated runs (uncommitted changes to registry/v*) cause merge conflicts. Also: remove `2>/dev/null || true` suppression from merge — log merge errors and set WORKER_FAILED=1 instead of silently continuing.

**Runtime fix RT-6**: Validate all handoff JSON files with `jq empty` before parsing. If the orchestrator or worker writes malformed JSON, the pipeline should fail gracefully (reset state to IDLE with error) instead of crashing with a jq parse error under `set -e`.

**Runtime fix RT-7**: All implementation code (`ml/`, `agents/`, `registry/`, `CLAUDE.md`, `.gitignore`, etc.) MUST be committed to git before running the pipeline. Worker worktrees are created from HEAD — if files are untracked, they don't exist in the worktree, and merging the worker branch back fails with "untracked working tree files would be overwritten by merge." Also add `__pycache__/` and `*.pyc` to `.gitignore` to prevent pycache conflicts during merge.

Pre-loop HUMAN_SYNC reset (CB-4 fix): if the current state is HUMAN_SYNC (starting a new batch after a completed 3-iteration batch), reset to IDLE before entering the loop:
```bash
if [[ "$current_state" == "HUMAN_SYNC" ]]; then
  source agents/state_utils.sh
  cas_transition HUMAN_SYNC IDLE '{"batch_id":null,"iteration":0}'
fi
```

Post-loop cleanup — remove worktrees created during this batch:
```bash
# Cleanup worktrees from this batch
for i in 1 2 3; do
  WT="${PROJECT_DIR}/.claude/worktrees/iter${i}-${BATCH_ID}"
  [[ -d "$WT" ]] && git worktree remove "$WT" --force 2>/dev/null || true
done
```

**Step 2: Commit**

```bash
git add agents/run_single_iter.sh agents/run_pipeline.sh
git commit -m "agents: add run_single_iter.sh and run_pipeline.sh (master loop)"
```

---

### Task 19: Create agent launch scripts (4 files)

**Files:**
- Create: `agents/launch_orchestrator.sh` (§5.1)
- Create: `agents/launch_worker.sh` (§5.2)
- Create: `agents/launch_reviewer_claude.sh` (§5.4)
- Create: `agents/launch_reviewer_codex.sh` (§5.5)

All must support `--dry-run`. All use stdin redirect (never `$(cat ...)`) EXCEPT `launch_reviewer_codex.sh` which must use `$(cat ...)` because `codex exec` takes a positional prompt argument, not stdin. Codex uses `${CODEX_MODEL}` from config.

`launch_orchestrator.sh` must support WORKER_FAILED injection when `--phase synthesize`:
```bash
if [[ "$PHASE" == "synthesize" ]]; then
  { echo "WORKER_FAILED=${WORKER_FAILED}"; cat "${PROMPT}"; } | claude ...
else
  claude ... < "${PROMPT}"
fi
```

**Step 1: Implement all 4 launch scripts**

All launch scripts must validate required env vars at the top:
```bash
[[ -n "$BATCH_ID" && -n "$N" && -n "$VERSION_ID" ]] \
  || { echo "ERROR: BATCH_ID, N, VERSION_ID must be set"; exit 1; }
```
This is especially critical for `launch_reviewer_codex.sh` where triple-nested quoting (tmux > codex exec > `$(cat ...)`) makes env var substitution errors silent and hard to debug.

**Step 2: Write agents/test_arg_parser.sh from verification plan §3.4 + §3.6**

Tests from §3.4: `--phase plan`, `--phase=synthesize`, invalid phase, default phase.
Tests from §3.6 (CWD independence): cd /tmp, run dry-run, verify `cd "${PROJECT_DIR}"` in output, verify Codex handoff uses relative `artifact_path`.
Test launch_worker.sh --dry-run: verify it prints the `git worktree add` command but does NOT create an actual worktree directory.

**Step 3: Write agents/test_guards.sh from verification plan §3.5 + §3.7**

Tests from §3.5: PIPELINE_LOCKED guard positive + negative.
Tests from §3.7 (watchdog false-positive guard): create handoff file, set state to WORKER_RUNNING 1 hour ago, run watchdog, verify no timeout artifact created.

**Step 4: Run tests**

Run: `bash agents/test_arg_parser.sh && bash agents/test_guards.sh`
Expected: no FAIL lines

**Step 5: Commit**

```bash
git add agents/launch_*.sh agents/test_arg_parser.sh agents/test_guards.sh
git commit -m "agents: add launch scripts (orchestrator, worker, reviewers) + tests"
```

---

### Task 20: Create agents/watchdog.sh + agents/install_cron.sh

**Files:**
- Create: `agents/watchdog.sh` (§11.2, lines 1034–1116)
- Create: `agents/install_cron.sh` (§11.1)

Watchdog: READ-ONLY, audit probe, crash detection, timeout handling, STATE_TO_HANDOFF map, disk check.

**Design doc bug fix (SB-3)**: The elapsed-timeout handler in design doc §11.2 (line 1107) checks `elapsed > max_s` but does NOT check `HANDOFF_EXISTS`. If an agent completed and wrote its handoff but the tmux session already exited, the watchdog would write a spurious timeout artifact. Fix: add `HANDOFF_EXISTS == false` guard to the elapsed-timeout branch:
```bash
if (( elapsed > max_s )) && [[ "$HANDOFF_EXISTS" == "false" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    echo "{...}" > "$TIMEOUT_FILE"
  fi
fi
```
This is required for VP test 3.7 to pass.

**Step 1: Implement both files**

**Step 2: Commit** (batched with Task 21)

---

### Task 21: Create agents/check_clis.sh + agents/test_pipeline_integrity.sh

**Files:**
- Create: `agents/check_clis.sh` (§13.1)
- Create: `agents/test_pipeline_integrity.sh` (§13.2)

**Step 1: Implement**

check_clis.sh: test Claude CLI + Codex CLI with `${CODEX_MODEL}`.
test_pipeline_integrity.sh: 12 assertions for 1-iter, plus flags:
- `--iterations N` (default 1; use 3 for full Tier 3 test including HUMAN_SYNC + archive)
- `--inject-worker-failure` (Tier 4: injects failed worker handoff, verifies synthesis handles it)
- `--inject-codex-timeout` (Tier 5: injects timeout artifact instead of Codex handoff, verifies degradation)
- Orphaned tmux session check at end

**Step 2: Commit**

```bash
git add agents/watchdog.sh agents/install_cron.sh agents/check_clis.sh agents/test_pipeline_integrity.sh
git commit -m "agents: add watchdog, cron installer, CLI check, integrity test"
```

---

### Task 22: Phase D checkpoint — all shell tests pass

Run: `bash agents/state_utils.sh test && bash agents/test_arg_parser.sh && bash agents/test_guards.sh`
Expected: all pass

---

## Phase E: Prompts (Tasks 23–24)

### Task 23: Create all 5 prompt files

**Files:**
- Create: `agents/prompts/orchestrator_plan.md` (§10.1)
- Create: `agents/prompts/orchestrator_synthesize.md` (§5.1 synthesis)
- Create: `agents/prompts/worker.md` (§10.2)
- Create: `agents/prompts/reviewer_claude.md` (§10.3)
- Create: `agents/prompts/reviewer_codex.md` (§10.4 — prints to stdout, no file writes)

Each prompt follows the IDENTITY / READ / TASK / WRITE (or WHEN DONE for Codex) / CONSTRAINTS structure from the design doc.

**Placeholder resolution** (v10 review finding #3): Each prompt must include a note at the top:
```
NOTE: Variables like {N}, {batch_id}, ${VERSION_ID} are NOT shell-substituted.
Read them from state.json at the start of your task:
  jq -r '.iteration' state.json        → N
  jq -r '.batch_id' state.json         → batch_id
  jq -r '.version_id // empty' state.json → VERSION_ID
```
**Worker prompt exception (HP-3 fix)**: The worker runs in a git worktree. The worktree's state.json is a stale committed copy (`state: "IDLE", version_id: null`). The worker prompt (`worker.md`) MUST override the generic header above with:
```
NOTE: You are running in a git worktree. The worktree's state.json is STALE.
Always read state from the PROJECT_DIR copy:
  VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")
  BATCH_ID=$(jq -r '.batch_id' "${PROJECT_DIR}/state.json")
  N=$(jq -r '.iteration' "${PROJECT_DIR}/state.json")
```

**Codex prompt exception**: `reviewer_codex.md` — Codex runs `--sandbox read-only` and cannot execute jq. The Codex launch script must use `envsubst` for targeted variable substitution (HP-5 fix — `$(cat ...)` shell expansion corrupts literal `$` in the prompt):
```bash
# In launch_reviewer_codex.sh: expand ONLY the 3 intended variables
PROMPT_TEXT=$(BATCH_ID="${BATCH_ID}" N="${N}" VERSION_ID="${VERSION_ID}" \
  envsubst '${BATCH_ID} ${N} ${VERSION_ID}' \
  < "${PROJECT_DIR}/agents/prompts/reviewer_codex.md")
```
Do NOT use `$(cat ...)` — it expands ALL `$` characters in the prompt, corrupting references to `$threshold`, `$SMOKE_TEST`, etc.

**Synthesis handoff schema (MG-3 fix)**: The synthesis orchestrator's handoff (`orchestrator_synth_done.json`) must include a structured `decisions` field so the controller can act without parsing prose:
```json
{
  "status": "done",
  "artifact_path": "memory/direction_iter2.md",
  "sha256": "...",
  "decisions": {
    "promote_version": null,
    "gate_change_requests": [],
    "next_hypothesis": "..."
  }
}
```
Controller reads: `jq -r '.decisions.promote_version // empty' orchestrator_synth_done.json`. If non-null, calls `promote_version()`. The orchestrator_synthesize.md prompt must instruct the agent to produce this exact JSON structure.

**WORKER_FAILED injection for orchestrator_synthesize.md**: The launch script (`launch_orchestrator.sh --phase synthesize`) must prepend the WORKER_FAILED value to the prompt via stdin:
```bash
{ echo "WORKER_FAILED=${WORKER_FAILED}"; cat "${PROMPT}"; } | claude --print --model opus ...
```
The prompt must document: "The first line of your input is WORKER_FAILED=0 or WORKER_FAILED=1. Branch accordingly."

**Step 1: Implement all 5 prompts**

**Step 2: Verify all contain critical sections**

Run: `for f in agents/prompts/*.md; do echo "=== $f ==="; grep -c "IDENTITY\|READ\|CONSTRAINT" "$f"; done`
Expected: each file has 3+ matches

**Step 3: Commit**

```bash
git add agents/prompts/
git commit -m "agents: add prompt files (orchestrator, worker, reviewers)"
```

---

### Task 24: (no separate task — covered in Task 23 commit)

---

## Phase F: Integration + v0 Baseline (Tasks 25–28)

### Task 25: Full ML test suite pass

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow && SMOKE_TEST=true python -m pytest ml/tests/ -v`
Expected: all pass. Fix any issues.

---

### Task 26: Create v0 baseline (SMOKE_TEST mode)

**Note**: This smoke v0 baseline exists to verify pipeline infrastructure end-to-end. Gate floors derived from synthetic data are meaningless for production. Before running any real-data batch, a real v0 must be created with `SMOKE_TEST=false` (requires Ray cluster), and `populate_v0_gates.py` must be re-run against real metrics.

**Step 1: Run v0 baseline**

```bash
SMOKE_TEST=true python ml/pipeline.py --version-id v0 --auction-month 2021-07 \
  --class-type onpeak --period-type f0
```

**Step 2: Populate gate floors**

```bash
python ml/populate_v0_gates.py
```

**Step 3: Verify**

Run: `jq '.gates | to_entries[] | select(.value.pending_v0 == true)' registry/gates.json`
Expected: empty (no pending entries)

Run: `jq '."S1-BRIER"' registry/v0/metrics.json` — note value
Run: `jq '.gates."S1-BRIER".floor' registry/gates.json` — must be > Brier value

**Step 4: Commit**

```bash
git add registry/
git commit -m "foundation: create v0 baseline (SMOKE_TEST) and populate gate floors"
```

---

### Task 27: Integration smoke test (Tiers 1–2)

**Step 1: Tier 1 — Unit tests**

```bash
bash agents/state_utils.sh test
bash agents/test_arg_parser.sh
bash agents/test_guards.sh
SMOKE_TEST=true python -m pytest ml/tests/ -v
```

**Step 2: Tier 2 — 1-iteration smoke loop** (requires Claude + Codex CLIs)

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 1
```

Verify 12 assertions from design §13.2. Final state == IDLE (design doc §13.2 says ORCHESTRATOR_PLANNING — VP needs update to match IDLE; see Task 17 step 18 note).

**Step 3: Commit if fixes needed**

```bash
git commit -m "integration: Tier 1-2 smoke tests pass"
```

---

### Task 28: Integration smoke test (Tiers 3–5) — run after Tier 2 passes

**Step 1: Tier 3 — Full 3-iteration smoke**

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 3
```

Verify: final state == HUMAN_SYNC, `memory/archive/${batch_id}/executive_summary.md` exists, `memory/archive/index.md` updated, warm/ files reset.

**Step 2: Tier 4 — Worker failure injection**

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 1 --inject-worker-failure
```

Verify: WORKER_FAILED=1 propagated, synthesis runs with failure branch, no reviews created.

**Step 3: Tier 5 — Codex timeout degradation**

```bash
SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 1 --inject-codex-timeout
```

Verify: timeout artifact written, synthesis proceeds with Claude review only.

**Step 4: Commit**

```bash
git commit -m "integration: Tier 3-5 smoke tests pass"
```

---

## Verification

After all tasks complete:

1. **ML unit tests**: `SMOKE_TEST=true python -m pytest ml/tests/ -v` — all pass
2. **Shell unit tests**: `bash agents/state_utils.sh test` — all pass
3. **Arg parser + CWD tests**: `bash agents/test_arg_parser.sh` — all pass (includes §3.4 + §3.6)
4. **Guard + watchdog tests**: `bash agents/test_guards.sh` — all pass (includes §3.5 + §3.7)
5. **v0 baseline exists**: `jq '."S1-AUC"' registry/v0/metrics.json` confirms metrics present
6. **Gates populated**: `jq '.gates | to_entries[] | select(.value.pending_v0 == true)' registry/gates.json` returns empty
7. **Tier 2 smoke**: `SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 1` — 12 assertions pass
8. **Tier 3 smoke**: `SMOKE_TEST=true bash agents/test_pipeline_integrity.sh --iterations 3` — HUMAN_SYNC reached, archive created
9. **Tier 4 smoke**: `--inject-worker-failure` — synthesis handles failed worker
10. **Tier 5 smoke**: `--inject-codex-timeout` — degradation path works

---

## Key Reference Files

| Purpose | Path |
|---------|------|
| Design doc (authoritative) | `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` |
| Verification plan | `docs/plans/2026-02-26-verification-plan.md` |
| Source config (features, XGBoost params) | `../research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/config.py` |
| Source models (threshold, training) | `../research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/models.py` |
| Source evaluation (metrics) | `../research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/evaluation.py` |
| v10 review (remaining findings) | `docs/reviews/2026-02-26-agentic-ml-pipeline-design-review-v10.md` |
