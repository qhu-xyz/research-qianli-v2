# Shadow Price Prediction — Pipeline Design & Version Control

## 1. Why We Need Version Control

Our shadow price prediction pipeline produces models that generate production trading signals (SPICE 6.7B). Without formal version control:

- **No reproducibility**: We can't trace which config/features/thresholds produced a given signal
- **No comparison baseline**: We can't measure whether a code change improved or regressed performance
- **No audit trail**: We can't answer "why did the model predict X?" for any historical prediction
- **No safe promotion**: We have no gates preventing a bad model from reaching production

The registry system addresses these by storing every model version's config, metrics, and provenance artifacts with integrity verification.

### 1.1 Reproducibility Scope

**What we guarantee**: Given a version ID, we can identify the exact config, feature set, training data characteristics, and threshold decisions used. Re-running the pipeline with the same config on the same data will produce statistically equivalent (but not bitwise identical) models.

**What we do NOT guarantee**: Bitwise reproducibility across retrains. XGBoost with multi-threading, floating-point ordering, and Ray-distributed data loading introduces non-determinism. Exact reproduction would require: fixed random seeds at every level, deterministic training flags, pinned library versions, and identical hardware — which is impractical in our research environment.

**Mitigation**: Each version stores enough provenance (config checksum, code commit, training data stats, threshold decisions, feature importances) that behavioral equivalence can be verified even if bitwise identity cannot.

## 2. Pipeline Architecture

### 2.1 The Two ML Tasks

The pipeline is a **two-stage cascade**:

**Stage 1 — Classification (binding detection)**
- **Target**: `y = (shadow_price > 0) → binary {0, 1}` (`models.py:865`)
- **Goal**: Predict which constraint-outage pairs will have non-zero shadow prices (will "bind")
- **Model**: XGBoost classifier per branch/horizon group
- **Features**: 9 step1 features (density probability metrics)
- **Threshold**: Optimized per-branch using F-beta score
- **Key metrics**: AUC-ROC, precision, recall, F-beta

**Stage 2 — Regression (shadow price magnitude)**
- **Target**: `y = log1p(shadow_price)` where shadow_price > 0 (`models.py:1095`)
- **Goal**: Predict magnitude for pairs classified as binding in Stage 1
- **Trained on**: Only true positives (rows where actual shadow_price > 0)
- **Model**: XGBoost regressor per branch/horizon group
- **Features**: 13 step2 features (density metrics + step1 probability outputs)
- **Key metrics**: Spearman correlation, MAE, RMSE, bias

**Combined evaluation**: RMSE/MAE across all rows, ranking metrics (NDCG, value capture at top-K).

### 2.2 Pipeline Workflow

```
[1] Data Loading (data_loader.py)
     │  Load MCPs, cleared trades, outages via Ray/pbase
     │  Sources: MisoCalculator, MisoApTools, density parquets
     ▼
[2] Feature Engineering (data_loader.py)
     │  Build 9 step1 features: density_prob_exceed_{50,85,100,120},
     │  density_mean, density_skewness, density_std, prob_below_100, hist_da
     │  Build 13 step2 features: step1 features + classifier probability outputs
     ▼
[3] Labeling (data_loader.py)
     │  Binary label: shadow_price > 0 → binding (1), else non-binding (0)
     │  Regression label: log1p(shadow_price) for binding rows
     ▼
[4] Training — Stage 1: Classifier (models.py)
     │  Per horizon group → per branch:
     │    - Feature selection (monotonic constraint filtering)
     │    - XGBoost classifier with CV
     │    - Threshold optimization (F-beta on validation set)
     │  Produces: classifier ensembles, threshold decisions, feature importances
     ▼
[5] Training — Stage 2: Regressor (models.py)
     │  Per horizon group → per branch:
     │    - Train only on binding rows (shadow_price > 0)
     │    - XGBoost regressor with CV
     │  Produces: regressor ensembles, feature importances
     ▼
[6] Inference (prediction.py)
     │  For each test row:
     │    - Stage 1: classify binding probability
     │    - Apply threshold → binary prediction
     │    - Stage 2: regress shadow price for predicted-binding rows
     │    - Non-binding rows get shadow_price = 0
     ▼
[7] Evaluation (evaluation.py)
     │  score_results_df() → comprehensive metrics:
     │    - stage1: AUC-ROC, precision, recall, F-beta, avg_precision, brier
     │    - stage2: spearman, MAE, RMSE, bias (on true positives only)
     │    - combined: RMSE_all, MAE_all
     │    - ranking: NDCG, value_capture at top-K
     ▼
[8] Registry (registry.py)
     │  Store: config.json, features.json, metrics.json, manifests
     │  Gate check: must pass all hard gates AND beat champion
     │  Promote: update manifest.json → new champion
     ▼
[9] Signal Generation (signal_generator.py)
     │  Convert predictions → production signal format
```

### 2.3 Module Responsibilities

| Module | File | Role |
|--------|------|------|
| Config | `config.py` | All hyperparameters as frozen dataclasses |
| ISO Config | `iso_configs.py` | MISO-specific paths, schedules, horizon groups |
| Data Loader | `data_loader.py` | Load data via Ray/pbase, build features, assign labels |
| Models | `models.py` | Train classifiers/regressors, feature selection, threshold optimization |
| Anomaly Detection | `anomaly_detection.py` | Detect flow anomalies for special handling |
| Prediction | `prediction.py` | Run inference (classify → regress → merge) |
| Evaluation | `evaluation.py` | Score predictions, compute all metrics |
| Pipeline | `pipeline.py` | Orchestrate end-to-end: load → train → predict → evaluate |
| Registry | `registry.py` | Version control, gate checking, promotion |
| Signal Generator | `signal_generator.py` | Convert to production signal format |
| Naming | `naming.py` | Canonical file path generators (NEW) |

All modules live under `src/shadow_price_prediction/`.

## 3. Version Control System

### 3.1 Artifact Contract

Each model version lives in `registry/versions/{model_id}/` with this **exact** file tree:

```
registry/versions/{model_id}/
├── meta.json                    # REQUIRED: ID, date, description, version_hash, source_commit
├── config.json                  # REQUIRED: full PredictionConfig snapshot
├── features.json                # REQUIRED: step1 + step2 feature lists
├── metrics.json                 # REQUIRED: all 32-period benchmark scores
├── threshold_manifest.json      # REQUIRED: per-branch threshold decisions (single file, all periods)
├── feature_importance.json      # REQUIRED: per-model feature importances (single file, all periods)
└── train_manifest.json          # REQUIRED: training data provenance (single file, all periods)
```

**Note**: All manifest files are single root-level files (not per-period subdirectories). Each contains a top-level dict keyed by `{YYYYMM}_{class_type}` for per-period data.

Compared to the teammate's versioning example (`versioning_example.png`):

| Teammate has | We have | Status |
|-------------|---------|--------|
| `config.json` | `config.json` | Already implemented |
| `features.json` | `features.json` | Already implemented |
| `metrics.json` | `metrics.json` | Already implemented |
| `threshold_manifest.json` | `threshold_manifest.json` | Adding in this plan |
| `train_manifest.json` | `train_manifest.json` | Adding in this plan |
| `feature_importance.json` | `feature_importance.json` | Adding in this plan |
| `model.pkl` | — (skipped) | See Section 3.2 |

### 3.2 Why We Don't Store model.pkl

The teammate persists a single `model.pkl` per version because they have **one model per version**. Our pipeline retrains **per (auction_month, class_type, horizon_group, branch)**:

- 8 auction months × 2 class types = 16 training runs
- Each run: ~3 horizon groups × (default + N branch models) × 2 stages
- **Total: 300-800 individual XGBoost models per version**

Persisting all would be large and unnecessary. Best-effort reproducibility is provided by:
1. `config.json` with SHA-256 checksum (exact hyperparameters)
2. `train_manifest.json` (training data row counts, date ranges, feature stats)
3. `threshold_manifest.json` (exact threshold decisions per branch)
4. `feature_importance.json` (model behavior fingerprint — useful for drift detection, though importances are not stable across retrains)
5. `meta.json` includes `source_commit` (git SHA linking to exact code)

### 3.3 Version Naming Convention

```
v{NNN}-{algo}-{YYYYMMDD}[-{SEQ}]

Examples:
  v000-legacy-20260220          # Baseline (SEQ omitted for legacy)
  v001-xgb-20260220-001        # First iteration (XGBoost, Feb 20)
  v002-xgb-20260220-002        # Second iteration
  v003-lgbm-20260221-001       # Third iteration (switched to LightGBM)
```

- `NNN`: monotonic version number (000, 001, 002, ...)
- `algo`: model algorithm (`xgb`, `lgbm`, `catboost`, `legacy`)
- `YYYYMMDD`: creation date
- `SEQ` (optional): sequence within same day (001, 002, ...). Omitted for legacy/special versions.

### 3.4 Integrity: Version-Level Hash

Each version stores a **version hash** in `meta.json` that covers all required artifacts:

```json
{
  "model_id": "v001-xgb-20260220-001",
  "created": "2026-02-20T15:02:30",
  "description": "Iteration 1: Switch threshold from F0.5 to F2.0",
  "source_commit": "8e93306",
  "algo": "xgb",
  "status": "candidate",
  "config_checksum": "sha256:39ef5d6f...",
  "version_hash": "sha256:a1b2c3d4..."
}
```

**`version_hash`** = SHA-256 of the concatenated checksums of: `config.json`, `features.json`, `metrics.json`, `threshold_manifest.json`, `feature_importance.json`, `train_manifest.json`. This is computed after all artifacts are written and stored in `meta.json`.

**`manifest.json`** references `version_hash` (not just `config_checksum`):
```json
{
  "active_version": "v000-legacy-20260220",
  "active_version_hash": "sha256:a1b2c3d4...",
  "active_config_checksum": "sha256:39ef5d6f...",
  "updated": "2026-02-20T14:30:00",
  "history": [
    {"version": "v000-legacy-20260220", "promoted": "2026-02-20T14:30:00"}
  ]
}
```

The audit script recomputes `version_hash` from the on-disk artifacts and compares to `meta.json`. Any tampering of any artifact will be detected.

## 4. Gate System (Promotion Rules)

### 4.1 Hard Gates

| Gate | Metric | Floor | Direction | Rationale |
|------|--------|-------|-----------|-----------|
| S1-AUC | Stage 1 AUC-ROC | ≥ 0.80 | Higher better | Classifier must have reasonable discrimination |
| S1-REC | Stage 1 Recall | ≥ 0.30 | Higher better | Must catch at least 30% of binding constraints |
| S2-SPR | Stage 2 Spearman | ≥ 0.30 | Higher better | Regression ranking must be meaningful |
| C-VC@1000 | Value Capture top-1000 | ≥ 0.50 | Higher better | Must capture majority of economic value |
| C-RMSE | Combined RMSE (all rows) | ≤ 2000.0 | Lower better | Overall prediction accuracy ceiling |

All gates have defined floors. C-RMSE ceiling of 2000.0 is based on the legacy baseline mean RMSE of ~1556 with a ~30% margin.

### 4.2 Promotion Criteria

A candidate version is promoted to champion if:

1. **All gate floors pass** — every metric above its floor, checked **per class type** (not just the average). Both onpeak and offpeak must individually pass all floors.
2. **No regression** — for each enforced gate, the candidate's **per-class mean** must not be worse than the champion's by more than the noise tolerance:
   - Tolerance: candidate metric must be ≥ `champion_metric × 0.98` for "higher is better" gates (2% degradation tolerance)
   - Tolerance: candidate metric must be ≤ `champion_metric × 1.02` for "lower is better" gates
   - This accounts for benchmark variance across the 32-period evaluation without requiring formal statistical testing (which would need more periods than we have)
3. **At least one improvement** — the candidate must show a meaningful improvement (> 2% relative) on at least one enforced gate vs champion, otherwise the change is noise.

### 4.3 Benchmark Scope

Every version is evaluated on 32 test periods:

| Planning Year | Auction Months | Seasons |
|--------------|----------------|---------|
| PY20 | 2020-07, 2020-10, 2021-01, 2021-04 | Summer, Fall, Winter, Spring |
| PY21 | 2021-07, 2021-10, 2022-01, 2022-04 | Summer, Fall, Winter, Spring |

× 2 class types (onpeak, offpeak) × 2 period types (f0, f1) = **32 runs total**.

## 5. File Naming Conventions

### 5.1 Registry Artifacts

| Artifact | Path Pattern |
|----------|-------------|
| Version directory | `registry/versions/{model_id}/` |
| Config snapshot | `registry/versions/{model_id}/config.json` |
| Feature lists | `registry/versions/{model_id}/features.json` |
| Benchmark metrics | `registry/versions/{model_id}/metrics.json` |
| Version metadata | `registry/versions/{model_id}/meta.json` |
| Threshold decisions | `registry/versions/{model_id}/threshold_manifest.json` |
| Feature importances | `registry/versions/{model_id}/feature_importance.json` |
| Training data provenance | `registry/versions/{model_id}/train_manifest.json` |

### 5.2 Experiment Artifacts

| Artifact | Path Pattern |
|----------|-------------|
| Experiment output dir | `{EXPERIMENT_BASE_DIR}/{model_id}/` |
| Benchmark parquet | `results_{YYYYMM}_{class_type}_{period_type}.parquet` |
| Aggregated CSV | `{model_id}_agg.csv` |

`EXPERIMENT_BASE_DIR` is configured via environment variable or argument (default: `/opt/temp/tmp/pw_data/spice6/experiments`). Not hardcoded in naming.py.

### 5.3 naming.py Module

All path patterns are centralized in `src/shadow_price_prediction/naming.py`. This module contains **only pure functions** that take directory roots as arguments — no hardcoded absolute paths.

```python
from pathlib import Path

def result_parquet_name(auction_month: str, class_type: str, period_type: str) -> str:
    """e.g. 'results_202007_onpeak_f0.parquet'"""
    am_compact = auction_month.replace("-", "")
    return f"results_{am_compact}_{class_type}_{period_type}.parquet"

def result_parquet_path(output_dir: str, auction_month: str, class_type: str, period_type: str) -> Path:
    return Path(output_dir) / result_parquet_name(auction_month, class_type, period_type)

def version_dir(registry_root: str, model_id: str) -> Path:
    return Path(registry_root) / "versions" / model_id

def experiment_output_dir(base_dir: str, model_id: str) -> Path:
    return Path(base_dir) / model_id

def threshold_manifest_path(ver_dir: Path) -> Path:
    return ver_dir / "threshold_manifest.json"

def feature_importance_path(ver_dir: Path) -> Path:
    return ver_dir / "feature_importance.json"

def train_manifest_path(ver_dir: Path) -> Path:
    return ver_dir / "train_manifest.json"
```

## 6. New Registry Artifacts (Details)

### 6.1 threshold_manifest.json

Records the threshold decision made for each branch during classifier training. Single file, keyed by `{YYYYMM}_{class_type}`:

```json
{
  "202007_onpeak": {
    "horizon_group_short": {
      "default": {
        "threshold": 0.42,
        "f_beta_used": 2.0,
        "f_beta_score": 0.65,
        "val_size": 1200,
        "val_binding_rate": 0.08
      },
      "branch_EKPC_to_AEP": {
        "threshold": 0.38,
        "f_beta_used": 2.0,
        "f_beta_score": 0.71,
        "val_size": 340,
        "val_binding_rate": 0.12
      }
    }
  }
}
```

### 6.2 feature_importance.json

Records XGBoost feature importances from each trained model. Note: feature importances are **not stable** across retrains and should be treated as a behavioral fingerprint, not a model identity check. Useful for drift detection and explainability, not for verifying exact model reproduction.

```json
{
  "stage1": {
    "202007_onpeak": {
      "horizon_group_short": {
        "default": {
          "gain": {"density_prob_exceed_50": 0.45, "density_mean": 0.22},
          "cover": {"density_prob_exceed_50": 0.30},
          "weight": {"density_prob_exceed_50": 0.15}
        }
      }
    }
  },
  "stage2": {
    "202007_onpeak": {
      "horizon_group_short": {
        "default": {
          "gain": {"density_mean": 0.38, "step1_prob": 0.25}
        }
      }
    }
  }
}
```

### 6.3 train_manifest.json

Records training data provenance for all periods in a single file:

```json
{
  "202007_onpeak": {
    "auction_month": "2020-07",
    "class_type": "onpeak",
    "training_date_range": ["2019-01", "2020-06"],
    "n_rows_total": 45000,
    "n_binding": 3200,
    "binding_rate": 0.071,
    "feature_stats": {
      "density_prob_exceed_50": {"mean": 0.12, "std": 0.08, "min": 0.0, "max": 0.95},
      "density_mean": {"mean": 45.3, "std": 120.5, "min": -50.0, "max": 8500.0}
    }
  },
  "202007_offpeak": {
    "auction_month": "2020-07",
    "class_type": "offpeak",
    "training_date_range": ["2019-01", "2020-06"],
    "n_rows_total": 38000,
    "n_binding": 1800,
    "binding_rate": 0.047,
    "feature_stats": {}
  }
}
```

## 7. Auditing Scripts

### 7.1 audit_registry.py — "Is the registry healthy?"

```bash
python notebook/audit_registry.py
```

Checks:
- All version directories have all 7 required files (meta.json, config.json, features.json, metrics.json, threshold_manifest.json, feature_importance.json, train_manifest.json)
- `config_checksum` in meta.json matches SHA-256 of config.json on disk
- `version_hash` in meta.json matches recomputed hash over all artifacts
- `source_commit` in meta.json is a valid git revision in the repo
- `manifest.json` points to a valid version that exists
- Active version's `version_hash` matches manifest's `active_version_hash`

Output:
```
Registry Audit Report
=====================
Versions found: 3
Active champion: v000-legacy-20260220

v000-legacy-20260220 ... PASS (7/7 files, checksum ✓, version_hash ✓, commit abc1234 ✓)
v001-xgb-20260220-001 ... PASS (7/7 files, checksum ✓, version_hash ✓, commit 8e93306 ✓)
v002-xgb-20260220-002 ... WARN (5/7 files: metrics MISSING, train_manifest MISSING)

Manifest integrity: PASS (active version exists, version_hash matches)
```

### 7.2 audit_experiment.py — "Is this experiment complete?"

```bash
python notebook/audit_experiment.py --version-id v001-xgb-20260220-001
```

Checks:
- All 32 benchmark parquets exist in the experiment output directory
- Parquet schemas have expected columns (`actual_binding`, `predicted_binding`, `predicted_shadow_price`, etc.)
- Row counts are reasonable (> 0, < 500K per parquet)
- No nulls in critical columns (`actual_binding`, `predicted_binding`)
- Spot-check: re-score 2 random parquets and verify metrics are within 0.01 of registry values

Output:
```
Experiment Audit: v001-xgb-20260220-001
========================================
Parquets: 32/32 present

Schema checks:
  results_202007_onpeak_f0.parquet ... PASS (45,231 rows, schema OK)
  results_202007_onpeak_f1.parquet ... PASS (42,108 rows, schema OK)
  ...

Spot-check re-scoring:
  results_202104_offpeak_f0.parquet ... PASS (AUC drift < 0.001)
  results_202110_onpeak_f1.parquet ... PASS (AUC drift < 0.001)

Overall: PASS (32/32 parquets valid, spot-check passed)
```

### 7.3 audit_config.py — "Does the code match the registry?"

```bash
python notebook/audit_config.py
```

Checks:
- Loads active champion's stored `config.json`
- Creates a fresh `PredictionConfig()` from current code defaults
- Compares all fields and flags any drift
- Verifies feature list ordering matches between config.json and features.json
- Checks `source_commit` against current HEAD (warns if code has diverged)

Output:
```
Config Audit: v000-legacy-20260220 (active champion)
=====================================================
Checking code defaults vs stored config...

threshold.threshold_beta: MATCH (0.5)
features.step1_features: MATCH (9 features, order matches)
models.default_classifiers: MATCH
...

Source commit: abc1234 (champion) vs def5678 (HEAD) — WARN: code has diverged
Feature ordering: MATCH (config.json ↔ features.json consistent)

Overall: WARN (config matches, but code has diverged from champion commit)
```

## 8. Dead Code Cleanup

Before building new features, clean up dead code. Evidence methodology: full-text grep across `src/`, `notebook/`, and top-level `*.py` files (excluding `_legacy/` which is a frozen snapshot). Items are only deleted if zero references exist outside `_legacy/` and `__init__.py` re-exports.

| What | Location | Evidence | Action |
|------|----------|----------|--------|
| `tuning_utils.py` | `src/.../tuning_utils.py` | 0 imports in `src/` or `notebook/`; all references in `_legacy/` only | Delete |
| `StackingModel` class | `models.py:73` | Only reference: `__init__.py` export + one `isinstance` check in models.py itself. No notebook, no pipeline code instantiates it. | Delete class + references |
| `HyperparameterTuner` | `tuning.py` | Only reference: `__init__.py` export. No notebook or script imports it. | Delete file (tuning.py becomes empty) |
| Unused sklearn re-exports | `__init__.py:32-37` | `GradientBoosting*`, `RandomForest*` — grep shows 0 imports in any notebook or script | Remove from `__init__.py` |
| `save_results` param | `pipeline.py:398` | Grep: no caller passes `save_results=`. Comment says "Deprecated". | Remove parameter |
| Stale comments | `config.py:488,512-513` | `# Moved to IsoConfig` — historical notes for removed fields | Remove |
| Stale comment | `data_loader.py:15` | `# memory = joblib.Memory(...)` — commented-out caching | Remove |
| `run_f0_smoke_test.py` | `notebook/run_f0_smoke_test.py` | Superseded by `run_experiment.py --mode smoke`. Not referenced by any other script. | Delete |

Keep: `_legacy/` directory (needed by `run_legacy_baseline.py`), `docs/` strategy docs (reference material).

## 9. Execution Plan

| Step | What | Files |
|------|------|-------|
| 0 | Dead code cleanup (Section 8) | Multiple `src/` files, `notebook/run_f0_smoke_test.py` |
| 1 | Create `naming.py` (Section 5.3) | `src/.../naming.py` (NEW) |
| 2 | Add `record_train_manifest()`, `record_threshold_manifest()`, `record_feature_importance()`, `compute_version_hash()` to `ModelVersion` | `src/.../registry.py` |
| 3 | Expose threshold decisions + feature importances from `train_classifiers()`/`train_regressors()` | `src/.../models.py` |
| 4 | Capture and return new data from `_process_auction_month()` | `src/.../pipeline.py` |
| 5 | Collect new artifacts in worker subprocess | `notebook/_experiment_worker.py` |
| 6 | Record new artifacts in registry during scoring | `notebook/run_experiment.py` |
| 7 | Create 3 audit scripts (Section 7) | `notebook/audit_registry.py`, `audit_experiment.py`, `audit_config.py` (NEW) |
| 8 | Update gate definitions: set C-RMSE floor, add per-class enforcement, add noise tolerance | `src/.../registry.py` |
| 9 | Update `model_promotion.md` + `runbook.md` with new artifacts and audit instructions | `document/model_promotion.md`, `document/runbook.md` |

## 10. What We Deliberately Skip

- **`model.pkl`**: 300-800 models per version; config + provenance provides best-effort reproducibility (Section 3.2)
- **Bitwise reproducibility**: Would require deterministic mode, fixed seeds everywhere, pinned hardware — impractical for research (Section 1.1)
- **Multi-algorithm switching UI**: `EnsembleConfig` already supports arbitrary models via `ModelSpec`
- **Structured logging**: Replacing `print()` with `structlog` is low ROI for research code
- **God-class refactor of `ShadowPriceModels`**: Real problem but too invasive; risks breaking pipeline
- **Dataset snapshot hashes**: Training data comes from Ray/pbase (remote, not under our version control). We record provenance stats (row counts, date ranges, feature distributions) but cannot hash the raw data without materializing it locally.

## 11. Verification

After implementation:
1. `python notebook/audit_registry.py` → all versions pass integrity checks (version_hash verified)
2. `python notebook/audit_config.py` → current code matches active champion config
3. `run_experiment.py --mode smoke --version-id test-audit` → produces new manifest files
4. Check `registry/versions/test-audit/` contains all 7 required files: meta.json, config.json, features.json, metrics.json, threshold_manifest.json, feature_importance.json, train_manifest.json
5. Verify `meta.json` contains valid `version_hash` and `source_commit`
6. Re-run `audit_registry.py` → test-audit version passes all checks

## 12. Example Workflow: Running a Model Iteration

```bash
# 1. Activate environment
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
REPO=/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli

# 2. Register a new version
PYTHONPATH=$REPO/src:$PYTHONPATH python -c "
from shadow_price_prediction.registry import ModelRegistry
reg = ModelRegistry('$REPO/registry')
ver = reg.create_version('v003-xgb-20260221-001', algo='xgb',
    description='Iteration 3: add prob_exceed_85 feature')
"

# 3. Smoke test
PYTHONPATH=$REPO/src:$PYTHONPATH python $REPO/notebook/run_experiment.py \
    --mode smoke --version-id v003-xgb-20260221-001 \
    --overrides '{"step1_features": [["density_prob_exceed_50", 1], ["prob_exceed_85", 1]]}'

# 4. Full 32-period benchmark
PYTHONPATH=$REPO/src:$PYTHONPATH python $REPO/notebook/run_experiment.py \
    --mode full --version-id v003-xgb-20260221-001 \
    --overrides '{"step1_features": [["density_prob_exceed_50", 1], ["prob_exceed_85", 1]]}'

# 5. Score and check gates (per-class enforcement)
PYTHONPATH=$REPO/src:$PYTHONPATH python $REPO/notebook/run_experiment.py \
    --mode score --version-id v003-xgb-20260221-001

# 6. Audit
python $REPO/notebook/audit_registry.py
python $REPO/notebook/audit_experiment.py --version-id v003-xgb-20260221-001

# 7. Promote (if all gates pass per-class + beat champion within tolerance)
PYTHONPATH=$REPO/src:$PYTHONPATH python -c "
from shadow_price_prediction.registry import ModelRegistry
reg = ModelRegistry('$REPO/registry')
result = reg.promote('v003-xgb-20260221-001')
print(result.summary_table())
"
```
