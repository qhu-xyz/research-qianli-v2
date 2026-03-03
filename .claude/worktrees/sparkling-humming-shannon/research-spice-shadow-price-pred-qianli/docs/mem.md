# Shadow Price Prediction — Progress & Context

> **Last updated**: 2026-02-25
> **Active champion**: v000-legacy (promoted 2026-02-20)
> **Best candidate**: v008-focused-classifier (first promotable version)
> **Current state**: v012-unified-regressor in progress (13/32 parquets)

## What This Project Does

Predicts MISO monthly DA shadow prices for FTR trading. Two-stage ML pipeline:
1. **Classifier** (XGBoost): predicts binding probability per constraint per outage-date
2. **Regressor** (XGBoost): predicts shadow price magnitude for predicted-binding constraints

Input: density model outputs (flow probability distributions) + historical DA shadow prices.
Output: constraint-level monthly signals with predicted shadow price and binding probability.

## Key Architecture Decisions

- **Per-branch models**: Each constraint branch gets its own classifier + regressor if
  enough training data exists. Falls back to a "default" (pooled) model otherwise.
- **Horizon groups**: f0 (current month), f1 (next month), long (2+ months). Separate
  models per group because f0 and f1 have very different predictability.
- **Threshold optimization**: Per-group F-beta threshold on validation set. Branch models
  mostly fall back to threshold=0.50 (insufficient per-branch val data).
- **32-run benchmark**: 8 auction months x 2 class types x 2 period types = 32 runs.
  All experiments run this full benchmark for comparability.

## Version Progression

| Version | Key Change | AUC | REC | PREC | SPR | VC@1K | RMSE | Pass | Promote |
|---------|-----------|-----|-----|------|-----|-------|------|------|---------|
| v000 (champ) | Legacy baseline | 0.695 | 0.390* | 0.302 | 0.400 | 0.848 | 1328 | Yes | - |
| v005 | 9/3 split, fixes | 0.682 | 0.287 | - | 0.424 | 0.822 | 1098 | Yes | No |
| v006 | Density-only, 10/2 | 0.689 | 0.290 | - | 0.403 | 0.836 | 1134 | Yes | No |
| v007 | +8 features, tuned XGB | 0.681 | 0.305 | 0.330 | 0.401 | 0.819 | 1057 | Yes | No |
| v008 | Focused clf, rich reg | 0.689 | 0.289 | 0.345 | 0.412 | 0.843 | 1109 | Yes | **Yes** |
| v009 | threshold=0.7 | 0.689 | 0.252 | 0.363 | 0.411 | 0.843 | 1103 | **No** | No |
| v010-thr050 | threshold=0.5 | 0.689 | 0.279 | 0.344 | 0.408 | 0.843 | 1111 | Yes | No |
| v010-vw | Value-weighted clf+reg | 0.688 | 0.269 | 0.335 | 0.419 | 0.841 | 1106 | **No** | No |
| v011 | VW reg only + EV scoring | 0.689 | 0.289 | 0.345 | 0.404 | 0.843 | 1098 | Yes | No |
| v012 | Unified regressor (WIP) | - | - | - | - | - | - | - | - |

*v000 recall is inflated by threshold=0.50 fallback (val_months=0, no optimization)

## Findings & Closed Investigations

### 1. Threshold tuning is a dead end (v009, v010-thr050)

Branch models (hundreds per segment) all fall back to threshold=0.50 (insufficient
per-branch validation data). They dominate aggregate metrics. Changing the override
has minimal effect:
- threshold=0.7: kills recall (raised branch fallbacks), fails S1-REC gate
- threshold=0.5: nearly identical to v008 (branches already at 0.50)
- v008's mixed regime (f0 default ~0.91, all branches 0.50) is the natural optimum

### 2. Asymmetric classifier/regressor design works (v007 vs v008)

Classifiers benefit from simplicity (fewer features, depth=4, more training data).
Regressors benefit from richness (27 features, depth=5). v008 decoupled these:
- Classifier: 14 features, v000 hyperparams + regularization
- Regressor: 27 features, deeper model, v007 hyperparams

### 3. Value-weighted training helps regressor only (v010-vw vs v011)

Weighting binding samples by shadow price magnitude in classifier training
de-emphasizes low-value binding constraints, hurting recall. But value-weighting
the regressor (v011) improves RMSE without affecting classifier recall.

### 4. Per-period gate enforcement catches hidden weakness

Gates are now checked per (class_type, period_type) segment. onpeak/f1 is consistently
the weakest segment (hardest to predict, lowest recall, highest RMSE). Multiple versions
that pass aggregate gates fail when onpeak/f1 is checked independently.

## Known Open Issues

| ID | Severity | Description |
|----|----------|-------------|
| BUG-6 | Low | Threshold artifact race condition: f0/f1 workers overwrite same file. Predictions use correct in-memory thresholds. |
| BUG-7 | Medium | S2-SPR gate silently drops NaN months (inflates metric) |
| IMP-11 | Medium | No formal hyperparameter search (Optuna). v008 params are from manual A/B testing. |
| IMP-15 | Medium | Val top-K anomaly unexplained (Lift < 1.7x on val vs 3-6x on holdout) |

## Infrastructure Notes

### How to run an experiment

```bash
# Activate venv
source /home/xyz/workspace/pmodel/.venv/bin/activate

# From project root:
PYTHONPATH=src:$PYTHONPATH python scripts/run_experiment.py \
  --mode full \
  --version-id v0XX-description \
  --overrides '{"threshold_override": 0.7}' \
  --concurrency 4

# Score after completion:
PYTHONPATH=src:$PYTHONPATH python scripts/run_experiment.py \
  --mode score \
  --version-id v0XX-description
```

Note: version must exist in registry before scoring. Create with:
```python
from shadow_price_prediction.registry import ModelRegistry
reg = ModelRegistry('versions')
ver = reg.create_version(algo='xgb', description='...', source_commit='...',
                         model_id_override='v0XX-description')
```

### Experiment output

Parquets: `/opt/temp/tmp/pw_data/spice6/experiments/{version_id}/`
Worker logs: `log_{YYYYMM}_{class_type}_{period_type}.txt` in same directory.

### Key config overrides (via --overrides JSON)

| Override | Effect |
|----------|--------|
| `threshold_override` | Fixed threshold for all groups (bypasses optimization) |
| `threshold_beta` | F-beta parameter for threshold optimization |
| `train_months` / `val_months` | Training/validation split (sum = lookback window) |
| `step1_features` / `step2_features` | Feature lists as `[[name, mono], ...]` |
| `default_clf_*` / `branch_clf_*` | XGBoost hyperparams for classifiers |
| `default_reg_*` / `branch_reg_*` | XGBoost hyperparams for regressors |

### Gate system

11 hard gates with per-segment enforcement (onpeak/f0, onpeak/f1, offpeak/f0, offpeak/f1).
Each segment must pass independently. `NOISE_TOLERANCE = 0.02` (2%) for champion comparison.
A version is promotable if it passes ALL floor gates AND beats the champion within noise
tolerance on all metrics, with at least one genuine improvement.

Since v010+, gates are split into PROMOTION_GATES (threshold-independent: AUC, VC@K, RMSE)
and MONITORING_GATES (threshold-dependent: recall, precision, Spearman-on-TPs, CAP@K).
This prevents threshold choice from distorting version comparisons.

## Where to Find Things

| What | Where |
|------|-------|
| Pipeline source | `src/shadow_price_prediction/` |
| Experiment scripts | `scripts/run_experiment.py`, `scripts/_experiment_worker.py` |
| Version registry | `versions/` (metrics.json, config.json, NOTES.md per version) |
| Experiment output | `/opt/temp/tmp/pw_data/spice6/experiments/{version_id}/` |
| Pipeline architecture | `docs/runbook.md` (19 sections) |
| Bugs & improvements | `docs/critique.md` (8 bugs, 21 improvements tracked) |
| Design docs | `docs/design/` (historical design documents) |
| Current champion | `versions/manifest.json` → v000-legacy |

## What To Work On Next

1. **Complete v012-unified-regressor** (13/32 done): trains regressor on ALL samples
   (binding + non-binding) with target=log1p(shadow_price). Eliminates the hard
   binary gate between classifier and regressor.

2. **Formal hyperparameter search** (IMP-11): v008 params are from manual experiments.
   Optuna with TimeSeriesSplit CV could find better configurations.

3. **New data sources**: Current features are all density-model-derived + historical DA.
   Shift factors, constraint metadata, or generation/load forecasts could provide
   genuinely new signal that density features cannot capture.

4. **Fix BUG-6** (threshold artifact race): Add period_type to threshold filename in
   naming.py. Low priority but improves post-hoc analysis reliability.
