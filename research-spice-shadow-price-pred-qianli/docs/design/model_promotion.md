# Model Promotion Pipeline

> **Last updated**: 2026-02-21
> **Module**: `src/shadow_price_prediction/registry.py`
> **Status**: Active

---

## 1. Overview

The model promotion pipeline manages versioned experiments for the shadow price prediction system. It provides:

- **Versioning**: Each experiment gets a unique ID, stored as a directory of 7 JSON files
- **Gating**: Hard metric floors with per-class enforcement and noise tolerance
- **Promotion**: Atomic champion swap with version-level hash integrity verification
- **Comparison**: Side-by-side version comparison
- **Checksum verification**: Config checksums + version-level hashes prevent drift
- **Audit scripts**: `audit_registry.py`, `audit_experiment.py`, `audit_config.py`

---

## 2. Naming Convention

```
v{NNN}-{algo}-{YYYYMMDD}-{SEQ}
```

| Component | Description | Example |
|-----------|-------------|---------|
| `NNN` | Monotonic version number (000, 001, 002...) | `v000` |
| `algo` | Algorithm identifier | `legacy`, `xgb`, `lgbm`, `catboost` |
| `YYYYMMDD` | Creation date | `20260220` |
| `SEQ` | Sequence number for same algo+date | `001` |

Special case: The baseline is `v000-legacy-20260220` (no sequence suffix for the bootstrap).

---

## 3. Directory Structure

```
registry/
├── manifest.json                           # Active champion pointer + version hash
├── legacy_baseline.json                    # Original flat benchmark (kept for reference)
├── legacy_baseline_agg.csv                 # Flat CSV (kept for reference)
└── versions/
    ├── v000-legacy-20260220/
    │   ├── meta.json                       # ID, date, commit, checksums, status
    │   ├── config.json                     # Full pipeline configuration
    │   ├── features.json                   # Feature lists + monotonicity
    │   ├── metrics.json                    # Gate values + per-run scores
    │   ├── threshold_manifest.json         # Per-branch threshold decisions
    │   ├── feature_importance.json         # Per-model feature importances
    │   └── train_manifest.json             # Training data provenance
    └── v001-xgb-20260221-001/
        ├── meta.json
        ├── config.json
        ├── features.json
        ├── metrics.json
        ├── threshold_manifest.json
        ├── feature_importance.json
        └── train_manifest.json
```

### manifest.json

The production manifest — the single source of truth for which version is active.

```json
{
  "active_version": "v000-legacy-20260220",
  "active_checksum": "sha256:9fe412762bf63eed",
  "active_version_hash": "sha256:a1b2c3d4e5f67890...",
  "promoted_at": "2026-02-20T14:51:16",
  "history": [
    {"version": "v000-legacy-20260220", "promoted_at": "...", "demoted_at": null}
  ]
}
```

### meta.json

Per-version metadata.

```json
{
  "model_id": "v000-legacy-20260220",
  "created": "2026-02-20T14:51:16",
  "description": "Legacy unmodified pipeline baseline",
  "source_commit": "b32bf6b",
  "algo": "legacy",
  "status": "active",
  "config_checksum": "sha256:9fe412762bf63eed",
  "version_hash": "sha256:a1b2c3d4e5f67890..."
}
```

Status transitions: `candidate` → `active` → `retired`

The `version_hash` is a SHA-256 computed over all 6 artifact files (config, features, metrics, threshold_manifest, feature_importance, train_manifest). It is verified at promotion time and stored in both `meta.json` and `manifest.json`.

### config.json

Full serialized `PredictionConfig`. Model classes are stored as qualified name strings (e.g., `"xgboost.XGBClassifier"`). A SHA-256 checksum is computed over this file and stored in both `meta.json` and `manifest.json`.

### features.json

Feature lists with monotonicity constraints for both stages.

### metrics.json

Contains:
- `gate_values` — per-class-type gate metric values (the 5 hard gates)
- `benchmark_scope` — what was tested (planning years, months, class types, period types)
- `per_run_scores` — full metric breakdown for every scored run

### threshold_manifest.json

Per-branch threshold decisions extracted during classifier training:

```json
{
  "202007_onpeak": {
    "horizon_group_short": {
      "default": {"threshold": 0.42, "f_beta_used": 2.0, "f_beta_score": 0.65},
      "branch_EKPC_to_AEP": {"threshold": 0.38, "f_beta_used": 2.0, "f_beta_score": 0.71}
    }
  }
}
```

### feature_importance.json

XGBoost feature importances (gain-based) from each trained model:

```json
{
  "stage1": {
    "horizon_group_short": {
      "default": {"gain": {"density_prob_exceed_50": 0.45, "density_mean": 0.22}}
    }
  },
  "stage2": { "..." }
}
```

### train_manifest.json

Training data provenance for reproducibility:

```json
{
  "202007_onpeak": {
    "auction_month": "2020-07", "class_type": "onpeak",
    "n_rows_total": 45000, "n_binding": 3200, "binding_rate": 0.071,
    "feature_stats": {"density_prob_exceed_50": {"mean": 0.12, "std": 0.08}}
  }
}
```

---

## 4. Hard Gates

Five metrics serve as promotion gates. A candidate must pass all enforced gates for **each class type individually** (onpeak AND offpeak) AND beat the current champion within a 2% noise tolerance.

| Gate | Stage | Metric | Direction | Floor | What moves it |
|------|-------|--------|-----------|-------|---------------|
| **S1-AUC** | Classifier | AUC-ROC | higher | 0.80 | hist_da + seasonal features |
| **S1-REC** | Classifier | Recall | higher | 0.30 | threshold_beta 0.5 → 2.0 |
| **S2-SPR** | Regressor | Spearman (TP) | higher | 0.30 | hist_da + seasonal + clamp |
| **C-VC@1000** | Combined | ValCap@1000 (constraint) | higher | 0.50 | Better probability ranking |
| **C-RMSE** | Combined | RMSE (all) | lower | 2000.0 | Regression clamp |

**Per-class enforcement**: Each class type (onpeak, offpeak) must individually pass all floor gates. A candidate that passes floors on average but fails on one class type is NOT promotable. The displayed value is the cross-class average.

**Noise tolerance**: When comparing to the champion, a candidate may be up to 2% worse on any individual gate without failing (`NOISE_TOLERANCE = 0.02`). However, the candidate must show at least one gate with >2% genuine improvement to be promoted — "not worse" alone is insufficient.

### Legacy Baseline Gate Values (champion)

| Gate | Onpeak | Offpeak | Mean | Floor | Status |
|------|-------:|--------:|-----:|------:|--------|
| S1-AUC | 0.6905 | 0.7002 | 0.6954 | 0.80 | BELOW |
| S1-REC | 0.2702 | 0.2766 | 0.2734 | 0.30 | BELOW |
| S2-SPR | 0.3774 | 0.4466 | 0.4120 | 0.30 | ABOVE |
| C-VC@1000 | 0.8379 | 0.8637 | 0.8508 | 0.50 | ABOVE |
| C-RMSE | $1,254.52 | $1,857.06 | $1,555.79 | 2000.0 | ABOVE |

---

## 5. Promotion Rules

A candidate version is **promotable** when:

1. **All hard gates pass per class type** — every gate with a floor/ceiling threshold is met for BOTH onpeak and offpeak individually
2. **Beats champion (noise-tolerant)** — for each enforced gate, candidate value within 2% of champion (or better). At least one gate must show >2% genuine improvement.
3. **Checksum verification passes** — config.json hash matches meta.json
4. **Version hash verification passes** — SHA-256 over all 6 artifact files matches meta.json

Force-promotion (skipping gates) is available for bootstrapping only.

### Promotion flow

```
1. Run experiment  →  produces results parquets + worker artifacts
2. Score results   →  compute all metrics via score_results_df()
3. Register        →  reg.create_version() + record_config/features/metrics
4. Aggregate       →  merge threshold/importance/provenance from workers
5. Hash            →  ver.compute_version_hash() over all 6 artifacts
6. Check gates     →  reg.check_gates(model_id)
7. Promote         →  reg.promote(model_id) if gate_result.promotable
8. Audit           →  audit_registry.py + audit_experiment.py
9. Update docs     →  runbook.md Run Log + critique.md status changes
```

---

## 6. Integrity Verification

Two levels of checksum protect against drift:

### Config checksum (per-file)

Prevents parameter drift between what's registered and what's loaded at runtime.

```python
# At registration time:
checksum = reg.create_version(...).record_config(config_dict)
# checksum stored in meta.json AND manifest.json

# At load time (before running pipeline):
assert reg.verify_active(), "Config checksum mismatch!"
```

The checksum is a truncated SHA-256 (16 hex chars) of the canonical JSON serialization of config.json (sorted keys, deterministic).

### Version hash (all artifacts)

Prevents any artifact from being modified after registration. Computed over the 6 artifact files: config.json, features.json, metrics.json, threshold_manifest.json, feature_importance.json, train_manifest.json.

```python
# After all artifacts are recorded:
ver.compute_version_hash()  # stores SHA-256 in meta.json

# To verify:
assert ver.verify_version_hash(), "Artifact tamper detected!"
```

The version hash is stored in both `meta.json` and `manifest.json` (for the active champion).

---

## 7. Usage Examples

### Create and register a new experiment

```python
from shadow_price_prediction import ModelRegistry, PredictionConfig

reg = ModelRegistry("registry/")
config = PredictionConfig(...)  # your experiment config

# Create version
ver = reg.create_version(
    algo="xgb",
    description="Added hist_da feature + F2.0 threshold",
    config_dict=ModelRegistry.config_to_dict(config),
    features_dict=ModelRegistry.features_to_dict(config),
)
print(f"Created: {ver.model_id}")  # v001-xgb-20260221-001

# After running benchmark and scoring:
ver.record_metrics(
    gate_values={"onpeak": {...}, "offpeak": {...}},
    benchmark_scope={...},
    per_run_scores=[...],
)
```

### Check gates and promote

```python
result = reg.check_gates(ver.model_id)
print(result.summary_table())

if result.promotable:
    promo = reg.promote(ver.model_id)
    print(f"Promoted: {promo.success}, previous: {promo.previous_champion}")
else:
    print("Not promotable yet")
```

### Compare versions

```python
print(reg.compare("v000-legacy-20260220", "v001-xgb-20260221-001"))
```

### Verify production integrity

```python
# Config checksum
assert reg.verify_active(), "Active model config has been modified!"

# Version hash (all artifacts)
ver = reg.get_version(active_id)
assert ver.verify_version_hash(), "Artifact tamper detected!"
```

---

## 8. Integration with Pipeline Runs

The typical workflow for running an experiment:

```bash
# 1. Activate environment
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# 2. Run pipeline (produces parquets + worker artifacts)
PYTHONPATH=/.../src:$PYTHONPATH python /.../notebook/run_experiment.py \
    --config v001-xgb-20260221-001 \
    --mode full

# 3. Score, aggregate artifacts, and register (automatic)
#    The run script calls record_metrics(), aggregates worker
#    artifacts (thresholds, importances, provenance), and
#    computes the version hash.

# 4. Audit
python /.../notebook/audit_registry.py
python /.../notebook/audit_experiment.py --version-id v001-xgb-20260221-001
python /.../notebook/audit_config.py

# 5. Check gates
PYTHONPATH=/.../src:$PYTHONPATH python -c "
from shadow_price_prediction import ModelRegistry
reg = ModelRegistry('registry/')
result = reg.check_gates('v001-xgb-20260221-001')
print(result.summary_table())
"
```

---

## 9. Audit Scripts

Three audit scripts verify pipeline integrity at different levels:

### audit_registry.py — "Is the registry healthy?"

```bash
python notebook/audit_registry.py [--registry-dir path/to/registry]
```

Checks all version directories for required files (7 per version), config checksums, version hashes, source commit validity, and manifest integrity.

### audit_experiment.py — "Is this experiment complete?"

```bash
python notebook/audit_experiment.py --version-id v001-xgb-20260221-001
```

Checks all 32 benchmark parquets exist, have the expected schema columns (`actual_binding`, `predicted_binding`, `predicted_shadow_price`), reasonable row counts, and no nulls in critical columns.

### audit_config.py — "Does the code match the registry?"

```bash
python notebook/audit_config.py [--registry-dir path/to/registry]
```

Compares the active champion's stored `config.json` against a fresh `PredictionConfig()` from current code defaults. Flags any parameter drift, source commit divergence, and feature ordering mismatches.

---

## 10. Design Decisions

| Decision | Rationale |
|----------|-----------|
| JSON over pickle for config | Human-readable, diffable, no security risk |
| File-based over database | No infrastructure dependency, works in any environment |
| Truncated SHA-256 (16 hex chars) for config | Sufficient for collision resistance in a small registry |
| Full SHA-256 (32 hex chars) for version hash | Covers all 6 artifacts, higher collision bar |
| Force-promote for baseline | Baseline won't pass absolute floors — that's the point |
| Per-class gate enforcement | Prevents masking a regression in one class type with a gain in another |
| 2% noise tolerance | Prevents rejecting genuinely equivalent models due to training noise |
| Require >2% improvement on at least one gate | "Not worse" alone is insufficient for promotion |
| C-RMSE floor = $2,000 | Legacy baseline is $1,556; floor set ~30% above to allow headroom |
| No model.pkl | 300-800 XGBoost models per version; config + provenance sufficient |
| Separate features.json | Quick inspection without parsing full config |
| Keep legacy_baseline.json | Reference artifact from the original scoring run |
| Worker-level artifact collection | Per-run JSONs aggregated at scoring time; no pipeline return signature changes |
