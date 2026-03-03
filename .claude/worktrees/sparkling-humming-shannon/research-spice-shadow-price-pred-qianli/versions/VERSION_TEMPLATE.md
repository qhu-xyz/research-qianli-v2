# Version Directory Contract

Each version directory `versions/vNNN-descriptive-name/` MUST contain these files:

## Required

| File | Description |
|------|-------------|
| `config.json` | Full frozen `PredictionConfig` as JSON. This is the single source of truth for what was run. |
| `meta.json` | `{model_id, created, description, source_commit, status, config_checksum}` |
| `metrics.json` | Full scoring output: 32-run per-period scores + aggregated gate values |
| `NOTES.md` | Hypothesis, what changed vs previous version, results table, conclusion |

## Optional

| File | Description |
|------|-------------|
| `overrides.json` | Only the config fields that differ from the previous version |
| `features.json` | Feature list (if different from what config.json implies) |
| `threshold_manifest.json` | Per-run threshold decisions |
| `feature_importance.json` | Per-model feature importances (diagnostic) |
| `train_manifest.json` | Training data provenance |
| `artifacts/` | Any additional artifacts (plots, pickled models, etc.) |

## NOTES.md Template

```markdown
# vNNN — Short Title

**Hypothesis**: What we expected to improve and why.

**Changes vs vPREV**:
- Change 1
- Change 2

## Results

| Gate | vPREV | vNNN | Delta |
|------|-------|------|-------|
| S1-AUC | x.xx | x.xx | +x.xx |
| S1-REC | x.xx | x.xx | +x.xx |
| S2-SPR | x.xx | x.xx | +x.xx |
| C-VC@1000 | x.xx | x.xx | +x.xx |
| C-RMSE | xxxx | xxxx | -xxx |

**Gates passed**: X/5
**Beats champion**: Yes/No (which gates by >2%)

**Conclusion**: What we learned. Promote or iterate.
```
