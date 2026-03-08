# Annual FTR Constraint Tier Prediction — Runbook

## Environment Setup

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
```

All scripts must be run with `PYTHONPATH=.` so the `ml` package resolves:

```bash
PYTHONPATH=. python scripts/<script>.py [--screen]
```

## Running Experiments

### Screen mode (4 groups, ~2 min)
```bash
PYTHONPATH=. python scripts/run_v1_experiment.py --screen
```

### Full eval (12 groups, ~5 min)
```bash
PYTHONPATH=. python scripts/run_v1_experiment.py
```

### Available versions
| Script | Version | Features | Labels | Description |
|--------|---------|----------|--------|-------------|
| `run_v0_baseline.py` | v0 | formula (rank_ori) | — | V6.1 formula baseline |
| `run_v1_experiment.py` | v1 | Set A (6 V6.1 feat) | rank | ML with raw rank labels |
| `run_v2_experiment.py` | v2 | Set B (11 = A+spice6) | rank | ML with spice6 density |
| `run_v3_experiment.py` | v3 | Set A (6 V6.1 feat) | tiered | ML with tiered labels |
| `run_v4_experiment.py` | v4 | Set AF (7 = A+rank_ori) | rank | ML with formula-as-feature |
| `run_v5_experiment.py` | v5 | Set AF (7 = A+rank_ori) | tiered | Both improvements combined |

### Comparing versions
```bash
PYTHONPATH=. python -m ml.compare
```

### Holdout evaluation (2025, 4 groups)
```bash
PYTHONPATH=. python ml/benchmark.py --version-id v1_holdout --eval-groups 2025-06/aq1 2025-06/aq2 2025-06/aq3 2025-06/aq4
```

## Caching

### Enriched data cache (`cache/enriched/`)
- V6.1 signal + spice6 density per (year, aq), cached as parquet
- Auto-created on first load, reused across versions
- Safe to delete and regenerate (no ground truth stored)

### Ground truth cache (`cache/ground_truth/`)
- Realized DA shadow prices per (year, aq)
- Created by `scripts/cache_all_ground_truth.py`
- Required for all experiment runs

### Model caching
- Benchmark trains once per eval year, evaluates 4 quarters with same model
- No model files persisted to disk — retrained each run

## Registry

```
registry/
  v0/          # formula baseline
  v1/          # champion (ML, 6 features)
  v2/          # ML + spice6
  v3/          # tiered labels
  v4/          # formula-as-feature
  v5/          # tiered + formula
  gates.json   # quality gates (calibrated from v0)
  champion.json
  comparisons/ # auto-generated comparison JSONs
```

Each version dir contains:
- `config.json` — LTR config + eval config
- `metrics.json` — per-group + aggregate metrics
- `meta.json` — version metadata
