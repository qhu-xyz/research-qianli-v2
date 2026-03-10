# Annual FTR Constraint Tier Prediction — Runbook

## Environment Setup

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-signal
export PYTHONPATH=.
```

All scripts must be run with `PYTHONPATH=.` so the `ml` package resolves.

**CRITICAL**: LightGBM must use `num_threads=4`. The container has 64 CPUs and auto-detection causes massive thread contention (57s → 0.1s). This is already hardcoded in `ml/train.py`.

---

## Running Experiments

### Screen mode (4 groups, ~1s)
```bash
PYTHONPATH=. python scripts/run_v1_experiment.py --screen
```

### Full eval (12 groups, ~3s)
```bash
PYTHONPATH=. python scripts/run_v1_experiment.py
```

### Available versions
| Script | Version | Features | Labels | Description | VC@20 | Gates |
|--------|---------|----------|--------|-------------|-------|-------|
| `run_v0_baseline.py` | v0 | formula (rank_ori) | — | V6.1 formula baseline | 0.2323 | PASS |
| `run_v1_experiment.py` | v1 | Set A (6 V6.1 feat) | rank | First ML version | 0.2934 | PASS |
| `run_v2_experiment.py` | v2 | Set B (11 = A+spice6) | rank | ML + spice6 density | 0.2904 | FAIL |
| `run_v3_experiment.py` | v3 | Set A (6 V6.1 feat) | tiered | Tiered labels only | 0.2871 | FAIL |
| `run_v4_experiment.py` | v4 | Set AF (7 = A+rank_ori) | rank | Formula-as-feature only | 0.3030 | FAIL |
| `run_v5_experiment.py` | v5 | Set AF (7 = A+rank_ori) | tiered | Both improvements | 0.3075 | FAIL |

**Gate failures**: v2-v5 all fail Recall@100 L2 tail gate (2 groups below floor, max=1).

### Comparing versions
```bash
PYTHONPATH=. python -m ml.compare
```
Outputs markdown comparison table to `reports/` and JSON to `registry/comparisons/`.

### Holdout evaluation (2025, 4 groups)
```bash
PYTHONPATH=. python ml/benchmark.py --version-id v1_holdout --eval-groups 2025-06/aq1 2025-06/aq2 2025-06/aq3 2025-06/aq4
```

---

## Caching

### Enriched data cache (`cache/enriched/`)
- V6.1 signal + spice6 density per (year, aq), cached as parquet
- Auto-created on first load, reused across versions
- Safe to delete and regenerate (no ground truth stored)
- Avoids re-scanning 18GB density distribution parquet on NFS

### Ground truth cache (`cache/ground_truth/`)
- Realized DA shadow prices per (year, aq)
- Created by `scripts/cache_all_ground_truth.py` (requires Ray)
- Required for all experiment runs
- 28 parquet files (2019-2025, aq1-aq4)

### Model caching
- Benchmark trains once per eval year, evaluates 4 quarters with same model
- No model files persisted to disk — retrained each run (~3s total)

---

## Registry

```
registry/
  v0/          # formula baseline (calibrates gates)
  v1/          # ML, 6 features (only ML version passing all gates)
  v2/          # ML + spice6 (fails Recall@100 tail)
  v3/          # tiered labels (fails Recall@100 tail)
  v4/          # formula-as-feature (fails Recall@100 tail)
  v5/          # tiered + formula (fails Recall@100 tail)
  v1_holdout/  # v1 holdout (2025) results
  gates.json   # quality gates (calibrated from v0)
  champion.json
  comparisons/ # auto-generated comparison JSONs
```

Each version dir contains:
- `config.json` — LTR config + eval config
- `metrics.json` — per-group + aggregate metrics
- `meta.json` — version metadata

---

## Key Files

| File | Purpose |
|------|---------|
| `experiment-setup.md` | Full problem spec, data sources, features, evaluation design |
| `mem.md` | Working memory — results, findings, next steps |
| `audit.md` | 20-point integrity audit (leakage, data, target, eval, design) |
| `runbook.md` | This file — how to run everything |
| `ml/config.py` | Feature sets, monotone constraints, eval splits, data paths |
| `ml/benchmark.py` | Multi-group runner (train per year, eval per quarter) |
| `ml/evaluate.py` | All 13 metrics, aggregation logic |
| `ml/pipeline.py` | Train/eval workflow (expanding window) |
| `ml/data_loader.py` | V6.1 load, Spice6 enrichment, caching |
| `ml/ground_truth.py` | Ray-based realized DA fetch, mapping, caching |
| `ml/train.py` | LightGBM/XGBoost training, label transforms |
| `ml/features.py` | Feature matrix extraction, query group computation |
| `ml/compare.py` | Gate checking, comparison table, three-layer detail |
