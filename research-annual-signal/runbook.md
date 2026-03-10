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

### Key scripts

| Script | Description |
|--------|-------------|
| `run_v16_champion_analysis.py` | **Champion holdout analysis** (per-group, tail risk, gating) |
| `run_v15_multi_metric.py` | Multi-metric dev+holdout eval for 7 variants + blends |
| `run_v8_binding_freq.py` | Binding frequency experiments (v8a-v8d) |
| `run_v9_feature_eng.py` | Feature engineering experiments (v9a-v9j) |
| `run_v13_backfill_strategies.py` | Backfill strategy exploration |
| `run_v14_combined_signals.py` | Combined signal exploration |
| `run_v0_baseline.py` | Formula baseline (v0, v0b, v0c) |
| `run_v1_experiment.py`..`run_v5_experiment.py` | Early ML experiments (superseded) |

### Running the champion analysis
```bash
PYTHONPATH=. python scripts/run_v16_champion_analysis.py
```
Runs 4 variants on holdout (v0b, backfill+offpeak, backfill_lean, v10e), prints per-group metrics, tail risk, gating check, feature importance, and saves champion config.

---

## Caching

### Enriched data cache (`cache/enriched/`)
- V6.1 signal + spice6 density per (year, aq), cached as parquet
- Auto-created on first load, reused across versions
- Safe to delete and regenerate

### Ground truth cache (`cache/ground_truth/`)
- Realized DA shadow prices per (year, aq)
- Created by `scripts/cache_all_ground_truth.py` (requires Ray)
- Required for all experiment runs
- 28 parquet files (2019-2025, aq1-aq4)

### Binding frequency
- NOT cached on disk — computed on-the-fly from `research-stage5-tier/data/realized_da/`
- In-memory caches (_BRIDGE_CACHE, _BINDING_SETS_CACHE) reuse across groups within a run
- 107 months of data (2017-04 through 2026-02), onpeak + offpeak

---

## Registry

```
registry/
  champion.json            # -> v16_champion
  v16_champion/config.json # full champion config + holdout metrics
  v15_multi_metric/        # multi-metric comparison (all variants)
  gates.json               # quality gates (calibrated from v0)
  v0/..v5/                 # early experiments
  v7a/..v7d/               # ML rebase
  v8a/..v9j/               # BF + feature engineering
  v13../v14..               # exploration summaries
```

---

## Key Files

| File | Purpose |
|------|---------|
| `mem.md` | Working memory — champion results, findings, architecture |
| `runbook.md` | This file — how to run everything |
| `audit.md` | 20-point integrity audit |
| `ml/binding_freq.py` | Binding frequency module (onpeak, offpeak, decayed) |
| `ml/config.py` | Feature sets, monotone constraints, eval splits, data paths |
| `ml/evaluate.py` | All metrics (VC@K, Recall@K, NDCG, Spearman, Tier-AP) |
| `ml/data_loader.py` | V6.1 load, Spice6 enrichment, caching |
| `ml/ground_truth.py` | Realized DA fetch, MISO_SPICE_CONSTRAINT_INFO mapping |
| `ml/train.py` | LightGBM LambdaRank training, tiered labels |
| `ml/features.py` | Feature matrix extraction, query group computation |
| `ml/compare.py` | Gate checking, comparison table, three-layer detail |
