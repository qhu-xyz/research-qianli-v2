# CLAUDE.md — research-pjm-signal-f0p-1 (PJM V7.1 — Expanded Universe)

Inherits rules from parent: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## What This Project Does

Build a PJM V7.1 constraint-tier signal with an **expanded constraint universe** sourced from raw spice6 density data (~3,100 branches) instead of V6.2B's filtered ~450 branches. The V6.2B universe misses 65% of binding constraints and 53% of binding value.

## Predecessor

`research-pjm-signal-f0p-0/` — PJM V7.0b (V6.2B universe, v0b formula champion).
Key finding: the V6.2B universe itself is the bottleneck, not the ML model.

## Data Sources

| Source | Path | What it provides |
|--------|------|------------------|
| Density score | `spice6/density/.../score.parquet` | Binding probability per constraint (~11,600 cids/month, ~22k with flow_dir) |
| Density distribution | `spice6/density/.../density.parquet` | 77-bin probability distribution (~11,200 cids) |
| Limit | `spice6/density/.../limit.parquet` | Physical thermal limits (~17,500 cids) |
| constraint_info | `spice6/constraint_info/.../` | constraint_id → branch_name mapping (~18k cids → ~4,800 branches) |
| ml_pred | `spice6/ml_pred/.../final_results.parquet` | ML predictions: binding_prob, predicted_sp, hist_da, prob_exceed_* (~12,900 rows, ~3,100 branches) |
| V6.2B signal | `signal_data/pjm/.../V6.2B.R1/` | Pre-filtered signal (~667 cids, ~475 branches) — used for comparison only |
| Realized DA cache | `research-pjm-signal-f0p-0/data/realized_da/` | Ground truth: branch_name → realized_sp (86 months, 2019-01 to 2026-02) |

### Data Availability
- Density: 106 months (2017-06 to 2026-03)
- ml_pred: 92 months (2018-06 to 2026-01), all 3 class types
- Realized DA: 86 months (2019-01 to 2026-02), all 3 class types
- constraint_info: stored under class_type=onpeak only (physical topology, class-invariant)

## Critical Rules (from f0p-0)

### Production Timing Lag
For period type fN, **lag = N + 1**. Signal for month M submitted ~mid(M-1). Latest complete DA is M-2.

### Branch-Level Join (NOT constraint_id)
DA joins on branch_name via constraint_info mapping. Naive constraint_id join captures only ~46% of value.

### PJM Timezone
PJM uses **US/Eastern** (not US/Central).

### LightGBM Threading
Always set `"num_threads": 4`. Container has 64 CPUs; auto-detect causes deadlocks.

### Memory Safety
Use polars over pandas. Free intermediates with `del df; gc.collect()`. Print `mem_mb()` at each stage.

## Aggregation Levels (Key Design Decision)

Raw data has multiple constraint_ids per branch (different contingencies). Options:
1. **Branch level** (~3,100 branches): deduplicate to one row per branch_name — same as f0p-0
2. **Constraint level** (~11,600 cids): keep all contingencies — risks correlated training examples
3. **Filtered branch** (~1,000-2,000): pre-filter branches with any signal (score > threshold or ml_pred > 0)

Decision TBD — see design doc.

## Metrics Design (Key Change from f0p-0)

Old f0p-0 metrics assumed a fixed universe (VC@20 = value in top-20 of ~450 constraints).
New metrics must handle variable-size universes and cross-universe comparison.

See `docs/design.md` for full metrics specification.

## Virtual Environment

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```
