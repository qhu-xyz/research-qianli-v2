# PJM Banding Phase Plan

**Date:** 2026-03-16
**Baseline:** `mtm_1st_mean * 12` for all 4 rounds (FINAL — see knowledge.md)
**Method:** Asymmetric empirical quantile bands (same as MISO v10)

---

## Task 1: Write `pjm/scripts/run_v1_bands.py`

Adapt from `miso/scripts/run_v9_bands.py`. Key changes:

| Parameter | MISO | PJM |
|-----------|------|-----|
| `CLASSES` | `["onpeak", "offpeak"]` | `["onpeak", "dailyoffpeak", "wkndonpeak"]` |
| Rounds | 3 (loop over quarters per round) | 4 (single period_type `a` per round) |
| `MCP_COL` | `mcp_q` (quarterly) | `mcp_a` (annual) |
| Scale | `mcp_mean * 3`, `nodal_f0 * 3` | `mcp` (native annual), `mtm_1st_mean * 12` |
| Data source | `aq*_all_baselines.parquet` (R1), `all_residuals_v2.parquet` (R2/R3) | `pjm_annual_with_mcp.parquet` (all rounds) |
| R1 baseline | `nodal_f0 * 3` (needs lookup) | `mtm_1st_mean * 12` (in data) |
| Fallback | H baseline for missing nodal_f0 | None needed |
| `MIN_CELL_ROWS` | 500 | 500 (same — PJM has more paths per cell) |
| `N_BINS` | 5 | 5 |
| Dev PYs | 2020-2024 | 2017-2024 |
| Holdout | 2025 | 2025 |
| `MIN_TRAIN_PYS` | 2 | 2 |

Script structure:
```
Constants (PJM-specific)
Utility functions (reuse from MISO via import or copy)
Core: calibrate_asymmetric_per_class (same algorithm)
Core: apply_asymmetric_bands_per_class_fast (same algorithm)
Evaluation: evaluate_coverage, evaluate_per_class_coverage (same)
Experiment runner: run_experiment (adapted for single period_type)
Round runner: run_round (simplified — no quarter loop, just per-round)
Main: load data, run 4 rounds, save results
```

## Task 2: Write `pjm/scripts/test_v1_bands.py`

Per CLAUDE.md testing requirements (90%+ coverage):

1. Core calibration: 3 classes × 5 bins, quantile pairs at 8 levels
2. Band application: 16 columns, no nulls, containment (P99 ⊃ P95 ⊃ ... ⊃ P10)
3. Coverage monotonicity: P10 < P30 < ... < P99
4. Scale correctness: annual values (baseline × 12, widths annual)
5. Edge cases: zero baseline, single class, PY with missing classes (PY2017-2022 onpeak-only)
6. CP assignment: buy/sell monotonicity
7. Class parity: all 3 classes get bands
8. Temporal CV: train < test, min_train_pys respected
9. Artifact round-trip: save → load → apply → same results
10. Integration: output schema matches production contract

Run tests BEFORE committing results.

## Task 3: Dev Run

Run v1 for all 4 rounds with temporal expanding CV.

Config:
- Dev PYs: 2017-2024
- min_train_pys: 2
- n_bins: 5
- 8 coverage levels (P10-P99)
- Save to `pjm/versions/v1_dev/r{1-4}/`

Expected output per round:
- `metrics.json` — full monitoring grid
- `config.json` — method + parameters
- `calibration_artifact.json` — for inference

Report walltimes.

## Task 4: Dev Results Report

`pjm/docs/v1-dev-results-report.md`

**Two equally important metrics: coverage accuracy AND band width.**
Coverage tells us if we're capturing the right percentage. Width tells us if the bands are economically useful (narrower = better, as long as coverage holds).

### Coverage Analysis (is target reached?)

**Table 1: Overall coverage vs target — all 8 levels × 4 rounds**
```
Round | P10 err | P30 err | P50 err | P70 err | P80 err | P90 err | P95 err | P99 err
```
Flag any cell where |error| > 5pp.

**Table 2: Per-bin coverage at P95 — 5 bins × 4 rounds**
Find the weakest bin per round. q5 expected to under-cover.
```
Round | q1 | q2 | q3 | q4 | q5 | WEAKEST
```

**Table 3: Per-PY coverage at P95 — all PYs × 4 rounds**
Find the weakest year per round. PY2022 expected to be worst.
```
Round | PY2019 | PY2020 | ... | PY2024 | WEAKEST | RANGE
```

**Table 4: Per-class coverage at P95 — 3 classes × 4 rounds**
Find the weakest class per round. onpeak expected to be weakest.
```
Round | onpeak | dailyoffpeak | wkndonpeak | WEAKEST | GAP
```

**Table 5: Per-PY × Per-bin P95 breakdown (the full grid)**
For each round, a matrix of PY × bin showing P95 coverage.
Highlight any cell below 85%.

### Width Analysis (are bands economically useful?)

**Table 6: Width at all levels — all 8 levels × 4 rounds (annual $)**
```
Round | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99
```

**Table 7: Per-bin P95 width — 5 bins × 4 rounds**
Find the widest bin per round. q5 expected to dominate.
```
Round | q1 | q2 | q3 | q4 | q5 | q5/q1 ratio
```

**Table 8: Per-PY P95 width — all PYs × 4 rounds**
Find the widest year. Track if width is shrinking over time.
```
Round | PY2019 | ... | PY2024 | TREND
```

**Table 9: Per-class P95 width — 3 classes × 4 rounds**
```
Round | onpeak | dailyoffpeak | wkndonpeak | onpeak/wkndon ratio
```

### Combined Analysis

**Table 10: Coverage-Width tradeoff summary**
For each round: avg P95 coverage, avg P95 width, worst cell (coverage), widest cell (width).
```
Round | Avg Cov | Avg Width | Worst Cov Cell | Widest Cell
```

**Red flags:** Any cell where BOTH coverage < 90% AND width is in the top 20% — these are paths getting bad coverage despite wide bands.

### Comparison with MISO

**Table 11: PJM vs MISO side-by-side**
```
Metric | PJM R1 | MISO R1 | PJM R2 | MISO R2 | ...
```

## Task 5: Holdout Validation (PY 2025)

Config:
- All PYs: 2017-2025
- min_train_pys: 3
- Save to `pjm/versions/v1_holdout/r{1-4}/`

Compare dev vs holdout coverage. Success: PY 2025 P95 within 5pp of dev average.

## Task 6: Empirical Clearing Probabilities

Add empirical CPs to holdout artifacts:
- Segmented by trade_type (buy/sell) × flow_type (prevail/counter) × bin_group (q1-q4/q5)
- 8 keys per round (same as MISO)

## Execution Order

1. Write script → 2. Write tests → Run tests → 3. Dev run → 4. Report → 5. Holdout → 6. CPs → Commit all

Per CLAUDE.md: tests pass BEFORE committing results.
