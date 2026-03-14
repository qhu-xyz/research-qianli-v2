# V9 Simplified Asymmetric Bands — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a simplified asymmetric band model (5 quantile bins, per-class only, no sign split, no correction) that reduces band width by ~20-25% vs the current promoted v3.

**Architecture:** Single self-contained script that loads data, runs temporal CV on dev PYs, and reports a comprehensive monitoring grid. No dependency on other band scripts.

**Tech Stack:** Python 3.13, Polars 1.31

---

## Directional Shift

**Previous plan** (2026-03-13) combined repo reorganization (renaming v3→v2, archiving, import surgery) with the new v9 experiment. Six review issues remained unresolved.

**New direction:** Drop all reorganization. Focus exclusively on the v9 banding model. The current messy directory layout stays as-is. If v9 results are acceptable (width reduction + okay coverage), we reorganize later. If not, we iterate on the model first.

**Baseline is given.** R1 uses stitched nodal f0 (`nodal_f0`), R2/R3 use prior round MCP (`mtm_1st_mean`). This is not changing.

---

## Background

### Method: Asymmetric Per-Class Quantile Bands

For each path, compute `residual = mcp_actual - baseline`. Bin paths by `|baseline|` into 5 quantile bins. Within each (bin, class) cell, compute signed quantile pairs:

```
lo = quantile(residual, (1 - target_coverage) / 2)
hi = quantile(residual, (1 + target_coverage) / 2)
band = [baseline + lo, baseline + hi]
```

This is narrower than symmetric bands because it doesn't waste width on the thin tail of a skewed distribution.

**Cells:** 5 bins × 2 classes = 10 cells per quarter per round.
**Fallback:** If a cell has < 500 rows, fall back to bin-pooled (both classes combined). Log an explicit warning.

### Coverage Levels

8 levels (P99 is new vs prior scripts which stopped at P95):

| Target | Lower quantile | Upper quantile |
|--------|:-:|:-:|
| P10 | 0.450 | 0.550 |
| P30 | 0.350 | 0.650 |
| P50 | 0.250 | 0.750 |
| P70 | 0.150 | 0.850 |
| P80 | 0.100 | 0.900 |
| P90 | 0.050 | 0.950 |
| P95 | 0.025 | 0.975 |
| P99 | 0.005 | 0.995 |

### Dev / Holdout Split

| Round | All PYs | Dev PYs | Holdout | min_train_pys | Usable dev folds |
|-------|---------|---------|---------|:--:|:--:|
| R1 | 2020-2025 | 2020-2024 | 2025 | 2 | 3 (PY 2022, 2023, 2024) |
| R2 | 2019-2025 | 2019-2024 | 2025 | 2 | 4 (PY 2021, 2022, 2023, 2024) |
| R3 | 2019-2025 | 2019-2024 | 2025 | 2 | 4 (PY 2021, 2022, 2023, 2024) |

**min_train_pys=2 for dev** (lowered from 3) to get one more fold per round. This means PY 2022 for R1 is trained on only 2020-2021 — noisy but useful for seeing directional behavior. Production will have 5+ training PYs so cold-start noise is a dev-only issue.

### Data Paths

| Data | Path | Rows |
|------|------|------|
| R1 per-quarter | `/opt/temp/qianli/annual_research/crossproduct_work/aq{1-4}_all_baselines.parquet` | ~130-165K each |
| R2/R3 combined | `/opt/temp/qianli/annual_research/all_residuals_v2.parquet` | 11.5M total |

### Monitoring Grid

Report actual coverage for every cell in:

| Dimension | Values |
|-----------|--------|
| Round | R1, R2, R3 |
| Quarter | aq1, aq2, aq3, aq4 |
| PY | Each test PY individually |
| Bin | q1-q5 |
| Coverage level | P10, P30, P50, P70, P80, P90, P95, P99 |
| Class | onpeak, offpeak (parity gap) |

**Red flags to watch:**
- Per-bin P95/P99 coverage in q4/q5 (large |baseline|) — most likely to under-cover
- Any PY with P95 coverage < 88%
- Coverage non-monotonicity (P50 > P70 for same cell)
- Class parity gap > 3pp at P95
- R2/R3 coverage deviating from target by > 3pp (they have more data, should be tighter)

### Environment

All commands require the pmodel venv:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-annual-band
```

---

## Task 1: Write `scripts/run_v9_bands.py`

**Files:**
- Create: `scripts/run_v9_bands.py`
- Reference: `scripts/run_v9_bands.py` (current v9 = class+sign version, copy and simplify)

The new script is self-contained — no imports from other band scripts.

- [ ] **Step 1: Write the script**

Structure (single file, ~600-800 lines):

```
Constants:
  COVERAGE_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
  COVERAGE_LABELS = ["p10", "p30", "p50", "p70", "p80", "p90", "p95", "p99"]
  MIN_CELL_ROWS = 500
  CLASSES = ["onpeak", "offpeak"]
  QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
  DEV_R1_PYS = [2020, 2021, 2022, 2023, 2024]
  DEV_R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024]
  MIN_TRAIN_PYS = 2

Utility functions:
  mem_mb() -> float
  sanitize_for_json(obj) -> any
  assign_bins(abs_baseline, boundaries, labels) -> Series
  compute_quantile_boundaries(series, n_bins) -> (boundaries, labels)

Core functions:
  calibrate_asymmetric_per_class(df, baseline_col, mcp_col, class_col,
      boundaries, labels, coverage_levels) -> dict
    - For each (bin, class) cell: compute signed quantile pairs
    - Fallback: if n_cls < MIN_CELL_ROWS, use pooled + print WARNING
    - Raise ValueError if pooled also empty
    - Returns {bin_label: {class: {p10: (lo, hi), ...}, _pooled: {...}}}

  apply_asymmetric_bands_per_class_fast(df, bin_pairs, baseline_col,
      class_col, boundaries, labels) -> DataFrame
    - Build lookup table keyed by (bin, class)
    - Join + compute lower/upper columns
    - Raise ValueError if calibration data missing (no silent .get defaults)

Evaluation functions:
  evaluate_coverage(df, mcp_col, baseline_col, coverage_levels,
      boundaries, labels) -> dict
    - Overall and per-bin coverage at all 8 levels

  evaluate_per_class_coverage(df, mcp_col, class_col,
      coverage_levels) -> dict
    - Per-class coverage at all 8 levels

  evaluate_per_py_detail(df, mcp_col, baseline_col, class_col, py_col,
      coverage_levels, boundaries, labels) -> dict
    - Per-PY breakdown with per-bin coverage (the deep monitoring grid)

Experiment runner:
  run_experiment(df, quarter, pys, n_bins, baseline_col,
      cv_mode="temporal", min_train_pys=2) -> dict
    - Temporal expanding CV only (NO LOO)
    - Validate class_type at entry (raise ValueError, not assert)
    - For each test_py: calibrate on train, apply to test, evaluate
    - Aggregate: coverage + widths across filtered folds
    - Return per_py detail + aggregate

Round runner:
  run_round(round_num, baseline_col, data_loader, pys, n_bins,
      version_id) -> dict
    - Run experiment per quarter
    - Print comparison tables (vs promoted v3, vs v5)
    - Print per-class and per-bin coverage tables
    - Save metrics.json, config.json, calibration_artifact.json
    - NO subprocess calls to pipeline.py

Main:
  main()
    - R1: load aq{1-4}_all_baselines.parquet, baseline=nodal_f0
    - R2: load all_residuals_v2.parquet filtered to round==2, baseline=mtm_1st_mean
    - R3: same filtered to round==3
    - Use DEV PYs only (exclude 2025)
    - Print final summary with monitoring grid
```

**Code quality rules enforced:**
- `raise ValueError` for class_type validation (not assert)
- Explicit `print(f"WARNING: ...")` on every fallback
- No `.get(key, {})` on critical calibration data — raise ValueError
- All coverage levels including P99

- [ ] **Step 2: Verify script parses and is clean**

```bash
python -c "import ast; ast.parse(open('scripts/run_v9_bands.py').read()); print('Parse OK')"
# Must return nothing:
grep -n 'sign_seg\|SIGN_SEGS\|add_sign_seg' scripts/run_v9_bands.py
# Must return nothing:
grep -n 'loo\|LOO\|leave.*one.*out' scripts/run_v9_bands.py
# Must return nothing:
grep -n 'min_train_pys=3' scripts/run_v9_bands.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_v9_bands.py
git commit -m "add v9 simplified asymmetric band script (per-class only, dev PYs, 8 coverage levels)"
```

---

## Task 2: Write `scripts/test_v9_bands.py`

**Files:**
- Create: `scripts/test_v9_bands.py`

- [ ] **Step 1: Write tests**

Tests using synthetic data (no real data dependency):

1. **test_calibrate_output_shape**: 5 bins × 2 classes, all have quantile pairs at all 8 levels, no tuple keys (sign_seg)
2. **test_apply_produces_16_band_columns**: lower/upper × 8 levels = 16 columns added, no sign_seg column
3. **test_coverage_monotonicity**: P10 coverage < P30 < P50 < ... < P99 on synthetic data
4. **test_fallback_triggers_and_records**: small cell falls back to pooled, `_fallback_stats["to_pooled"] > 0`
5. **test_bad_class_type_raises**: class_type="peak" raises ValueError
6. **test_asymmetric_narrower_than_symmetric**: for skewed residuals, asymmetric total width < symmetric total width
7. **test_p99_quantiles_wider_than_p95**: P99 band contains P95 band

- [ ] **Step 2: Run tests**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts
python test_v9_bands.py
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_v9_bands.py
git commit -m "add v9 band tests: 7 tests covering calibration, apply, coverage, fallback, validation"
```

---

## Task 3: Run v9 on dev PYs

**Files:**
- Run: `scripts/run_v9_bands.py`
- Creates: `versions/bands/v9/r1/`, `v9/r2/`, `v9/r3/` (metrics.json, config.json, calibration_artifact.json)

- [ ] **Step 1: Run**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/run_v9_bands.py 2>&1 | tee /tmp/v9_dev_output.log
```

Expected runtime: ~2-5 minutes. Memory: < 10 GiB.

- [ ] **Step 2: Verify outputs exist**

```bash
for r in r1 r2 r3; do
    echo "v9/$r:"
    ls versions/bands/v9/$r/{config,metrics,calibration_artifact}.json 2>/dev/null && echo "  OK" || echo "  MISSING"
done
```

- [ ] **Step 3: Extract and review the monitoring grid**

From the script output (also in metrics.json), build the full reporting tables:

**Table A: Overall coverage accuracy (dev temporal, all 8 levels)**
```
Round | Quarter | P10 | P30 | P50 | P70 | P80 | P90 | P95 | P99
```

**Table B: Per-PY P95 coverage (stability check)**
```
Round | Quarter | PY2022 | PY2023 | PY2024 | Range | Worst
```

**Table C: Per-bin P95 coverage (bin calibration check)**
```
Round | Quarter | q1 | q2 | q3 | q4 | q5
```

**Table D: P95 half-width comparison vs promoted v3**
```
Round | Quarter | v3 width | v9 width | Δ%
```

**Table E: Class parity at P95**
```
Round | Quarter | onpeak | offpeak | gap
```

- [ ] **Step 4: Write NOTES.md for v9/r1, v9/r2, v9/r3**

Include Tables A-E with actual numbers. Follow format of existing NOTES.md files.

- [ ] **Step 5: Commit**

```bash
git add versions/bands/v9/
git commit -m "run v9 dev: simplified asymmetric bands, per-class only, 8 coverage levels"
```

---

## Task 4: Review and decide

**NOT a code task.** After Task 3, present results to user for review.

- [ ] **Step 1: Present monitoring grid tables A-E**

Flag any red flags:
- Per-bin P95/P99 undershoot in q4/q5
- PY-to-PY spread > 10pp at P95
- Coverage non-monotonicity
- Class parity gap > 3pp
- R2/R3 deviating > 3pp from targets

- [ ] **Step 2: Get user decision**

Options:
- **Accept**: proceed to holdout validation (Task 5)
- **Iterate**: adjust bins, MIN_CELL_ROWS, or coverage method
- **Reject**: try different approach

---

## Task 5: Holdout validation (only after user approval)

**Files:**
- Modify: `scripts/run_v9_bands.py` (change PYs to include 2025, set min_train_pys=3)
- Creates: `versions/bands/v9_holdout/r1/`, etc.

- [ ] **Step 1: Run on full PYs including holdout**

Change constants to:
```python
R1_PYS = [2020, 2021, 2022, 2023, 2024, 2025]
R2R3_PYS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MIN_TRAIN_PYS = 3  # back to 3 for production-relevant eval
```

Save to `v9_holdout/` to keep dev results intact.

- [ ] **Step 2: Report holdout results**

Same Tables A-E but now including PY 2025 fold.
Compare dev vs holdout coverage — if holdout is much worse, the model may be overfitting to dev.

- [ ] **Step 3: Commit**

```bash
git add versions/bands/v9_holdout/
git commit -m "v9 holdout validation: full PYs including 2025"
```
