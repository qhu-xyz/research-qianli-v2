# Band Version Reorganization + New Simplified Asymmetric Version

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the band versions directory into a uniform `v{N}/r{M}` layout (archiving old v2 and duplicate r-first layout), then implement and run a new simplified asymmetric band model (asymmetric, 5 bins, per-class only, no sign split, no correction).

**Architecture:** Two-phase project. Phase 1 renames existing directories and scripts (no re-execution). Phase 2 writes and runs a new band experiment script. Both phases are independent — Phase 2 can start from v9's code but references the new version numbers from Phase 1.

**Tech Stack:** Python 3.13, Polars 1.31, pipeline.py CLI for validation

---

## Background for Reviewers

### What is this repo?

This is `research-annual-band`, a research repo for calibrating **pricing bands** around MISO FTR (Financial Transmission Rights) annual auction clearing prices. The bands represent confidence intervals: "we expect the actual MCP (market clearing price) to fall within `[baseline ± width]` with P95 probability."

### RTO and Auction Structure

- **RTO:** MISO (Midcontinent ISO)
- **Auction type:** Annual quarterly — 3 rounds, each bidding 4 quarters (aq1-aq4)
- **Planning year:** June–May (PY 2024 = Jun 2024 – May 2025)
- **Class types:** `onpeak`, `offpeak` (both traded in annual)
- **Data:** ~130-165K paths per quarter for R1, ~340-700K for R2/R3

### Baseline Strategy

| Round | Baseline Column | What It Is |
|-------|----------------|------------|
| R1 | `nodal_f0` | Stitched nodal f0 forward prices (sink_f0 - source_f0 averaged over 3 delivery months). No prior round exists. |
| R2 | `mtm_1st_mean` | R1's clearing price for same path (known at R2 bid time) |
| R3 | `mtm_1st_mean` | R2's clearing price for same path (known at R3 bid time) |

### How Bands Work

For each path, we compute `residual = mcp_actual - baseline`. We bin paths by `|baseline|` into quantile bins, then compute empirical quantiles of the residuals within each bin to set band widths.

- **Symmetric (current promoted v3):** `width = quantile(|residual|, 0.95)` → band = `[baseline - width, baseline + width]`
- **Asymmetric (proposed):** `lo = quantile(residual, 0.025)`, `hi = quantile(residual, 0.975)` → band = `[baseline + lo, baseline + hi]`. Narrower because it doesn't waste width on the thin tail of a skewed distribution.

### Cross-Validation

- **Temporal expanding:** Train on PYs strictly before test PY. **This is the ONLY CV method used.** All rounds use temporal — no LOO.
- **min_train_pys=3:** Exclude temporal folds with <3 training PYs from aggregate metrics (cold-start folds are unreliable).
- **Hold-out years:** Reserved for **final validation only**, not for experiment iteration. Use all available PYs during temporal CV for development. Hold-out is a one-shot gate at the end.

### Round Isolation

Each round is trained and evaluated independently — R1 only uses R1 data, R2 only R2, etc. No cross-round data mixing.

### Current Directory Problem

Two coexisting layouts:
- **v-first** (`versions/bands/v{N}/r{M}/`): v1-v8, used by scripts v5-v9
- **r-first** (`versions/bands/r{M}/v{N}/`): r1/v1-v3, r2/v1-v2, r3/v1-v2

The r-first layout is an exact duplicate. Old v2 (width reduction via quantile bins) was R1-only and was absorbed into v3. Six directories (v6, v7) lack NOTES.md.

### Version Mapping (Old → New)

| Old | New | Method | Rounds |
|-----|-----|--------|:---:|
| v1 | v1 | Symmetric, fixed 4 bins, pooled | r1,r2,r3 |
| v2 | *archived* | Width reduction (R1-only, absorbed into v3) | r1 only |
| v3 | v2 | Symmetric, quantile bins, per-class (**promoted**) | r1,r2,r3 |
| v4 | v3 | Symmetric, quantile bins, class+sign | r1,r2,r3 |
| v5 | v4 | Asymmetric, 8-bin, class+sign | r1,r2,r3 |
| v6 | v5 | Asymmetric, 6-bin, correction experiments | r1,r2,r3 |
| v7 | v6 | Asymmetric, 8-bin, bidir + temporal | r1,r2,r3 |
| v8 | v7 | Asymmetric, 5-bin, bidir correction | r1,r2,r3 |
| v9 (script only) | v8 | Asymmetric, 5-bin, class+sign, no correction | needs run |
| *new* | v9 | **Asymmetric, 5-bin, per-class only, no correction** | needs run |

### New v9 Design Rationale

The promoted v3 (now v2) is symmetric and too wide. Asymmetric v5 (now v4) achieved 21-32% width reduction but with 32 cells (8 bins × 2 classes × 2 signs), causing noisy quantile estimates and unstable coverage.

v9 simplifies to **5 bins × 2 classes = 10 cells**:
- **Asymmetric** signed quantile pairs (the main source of width reduction)
- **5 quantile bins** (stable with 6 PYs of R1 data)
- **Per-class only** (no sign split — sign was unstable, v4/R2 winner was symmetric)
- **No correction** (bidirectional correction added complexity but barely helped)
- **Temporal CV, min_train_pys=3**

Expected: ~20-25% narrower bands vs promoted, ~92-93% P95 coverage.

### Migration Target: pmodel ftr24/v1

The finalized annual band version must eventually be migrated into `pmodel/src/pmodel/base/ftr24/v1/` alongside the existing monthly `band_generator.py`. Key differences from the monthly pipeline:

| Aspect | Monthly (f0p) | Annual |
|--------|--------------|--------|
| Method | LightGBM ML + empirical clearing probs | Empirical quantile bands (no ML) |
| Entry point | `generate_bands(df)` → bid points + clearing probs | TBD — needs `generate_annual_bands(df)` API |
| Training data | Pre-saved parquet per month/ptype/class | Per-round historical auction data |
| Band type | Symmetric width around baseline | Asymmetric signed quantile pairs |
| Output | `bid_price_1..10`, `clearing_prob_1..10` | `lower_p50..p95`, `upper_p50..p95` |

**Migration gaps to track during research:**
1. **Data serialization:** The research script loads from `/opt/temp/qianli/annual_research/`. Production needs either pre-computed band widths saved as artifacts or a self-contained calibration step.
2. **API contract:** `generate_annual_bands(df, round, class_type)` must accept the same DataFrame schema as the monthly pipeline (trades with `source_id`, `sink_id`, `mtm_1st_mean`, etc.).
3. **Band width artifacts:** Save calibrated (bin_boundaries, per-cell quantile pairs) as JSON/parquet artifacts that can be loaded at inference time without re-running calibration.
4. **Inference vs calibration separation:** Research script does both. Production needs: (a) a calibration script run offline to produce artifacts, (b) an inference function that applies pre-calibrated bands to new trades.

### Code Quality Rules (from CLAUDE.md)

- **Always use `uv` to run Python** — not system python
- **No silent fallbacks** — if a cell has < MIN_CELL_ROWS, log an explicit warning with counts, don't silently fall back
- **Never confuse onpeak/offpeak** — always validate class_type is one of `["onpeak", "offpeak"]` at function boundaries
- **No defaults that hide bugs** — don't use `default=` in `.get()` calls for critical fields; raise ValueError if missing

### Data Paths

| Data | Path |
|------|------|
| R1 per-quarter | `/opt/temp/qianli/annual_research/crossproduct_work/aq{1-4}_all_baselines.parquet` |
| R2/R3 combined | `/opt/temp/qianli/annual_research/all_residuals_v2.parquet` |

### Script Import Dependencies

```
run_phase3_bands.py ← base functions (assign_bins, calibrate_bin_widths, etc.)
    ↑
run_phase3_v2_bands.py ← quantile boundaries, LOO/temporal calibration
    ↑                  ↑
run_v3_bands.py    run_r2r3_bands.py
    ↑
run_v4_bands.py

run_v5_bands.py through run_v9_bands.py ← self-contained (no cross-imports)
```

Scripts v3 and v4 (old names) import from `run_phase3_v2_bands.py` (being archived) and `run_v3_bands.py` (being renamed). Shared functions must be extracted first.

Analysis scripts (`analyze_segments.py`, `analyze_sign_segments.py`, `investigate_sign_split.py`) also import from these files.

---

## Phase 1: Repository Reorganization

### Task 1: Extract shared functions into `scripts/band_utils.py`

**Files:**
- Create: `scripts/band_utils.py`
- Read: `scripts/run_phase3_bands.py` (lines 30-260)
- Read: `scripts/run_phase3_v2_bands.py` (lines 86-220)
- Read: `scripts/run_v3_bands.py` (lines 60-200)

**Why:** `run_phase3_v2_bands.py` is being archived but its functions are imported by 6 other scripts. Extract shared functions into a utility module before archiving.

- [ ] **Step 1: Create `scripts/band_utils.py`**

Extract these functions from `run_phase3_bands.py`:
- `assign_bins(abs_baseline, boundaries, labels)` → Series
- `calibrate_bin_widths(df, boundaries, labels, ...)` → dict
- `apply_bands(df, bin_widths, baseline_col, boundaries, labels)` → DataFrame
- `evaluate_coverage(df, mcp_col, baseline_col, ...)` → dict
- `sanitize_for_json(obj)` → any
- `mem_mb()` → float
- Constants: `COVERAGE_LEVELS`, `COVERAGE_LABELS`, `BASELINE_COL`, `MCP_COL`, `PY_COL`

Extract these functions from `run_phase3_v2_bands.py`:
- `compute_quantile_boundaries(series, n_bins)` → (boundaries, labels)
- `loo_band_calibration_quantile(df, n_bins, ...)` → dict
- `temporal_band_calibration_quantile(df, n_bins, ...)` → dict

Extract these functions from `run_v3_bands.py`:
- `calibrate_bin_widths_per_class(df, boundaries, labels, ...)` → dict
- `apply_bands_per_class_fast(df, bin_widths, baseline_col, ...)` → DataFrame
- `evaluate_per_class_coverage(df, mcp_col, class_col, ...)` → dict
- `loo_per_class_quantile(df, n_bins, ...)` → dict
- `temporal_per_class_quantile(df, n_bins, ...)` → dict

The new file should import polars and math at top. Copy function bodies exactly — no refactoring.

- [ ] **Step 2: Verify `band_utils.py` parses**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python -c "import ast; ast.parse(open('/home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/band_utils.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/band_utils.py
git commit -m "extract shared band calibration functions into band_utils.py"
```

### Task 2: Update imports in dependent scripts

**Files:**
- Modify: `scripts/run_v3_bands.py` (import lines ~38-53)
- Modify: `scripts/run_v4_bands.py` (import lines ~41-62)
- Modify: `scripts/run_r2r3_bands.py` (import lines ~35-51)
- Modify: `scripts/analyze_segments.py` (import lines)
- Modify: `scripts/analyze_sign_segments.py` (import lines)
- Modify: `scripts/investigate_sign_split.py` (import lines)
- Modify: `scripts/test_sign_split_calibration.py` (import lines)

- [ ] **Step 1: Update imports in all 7 scripts**

For each script, replace:
```python
from run_phase3_bands import (
    assign_bins, calibrate_bin_widths, apply_bands, evaluate_coverage,
    sanitize_for_json, mem_mb, COVERAGE_LEVELS, COVERAGE_LABELS,
)
from run_phase3_v2_bands import (
    compute_quantile_boundaries, loo_band_calibration_quantile,
    temporal_band_calibration_quantile,
)
```
With:
```python
from band_utils import (
    assign_bins, calibrate_bin_widths, apply_bands, evaluate_coverage,
    sanitize_for_json, mem_mb, COVERAGE_LEVELS, COVERAGE_LABELS,
    compute_quantile_boundaries, loo_band_calibration_quantile,
    temporal_band_calibration_quantile,
)
```

For `run_v4_bands.py`, also replace `from run_v3_bands import (...)` with `from band_utils import (...)`.

**Note:** `test_annual_replication.py` import update is deferred to Task 5 (depends on Task 4 rename).

- [ ] **Step 2: Verify all scripts parse**

Run:
```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-band
for f in scripts/run_v3_bands.py scripts/run_v4_bands.py scripts/run_r2r3_bands.py scripts/analyze_segments.py scripts/analyze_sign_segments.py scripts/investigate_sign_split.py scripts/test_sign_split_calibration.py; do
    python -c "import ast; ast.parse(open('$f').read())" && echo "OK: $f" || echo "FAIL: $f"
done
```

Expected: All OK.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_v3_bands.py scripts/run_v4_bands.py scripts/run_r2r3_bands.py scripts/analyze_segments.py scripts/analyze_sign_segments.py scripts/investigate_sign_split.py scripts/test_sign_split_calibration.py
git commit -m "update imports to use band_utils instead of archived scripts"
```

### Task 3: Generate missing NOTES.md for v6 and v7

**Files:**
- Create: `versions/bands/v6/r1/NOTES.md`, `v6/r2/NOTES.md`, `v6/r3/NOTES.md`
- Create: `versions/bands/v7/r1/NOTES.md`, `v7/r2/NOTES.md`, `v7/r3/NOTES.md`
- Read: `scripts/run_v6_bands.py` (docstring lines 1-20)
- Read: `scripts/run_v7_bands.py` (docstring lines 1-20)
- Read: `versions/bands/v6/r1/config.json`, `v7/r1/config.json`
- Read: `versions/bands/v6/r1/metrics.json`, `v7/r1/metrics.json` (for coverage/width numbers)

**Template (follow v8/r1/NOTES.md format):**

```markdown
# Bands v{N}/r{M} — [Description from config.json]

**Date:** [from config.created]  |  **Script:** `scripts/run_v{N}_bands.py`

## Method
[From config.json method fields + script docstring]

## Coverage Accuracy (temporal, P95)
[Extract from metrics.json temporal_validation.aq{1-4}.aggregate_coverage.p95]

## P95 Mean Half-Width ($/MWh)
[Extract from metrics.json temporal_validation.aq{1-4}.aggregate_widths.p95.mean_width]

## Gate Results
[If gate_results in metrics, list them. If not, note "Gates not evaluated in metrics.json"]
```

- [ ] **Step 1: Generate all 6 NOTES.md files**

Extract data from config.json and metrics.json for each directory. Use the script docstring for method description.

- [ ] **Step 2: Commit**

```bash
git add versions/bands/v6/*/NOTES.md versions/bands/v7/*/NOTES.md
git commit -m "add missing NOTES.md for v6 and v7 band versions"
```

### Task 4: Archive old v2 and r-first layout, rename directories and scripts

**Files:**
- Move: `versions/bands/v2/` → `archive/versions_v2_width_reduction/`
- Move: `versions/bands/r1/`, `r2/`, `r3/` → `archive/versions_r_layout/`
- Move: `scripts/run_phase3_v2_bands.py` → `archive/versions_v2_width_reduction/`
- Rename: `versions/bands/v3` → `v2`, `v4` → `v3`, ..., `v8` → `v7`
- Rename: scripts `run_v3_bands.py` → `run_v2_bands.py`, etc.
- Rename: test scripts similarly

- [ ] **Step 1: Archive old v2 and r-first layout**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-band
mkdir -p archive/versions_v2_width_reduction archive/versions_r_layout
mv versions/bands/v2/ archive/versions_v2_width_reduction/
mv versions/bands/r1/ archive/versions_r_layout/
mv versions/bands/r2/ archive/versions_r_layout/
mv versions/bands/r3/ archive/versions_r_layout/
mv scripts/run_phase3_v2_bands.py archive/versions_v2_width_reduction/
```

- [ ] **Step 2: Rename version directories (forward order)**

```bash
# v2 archived, so v3→v2 is safe. Each subsequent mv vacates the source.
mv versions/bands/v3 versions/bands/v2
mv versions/bands/v4 versions/bands/v3
mv versions/bands/v5 versions/bands/v4
mv versions/bands/v6 versions/bands/v5
mv versions/bands/v7 versions/bands/v6
mv versions/bands/v8 versions/bands/v7
```

- [ ] **Step 3: Rename experiment scripts**

```bash
cd scripts
mv run_v3_bands.py run_v2_bands.py
mv run_v4_bands.py run_v3_bands.py
mv run_v5_bands.py run_v4_bands.py
mv run_v6_bands.py run_v5_bands.py
mv run_v7_bands.py run_v6_bands.py
mv run_v8_bands.py run_v7_bands.py
mv run_v9_bands.py run_v8_bands.py
```

- [ ] **Step 4: Rename test scripts**

```bash
mv test_v5_bands.py test_v4_bands.py
mv test_v6_bands.py test_v5_bands.py
mv test_v7_bands.py test_v6_bands.py
mv test_v8_bands.py test_v7_bands.py
mv test_v9_bands.py test_v8_bands.py
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "archive old v2 and r-first layout, renumber v3-v9 to v2-v8"
```

### Task 5: Update version references in config.json, NOTES.md, scripts

**Files:**
- Modify: All `versions/bands/v{N}/r{M}/config.json` (22 files)
- Modify: All `versions/bands/v{N}/r{M}/NOTES.md` (22 files)
- Modify: All `scripts/run_v{N}_bands.py` (7 files)
- Modify: All `scripts/test_v{N}_bands.py` (5 files)
- Modify: `scripts/test_5bins.py`, `scripts/test_annual_replication.py`
- Modify: `scripts/run_r2r3_bands.py`, `scripts/run_phase3_bands.py`

The full version mapping for search-and-replace:

| Context | Old reference | New reference |
|---------|--------------|---------------|
| config.json `"version"` | `"v3"` (in old v3 dirs) | `"v2"` |
| config.json `"part"` | `"bands/r1"` etc. | `"bands"` |
| config.json `"round"` | missing in some | add `"r{M}"` |
| NOTES.md titles | "R2 v3" | "v2/r2" |
| NOTES.md cross-refs | "vs v3", "v4 width" | "vs v2", "v3 width" |
| Script version_id | `version_id="v5"` | `version_id="v4"` |
| Script prior_version | `"bands/v4/r1"` | `"bands/v3/r1"` |
| Script ref paths | `"v3 (promoted)"` | `"v2 (promoted)"` |
| Script docstrings | `"Bands v5"` | `"Bands v4"` |
| Test imports | `from run_v7_bands` | `from run_v6_bands` |

**Important:** `baseline_version: "v3"` refers to the BASELINE namespace (not bands) and must NOT be changed.

- [ ] **Step 1: Update all config.json files**

For each `versions/bands/v{N}/r{M}/config.json`:
- Set `"version"` to current directory version name
- Set `"part"` to `"bands"` (remove round from part)
- Ensure `"round"` field exists and is correct
- Update `method.winner`, `method.experiments_tested` if they reference old version names
- Do NOT change `baseline_version`

- [ ] **Step 2: Update all NOTES.md files**

Apply version mapping to titles and cross-references. Use `sed` or manual edits.

- [ ] **Step 3: Update all scripts**

For each renamed script (`run_v2_bands.py` through `run_v8_bands.py`):
- Update docstring version number
- Update `version_id=` in main()
- Update `prior_version_part=` paths
- Update reference comparison paths (e.g., `"v3 (promoted)"` → `"v2 (promoted)"`)

For `run_r2r3_bands.py`: update any version paths in its main().
For test scripts: update `from run_vN_bands import` to use new name.
For `test_5bins.py`: update `from run_v7_bands import` → `from run_v6_bands import`.
For `test_annual_replication.py`: update `from run_v9_bands import` → `from run_v8_bands import`.

- [ ] **Step 4: Update promoted.json**

Write `versions/bands/promoted.json`:
```json
{
  "r1": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"},
  "r2": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"},
  "r3": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"}
}
```

- [ ] **Step 5: Update documentation**

Update version references in:
- `mem.md` (repo root)
- `runbook.md` (repo root) — §15 band calibration
- `segment_analysis.md` (repo root) — "v3 is working well" → "v2"
- `versions/bands/bg2_gate_revision_proposal.md` — v4/v5 → v3/v4
- `design-planning.md` (repo root) — minimal updates if it references band version numbers

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "update all version references after renumbering v3-v9 to v2-v8"
```

### Task 6: Verify reorganization

- [ ] **Step 1: Verify all JSON files are valid**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-band
find versions/bands -name "*.json" -exec python -c "import json,sys; json.load(open(sys.argv[1])); print('OK:', sys.argv[1])" {} \;
```

- [ ] **Step 2: Verify all scripts parse**

```bash
for f in scripts/run_*.py scripts/test_*.py; do
    python -c "import ast; ast.parse(open('$f').read())" && echo "OK: $f" || echo "FAIL: $f"
done
```

- [ ] **Step 3: Verify no stale version references**

```bash
# Old v3 bands references (should only appear in baseline_version and archived files)
grep -rn '"v3"' versions/bands/ scripts/ | grep -v baseline_version | grep -v archive/
# Old v4-v8 band references (should not appear)
grep -rn 'bands/v[4-9]' versions/bands/ scripts/
# Old v9 references
grep -rn 'v9' versions/bands/ scripts/
# r-first layout remnants
grep -rn 'bands/r[123]/v' versions/bands/ scripts/
```

Expected: No matches (or only in comments/docstrings that describe history).

- [ ] **Step 4: Verify imports actually resolve (not just parse)**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts
for f in run_v2_bands run_v3_bands run_v4_bands run_v5_bands run_v6_bands run_v7_bands run_v8_bands run_r2r3_bands; do
    python -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', '${f}.py'); print(f'OK: ${f}')" 2>&1 || echo "FAIL: $f"
done
```

- [ ] **Step 5: Verify directory structure is uniform**

```bash
for v in versions/bands/v*/; do
    echo "$v: $(ls -d ${v}r*/ 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"
done
```

Expected: v1-v7 each have r1 r2 r3. v8 does not exist yet (created in Phase 2).

- [ ] **Step 5: Run pipeline validate (if available)**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /home/xyz/workspace/research-qianli-v2/research-annual-band/pipeline/pipeline.py list bands
python /home/xyz/workspace/research-qianli-v2/research-annual-band/pipeline/pipeline.py validate bands
```

- [ ] **Step 6: Commit any fixes**

If any verification step reveals issues, fix them and commit.

---

## Phase 2: Run old v8 (renumbered from v9) and create new v9

### Task 7: Run old v8 (formerly v9 script, class+sign, no correction)

**Files:**
- Run: `scripts/run_v8_bands.py`
- Creates: `versions/bands/v8/r1/`, `v8/r2/`, `v8/r3/` (config.json, metrics.json)

This script already exists but was never run. It will create v8 results.

- [ ] **Step 1: Update v8 script version references**

The script (formerly run_v9_bands.py) references old version numbers. Verify that Task 5 already updated:
- `version_id="v8"` (was `"v9"`)
- `prior_version_part="bands/v7/r1"` (was `"bands/v8/r1"`)
- Reference comparisons use new version numbers

- [ ] **Step 2: Run v8**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/run_v8_bands.py
```

Expected runtime: ~2-5 minutes. Expected output: metrics.json and config.json in v8/r1, v8/r2, v8/r3.

- [ ] **Step 3: Write NOTES.md for v8/r1, v8/r2, v8/r3**

From the script output, capture coverage and width numbers. Follow the template from Task 3.

- [ ] **Step 4: Commit**

```bash
git add versions/bands/v8/
git commit -m "run v8 band experiment (asymmetric, 5-bin, class+sign, no correction)"
```

### Task 8: Create new v9 script (simplified asymmetric, per-class only)

**Files:**
- Create: `scripts/run_v9_bands.py`
- Reference: `scripts/run_v8_bands.py` (copy and simplify)

The new v9 script is a simplification of v8. Key changes:
1. **Remove sign stratification:** No `sign_seg` column, no `(cls, seg)` cell keys
2. **Calibration cells:** 5 bins × 2 classes = 10 cells (vs v8's 5 × 2 × 2 = 20)
3. **Fallback chain:** `(bin, class) → (bin, pooled)` (no sign level)
4. **Remove sign-related evaluation** (per_sign_coverage, widths_by_sign)

- [ ] **Step 1: Create `scripts/run_v9_bands.py`**

Copy `run_v8_bands.py` and make these changes:

**Constants section:**
- Remove: `SIGN_SEGS = ["prevail", "counter"]`
- Update docstring: "Bands v9: Simplified Asymmetric (per-class only, no sign split)"

**Remove functions:**
- `add_sign_seg()` — not needed
- `evaluate_per_sign_coverage()` — not needed

**Simplify `calibrate_asymmetric_per_class` (rename from `calibrate_asymmetric_per_class_sign`):**

```python
def calibrate_asymmetric_per_class(
    df: pl.DataFrame,
    baseline_col: str,
    mcp_col: str = MCP_COL,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
    coverage_levels: list[float] = COVERAGE_LEVELS,
) -> dict[str, dict]:
    """Per-(bin, class) signed quantile pairs. No sign stratification.

    Returns {bin_label: {class: {p50: (lo, hi), ...}, _pooled: {...}}}.
    Fallback: (bin, class) -> (bin, _pooled).
    """
    residual = df[mcp_col] - df[baseline_col]
    bins = assign_bins(df[baseline_col].abs(), boundaries, labels)

    work = pl.DataFrame({
        "residual": residual,
        "bin": bins,
        "class_type": df[class_col],
    })

    result = {}
    fallback_stats = {"total": 0, "to_pooled": 0}

    for label in labels:
        bin_data = work.filter(pl.col("bin") == label)

        # Pooled estimate (fallback)
        pooled_subset = bin_data["residual"]
        n_pooled = len(pooled_subset)
        pooled_pairs = {}
        for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
            if n_pooled > 0:
                lo = round(float(pooled_subset.quantile((1 - level) / 2)), 1)
                hi = round(float(pooled_subset.quantile((1 + level) / 2)), 1)
                pooled_pairs[clabel] = (lo, hi)
            else:
                pooled_pairs[clabel] = None
        pooled_pairs["n"] = n_pooled

        cell_pairs = {"_pooled": pooled_pairs}

        # Per-class estimate
        for cls in CLASSES:
            fallback_stats["total"] += 1
            cls_subset = bin_data.filter(pl.col("class_type") == cls)["residual"]
            n_cls = len(cls_subset)

            if n_cls >= MIN_CELL_ROWS:
                pairs = {}
                for level, clabel in zip(coverage_levels, COVERAGE_LABELS):
                    lo = round(float(cls_subset.quantile((1 - level) / 2)), 1)
                    hi = round(float(cls_subset.quantile((1 + level) / 2)), 1)
                    pairs[clabel] = (lo, hi)
                pairs["n"] = n_cls
            else:
                print(f"  WARNING: Cell ({label}, {cls}) has {n_cls} rows < {MIN_CELL_ROWS}, "
                      f"falling back to pooled ({n_pooled} rows)")
                if n_pooled == 0:
                    raise ValueError(f"Cell ({label}, {cls}): both class ({n_cls}) and pooled ({n_pooled}) have insufficient rows")
                pairs = dict(pooled_pairs)
                pairs["n"] = n_cls
                pairs["_fallback"] = "pooled"
                fallback_stats["to_pooled"] += 1
            cell_pairs[cls] = pairs

        result[label] = cell_pairs

    result["_fallback_stats"] = fallback_stats
    return result
```

**Simplify `apply_asymmetric_bands_per_class_fast` (rename from `..._per_class_sign_fast`):**

```python
def apply_asymmetric_bands_per_class_fast(
    df: pl.DataFrame,
    bin_pairs: dict[str, dict],
    baseline_col: str,
    class_col: str = CLASS_COL,
    boundaries: list[float] | None = None,
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Vectorized asymmetric band application using join on (bin, class)."""
    abs_bl = df[baseline_col].abs()
    bins = assign_bins(abs_bl, boundaries, labels)

    if "_bin" in df.columns:
        df = df.drop("_bin")
    df = df.with_columns(pl.Series("_bin", bins))

    rows = []
    for bin_label in labels:
        cell = bin_pairs.get(bin_label, {})
        for cls in CLASSES:
            entry = {"_bin": bin_label, CLASS_COL: cls}
            if cls in cell:
                data = cell[cls]
            elif "_pooled" in cell:
                data = cell["_pooled"]
            else:
                raise ValueError(f"No calibration data for bin={bin_label}, class={cls}")
            for clabel in COVERAGE_LABELS:
                lo_hi = data.get(clabel)
                if isinstance(lo_hi, (list, tuple)):
                    entry[f"_lo_{clabel}"] = lo_hi[0]
                    entry[f"_hi_{clabel}"] = lo_hi[1]
                else:
                    entry[f"_lo_{clabel}"] = None
                    entry[f"_hi_{clabel}"] = None
            rows.append(entry)

    lookup = pl.DataFrame(rows)
    df = df.join(lookup, on=["_bin", CLASS_COL], how="left")

    for clabel in COVERAGE_LABELS:
        df = df.with_columns([
            (pl.col(baseline_col) + pl.col(f"_lo_{clabel}")).alias(f"lower_{clabel}"),
            (pl.col(baseline_col) + pl.col(f"_hi_{clabel}")).alias(f"upper_{clabel}"),
        ])

    drop_cols = ["_bin"]
    drop_cols += [f"_lo_{clabel}" for clabel in COVERAGE_LABELS]
    drop_cols += [f"_hi_{clabel}" for clabel in COVERAGE_LABELS]
    return df.drop(drop_cols)
```

**Simplify `run_experiment`:**
- Remove all `add_sign_seg()` calls
- Remove `sign_seg` from work DataFrame
- Call `calibrate_asymmetric_per_class` (not `..._per_class_sign`)
- Call `apply_asymmetric_bands_per_class_fast` (not `..._per_class_sign_fast`)
- Remove `evaluate_per_sign_coverage` calls
- Remove `widths_by_sign` computation (the nested loop over `SIGN_SEGS`)
- Width summary: iterate over bins × classes only (not signs)
- **Remove LOO secondary validation entirely** — only run temporal CV
- **No silent fallbacks:** When a cell falls back to pooled, log: `logger.warning(f"Cell ({bin_label}, {cls}) has {n_cls} rows < {MIN_CELL_ROWS}, falling back to pooled ({n_pooled} rows)")`
- **Validate class_type at entry (raise, not assert):**
```python
actual_classes = set(df[CLASS_COL].unique().to_list())
if not actual_classes <= {"onpeak", "offpeak"}:
    raise ValueError(f"Unexpected class_type values: {actual_classes - {'onpeak', 'offpeak'}}")
```

```python
# Width summary (simplified — no sign dimension)
width_summary = {}
for clabel in COVERAGE_LABELS:
    half_widths = []
    for label in bin_labels:
        for cls in CLASSES:
            lo_hi = bin_pairs[label].get(cls, {}).get(clabel)
            if isinstance(lo_hi, (list, tuple)):
                half_widths.append((lo_hi[1] - lo_hi[0]) / 2)
    width_summary[clabel] = {
        "mean_width": round(sum(half_widths) / len(half_widths), 1) if half_widths else None,
        "per_bin": {},
    }
    for label in bin_labels:
        cls_hw = {}
        for cls in CLASSES:
            lo_hi = bin_pairs[label].get(cls, {}).get(clabel)
            if isinstance(lo_hi, (list, tuple)):
                cls_hw[cls] = round((lo_hi[1] - lo_hi[0]) / 2, 1)
            else:
                cls_hw[cls] = None
        vals = [v for v in cls_hw.values() if v is not None]
        cls_hw["avg"] = round(sum(vals) / len(vals), 1) if vals else None
        width_summary[clabel]["per_bin"][label] = cls_hw
```

**Update `run_round_v9` → `run_round`:**
- Remove all `add_sign_seg()` calls
- Remove per-sign coverage printing
- **Remove the entire "Phase 4: Secondary validation (LOO)" section** — temporal only
- **Remove `loo_validation` from metrics dict** — only `temporal_validation`
- Update config.json:
  - `"stratification": "per_class (onpeak/offpeak)"`
  - `"stratify_by": ["class_type"]`
  - `"fallback_chain": "(bin, class) -> (bin, pooled)"`
  - `"calibration": "signed quantile pair of (mcp - baseline) per (bin, class)"`
  - `"cv_method": "temporal_only"`

**Add artifact saving (for future pmodel migration):**

After the winner is selected and metrics are saved, also save the calibrated band parameters as a standalone artifact:

```python
# Save band calibration artifact for production inference
artifact = {
    "version": version_id,
    "round": round_num,
    "method": "asymmetric_per_class",
    "n_bins": winner_exp["n_bins"],
    "calibration": {}  # {quarter: {bin_boundaries, per_cell_quantile_pairs}}
}
# For each quarter, re-calibrate on ALL available PYs (full training set)
for quarter in QUARTERS:
    df = data_loader(quarter)
    all_pys = sorted(df[PY_COL].unique().to_list())
    boundaries, bin_labels = compute_quantile_boundaries(df[baseline_col], winner_exp["n_bins"])
    bin_pairs = calibrate_asymmetric_per_class(
        df, baseline_col, mcp_col, CLASS_COL,
        boundaries, bin_labels, coverage_levels,
    )
    artifact["calibration"][quarter] = {
        "boundaries": [b if not math.isinf(b) else "inf" for b in boundaries],
        "bin_labels": bin_labels,
        "bin_pairs": sanitize_for_json(bin_pairs),
        "n_rows": df.height,
        "pys_used": all_pys,
    }

artifact_path = v_dir / "calibration_artifact.json"
# atomic write
```

This artifact contains everything needed to apply bands to new trades without re-running calibration.

**Update `main()`:**
- `version_id="v9"`
- `prior_version_part="bands/v8/r1"` etc.
- Reference comparisons: `"v2 (promoted)"`, `"v4"`, `"v5"`, `"v6"`, `"v7"`, `"v8"`

- [ ] **Step 2: Verify script parses and has no sign_seg or LOO remnants**

```bash
python -c "import ast; ast.parse(open('scripts/run_v9_bands.py').read()); print('Parse OK')"
# Must return nothing:
grep -n 'sign_seg\|SIGN_SEGS\|add_sign_seg' scripts/run_v9_bands.py
# Must return nothing:
grep -n 'loo\|LOO\|leave.*one.*out' scripts/run_v9_bands.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_v9_bands.py
git commit -m "add v9 band script: simplified asymmetric (per-class only, no sign split)"
```

### Task 9: Run v9 experiment

**Files:**
- Run: `scripts/run_v9_bands.py`
- Creates: `versions/bands/v9/r1/`, `v9/r2/`, `v9/r3/`

- [ ] **Step 1: Run v9**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/run_v9_bands.py 2>&1 | tee /tmp/v9_bands_output.log
```

Expected runtime: ~2-5 minutes. Monitor memory (should stay under 10 GiB).

Capture from output:
- P95 coverage per round per quarter (temporal)
- P95 half-width per round per quarter
- Comparison vs v2 (promoted) — width reduction %
- Winner experiment name

- [ ] **Step 2: Verify results exist**

```bash
for r in r1 r2 r3; do
    echo "v9/$r:"
    ls versions/bands/v9/$r/{config,metrics}.json 2>/dev/null && echo "  OK" || echo "  MISSING"
done
```

- [ ] **Step 3: Write NOTES.md for v9/r1, v9/r2, v9/r3**

Template:

```markdown
# Bands v9/r{M} — Simplified Asymmetric (Per-Class Only)

**Date:** 2026-03-13  |  **Script:** `scripts/run_v9_bands.py`

## Method
Asymmetric signed quantile bands with per-(bin, class) calibration.
No sign stratification, no correction. 5 quantile bins × 2 classes = 10 cells.
Temporal CV with min_train_pys=3.

## Key Simplification vs v8
v8 used 5 bins × 2 classes × 2 signs = 20 cells. v9 drops sign split → 10 cells.
Fewer cells = more data per cell = more stable quantile estimates.

## Coverage Accuracy (temporal, P95)
[From output]

## P95 Mean Half-Width ($/MWh)
[From output]

## vs v2 (promoted, symmetric)
[Width reduction % from output]

## vs v8 (class+sign, no correction)
[Comparison from output — expect similar or slightly wider due to no sign split,
 but more stable across folds]
```

- [ ] **Step 4: Validate calibration artifacts**

```bash
python3 -c "
import json
from pathlib import Path
for r in ['r1', 'r2', 'r3']:
    p = Path(f'versions/bands/v9/{r}/calibration_artifact.json')
    if not p.exists():
        print(f'MISSING: {p}')
        continue
    a = json.load(open(p))
    for q in ['aq1', 'aq2', 'aq3', 'aq4']:
        cal = a['calibration'].get(q)
        if cal is None:
            print(f'MISSING quarter: {r}/{q}')
            continue
        n_bins = len(cal['bin_labels'])
        n_pairs = sum(1 for bl in cal['bin_labels'] for cls in ['onpeak','offpeak'] if cls in cal['bin_pairs'].get(bl, {}))
        print(f'  {r}/{q}: {n_bins} bins, {n_pairs} cells, {cal[\"n_rows\"]} rows')
print('Artifact validation complete')
"
```

- [ ] **Step 5: Run pipeline validate**

```bash
python /home/xyz/workspace/research-qianli-v2/research-annual-band/pipeline/pipeline.py validate bands
```

If validate fails, check which config.json field is missing/wrong and fix before committing.

- [ ] **Step 6: Commit**

```bash
git add versions/bands/v9/
git commit -m "run v9 band experiment: simplified asymmetric, per-class only, 20-25% narrower than promoted"
```

### Task 10: Final verification and documentation update

- [ ] **Step 1: Verify complete directory structure**

```bash
for v in versions/bands/v*/; do
    echo "$v: $(ls -d ${v}r*/ 2>/dev/null | xargs -I{} basename {} | tr '\n' ' ')"
done
```

Expected: v1-v9 each have r1 r2 r3.

- [ ] **Step 2: Update mem.md with v9 results**

Add v9 results to the band section of mem.md. Include width reduction vs promoted and coverage numbers.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "update documentation with v9 results and final reorganization"
```

### Task 11: Document migration gaps for pmodel integration

**Files:**
- Create: `docs/migration-to-pmodel.md`

After v9 is finalized, document what's needed to port the annual band generator into `pmodel/src/pmodel/base/ftr24/v1/`:

- [ ] **Step 1: Write migration gap document**

Include:
1. **API design:** Proposed `generate_annual_bands(df, round, class_type, artifact_path)` signature
2. **Artifact format:** The `calibration_artifact.json` saved in Task 9 — document its schema
3. **Inference function:** How to apply pre-calibrated bands to new trades (load artifact → assign bins → lookup quantile pairs → compute band edges)
4. **Data dependency:** What training data is needed, where it lives, how to refresh it
5. **Integration point:** Where in `autotuning.py` or `miso_base.py` the annual band generator would be called
6. **Column mapping:** Research uses `nodal_f0` / `mtm_1st_mean` / `mcp_mean`; production uses `mtm_1st_mean` / `mcp_monthly` etc. — document the mapping
7. **Class type handling:** Research iterates over `["onpeak", "offpeak"]`; production passes `class_type` as a parameter — ensure no confusion

- [ ] **Step 2: Commit**

```bash
git add docs/migration-to-pmodel.md
git commit -m "document migration gaps for annual band generator to pmodel"
```
