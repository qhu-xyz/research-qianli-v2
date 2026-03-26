# Band Version Reorganization — Design Spec

**Date:** 2026-03-13
**Scope:** `versions/bands/`, `scripts/`, documentation files
**Approach:** Rename in-place (no re-execution of experiments except v9)

---

## Problem

The `versions/bands/` directory has two coexisting layouts:
- **v-first** (`v{N}/r{M}/`): v1-v8, used by scripts v5-v9
- **r-first** (`r{M}/v{N}/`): r1/v1-v3, r2/v1-v2, r3/v1-v2, used by scripts v3-v4

The r-first layout is an exact duplicate of a subset of v-first. Additionally, old v2 (width reduction) is R1-only — it was an intermediate step absorbed into v3. Six directories (v6, v7) lack NOTES.md. Config schemas differ between layouts.

## Target State

```
versions/bands/
  v1/r1/  v1/r2/  v1/r3/    # Symmetric, fixed 4 bins, pooled
  v2/r1/  v2/r2/  v2/r3/    # Symmetric, quantile bins, per-class (was v3)
  v3/r1/  v3/r2/  v3/r3/    # Symmetric, quantile bins, class+sign (was v4)
  v4/r1/  v4/r2/  v4/r3/    # Asymmetric, 8-bin, class+sign (was v5)
  v5/r1/  v5/r2/  v5/r3/    # Asymmetric, 6-bin, correction experiments (was v6)
  v6/r1/  v6/r2/  v6/r3/    # Asymmetric, 8-bin, bidir + temporal (was v7)
  v7/r1/  v7/r2/  v7/r3/    # Asymmetric, 5-bin, bidir correction (was v8)
  v8/r1/  v8/r2/  v8/r3/    # Asymmetric, no correction, min_train_pys=3 (was v9, needs rerun)
  promoted.json
  bg2_gate_revision_proposal.md
```

Each `v{N}/r{M}/` has: `config.json`, `metrics.json`, `NOTES.md`.

## Version Mapping

| Old | New | Method | All 3 Rounds? |
|-----|-----|--------|:---:|
| v1 | v1 | Symmetric, fixed 4 bins, pooled | ✓ |
| v2 | *archived* | Width reduction (R1-only intermediate) | — |
| v3 | v2 | Symmetric, quantile bins, per-class | ✓ |
| v4 | v3 | Symmetric, quantile bins, class+sign | ✓ |
| v5 | v4 | Asymmetric, 8-bin, class+sign | ✓ |
| v6 | v5 | Asymmetric, 6-bin, correction exps | ✓ |
| v7 | v6 | Asymmetric, 8-bin, bidir + temporal | ✓ |
| v8 | v7 | Asymmetric, 5-bin, bidir correction | ✓ |
| v9 | v8 | Asymmetric, no correction, min_train_pys=3 | needs rerun |

Promoted version: old v3 = **new v2** (all 3 rounds).

---

## Execution Steps

### Step 1: Generate missing NOTES.md for v6 and v7

6 directories lack NOTES.md: `v6/r1`, `v6/r2`, `v6/r3`, `v7/r1`, `v7/r2`, `v7/r3`.

Generate from:
- Script docstrings (`run_v6_bands.py`, `run_v7_bands.py`)
- `config.json` method/experiments fields
- `metrics.json` coverage/width/stability data

Follow the format of existing NOTES.md files (v5, v8 as templates).

### Step 2: Extract shared functions into `scripts/band_utils.py`

`run_phase3_v2_bands.py` is being archived but 3 functions are imported by other scripts:
- `compute_quantile_boundaries`
- `loo_band_calibration_quantile`
- `temporal_band_calibration_quantile`

Also, `run_phase3_bands.py` provides base functions imported by v3/v4/r2r3:
- `assign_bins`, `calibrate_bin_widths`, `apply_bands`, `evaluate_coverage`
- `sanitize_for_json`, `mem_mb`, `COVERAGE_LEVELS`, `COVERAGE_LABELS`

And `run_v3_bands.py` (old) provides per-class functions imported by v4:
- `calibrate_bin_widths_per_class`, `apply_bands_per_class_fast`
- `evaluate_per_class_coverage`, `loo_per_class_quantile`, `temporal_per_class_quantile`

**Action:** Create `scripts/band_utils.py` containing all shared functions. Source scripts keep their copies (they're self-contained experiment records), but imports point to `band_utils`.

### Step 3: Update imports in dependent scripts

| Script (old name) | Imports to update |
|---|---|
| `run_v3_bands.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils` |
| `run_v4_bands.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils`, `run_v3_bands` → `band_utils` |
| `run_r2r3_bands.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils` |
| `analyze_segments.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils`, `run_v3_bands` → `band_utils` |
| `analyze_sign_segments.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils`, `run_v3_bands` → `band_utils` |
| `investigate_sign_split.py` | `run_phase3_bands` → `band_utils`, `run_phase3_v2_bands` → `band_utils` |
| `test_sign_split_calibration.py` | `run_phase3_v2_bands` → `band_utils`, `run_v3_bands` → `band_utils` |
| `test_annual_replication.py` | `run_v9_bands` → `run_v8_bands` (after rename) |

### Step 4: Verify all scripts parse

```bash
for f in scripts/run_*.py scripts/test_*.py; do
    python -c "import ast; ast.parse(open('$f').read())" && echo "OK: $f" || echo "FAIL: $f"
done
```

### Step 5: Archive old v2 and r-first layout

```bash
mkdir -p archive/versions_v2_width_reduction
mv versions/bands/v2/ archive/versions_v2_width_reduction/

mkdir -p archive/versions_r_layout
mv versions/bands/r1/ archive/versions_r_layout/
mv versions/bands/r2/ archive/versions_r_layout/
mv versions/bands/r3/ archive/versions_r_layout/
```

Also archive:
```bash
mv scripts/run_phase3_v2_bands.py archive/versions_v2_width_reduction/
```

Keep `run_phase3_bands.py` (it's the v1 R1 script and still needed as a record).

### Step 6: Rename version directories

Rename in forward order (lowest first). Each `mv v{N} v{N-1}` vacates v{N} before
the next command needs it as a target. Safe because v2 was already archived in Step 5:

```bash
# v2 already archived, so v3→v2 has no collision. Each subsequent mv is safe
# because the prior mv vacated the target.
mv versions/bands/v3 versions/bands/v2
mv versions/bands/v4 versions/bands/v3
mv versions/bands/v5 versions/bands/v4
mv versions/bands/v6 versions/bands/v5
mv versions/bands/v7 versions/bands/v6
mv versions/bands/v8 versions/bands/v7
# v9 doesn't exist yet as directory — will be created by rerun as v8
```

### Step 7: Rename scripts and test scripts

| Old filename | New filename |
|---|---|
| `run_v3_bands.py` | `run_v2_bands.py` |
| `run_v4_bands.py` | `run_v3_bands.py` |
| `run_v5_bands.py` | `run_v4_bands.py` |
| `run_v6_bands.py` | `run_v5_bands.py` |
| `run_v7_bands.py` | `run_v6_bands.py` |
| `run_v8_bands.py` | `run_v7_bands.py` |
| `run_v9_bands.py` | `run_v8_bands.py` |
| `test_v5_bands.py` | `test_v4_bands.py` |
| `test_v6_bands.py` | `test_v5_bands.py` |
| `test_v7_bands.py` | `test_v6_bands.py` |
| `test_v8_bands.py` | `test_v7_bands.py` |
| `test_v9_bands.py` | `test_v8_bands.py` |
| `test_5bins.py` | keep (imports from `run_v7_bands` → update to `run_v6_bands`) |

### Step 8: Update config.json in all version directories

Uniform schema for all configs:

```json
{
  "schema_version": 1,
  "version": "v{N}",           // new version number
  "part": "bands",             // always "bands", never "bands/r1"
  "round": "r{M}",             // always present
  "baseline_version": "v3",    // unchanged (refers to baseline namespace)
  ...
}
```

Changes per version:
- `"version"` → new version number
- Add `"round"` field if missing (r-first configs lacked it)
- Set `"part": "bands"` uniformly

### Step 9: Update version references in NOTES.md

For each NOTES.md:
- Update title to use new global version: e.g., "Bands v3/r2" not "R2 v3"
- Update cross-references: "vs v3" → "vs v2" (per mapping table)
- Update "Extends v3" → "Extends v2"
- Update script references: "run_v5_bands.py" → "run_v4_bands.py"

### Step 10: Update version references in scripts

For each renamed script:
- Update `version_id` in main() calls
- Update `prior_version_part` paths
- Update reference comparisons (e.g., `"v3 (promoted)"` → `"v2 (promoted)"`)
- Update docstring version numbers
- Update test script imports to use new script names

Also update internal version references in scripts that are NOT renamed:
- `run_r2r3_bands.py`: update any `bands/v{N}` paths and version_id values
- `run_phase3_bands.py`: update if it references other band versions

### Step 11: Update promoted.json

Master `versions/bands/promoted.json`:
```json
{
  "r1": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"},
  "r2": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"},
  "r3": {"version": "v2", "promoted_at": "2026-02-23", "notes": "was v3, renumbered"}
}
```

Remove per-round promoted.json files (they belonged to the r-first layout).

### Step 12: Update documentation

All paths relative to repo root `/home/xyz/workspace/research-qianli-v2/research-annual-band/`.

**`mem.md`** (repo root):
- Band sections: update version numbers per mapping table
- Update promoted version references

**`runbook.md`** (repo root) §15:
- Update current promoted results (v3 → v2)
- Update band methodology section
- Update CLI examples
- Update script paths

**`segment_analysis.md`** (repo root):
- Update "v3 is working well" → "v2"

**`versions/bands/bg2_gate_revision_proposal.md`**:
- Update v4/v5 references → v3/v4

**`design-planning.md`** (repo root):
- Minimal changes (mostly baseline-focused, band versioning deferred)

### Step 13: Verify

1. All JSON files valid: `python -c "import json; json.load(open(f))"`
2. All scripts parse: `python -c "import ast; ast.parse(...)"`
3. `python pipeline/pipeline.py list bands` — should show v1-v7 (v8 pending rerun)
4. `python pipeline/pipeline.py validate bands` — schema check
5. No stale version references — sweep for ALL old version numbers that should no longer appear:
   ```bash
   # Old v3 (now v2) — should not appear in version paths or version_id fields
   grep -rn '"v3"' versions/bands/ scripts/ | grep -v baseline_version | grep -v "v3 (baseline"
   # Old v4-v8 — these version numbers should not appear anywhere
   grep -rn 'bands/v[4-9]' versions/bands/ scripts/
   # Old v9 — should not exist
   grep -rn 'v9' versions/bands/ scripts/
   ```
   Note: `baseline_version: "v3"` is correct (refers to baseline namespace, not bands).
6. No r-first paths remain: `grep -r "bands/r[123]/v" versions/ scripts/` should find nothing
7. No broken imports: attempt to import each script module in Python

### Step 14: Rerun v8 (old v9)

After all renaming is complete:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
python /home/xyz/workspace/research-qianli-v2/research-annual-band/scripts/run_v8_bands.py
```

This populates `v8/r1/`, `v8/r2/`, `v8/r3/` with config.json, metrics.json. Write NOTES.md from results.

---

## Risk Mitigations

- **Git safety:** Commit current state before starting. Each major step (archive, rename dirs, rename scripts, update refs) is a separate commit for easy rollback.
- **No data loss:** Old v2 and r-first layout are archived, not deleted.
- **No recomputation:** Metrics are preserved exactly. Only version labels change.
- **Import chain:** Shared functions extracted to `band_utils.py` before archiving `run_phase3_v2_bands.py`.

## Out of Scope

- Re-running experiments v1-v7 (metrics preserved as-is)
- Adding gate_results to v5/v6 metrics.json (schema inconsistency, non-blocking)
- Updating design-planning.md naming scheme (d001 → v1, low priority)
