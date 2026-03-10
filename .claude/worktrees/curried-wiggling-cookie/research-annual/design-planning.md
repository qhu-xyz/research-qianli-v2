# Plan: Formalize Research Pipeline — Single-File Design (Rev 3)

**Rev 3 changes:** Fixes all 10 new findings from Rev 2 review (6.7/10). Fixed `compute_stability` signature, added staleness detection for `matched` section, specified `validate` checks, made `promote` enforce hard gates, clarified per-PY backfill source, added tie-breaking rule.

---

## Context

**Problem:** Our research pipeline is scattered — 6 procedural scripts, all results printed to stdout, no structured metrics, no version tracking, no promotion criteria. When we (or the next AI) return to this project, there's no way to know which experiment is "current best" or reproduce comparisons without re-running everything.

**Why version control matters:**
- We already have 3 baseline experiments (raw f0, α-scaled, α+residual) plus 3 dead ends (Ridge, convex, de-biased). Without formal tracking, these live only in `findings.md` prose — no machine-readable metrics, no reproducible comparison.
- Part II (pricing bands) hasn't started. When it does, we need to independently version band experiments without coupling them to baseline experiments.
- The tech lead wants a pattern similar to the shadow-price-prediction project: each version gets its own config + metrics + notes, with a clear promotion path.

**Two questions this design answers:**
1. *"Multiple experiments for Part I and Part II?"* → Independent version sequences (`b001`, `b002`... for baseline; `d001`, `d002`... for bands). Releases (composing baseline + bands) are **deferred** until Part II is underway.
2. *"Are metrics/gates enough?"* → Current metrics (MAE, Dir%, coverage, bias) are necessary but not sufficient. Adding: per-PY stability, tail risk, matched-path win rate, class parity.

---

## What we're building

**One Python file: `pipeline/pipeline.py`** that handles:
1. Metrics computation — structured dict output matching a frozen schema
2. Version directory management — create, list, promote
3. Gate checking — candidate vs promoted on **matched paths**, 9 gates
4. JSON I/O with atomic writes

Plus: **backfill 3 existing experiments** into `versions/baseline/`, **update docs**.

**Out of scope for this phase:** release composition (baseline + bands), band-specific gates, automated CI. These are deferred until Part II begins.

---

## Directory structure (after implementation)

```
research-annual/
  pipeline/
    __init__.py
    pipeline.py              # Single file: metrics, versioning, gates, comparison
  versions/
    baseline/
      b001-nodal-f0-raw/
        config.json
        metrics.json
        NOTES.md
      b002-alpha-scaled/
        config.json
        metrics.json
        NOTES.md
      b003-alpha-residual/   # Current best
        config.json
        metrics.json
        NOTES.md
      promoted.json          # {"version": "b003-alpha-residual", ...}
    bands/                   # Empty until Phase 3
```

---

## Frozen Schema Contract

### Version ID format

**Pattern:** `{prefix}{NNN}-{slug}`
- Prefix: `b` for baseline, `d` for bands
- NNN: 3-digit zero-padded sequence number
- Slug: lowercase alphanumeric + hyphens, required, descriptive
- Regex: `^[bd]\d{3}-[a-z0-9-]+$`
- The **full string** (e.g. `b003-alpha-residual`) is the canonical ID everywhere — in directory names, in JSON fields, in CLI arguments. No short forms like `b003`.

### Schema version

All JSON files include `"schema_version": 1`. If field names or semantics change, bump this number. Gate checker refuses to compare across schema versions.

### Field names — locked to `baseline_utils.py` output

The metrics schema uses a **subset** of keys returned by `eval_baseline()` in `scripts/baseline_utils.py`. The disposition of every `eval_baseline()` output field is listed below.

**Stored fields** (from `eval_baseline`):

| Field | Type | Notes |
|-------|------|-------|
| `n` | int | |
| `coverage_pct` | float | |
| `bias` | float | |
| `mae` | float | |
| `median_ae` | float | |
| `p95_ae` | float | |
| `p99_ae` | float | |
| `dir_all` | float (%) | |
| `dir_50` | float (%) | |
| `dir_100` | float (%) | |
| `mae_tiny` | float (binned) | |
| `mae_small` | float (binned) | |
| `mae_med` | float (binned) | |
| `mae_large` | float (binned) | |

**Dropped fields** (from `eval_baseline`, intentionally not stored):

| Field | Reason |
|-------|--------|
| `label` | Display-only string, not a metric |
| `n_dir` | Redundant — equals `n` minus rows where mcp=0 or pred=0 |
| `n_50` | Derivable from data, not needed for gates |
| `n_100` | Derivable from data, not needed for gates |

**Additional fields** computed by `pipeline.py`:

| Field | Computed by | Type |
|-------|------------|------|
| `win_rate` | `compute_matched_comparison` | float (%) — path-level win rate vs promoted. Ties (|cand_err| == |prom_err|) count as 0.5 wins for each side. |
| `n_matched` | `compute_matched_comparison` | int — rows in matched intersection |
| `mae_cv` | `compute_stability` | float — CV of per-PY MAEs |
| `worst_py` | `compute_stability` | str — PY with highest MAE |
| `worst_py_mae` | `compute_stability` | float |
| `median_py_mae` | `compute_stability` | float |
| `dir_all_range` | `compute_stability` | list[float, float] — min/max Dir% across PYs (JSON array) |

---

## config.json schema

```json
{
  "schema_version": 1,
  "version": "b003-alpha-residual",
  "description": "Alpha-scaled nodal f0 + prior-year residual correction (LOO by PY)",
  "created": "2026-02-21T14:30:00",
  "part": "baseline",

  "method": {
    "tier1a": {"formula": "alpha * nodal_f0 + beta * (py1_mcp - py1_f0)", "coverage": "~22%", "condition": "path recurs from PY-1 R1"},
    "tier1b": {"formula": "alpha * nodal_f0", "coverage": "~77%", "condition": "nodal coverage, no prior-year data"},
    "tier2":  {"formula": "H (DA congestion fallback)", "coverage": "~1%", "condition": "missing nodes"}
  },

  "parameters": {
    "alpha_grid": [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60],
    "beta_grid": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "cv_method": "LOO_by_PY",
    "py_range": [2020, 2025]
  },

  "data_sources": [
    {"path": "crossproduct_work/aq1_all_baselines.parquet", "row_count": 165898},
    {"path": "crossproduct_work/aq2_all_baselines.parquet", "row_count": 166535},
    {"path": "crossproduct_work/aq3_all_baselines.parquet", "row_count": 152522},
    {"path": "crossproduct_work/aq4_all_baselines.parquet", "row_count": 151210}
  ],

  "script": "scripts/run_phase2_improvement.py",

  "environment": {
    "git_hash": "c4e33be",
    "python_version": "3.11.9",
    "polars_version": "1.31.0"
  }
}
```

**Changes from Rev 1:** `git_hash` moved into `environment` block with `python_version` and `polars_version`. `data_sources` now includes `row_count` per file (cheap reproducibility fingerprint — no hashing needed for parquets already under our control). `schema_version` added at top level.

---

## metrics.json schema

All gate inputs must be computable from this file alone. No gate may require reloading raw data.

The example below shows the **schema structure**. Values in `overall` are from `findings.md` Phase 2b tables. Values in `per_py`, `per_class`, `stability`, and `matched` are **placeholders** — they will be populated at backfill time by re-running `run_phase2_improvement.py` with per-PY output enabled. The script already computes per-PY metrics internally for LOO; we just need to capture and store them.

```json
{
  "schema_version": 1,
  "version": "b003-alpha-residual",
  "evaluated_at": "2026-02-21T14:30:00",
  "compared_against": "b002-alpha-scaled",

  "overall": {
    "aq1": {"n": 32495, "coverage_pct": 98.8, "bias": 96, "mae": 663, "median_ae": 262, "p95_ae": 2639, "p99_ae": 5645, "dir_all": 81.6, "dir_50": "...", "dir_100": 84.9, "mae_tiny": "...", "mae_small": "...", "mae_med": "...", "mae_large": "..."},
    "aq2": {"n": 32266, "coverage_pct": 99.5, "bias": 95, "mae": 650, "median_ae": 240, "p95_ae": 2663, "p99_ae": 6062, "dir_all": 82.5, "dir_50": "...", "dir_100": 86.2, "...": "..."},
    "aq3": {"n": 30565, "coverage_pct": 99.7, "bias": 35, "mae": 570, "median_ae": 224, "p95_ae": 2249, "p99_ae": 4648, "dir_all": 84.4, "dir_50": "...", "dir_100": 88.1, "...": "..."},
    "aq4": {"n": 30395, "coverage_pct": 100.0, "bias": 34, "mae": 459, "median_ae": 184, "p95_ae": 1809, "p99_ae": 3972, "dir_all": 85.1, "dir_50": "...", "dir_100": 89.5, "...": "..."}
  },

  "matched": {
    "description": "Head-to-head on rows where both candidate and promoted have non-null predictions",
    "match_keys": ["path", "class_type", "planning_year"],
    "compared_against": "b002-alpha-scaled",
    "aq1": {"n_matched": "...", "mae": "...", "dir_all": "...", "win_rate": "...", "p99_ae": "..."},
    "aq2": {"n_matched": "...", "mae": "...", "dir_all": "...", "win_rate": "...", "p99_ae": "..."},
    "aq3": {"n_matched": "...", "mae": "...", "dir_all": "...", "win_rate": "...", "p99_ae": "..."},
    "aq4": {"n_matched": "...", "mae": "...", "dir_all": "...", "win_rate": "...", "p99_ae": "..."}
  },

  "per_py": {
    "aq1": {
      "2020": {"mae": "...", "bias": "...", "dir_all": "...", "alpha": "...", "beta": "..."},
      "2021": {"mae": "...", "bias": "...", "dir_all": "...", "alpha": "...", "beta": "..."},
      "...": "..."
    },
    "aq2": {"...": "..."},
    "aq3": {"...": "..."},
    "aq4": {"...": "..."}
  },

  "per_class": {
    "aq1": {"onpeak": {"mae": "...", "dir_all": "..."}, "offpeak": {"mae": "...", "dir_all": "..."}},
    "aq2": {"...": "..."},
    "aq3": {"...": "..."},
    "aq4": {"...": "..."}
  },

  "stability": {
    "aq1": {"mae_cv": "...", "worst_py": "...", "worst_py_mae": "...", "median_py_mae": "...", "dir_all_range": ["...", "..."]},
    "aq2": {"...": "..."},
    "aq3": {"...": "..."},
    "aq4": {"...": "..."}
  },

  "vs_h": {
    "aq1": {"mae_improvement_pct": -25.0, "dir_improvement_pp": 13.9},
    "aq2": {"mae_improvement_pct": -39.3, "dir_improvement_pp": 13.5},
    "aq3": {"mae_improvement_pct": -38.0, "dir_improvement_pp": 15.3},
    "aq4": {"mae_improvement_pct": -48.5, "dir_improvement_pp": 20.8}
  }
}
```

**Schema rules:**
- `overall`, `per_py`, `per_class`, `stability` are **required**. All 4 quarters must appear in each section. A missing quarter is a validation error.
- `matched` is **required** if a promoted version exists at evaluation time; **omitted** if no promoted exists (first version).
- `vs_h` is **required** — computed by `compute_full_evaluation` using the `h_col` parameter (defaults to `"mtm_1st_mean"`).
- `compared_against` appears both at top level and inside `matched` (redundant for clarity). Must match.
- Values labeled `"..."` above are placeholders. At backfill, all values will be populated by re-running evaluation with per-PY/per-class breakdowns enabled.

---

## Promotion Gates

**Design principle:** All gates are computable from the stored `metrics.json` alone. No gate requires reloading raw parquet data. The `matched` section is computed once at evaluation time and stored.

### Gate definitions

| # | Gate | Input section | Check | Severity |
|---|------|--------------|-------|----------|
| G1 | MAE improvement | `matched` | `candidate.matched.{q}.mae <= promoted.matched.{q}.mae` for all 4 quarters | HARD |
| G2 | Direction preserved | `matched` | `candidate.matched.{q}.dir_all >= promoted.matched.{q}.dir_all - 1.0` for all 4 quarters | HARD |
| G3 | Coverage floor | `overall` | `candidate.overall.{q}.coverage_pct >= 95.0` for all 4 quarters | HARD |
| G4 | Bias sign | `overall` | `candidate.overall.{q}.bias >= 0` for all 4 quarters | SOFT |
| G5 | Per-PY stability | `stability` | `candidate.stability.{q}.mae_cv < 0.30` for all 4 quarters | SOFT |
| G6 | Worst-PY bound | `stability` | `candidate.stability.{q}.worst_py_mae < 1.5 * candidate.stability.{q}.median_py_mae` for all 4 quarters | SOFT |
| G7 | Class parity | `per_class` | `abs(onpeak.mae - offpeak.mae) / avg(onpeak.mae, offpeak.mae) < 0.40` for all 4 quarters | ADVISORY |
| G8 | Win rate | `matched` | `candidate.matched.{q}.win_rate >= 50.0` for all 4 quarters | ADVISORY |
| G9 | Tail risk | `matched` | `candidate.matched.{q}.p99_ae <= promoted.matched.{q}.p99_ae * 1.10` for all 4 quarters | ADVISORY |

**Gate severity rules:**
- **HARD:** All must pass. Failure = auto-reject.
- **SOFT:** Failure requires written justification in NOTES.md. Human decides.
- **ADVISORY:** Printed as warnings. Informational only.

**Why G1/G2/G8 use `matched` instead of `overall`:** `runbook.md` Section 12 warns that baselines with different coverage are not comparable via overall metrics. A candidate with 96% coverage vs a promoted with 99% coverage could appear better simply by dropping hard paths. The `matched` section forces apples-to-apples comparison on the intersection of both coverages.

**Special case — first version (no promoted exists):** When there is no promoted version, skip G1/G2/G8/G9 (they're comparative). Only run G3/G4/G5/G6/G7 as absolute checks. The `matched` section is omitted from metrics.json.

---

## pipeline.py — API contract

### Section 1: Metrics

```python
def compute_metrics(mcp: pl.Series, pred: pl.Series, label: str, total_n: int) -> dict:
    """
    Same logic as eval_baseline() in baseline_utils.py.
    Returns dict with keys: n, coverage_pct, bias, mae, median_ae, p95_ae, p99_ae,
    dir_all, dir_100, mae_tiny, mae_small, mae_med, mae_large.
    """

def compute_matched_comparison(
    df: pl.DataFrame, candidate_col: str, promoted_col: str, quarter: str
) -> dict:
    """
    On rows where both candidate_col and promoted_col are non-null:
    - Compute candidate metrics via compute_metrics
    - Compute path-level win rate: % of rows where |candidate_error| < |promoted_error|
      Ties (|cand_err| == |prom_err|) count as 0.5 wins for each side.
    Returns dict with keys: n_matched, mae, dir_all, win_rate, p99_ae.
    """

def compute_stability(per_py_metrics: dict[str, dict]) -> dict:
    """
    Input: {py_str: {"mae": float, "dir_all": float, ...}} for each PY.
    Computes:
    - mae_cv: std(maes) / mean(maes)
    - worst_py: PY with highest MAE
    - worst_py_mae: that PY's MAE
    - median_py_mae: median of per-PY MAEs
    - dir_all_range: [min(dir_all), max(dir_all)] across PYs (stored as JSON array)
    """

def compute_full_evaluation(
    df: pl.DataFrame, candidate_col: str, promoted_col: str | None,
    quarters: list[str], mcp_col: str = "mcp_mean",
    quarter_col: str = "period_type", py_col: str = "planning_year",
    class_col: str = "class_type", h_col: str = "mtm_1st_mean"
) -> dict:
    """
    Master function. Returns full metrics.json dict with sections:
    - overall: compute_metrics per quarter (all rows)
    - matched: compute_matched_comparison per quarter (if promoted_col is not None)
    - per_py: compute_metrics per (quarter, PY) pair
    - per_class: compute_metrics per (quarter, class_type) pair
    - stability: compute_stability per quarter (from per_py metrics)
    - vs_h: MAE improvement % and Dir% improvement pp vs H baseline (h_col)
    """

def save_metrics(metrics: dict, version_dir: Path) -> None:
    """Atomic write: write to .tmp, then os.rename to metrics.json."""
```

### Section 2: Version management

```python
VERSION_RE = re.compile(r"^[bd]\d{3}-[a-z0-9-]+$")

def validate_version_id(version_id: str) -> None:
    """Raises ValueError if version_id doesn't match VERSION_RE."""

def create_version(part: str, version_id: str, description: str) -> Path:
    """
    Creates versions/{part}/{version_id}/ with a config.json stub containing:
    schema_version, version, description, created, part, environment (auto-detected).
    Returns the directory path.
    Raises FileExistsError if directory already exists.
    """

def list_versions(part: str) -> list[str]:
    """Returns sorted list of version directory names matching VERSION_RE."""

def get_promoted(part: str) -> dict | None:
    """Reads promoted.json. Returns {"version": "b003-alpha-residual", ...} or None."""

def promote(part: str, version_id: str, notes: str = "", force: bool = False) -> None:
    """
    Atomic write to promoted.json: {"version": version_id, "promoted_at": ..., "notes": ...}.
    Raises FileNotFoundError if version dir or metrics.json doesn't exist.

    Before writing, runs check_gates against current promoted (if one exists).
    - If any HARD gate fails: refuses to promote unless force=True.
    - If matched section is stale (compared_against mismatch): refuses unless force=True.
    - Soft/advisory failures print warnings but do not block.
    - If no promoted exists yet (first promotion): only absolute gates checked.
    CLI flag: --force to override HARD gate failures or staleness.
    """
```

### Section 3: Gate checking

```python
def check_gates(candidate_metrics: dict, promoted_metrics: dict | None) -> list[dict]:
    """
    Returns list of gate results:
    [{"gate": "G1", "name": "MAE improvement", "severity": "HARD",
      "passed": True, "detail": "4/4 quarters pass", "values": {...}}]
    If promoted_metrics is None, skips comparative gates (G1/G2/G8/G9).

    Quarter iteration: reads quarter keys from candidate_metrics["overall"].
    All 4 quarters (aq1-aq4) must be present; missing quarter = validation error.
    """

def print_gate_table(gate_results: list[dict]) -> None:
    """Pretty-prints gate pass/fail table to stdout."""

def compare(part: str, candidate_id: str, promoted_id: str | None = None) -> None:
    """
    Main entry point. Loads metrics.json for both versions, runs check_gates, prints table.
    If promoted_id is None, reads promoted.json.

    Staleness check: if candidate's matched.compared_against != promoted version being
    compared against, prints WARNING and skips comparative gates (G1/G2/G8/G9).
    Message: "matched section was computed against {X}, not {Y}. Re-evaluate to get
    fresh matched metrics. Only absolute gates (G3-G7) will run."
    """
```

### Section 4: CLI

```python
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    # create <part> <version_id> <description>
    # list <part>
    # compare <part> <candidate_id> [promoted_id]
    # promote <part> <version_id> [--force]
    # validate <part>
```

### `validate` command specification

`validate <part>` walks all version dirs in `versions/{part}/` and checks:

| Check | Severity | What |
|-------|----------|------|
| Version ID format | ERROR | Directory name matches `^[bd]\d{3}-[a-z0-9-]+$` |
| config.json exists | ERROR | File present and valid JSON |
| config.json schema_version | ERROR | `schema_version` field present, equals 1 |
| config.json required fields | ERROR | `version`, `description`, `created`, `part`, `environment` all present |
| config.json version match | ERROR | `config.version` matches directory name |
| metrics.json exists | WARN | Missing = version not yet evaluated (not an error) |
| metrics.json schema_version | ERROR | If file exists: `schema_version` present, equals 1 |
| metrics.json required sections | ERROR | If file exists: `overall`, `per_py`, `per_class`, `stability` present |
| metrics.json all quarters | ERROR | Each section has keys for all 4 quarters (`aq1`-`aq4`) |
| metrics.json field types | ERROR | `mae`, `bias`, `dir_all` etc. are numeric (not string, not null) |
| metrics.json matched coherence | WARN | If `matched` present: `compared_against` points to an existing version dir |
| promoted.json coherence | ERROR | If exists: `version` points to an existing version dir with metrics.json |
| NOTES.md exists | WARN | Missing = version not documented |

Exit code: 0 if no ERRORs, 1 if any ERROR found. WARNs are printed but don't fail.

---

## Atomic write pattern

All JSON writes (config.json, metrics.json, promoted.json) use:

```python
def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)  # atomic on same filesystem
```

This prevents corrupted files from interrupted writes.

---

## NOTES.md structure

NOTES.md is freeform but must contain at minimum:

| Section | Required | Purpose |
|---------|----------|---------|
| `## What` | YES | What the method does (1-3 sentences + formula) |
| `## Why` | YES | Motivation / hypothesis |
| `## Results` | YES | Key metrics vs previous version |
| `## Gate failures` | IF ANY | For each soft gate failure: which gate, the value, and justification for why it's acceptable |
| `## Dead ends` | NO | What didn't work and why (prevents re-trying) |
| `## Decision` | YES | Promoted / rejected / superseded |

**Example (b003-alpha-residual):**

```markdown
# b003-alpha-residual

## What
Alpha-scaled nodal f0 with prior-year residual correction.
- Tier 1a: `alpha * nodal_f0 + beta * (PY-1 mcp - PY-1 f0)` on ~22% recurring paths
- Tier 1b: `alpha * nodal_f0` on remaining ~77%
- Tier 2: H fallback on ~1%

## Why
- Alpha scaling (b002) corrects the persistent positive bias (+200-340) in raw nodal f0
- Prior-year residual exploits lag-1 autocorrelation (r=0.41-0.71) on recurring paths
- Combined gives 10-31% MAE improvement over raw f0

## Results vs b002-alpha-scaled
- aq1: 669 → 663 (-1%)
- aq2: 674 → 650 (-4%)
- aq3: 620 → 570 (-8%)
- aq4: 536 → 459 (-14%)

## Dead ends tried
- Signal blending (Ridge, convex, de-biased): all fail. See findings.md Phase 2c.
- Root cause: all signals are noisier versions of same congestion signal, no independent directional info.

## Decision
Promoted as current best. Improvement increases aq1→aq4 because later quarters have more stable congestion.
```

---

## Workflow: running a new experiment

```bash
# 1. Create version directory (creates dir + config.json stub)
python pipeline/pipeline.py create baseline b004-seasonal-alpha "Per-quarter alpha instead of global"

# 2. Edit config.json to fill in method/parameters/data_sources

# 3. Run experiment script
#    Scripts need a small addition at the end to save structured metrics:
#
#    from pipeline.pipeline import compute_full_evaluation, save_metrics
#    metrics = compute_full_evaluation(
#        df, candidate_col="prediction", promoted_col="nodal_f0",  # or None
#        quarters=["aq1","aq2","aq3","aq4"]
#    )
#    save_metrics(metrics, Path("versions/baseline/b004-seasonal-alpha"))
python scripts/run_experiment.py

# 4. Compare against current promoted
python pipeline/pipeline.py compare baseline b004-seasonal-alpha

# Output:
# Comparing b004-seasonal-alpha vs promoted (b003-alpha-residual)
# ┌──────┬──────────────────────┬──────────┬────────┬────────────────┬───────────┐
# │ Gate │ Check                │ Severity │ Result │ Value          │ Threshold │
# ├──────┼──────────────────────┼──────────┼────────┼────────────────┼───────────┤
# │ G1   │ MAE improvement      │ HARD     │ PASS   │ 4/4 qtrs       │ all 4     │
# │ G2   │ Direction preserved  │ HARD     │ PASS   │ min -0.3pp     │ -1.0pp    │
# │ G3   │ Coverage floor       │ HARD     │ PASS   │ min 98.8%      │ 95%       │
# │ G4   │ Bias sign            │ SOFT     │ PASS   │ all >= 0       │ >= 0      │
# │ G5   │ Per-PY stability     │ SOFT     │ PASS   │ max CV 0.22    │ 0.30      │
# │ G6   │ Worst-PY bound       │ SOFT     │ PASS   │ max 1.3x med   │ 1.5x      │
# │ G7   │ Class parity         │ ADVISORY │ WARN   │ max gap 0.38   │ 0.40      │
# │ G8   │ Win rate             │ ADVISORY │ PASS   │ min 53.2%      │ 50%       │
# │ G9   │ Tail risk (p99)      │ ADVISORY │ PASS   │ max +4%        │ +10%      │
# └──────┴──────────────────────┴──────────┴────────┴────────────────┴───────────┘
# RESULT: All HARD gates PASS, all SOFT gates PASS. Auto-promotable.

# 5. Promote (runs gate check automatically before writing)
python pipeline/pipeline.py promote baseline b004-seasonal-alpha
# → Runs check_gates, all HARD gates pass, no staleness → promoted.json updated

# If HARD gates fail or matched section is stale:
# python pipeline/pipeline.py promote baseline b004-seasonal-alpha --force
# → Promotes with WARNING printed

# 6. Validate all versions
python pipeline/pipeline.py validate baseline
# Checks: schema_version, required fields, version ID format, quarter completeness, etc.
# Exit 0 = all clear, exit 1 = errors found
```

---

## How Part II (bands) works independently

When Phase 3 (band calibration) begins:
- Band experiments get `d001-*`, `d002-*`, ... in `versions/bands/`
- Each band config includes `"baseline_version": "b003-alpha-residual"` — the baseline it was calibrated against
- Band-specific gate definitions will be designed when Phase 3 starts. They will differ from baseline gates (e.g., coverage per |MCP| bin, calibration accuracy vs target coverage, clearing probability accuracy). We do not define them now because we don't yet know the band method.
- Band promotion is independent of baseline promotion
- When both parts are mature, we add a `release` command that locks a baseline + band combination. This is explicitly **not in scope** for this phase.

---

## Implementation steps

| Step | Action | Files |
|------|--------|-------|
| 1 | Create `pipeline/pipeline.py` with sections 1-4 + atomic write helper | NEW: `pipeline/__init__.py`, `pipeline/pipeline.py` |
| 2 | Create version dirs + backfill b001/b002/b003 config.json + metrics.json + NOTES.md | NEW: 9 files in `versions/baseline/` |
| 3 | Set promoted.json → b003-alpha-residual | NEW: `versions/baseline/promoted.json` |
| 4 | Create empty `versions/bands/` | NEW: dir |
| 5 | Update `mem.md` (repo structure, pipeline section) | EDIT: `mem.md` |
| 6 | Update `runbook.md` (add Section 14: Pipeline & Versioning) | EDIT: `runbook.md` |

**Note on backfill:** Backfill requires re-running evaluation (via `compute_full_evaluation`) on the existing parquet data to populate per-PY, per-class, stability, and matched sections. The `overall` values can be verified against `findings.md` Phase 2b tables. Backfill order matters:
1. b001: no `matched` (no promoted exists yet). Promote b001.
2. b002: `matched` computed against b001 (currently promoted). Promote b002.
3. b003: `matched` computed against b002 (currently promoted). Promote b003.

This ensures each version's `compared_against` is valid and points to the version that was promoted at evaluation time.

---

## Verification

| # | Check | How |
|---|-------|-----|
| 1 | Schema compliance | `python pipeline/pipeline.py validate baseline` exits 0 for all 3 versions |
| 2 | Metrics accuracy | b001/b002/b003 `overall` MAE/Dir/bias values match `findings.md` Phase 1 (Table A for b001) and Phase 2b tables within ±1 (float rounding) |
| 3 | Gate logic (fail case) | `compare baseline b002-alpha-scaled b003-alpha-residual` → b002 is candidate, b003 is explicit promoted. Staleness warning printed (b002 was compared_against b001, not b003). Comparative gates skipped. Only G3-G7 run. |
| 4 | Gate logic (pass case) | `compare baseline b003-alpha-residual b002-alpha-scaled` → b003 is candidate, b002 is explicit promoted. b003's `compared_against` is b002 = match, no staleness. All 9 gates run. G1 passes (b003 MAE < b002 MAE). |
| 5 | Promotion guard | `promote baseline b002-alpha-scaled` → refuses (b002's matched is stale against current promoted b003). `promote baseline b002-alpha-scaled --force` → succeeds with warning. |
| 6 | Version ID validation | `create baseline bad_id ...` → rejected by VERSION_RE |
| 7 | Atomic write | promoted.json is valid JSON after promotion (no partial writes) |

---

## Review findings addressed

### Rev 1 review (review-codex.md, 4.8/10 → fixed in Rev 2)

| Finding | Resolution |
|---------|-----------|
| F1 G8 not computable | `matched` section stores `win_rate`, computed at evaluation time |
| F2 Naming inconsistency | Frozen to `eval_baseline()` keys: `dir_all`, `dir_100`, `coverage_pct` etc. |
| F3 No matched-path comparison | G1/G2/G8/G9 use `matched` section, not `overall`. Match keys defined. |
| F4 CLI release undefined | Deferred. Removed from this phase. |
| F5 create_version API/CLI conflict | Fixed: `create_version(part, version_id, description)` matches CLI. |
| F6 Version ID format inconsistent | Canonical format defined: `^[bd]\d{3}-[a-z0-9-]+$`. Full ID everywhere. |
| F7 No atomic writes | `_atomic_write_json` with tmp+rename+fsync for all state files. |
| F8 Incomplete reproducibility | `environment` block in config.json: git_hash, python/polars versions. `data_sources` with row_count. |
| F9 Single-file concern | Kept single file for this phase. Split criterion: if Part II adds >200 lines, extract `gates.py`. |
| F10 Manual verification | Added `validate` CLI command for schema checks. |
| F11 "Scripts unchanged" inaccurate | Fixed: "scripts need a small addition at the end." |
| F12 Bands section vague | Explicitly deferred band gate definitions to Phase 3 start. |

### Rev 2 review (6.7/10 → fixed in Rev 3)

| Finding | Resolution |
|---------|-----------|
| N1 `compute_stability` signature wrong | Fixed: input is `dict[str, dict]` with MAE + dir_all per PY, not `dict[str, float]` |
| N2 `vs_h` undocumented in API | Added to `compute_full_evaluation` docstring. `h_col` param already existed. |
| N3 Verification check #3 circular dependency | Backfill done in order (b001→b002→b003) with sequential promotion. Verification checks rewritten to use explicit promoted_id args. |
| N4 `per_py` example only shows aq1 | All 4 quarters shown in example (abbreviated with `"..."`) |
| N5 NOTES.md soft gate failure template | Added `## Gate failures` as required section when any soft gate fails |
| N6 Stale `matched` section | `compare` checks `matched.compared_against` vs target promoted. Warns and skips comparative gates if mismatch. `promote` refuses on staleness without `--force`. |
| N7 Per-PY metrics fabricated | Example now shows `"..."` placeholders. Backfill source documented: re-run eval with per-PY output. `overall` values verified against findings.md. |
| N8 `dir_all_range` type | Documented as `list[float, float]` (JSON array). `compute_stability` input type fixed (see N1). |
| N9 `promote` doesn't enforce gates | `promote` now runs `check_gates` before writing. Refuses on HARD failures or staleness unless `--force`. |
| N10 `validate` spec hollow | Full specification table added: 13 checks with ERROR/WARN severity and exit code behavior. |
| R2 Win rate tie-breaking | Ties count as 0.5 wins for each side. Documented in `compute_matched_comparison` and field table. |
| R3 Missing quarter in gates | Gate checker reads quarters from `overall` keys. Missing quarter = validation error (caught by `validate`). Documented in `check_gates`. |
| R5 Schema version bump | No migration path. `check_gates` refuses to compare across schema versions. Stated in "Schema version" section. |
