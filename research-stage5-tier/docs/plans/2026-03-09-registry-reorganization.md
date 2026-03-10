# Registry Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure registry/ and holdout/ from flat `{version_id}/` to hierarchical `{period_type}/{class_type}/{version_id}/` to support multi-period (f0/f1/f2/f3) and multi-class (onpeak/offpeak) experiments.

**Architecture:** Introduce a `registry_root()` helper that computes `registry/{period_type}/{class_type}` from parameters. All scripts, benchmark.py, compare.py derive their paths from this helper. Legacy v1-v10d experiments (superseded by v10e-lag1) move to `archive/registry/` and `archive/holdout/`. Active versions (v0, v10e, v10e-lag1, and their offpeak variants) migrate into the new hierarchy.

**Tech Stack:** Python, polars, pathlib, json, git

---

## Task 1: Create `ml/registry_paths.py` — Centralized Path Helper

**Files:**
- Create: `ml/registry_paths.py`
- Create: `ml/tests/test_registry_paths.py`

**Step 1: Write the failing tests**

```python
# ml/tests/test_registry_paths.py
"""Tests for registry path helpers."""
from pathlib import Path
import pytest
from ml.registry_paths import registry_root, holdout_root, version_dir, gates_path, champion_path


def test_registry_root_defaults():
    p = registry_root()
    assert p == Path("registry") / "f0" / "onpeak"


def test_registry_root_offpeak_f1():
    p = registry_root(period_type="f1", class_type="offpeak")
    assert p == Path("registry") / "f1" / "offpeak"


def test_holdout_root_defaults():
    p = holdout_root()
    assert p == Path("holdout") / "f0" / "onpeak"


def test_version_dir():
    p = version_dir("v10e-lag1", period_type="f0", class_type="onpeak")
    assert p == Path("registry") / "f0" / "onpeak" / "v10e-lag1"


def test_gates_path():
    p = gates_path(period_type="f0", class_type="onpeak")
    assert p == Path("registry") / "f0" / "onpeak" / "gates.json"


def test_champion_path():
    p = champion_path(period_type="f1", class_type="offpeak")
    assert p == Path("registry") / "f1" / "offpeak" / "champion.json"


def test_version_dir_custom_base(tmp_path):
    p = version_dir("v0", period_type="f0", class_type="onpeak", base_dir=tmp_path)
    assert p == tmp_path / "f0" / "onpeak" / "v0"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python -m pytest ml/tests/test_registry_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml.registry_paths'`

**Step 3: Write the implementation**

```python
# ml/registry_paths.py
"""Centralized path helpers for registry and holdout directories.

All experiment results live at:
    registry/{period_type}/{class_type}/{version_id}/
    holdout/{period_type}/{class_type}/{version_id}/

Each (period_type, class_type) slice has its own gates.json and champion.json.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults match the original f0/onpeak pipeline
_DEFAULT_PERIOD_TYPE = "f0"
_DEFAULT_CLASS_TYPE = "onpeak"


def registry_root(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}."""
    base = Path(base_dir) if base_dir is not None else Path("registry")
    return base / period_type / class_type


def holdout_root(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return holdout/{period_type}/{class_type}."""
    base = Path(base_dir) if base_dir is not None else Path("holdout")
    return base / period_type / class_type


def version_dir(
    version_id: str,
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/{version_id}."""
    return registry_root(period_type, class_type, base_dir) / version_id


def holdout_version_dir(
    version_id: str,
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return holdout/{period_type}/{class_type}/{version_id}."""
    return holdout_root(period_type, class_type, base_dir) / version_id


def gates_path(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/gates.json."""
    return registry_root(period_type, class_type, base_dir) / "gates.json"


def champion_path(
    period_type: str = _DEFAULT_PERIOD_TYPE,
    class_type: str = _DEFAULT_CLASS_TYPE,
    base_dir: Path | str | None = None,
) -> Path:
    """Return registry/{period_type}/{class_type}/champion.json."""
    return registry_root(period_type, class_type, base_dir) / "champion.json"
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python -m pytest ml/tests/test_registry_paths.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add ml/registry_paths.py ml/tests/test_registry_paths.py
git commit -m "feat: add centralized registry path helpers for multi-period/class support"
```

---

## Task 2: Migrate Files — Move Active Versions into Hierarchy, Archive Legacy

This task physically reorganizes the directory tree. No code changes.

**Active versions** (keep and migrate):
- `v0`, `v0-offpeak`, `v10e`, `v10e-lag1`, `v10e-lag1-offpeak` — these have current value
- Registry-level files: `gates.json`, `champion.json`

**Archive** (move to `archive/`):
- `v0_36`, `v1`, `v1_screen`, `v1b`, `v1b_screen`, `v3_screen`, `v4a*`, `v4b*`, `v5*`, `v6a*`, `v6b*`, `v6c*`, `v7`, `v8a`, `v8b`, `v8c_ensemble`, `v9`, `v9c` (registry)
- `v10`, `v10b`, `v10c`, `v10d`, `v10e` (subsumed by v10e-lag1), `v10f`, `v10g`
- `spawn_test`, `comparisons/`
- Holdout: `v5`, `v6b`, `v6c`, `v7`, `v8b`, `v8c`, `v9`, `v9c`, `v10e` (subsumed)

**Step 1: Create the new directory structure**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier

# New hierarchy for active versions
mkdir -p registry/f0/onpeak
mkdir -p registry/f0/offpeak
mkdir -p holdout/f0/onpeak
mkdir -p holdout/f0/offpeak

# Archive directories
mkdir -p archive/registry
mkdir -p archive/holdout
```

**Step 2: Archive legacy registry versions**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier

# Archive all superseded versions
for v in v0_36 v1 v1_screen v1b v1b_screen v3_screen \
         v4a v4a_screen v4b v4b_screen \
         v5 v5_screen v5_36 \
         v6a v6a_screen v6a_36 v6b v6b_screen v6b_36 v6c v6c_screen v6c_36 \
         v7 v8a v8b v8c_ensemble \
         v9 v9c \
         v10 v10b v10c v10d v10f v10g \
         spawn_test comparisons; do
    [ -e "registry/$v" ] && mv "registry/$v" "archive/registry/"
done
```

**Step 3: Archive legacy holdout versions**

```bash
for v in v5 v6b v6c v7 v8b v8c v9 v9c; do
    [ -e "holdout/$v" ] && mv "holdout/$v" "archive/holdout/"
done

# Also archive holdout-level config/comparison files
[ -e "holdout/config.json" ] && mv holdout/config.json archive/holdout/
[ -e "holdout/comparison.json" ] && mv holdout/comparison.json archive/holdout/
```

**Step 4: Migrate active versions into new hierarchy**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier

# Registry: f0/onpeak
mv registry/v0 registry/f0/onpeak/v0
mv registry/v10e registry/f0/onpeak/v10e
mv registry/v10e-lag1 registry/f0/onpeak/v10e-lag1
mv registry/gates.json registry/f0/onpeak/gates.json
mv registry/champion.json registry/f0/onpeak/champion.json

# Registry: f0/offpeak (strip the -offpeak suffix from version names)
mv registry/v0-offpeak registry/f0/offpeak/v0
mv registry/v10e-lag1-offpeak registry/f0/offpeak/v10e-lag1

# Holdout: f0/onpeak
mv holdout/v0 holdout/f0/onpeak/v0
mv holdout/v10e holdout/f0/onpeak/v10e
mv holdout/v10e-lag1 holdout/f0/onpeak/v10e-lag1

# Holdout: f0/offpeak (strip suffix)
mv holdout/v0-offpeak holdout/f0/offpeak/v0
mv holdout/v10e-lag1-offpeak holdout/f0/offpeak/v10e-lag1
```

**Step 5: Verify the new layout**

```bash
echo "=== Registry ==="
find registry -name "*.json" | sort
echo "=== Holdout ==="
find holdout -name "*.json" | sort
echo "=== Archive ==="
ls archive/registry/ archive/holdout/
```

Expected registry tree:
```
registry/f0/onpeak/v0/metrics.json
registry/f0/onpeak/v0/config.json
registry/f0/onpeak/v0/meta.json
registry/f0/onpeak/v10e/metrics.json
registry/f0/onpeak/v10e-lag1/metrics.json
registry/f0/onpeak/v10e-lag1/NOTES.md
registry/f0/onpeak/gates.json
registry/f0/onpeak/champion.json
registry/f0/offpeak/v0/metrics.json
registry/f0/offpeak/v10e-lag1/metrics.json
```

**Step 6: Commit**

```bash
git add -A registry/ holdout/ archive/
git commit -m "refactor: reorganize registry into {period_type}/{class_type}/{version_id} hierarchy

Active versions (v0, v10e, v10e-lag1) migrated to registry/f0/{onpeak,offpeak}/.
Legacy v1-v10d archived to archive/registry/.
Offpeak versions renamed: v0-offpeak -> f0/offpeak/v0."
```

---

## Task 3: Update `ml/benchmark.py` to Use New Paths

**Files:**
- Modify: `ml/benchmark.py`

**Step 1: Update `run_benchmark()` to accept period_type/class_type and use registry_paths**

The key change: `registry_dir` parameter becomes the project-level registry root, and `run_benchmark` computes the slice path internally.

```python
# At top, add import:
from ml.registry_paths import registry_root

# In run_benchmark(), change the path computation (around line 146):
# BEFORE:
#   registry_path = Path(registry_dir)
#   version_dir = registry_path / version_id
# AFTER:
    slice_dir = registry_root(period_type, class_type, base_dir=registry_dir)
    version_path = slice_dir / version_id
    version_path.mkdir(parents=True, exist_ok=True)
```

Also update the `main()` CLI defaults:
```python
# BEFORE: --registry-dir default="registry"
# AFTER:  --registry-dir default="registry" (unchanged — registry_root handles the rest)
```

**Step 2: Run existing benchmark tests (if any) to verify no breakage**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python -m pytest ml/tests/ -v -k benchmark`

**Step 3: Commit**

```bash
git add ml/benchmark.py
git commit -m "refactor: benchmark.py uses registry_paths for hierarchical output"
```

---

## Task 4: Update `ml/compare.py` to Use New Paths

**Files:**
- Modify: `ml/compare.py`

**Step 1: Update `load_all_versions()` and `run_comparison()`**

```python
# At top, add import:
from ml.registry_paths import registry_root, gates_path as _gates_path, champion_path as _champ_path

# In load_all_versions(): add period_type/class_type params
def load_all_versions(
    registry_dir: str | Path,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> dict[str, dict]:
    """Read all registry/{ptype}/{ctype}/v*/metrics.json files."""
    root = registry_root(period_type, class_type, base_dir=registry_dir)
    versions = {}
    for metrics_path in sorted(root.glob("v*/metrics.json")):
        version_id = metrics_path.parent.name
        with open(metrics_path) as f:
            versions[version_id] = json.load(f)
    return versions

# In run_comparison(): update defaults
def run_comparison(
    batch_id: str,
    iteration: int,
    registry_dir: str = "registry",
    period_type: str = "f0",
    class_type: str = "onpeak",
    output_path: str | None = None,
) -> dict:
    slice_root = registry_root(period_type, class_type, base_dir=registry_dir)
    gp = slice_root / "gates.json"
    cp = slice_root / "champion.json"
    # ... rest uses slice_root instead of registry_dir
```

Update `main()` CLI to accept `--ptype` and `--class-type`, remove `--gates-path` and `--champion-path` (derived from hierarchy).

**Step 2: Run compare tests**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python -m pytest ml/tests/ -v -k compare`

**Step 3: Commit**

```bash
git add ml/compare.py
git commit -m "refactor: compare.py uses hierarchical registry paths"
```

---

## Task 5: Update `scripts/run_v0_formula_baseline.py`

**Files:**
- Modify: `scripts/run_v0_formula_baseline.py`

**Step 1: Replace hardcoded path logic with registry_paths**

Key changes:
- Remove the `-offpeak` suffix from `version_id` — offpeak is handled by directory, not name
- Use `registry_root()` and `holdout_root()` for output paths
- `version_id` is always `"v0"` regardless of class_type

```python
# BEFORE (line 157-160):
#   suffix = "" if class_type == "onpeak" else f"-{class_type}"
#   version_id = f"v0{suffix}"
#   v0_dir = registry_dir / version_id

# AFTER:
from ml.registry_paths import registry_root, holdout_root

version_id = "v0"
v0_dir = registry_root(period_type="f0", class_type=class_type, base_dir=registry_dir)
v0_dir = v0_dir / version_id
v0_dir.mkdir(parents=True, exist_ok=True)

# Same pattern for holdout:
holdout_dir = holdout_root(period_type="f0", class_type=class_type,
                           base_dir=Path(__file__).resolve().parent.parent / "holdout")
holdout_dir = holdout_dir / version_id
```

**Step 2: Verify by dry-running with --help**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python scripts/run_v0_formula_baseline.py --help`

**Step 3: Commit**

```bash
git add scripts/run_v0_formula_baseline.py
git commit -m "refactor: v0 baseline uses hierarchical registry paths"
```

---

## Task 6: Update `scripts/run_v10e_lagged.py`

**Files:**
- Modify: `scripts/run_v10e_lagged.py`

**Step 1: Replace path logic**

Key changes:
- Remove `-offpeak` suffix from version_id — offpeak is the directory, not the name
- `REGISTRY` and `HOLDOUT` become functions of (period_type, class_type)
- Comparison section reads from new hierarchy paths

```python
# BEFORE (line 37-38):
#   REGISTRY = ROOT / "registry"
#   HOLDOUT = ROOT / "holdout"
# BEFORE (line 285-286):
#   suffix = "" if class_type == "onpeak" else f"-{class_type}"
#   version_id = f"v10e-lag{lag}{suffix}"

# AFTER:
from ml.registry_paths import registry_root, holdout_root

# In main():
    version_id = f"v10e-lag{lag}"  # No suffix — class_type is in the path
    reg_dir = registry_root("f0", class_type, base_dir=ROOT / "registry")
    ho_dir = holdout_root("f0", class_type, base_dir=ROOT / "holdout")

# In save_results(): use the slice directory directly
def save_results(label, per_month, eval_months, dest_dir, ...):
    d = dest_dir / label  # dest_dir is already registry/f0/onpeak or similar
    ...

# Comparison section: load from correct hierarchy
    v0_dev = json.load(open(reg_dir / "v0" / "metrics.json"))["aggregate"]["mean"]
```

**Step 2: Verify --help**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier && python scripts/run_v10e_lagged.py --help`

**Step 3: Commit**

```bash
git add scripts/run_v10e_lagged.py
git commit -m "refactor: v10e-lagged uses hierarchical registry paths"
```

---

## Task 7: Update `scripts/run_holdout_test.py`

**Files:**
- Modify: `scripts/run_holdout_test.py`

**Step 1: Update holdout paths**

This script is the most affected because it iterates multiple champions and writes results:

```python
# BEFORE:
#   HOLDOUT_DIR = Path(__file__).resolve().parent.parent / "holdout"

# AFTER:
from ml.registry_paths import holdout_root

def _holdout_dir(period_type="f0", class_type="onpeak"):
    return holdout_root(period_type, class_type,
                        base_dir=Path(__file__).resolve().parent.parent / "holdout")

# write_result: use _holdout_dir
# print_comparison: iterate _holdout_dir().iterdir()
```

Add `--class-type` and `--ptype` CLI arguments.

**Step 2: Commit**

```bash
git add scripts/run_holdout_test.py
git commit -m "refactor: holdout test uses hierarchical paths"
```

---

## Task 8: Update Remaining Scripts (Batch)

**Files:**
- Modify: `scripts/cache_realized_da.py` — no registry changes needed (data cache, not registry)
- Modify: `scripts/run_36mo_comparison.py` — update to use `registry_root()`
- Verify: `scripts/run_v1_experiment.py` through `scripts/run_v10_variants.py` — these are archived experiments, but if they still import benchmark.py, ensure they still work or add a deprecation notice

**Step 1: Update run_36mo_comparison.py**

This script likely reads from registry/ to compare versions. Update to use `registry_root()`.

**Step 2: Add deprecation header to archived experiment scripts**

For each of `run_v1_experiment.py`, `run_v1b_experiment.py`, ..., `run_v10_variants.py`, `run_v10_pruned_features.py`:

```python
# Add at top of file:
"""DEPRECATED: This experiment produced archived results in archive/registry/.
Results have been superseded by v10e-lag1. Do not re-run.
Original results preserved in archive/registry/{version_id}/.
"""
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "refactor: update remaining scripts for hierarchical registry"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `mem.md`
- Modify: `experiment-setup.md`
- Modify: `multi-period-extension.md`
- Modify: `docs/audit-report.md`

**Step 1: Update CLAUDE.md**

Add a section documenting the new registry structure:

```markdown
## Registry Structure (MANDATORY)

Results are organized hierarchically by period_type and class_type:

    registry/{period_type}/{class_type}/{version_id}/metrics.json
    holdout/{period_type}/{class_type}/{version_id}/metrics.json

Each (period_type, class_type) slice has its own:
- `gates.json` — quality gate thresholds calibrated from v0
- `champion.json` — current best version for that slice

Use `ml.registry_paths` helpers to construct paths — never hardcode.

Legacy experiments (v1-v10d) are in `archive/registry/`.
```

**Step 2: Update mem.md with new paths**

Replace references like `registry/v10e-lag1/` with `registry/f0/onpeak/v10e-lag1/`.

**Step 3: Update multi-period-extension.md Phase 2**

The "Parameterize for Period Type" section should reference the new structure as already done.

**Step 4: Update audit-report.md file references**

**Step 5: Commit**

```bash
git add CLAUDE.md mem.md experiment-setup.md multi-period-extension.md docs/audit-report.md
git commit -m "docs: update all references for hierarchical registry structure"
```

---

## Task 10: Update Parent CLAUDE.md and Auto-Memory

**Files:**
- Modify: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md` — add registry structure note
- Modify: `/home/xyz/.claude/projects/-home-xyz-workspace-research-qianli-v2/memory/MEMORY.md` — update paths

**Step 1: Update parent CLAUDE.md**

Add under "Versioned Experiments" section:
```markdown
### Registry Layout
Results: `registry/{period_type}/{class_type}/{version_id}/metrics.json`
Each slice has its own `gates.json` and `champion.json`.
Use `ml.registry_paths` — never hardcode paths.
```

**Step 2: Update MEMORY.md**

Replace all `registry/v10e-lag1/` references with `registry/f0/onpeak/v10e-lag1/` etc.

**Step 3: Commit**

```bash
git add /home/xyz/workspace/research-qianli-v2/CLAUDE.md
git commit -m "docs: update parent CLAUDE.md with registry hierarchy"
```

---

## Task 11: Verify End-to-End

**Step 1: Run all tests**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
python -m pytest ml/tests/ -v
```

Expected: All tests pass (registry_paths tests + existing tests).

**Step 2: Verify registry structure is correct**

```bash
# Check active versions exist in new locations
test -f registry/f0/onpeak/v0/metrics.json && echo "OK: v0 onpeak" || echo "MISSING"
test -f registry/f0/onpeak/v10e-lag1/metrics.json && echo "OK: v10e-lag1 onpeak" || echo "MISSING"
test -f registry/f0/offpeak/v0/metrics.json && echo "OK: v0 offpeak" || echo "MISSING"
test -f registry/f0/offpeak/v10e-lag1/metrics.json && echo "OK: v10e-lag1 offpeak" || echo "MISSING"
test -f registry/f0/onpeak/gates.json && echo "OK: gates" || echo "MISSING"
test -f registry/f0/onpeak/champion.json && echo "OK: champion" || echo "MISSING"

# Check holdout
test -f holdout/f0/onpeak/v0/metrics.json && echo "OK: holdout v0" || echo "MISSING"
test -f holdout/f0/onpeak/v10e-lag1/metrics.json && echo "OK: holdout v10e-lag1" || echo "MISSING"

# Check archive has the legacy versions
ls archive/registry/ | head -20
```

**Step 3: Verify compare.py works on new structure**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage5-tier
python ml/compare.py --batch-id reorg-verify --iteration 1 --ptype f0 --class-type onpeak
```

Expected: Loads v0, v10e, v10e-lag1 from `registry/f0/onpeak/`, produces comparison.

**Step 4: Final commit**

```bash
git add -A
git commit -m "verify: end-to-end registry reorganization complete"
```

---

## Summary of Version Name Changes

| Old Path | New Path |
|----------|----------|
| `registry/v0/` | `registry/f0/onpeak/v0/` |
| `registry/v0-offpeak/` | `registry/f0/offpeak/v0/` |
| `registry/v10e/` | `registry/f0/onpeak/v10e/` |
| `registry/v10e-lag1/` | `registry/f0/onpeak/v10e-lag1/` |
| `registry/v10e-lag1-offpeak/` | `registry/f0/offpeak/v10e-lag1/` |
| `registry/gates.json` | `registry/f0/onpeak/gates.json` |
| `registry/champion.json` | `registry/f0/onpeak/champion.json` |
| `holdout/v0/` | `holdout/f0/onpeak/v0/` |
| `holdout/v10e-lag1/` | `holdout/f0/onpeak/v10e-lag1/` |
| `holdout/v0-offpeak/` | `holdout/f0/offpeak/v0/` |
| `holdout/v10e-lag1-offpeak/` | `holdout/f0/offpeak/v10e-lag1/` |
| `registry/v1/` ... `registry/v10d/` | `archive/registry/v1/` ... |
