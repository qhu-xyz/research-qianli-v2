# Repo Reorganization Plan: MISO + PJM (v2)

**Date**: 2026-03-21
**Goal**: Restructure to support both MISO and PJM annual signals.
**Current state**: All code is MISO-specific. 98/99 tests pass (1 failure: `test_bridge_hive_scan_not_used`). V7.0B.R1 published.

---

## 1. Import Strategy: Option A

Keep `ml` as the Python package name. Each RTO gets its own `ml/` under its directory. Scripts use PYTHONPATH to select which RTO's `ml` to import.

```
PYTHONPATH=miso  → from ml.config import ...  (MISO config)
PYTHONPATH=pjm   → from ml.config import ...  (PJM config)
```

**Why**: Zero Python import rewrites across 243 lines in 47 files.

---

## 2. Root-Relative Path Contracts That Break

Python `from ml.X import Y` imports are fine under Option A. The real work is filesystem paths that assume the working directory or `PROJECT_ROOT` resolves to the repo root.

### 2.1 `PROJECT_ROOT` in `ml/config.py`

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # line 11
DA_CACHE_DIR = PROJECT_ROOT / "data" / "realized_da"   # line 12
COLLAPSED_CACHE_DIR = PROJECT_ROOT / "data" / "collapsed"  # line 13
REGISTRY_DIR = PROJECT_ROOT / "registry"                # line 14
```

After move: `__file__` = `miso/ml/config.py`, `.parent.parent` = `miso/`. This resolves to `miso/data/` and `miso/registry/` — correct. **No change needed.**

### 2.2 Relative output paths in scripts (CWD-dependent)

These scripts write to relative paths and assume CWD = repo root:

| File | Path assumption | After move |
|------|----------------|------------|
| `scripts/phase6/run_model_ladder.py:194` | `registry/{class}/m2_{split}` | Must run from `miso/` or fix to use `PROJECT_ROOT` |
| `scripts/phase6/run_learned_weights.py:252` | `registry/{class}/phase7_{split}` | Same |
| `scripts/v7_step1_sanity.py:25` | `Path("data/v7_verification")` | Same |
| `scripts/v7_step2_da_merge.py:36` | `Path("data/v7_verification")` | Same |
| `scripts/v7_mapping_tables.py:29` | `Path("data/v7_verification")` | Same |

**Fix**: Change these to use `PROJECT_ROOT / "registry"` or `PROJECT_ROOT / "data"` from `ml.config` instead of relative paths.

### 2.3 `sys.path.insert` in scripts

| File | What it does | After move |
|------|-------------|------------|
| `scripts/calibrate_threshold.py:27` | `sys.path.insert(0, Path(__file__).parent.parent)` | Still resolves to `miso/` — correct |
| `scripts/v7_step2_da_merge.py:26` | `sys.path.insert(0, dirname(__file__) + "/..")` | Still resolves to `miso/` — correct |
| `scripts/v7_mapping_tables.py:21` | Same pattern | Correct |
| `scripts/publish_annual_signal.py:23-24` | Hardcoded psignal/pbase paths | Unchanged |
| `ml/signal_publisher.py:307-308` | Hardcoded psignal/pbase paths | Unchanged |

### 2.4 PYTHONPATH in script docstrings

10 scripts have `PYTHONPATH=.` in their usage docstrings. After move, these must say `PYTHONPATH=miso` (or the scripts must be run from `miso/` as CWD).

| File | Current | After move |
|------|---------|------------|
| `scripts/run_phase5_reeval.py:8` | `PYTHONPATH=.` | `PYTHONPATH=miso` |
| `scripts/v7_step2_da_merge.py:12` | `PYTHONPATH=.` | `PYTHONPATH=miso` |
| `scripts/v7_mapping_tables.py:10` | `PYTHONPATH=.` | `PYTHONPATH=miso` |
| `scripts/publish_annual_signal.py:7-9` | `PYTHONPATH=.` | `PYTHONPATH=miso` |
| `scripts/phase6/run_learned_weights.py:6-8` | `PYTHONPATH=.` | `PYTHONPATH=miso` |
| `scripts/phase6/run_model_ladder.py:8-9` | `PYTHONPATH=.` | `PYTHONPATH=miso` |

### 2.5 Docs referencing current paths

CLAUDE.md, README.md, and various docs reference paths like `ml/config.py`, `scripts/`, `registry/`. These need `miso/` prefix added.

---

## 3. What Moves Where

| Current path | New path | Notes |
|-------------|----------|-------|
| `ml/` | `miso/ml/` | Python package — no import changes |
| `scripts/` | `miso/scripts/` | Fix relative output paths + docstrings |
| `tests/` | `miso/tests/` | No changes needed |
| `registry/` | `miso/registry/` | Referenced via `PROJECT_ROOT` — auto-resolves |
| `data/` | `miso/data/` | gitignored, caches rebuild on first run |
| `cache/` | `miso/cache/` | gitignored |
| `human-inputs/` | `miso/human-inputs/` | No code references |
| `reviews/` | `miso/reviews/` | No code references |
| `README.md` | Rewrite at top level | New structure overview |

### Docs split

| Current | New | Reason |
|---------|-----|--------|
| `docs/*.md` (MISO-specific) | `miso/docs/` | Signal docs, verification, coverage, handoff |
| (new) `docs/` | Top-level cross-RTO | Reorg plan, future comparison reports |

---

## 4. Execution Order

### Phase 1: Fix path contracts (before moving)

1. **Fix relative output paths** in 5 scripts: replace `Path("data/...")` and `f"registry/..."` with `ml.config.PROJECT_ROOT / "data/..."` and `ml.config.REGISTRY_DIR / ...`
2. **Verify**: run 98/99 tests with these fixes (still at repo root)

### Phase 2: Move files

3. **git mv** all directories into `miso/` in one pass
4. **Create** `pjm/` skeleton (empty dirs + `__init__.py` + README)
5. **Split docs**: MISO-specific → `miso/docs/`, cross-RTO → top-level `docs/`

### Phase 3: Fix references

6. **Update script docstrings**: `PYTHONPATH=.` → `PYTHONPATH=miso`
7. **Update .gitignore**: `data/` → `miso/data/`, `pjm/data/`
8. **Rewrite top-level README**: document the two-RTO structure + PYTHONPATH convention

### Phase 4: Verify

9. **Run tests**: `PYTHONPATH=miso pytest miso/tests/` — must be 98/99 (same as before)
10. **Dry-test** one script: `PYTHONPATH=miso python miso/scripts/v7_step1_sanity.py`
11. **Verify** `PROJECT_ROOT` resolves: `PYTHONPATH=miso python -c "from ml.config import PROJECT_ROOT; print(PROJECT_ROOT)"`

### Phase 5: Commit

12. **Commit atomically**: one commit with all moves + fixes
13. **Push to both repos**

---

## 5. What PJM Will Need (future, not part of this refactor)

PJM annual signal will need its own:
- `pjm/ml/config.py` — PJM data paths, density bins, thresholds, planning years
- `pjm/ml/bridge.py` — PJM uses different bridge structure
- `pjm/ml/ground_truth.py` — PJM DA format differs
- `pjm/ml/data_loader.py` — PJM density distribution loading
- `pjm/ml/history_features.py` — PJM BF computation
- `pjm/ml/signal_publisher.py` — PJM signal schema may differ

Can reuse (copy initially):
- `evaluate.py` — metric functions are RTO-agnostic
- `train.py` — LightGBM training is RTO-agnostic
- `registry.py` — save/load is RTO-agnostic

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Scripts fail when not run from `miso/` CWD | Phase 1 fixes all CWD-dependent paths to use `PROJECT_ROOT` |
| PYTHONPATH confusion | Document clearly in README + CLAUDE.md |
| `data/` local state lost | gitignored, caches rebuild. Verification artifacts at `/opt/temp/` are unaffected |
| Developer runs `pytest` from repo root | Will fail — document correct invocation in README |
| IDE/test runner defaults | Document PYTHONPATH setting for VSCode/Cursor |

---

## 7. What NOT to do

- Do NOT rename the `ml` package
- Do NOT create a `shared/` package yet
- Do NOT move files before fixing path contracts (Phase 1 before Phase 2)
- Do NOT do the move in multiple commits
- Do NOT assume "no import rewrites" means "nothing breaks"
