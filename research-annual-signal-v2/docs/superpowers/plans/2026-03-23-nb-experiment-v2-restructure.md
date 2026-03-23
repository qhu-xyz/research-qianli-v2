# NB-hist-12 Experiment V2: Per-Ctype Restructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the NB-hist-12 reserved-slot experiment with correct per-ctype everything: 2 NB models (per-ctype dormant universes), class-specific v0c, class-specific V4.4 benchmark, per-group LambdaRank labels, and proper repo integration.

**Architecture:** Two NB models — one per class type. The onpeak model trains on `bf_12 == 0` branches with `onpeak_sp` target; the offpeak model trains on `bfo_12 == 0` branches with `offpeak_sp` target. Features are class-agnostic (density bins + combined DA history), but training population, target, and candidate pool are class-specific. Evaluation uses `build_class_model_table` for fully class-specific v0c (class-specific `da_rank_value`, `shadow_price_da`, BF). V4.4 loaded per-ctype as benchmark. All configs maintain fixed K via v0c backfill when NB/V4.4 can't fill reserved slots.

**Tech Stack:** polars, numpy, lightgbm (LambdaRank), `ml/phase6/features.py` (`build_class_model_table`), `ml/features.py` (`build_model_table` for combined data only during build phase)

---

## What changed from V1

| Issue | V1 (broken) | V2 (fix) |
|-------|-------------|----------|
| NB model count | 1 combined model | **2 models**: onpeak (`bf_12==0`, target=`onpeak_sp`) and offpeak (`bfo_12==0`, target=`offpeak_sp`) |
| NB dormant definition | `bf_combined_12 == 0` | **Per-ctype**: onpeak dormant = `bf_12 == 0`, offpeak dormant = `bfo_12 == 0` |
| V4.4 as feature | Used onpeak-only V4.4 deviation features in NB model | **Dropped from model** — not reproducible, limits NB universe |
| V4.4 comparison | Compared onpeak V4.4 rank against combined SP | **Per-ctype benchmark**: onpeak V4.4 for onpeak, offpeak V4.4 for offpeak |
| V4.4 underfill | Unfilled slots left empty (K not fixed) | **v0c backfill**: unfilled V4.4 NB slots are filled by next-best v0c picks to maintain fixed K |
| v0c formula | Combined `bf_combined_12` + combined `da_rank_value` | **Fully class-specific**: `bf_12`/`bfo_12`, class-specific `da_rank_value` and `shadow_price_da` from `build_class_model_table` |
| Target SP | Combined `realized_shadow_price` for eval | **Per-ctype**: from class-specific builder's `realized_shadow_price` |
| Abs_SP denominator | Combined `total_da_sp_quarter` | **Per-ctype**: from class-specific builder (sourced from GT `onpeak_total_da_sp` / `offpeak_total_da_sp`) |
| LambdaRank labels | Global tertiles across all training quarters | **Per-group** tertiles (one tier distribution per (PY, aq) group) |
| NB universe | Only V4.4-covered dormant branches | **All** per-ctype dormant branches |
| NB-only V4.4 denominator | Undefined | **Full per-ctype dormant universe** — V4.4 missing branches scored -inf, kept in denominator |
| Report matrix | Collapsed dimensions | **Explicit**: (2024, 2025) × (onpeak, offpeak) × (K=200, K=400) = 8 tables + 4 aggregate |
| NDCG / label_tier | Combined labels | **Per-ctype**: `build_class_model_table` recomputes `label_tier` from class-specific SP. NDCG valid per-ctype. |
| NB metrics masks | Conflated dormant with binder | **Two explicit masks**: `is_dormant = (class_bf == 0)` for candidate pool; `is_nb_binder = is_dormant & (sp > 0)` for binder metrics. Do NOT reuse pipeline's `is_nb_12` / `nb_onpeak_12`. |
| Fill count reporting | Only V4.4 | **ALL reserved-slot configs** (R*_nb AND R*_v44): how many of N_nb were genuine NB picks vs v0c backfill |
| Case studies | Combined SP, combined ranks | **Per-ctype**: per-ctype SP, per-ctype ranks for v0c/V4.4/ML_nb |
| Repo integration | Loose script, no registry | Registry entries per ctype, proper docs |

## What stays the same

- **NB model features**: Class-agnostic (density bins + combined DA history) — same 8 features for both models
- **Rolling CV**: train 2021-2022 → eval 2023, train 2021-2023 → eval 2024, train 2021-2024 → eval 2025
- **Reserved-slot mechanism**: v0c picks top N_v0c, then NB/V4.4 picks top N_nb from remaining dormant branches
- **K levels**: 200 (Tier 0) and 400 (Tier 0+1)

## Why 2 NB models

The per-ctype dormant universes are different:
- Onpeak dormant (`bf_12 == 0`): includes branches that bind in offpeak but not onpeak
- Offpeak dormant (`bfo_12 == 0`): includes branches that bind in onpeak but not offpeak
- Combined dormant (`bf_combined_12 == 0`): excludes both of the above — strictly smaller

A branch that is dormant in onpeak but active in offpeak is exactly the kind of branch we want the onpeak NB model to detect. Using `bf_combined_12 == 0` would exclude it. The training population, target label distribution, and candidate pool all differ by ctype, which is sufficient to justify separate models even with identical feature families.

## NB model features (shared across both models)

| Feature | Source | Notes |
|---------|--------|-------|
| `bin_80_cid_max` | density | Right-tail density bins |
| `bin_90_cid_max` | density | |
| `bin_100_cid_max` | density | |
| `bin_110_cid_max` | density | |
| `rt_max` | derived | `max(bin_80, bin_90, bin_100, bin_110)` |
| `count_active_cids` | metadata | Number of active SPICE CIDs for branch |
| `shadow_price_da` | history (combined) | Historical DA SP — combined |
| `da_rank_value` | history (combined) | Rank of historical DA SP — combined |

8 features. Note: `shadow_price_da` and `da_rank_value` are from the combined builder, not class-specific. This is a pragmatic constraint — the class-specific builder recomputes these per ctype, but for NB training we use the combined version because (a) `build_model_table` is already available for all PYs and (b) for dormant branches with >12mo stale history, the combined and per-ctype values are correlated but not identical. We acknowledge this is an approximation, not a data-equivalence claim.

## Two data builders used

| Builder | Used for | Class-specific? |
|---------|----------|----------------|
| `ml.features.build_model_table(py, aq)` | NB model training data — provides combined features for NB training population selection + combined `onpeak_sp`/`offpeak_sp` for per-ctype target extraction | No — combined `da_rank_value`, `shadow_price_da` |
| `ml.phase6.features.build_class_model_table(py, aq, ct)` | Per-ctype evaluation — provides class-specific `da_rank_value`, `shadow_price_da`, BF, target, `total_da_sp_quarter` for v0c scoring and metric computation | Yes — fully class-specific |

**NB training data flow:**
1. `build_model_table(py, aq)` → full table with combined features + `onpeak_sp`, `offpeak_sp`, `bf_12`, `bfo_12`
2. For onpeak NB: filter `bf_12 == 0`, target = `onpeak_sp`
3. For offpeak NB: filter `bfo_12 == 0`, target = `offpeak_sp`

**Eval data flow:**
1. `build_class_model_table(py, aq, ct)` → class-specific table
2. v0c computed from class-specific `da_rank_value`, `rt_max`, class BF
3. NB scores looked up by branch name (from per-ctype trained model)
4. Metrics computed using class-specific target and class-specific `total_da_sp_quarter`

## Configs to test

| Config | K=200 split | K=400 split | NB scorer |
|--------|-------------|-------------|-----------|
| `pure_v0c` | 200 v0c | 400 v0c | — |
| `pure_v44` | 200 V4.4 | 400 V4.4 | — |
| `R30_nb` | 170 v0c + 30 NB | 350 v0c + 50 NB | ML_nb (per-ctype) |
| `R50_nb` | 150 v0c + 50 NB | 300 v0c + 100 NB | ML_nb (per-ctype) |
| `R30_v44` | 170 v0c + 30 V4.4-NB | 350 v0c + 50 V4.4-NB | V4.4 (per-ctype, from dormant only) |
| `R50_v44` | 150 v0c + 50 V4.4-NB | 300 v0c + 100 V4.4-NB | V4.4 (per-ctype, from dormant only) |

**V4.4 underfill handling:** When V4.4 cannot fill all N_nb reserved slots (because some dormant branches have no V4.4 coverage), the remaining slots are backfilled by the next-best v0c branches not already selected. This ensures all configs have exactly K branches selected, making VC/Recall/Prec comparable across configs. The number of slots actually filled by V4.4 (vs backfilled by v0c) is reported separately.

**NB-only comparison (Part 1):** VC/Rec computed on the full per-ctype dormant universe. V4.4 missing branches are scored -inf (they rank last). This means V4.4's NB-only metrics reflect its coverage gap — it can only rank the ~60% of dormant branches it covers, while ML_nb ranks all of them. This is the fair comparison: both scorers face the same dormant universe.

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/nb_experiment_v2.py` | Create | Main experiment: data build (both builders), train 2 NB models, eval per-ctype, V4.4 benchmark, case studies, registry save |
| `tests/test_nb_experiment_v2.py` | Create | Tests: per-group tiering, v0c per-ctype, slot allocation (including backfill), per-ctype NB population, V4.4 real loader |
| `registry/onpeak/nb_v2/metrics.json` | Create | Onpeak results |
| `registry/onpeak/nb_v2/config.json` | Create | Experiment config |
| `registry/offpeak/nb_v2/metrics.json` | Create | Offpeak results |
| `registry/offpeak/nb_v2/config.json` | Create | Experiment config |
| `docs/2026-03-23-nb-v2-experiment-report.md` | Create | Full results: 8 tables + aggregates + case studies |
| `README.md` | Modify | Add NB V2 section |

No changes to `ml/` — `build_model_table` and `build_class_model_table` already provide everything needed.

---

## Tasks

### Task 1: Unit tests for core logic

**Files:**
- Create: `tests/test_nb_experiment_v2.py`

- [ ] **Step 1: Write tests**

```python
import numpy as np
import os
import pytest

# Functions defined in scripts/nb_experiment_v2.py
from scripts.nb_experiment_v2 import (
    assign_tiers_per_group, compute_v0c, allocate_reserved_slots, _minmax,
    load_v44,
)


# ── Per-group tiering ──────────────────────────────────────────────────

def test_assign_tiers_per_group_basic():
    """Each group gets its own 0/1/2/3 distribution independently."""
    sp = np.array([0, 0, 0, 10, 20, 30,       # group 1: small SP
                   0, 0, 0, 1000, 2000, 3000])  # group 2: large SP
    groups = np.array([6, 6])
    labels = assign_tiers_per_group(sp, groups)
    assert list(labels[:6]) == [0, 0, 0, 1, 2, 3]
    assert list(labels[6:]) == [0, 0, 0, 1, 2, 3]


def test_assign_tiers_per_group_no_binders():
    """Group with no binders gets all zeros."""
    sp = np.array([0, 0, 0,   10, 20, 30])
    groups = np.array([3, 3])
    labels = assign_tiers_per_group(sp, groups)
    assert list(labels[:3]) == [0, 0, 0]
    assert list(labels[3:]) == [1, 2, 3]


def test_assign_tiers_per_group_single_binder():
    """Group with 1 binder assigns tier 3 (top)."""
    sp = np.array([0, 0, 100])
    groups = np.array([3])
    labels = assign_tiers_per_group(sp, groups)
    assert labels[2] == 3
    assert labels[0] == 0


# ── v0c formula ────────────────────────────────────────────────────────

def test_v0c_onpeak_vs_offpeak():
    """v0c produces different scores when bf differs (onpeak bf_12 vs offpeak bfo_12)."""
    da = np.array([1.0, 2.0, 3.0])
    rt = np.array([0.1, 0.5, 0.9])
    bf_12 = np.array([0.0, 0.5, 1.0])
    bfo_12 = np.array([1.0, 0.0, 0.0])
    on_score = compute_v0c(da, rt, bf_12)
    off_score = compute_v0c(da, rt, bfo_12)
    assert not np.allclose(on_score, off_score)


def test_v0c_weights():
    """v0c = 0.40*(1-minmax(da)) + 0.30*minmax(rt) + 0.30*minmax(bf)."""
    da = np.array([10.0, 20.0])
    rt = np.array([0.0, 1.0])
    bf = np.array([0.0, 1.0])
    scores = compute_v0c(da, rt, bf)
    expected = np.array([0.40*1+0.30*0+0.30*0, 0.40*0+0.30*1+0.30*1])
    np.testing.assert_allclose(scores, expected)


# ── Reserved-slot allocation ───────────────────────────────────────────

def test_reserved_slot_allocation_basic():
    """NB slots filled from dormant population only, not already in v0c picks."""
    n = 100
    rng = np.random.RandomState(42)
    v0c_scores = rng.rand(n)
    nb_scores = rng.rand(n)
    is_dormant = np.zeros(n, dtype=bool)
    is_dormant[80:] = True
    selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=8, n_nb=2)
    assert len(selected) == 10  # always K = n_v0c + n_nb
    assert nb_filled == 2  # enough dormant to fill
    v0c_picks = set(int(x) for x in np.argsort(v0c_scores)[::-1][:8])
    nb_picks = selected - v0c_picks
    for idx in nb_picks:
        assert is_dormant[idx], f"NB pick {idx} is not dormant"


def test_reserved_slot_backfill():
    """When NB scorer can't fill all slots, remaining are backfilled by v0c. nb_filled < n_nb."""
    n = 20
    v0c_scores = np.arange(n, dtype=float)  # higher = better
    nb_scores = np.full(n, -np.inf)
    is_dormant = np.zeros(n, dtype=bool)
    is_dormant[:2] = True  # only 2 dormant, both low v0c
    nb_scores[0] = 1.0
    nb_scores[1] = 0.5
    # Request 15 v0c + 5 NB: only 2 dormant available not already in v0c
    selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=15, n_nb=5)
    assert len(selected) == 20  # K = 15 + 5 = 20, backfill ensures fixed K
    assert nb_filled <= 2  # at most 2 dormant available


def test_reserved_slot_total_k_fixed():
    """Total selected always equals n_v0c + n_nb regardless of NB coverage."""
    for n_dormant in [0, 5, 50, 200]:
        n = 300
        v0c_scores = np.random.RandomState(42).rand(n)
        nb_scores = np.full(n, -np.inf)
        is_dormant = np.zeros(n, dtype=bool)
        is_dormant[:n_dormant] = True
        nb_scores[:n_dormant] = np.random.RandomState(42).rand(n_dormant)
        selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=150, n_nb=50)
        assert len(selected) == 200, f"K not fixed at 200 for n_dormant={n_dormant}: got {len(selected)}"
        assert nb_filled <= min(n_dormant, 50)


# ── Per-ctype NB population ───────────────────────────────────────────

def test_per_ctype_dormant_differs():
    """Onpeak dormant (bf_12==0) and offpeak dormant (bfo_12==0) are different sets."""
    bf_12 = np.array([0, 0, 0.5, 0.5, 0])
    bfo_12 = np.array([0.5, 0, 0, 0, 0])
    on_dormant = bf_12 == 0
    off_dormant = bfo_12 == 0
    assert list(on_dormant) == [True, True, False, False, True]
    assert list(off_dormant) == [False, True, True, True, True]
    assert not np.array_equal(on_dormant, off_dormant)


# ── V4.4 loader (real path) ───────────────────────────────────────────

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"

@pytest.mark.skipif(
    not os.path.exists(f"{V44_BASE}/2025-06/aq1/onpeak/"),
    reason="V4.4 data not available",
)
def test_v44_loader_returns_different_ranks_per_ctype():
    """load_v44 returns different rank values for onpeak vs offpeak."""
    on = load_v44("2025-06", "aq1", "onpeak")
    off = load_v44("2025-06", "aq1", "offpeak")
    assert len(on) > 0, "onpeak V4.4 empty"
    assert len(off) > 0, "offpeak V4.4 empty"
    assert len(on) == len(off), "onpeak/offpeak V4.4 different branch count"
    # At least some branches should have different ranks
    common = set(on.keys()) & set(off.keys())
    n_diff = sum(1 for b in common if on[b].get("rank") != off[b].get("rank"))
    assert n_diff > len(common) * 0.5, f"Only {n_diff}/{len(common)} ranks differ — suspiciously similar"


@pytest.mark.skipif(
    not os.path.exists(f"{V44_BASE}/2025-06/aq1/onpeak/"),
    reason="V4.4 data not available",
)
def test_v44_loader_both_ctypes_have_data():
    """Both onpeak and offpeak V4.4 directories exist and load non-empty."""
    for ct in ["onpeak", "offpeak"]:
        data = load_v44("2025-06", "aq1", ct)
        assert len(data) > 1000, f"{ct} V4.4 has only {len(data)} branches"
        # Each entry should have 'rank' and 'equipment'
        sample = next(iter(data.values()))
        assert "rank" in sample, f"{ct} V4.4 missing 'rank' column"


# ── Per-ctype target invariant ────────────────────────────────────────

def test_per_ctype_target_sum():
    """onpeak_sp + offpeak_sp == realized_shadow_price (combined) within tolerance."""
    combined = np.array([100.0, 0.0, 50.0, 200.0])
    onpeak = np.array([60.0, 0.0, 30.0, 150.0])
    offpeak = np.array([40.0, 0.0, 20.0, 50.0])
    np.testing.assert_allclose(combined, onpeak + offpeak)
```

- [ ] **Step 2: Run tests — expect FAIL (imports missing)**

Run: `source /home/xyz/workspace/pmodel/.venv/bin/activate && PYTHONPATH=. python -m pytest tests/test_nb_experiment_v2.py -v`
Expected: ImportError

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_nb_experiment_v2.py
git commit -m "test: add NB experiment V2 unit tests (red)"
```

### Task 2: Build the experiment script

**Files:**
- Create: `scripts/nb_experiment_v2.py`

The script has 6 phases:

#### Phase 1: Build data (combined builder — all PYs)

```python
from ml.features import build_model_table

# Build once for all PYs — provides combined features + per-ctype SP + per-ctype BF
for py in all_pys:
    for aq in aqs:
        table = build_model_table(py, aq)
        for r in table.iter_rows(named=True):
            all_rows.append({
                "branch": r["branch_name"], "py": py, "aq": aq,
                # Per-ctype targets (from combined builder — it exposes both)
                "sp_onpeak": r["onpeak_sp"],
                "sp_offpeak": r["offpeak_sp"],
                "sp_combined": r["realized_shadow_price"],
                # Per-ctype dormant flags
                "bf_12": r["bf_12"],
                "bfo_12": r["bfo_12"],
                "bf_combined_12": r["bf_combined_12"],
                # NB model features (combined, class-agnostic)
                "bin_80_max": r["bin_80_cid_max"],
                "bin_90_max": r["bin_90_cid_max"],
                "bin_100_max": r["bin_100_cid_max"],
                "bin_110_max": r["bin_110_cid_max"],
                "rt_max": max(r["bin_80_cid_max"], r["bin_90_cid_max"],
                              r["bin_100_cid_max"], r["bin_110_cid_max"]),
                "count_active_cids": r["count_active_cids"],
                "shadow_price_da": r["shadow_price_da"],  # combined
                "da_rank_value": r["da_rank_value"],       # combined
            })
# Assert per-ctype target invariant
assert (df["sp_onpeak"] + df["sp_offpeak"] - df["sp_combined"]).abs().max() < 0.01
```

#### Phase 2: Train 2 NB models per eval year

```python
NB_FEATURES = [
    "bin_80_max", "bin_90_max", "bin_100_max", "bin_110_max",
    "rt_max", "count_active_cids", "shadow_price_da", "da_rank_value",
]

for eval_py, train_pys in eval_configs:
    for ct in ["onpeak", "offpeak"]:
        dormant_col = "bf_12" if ct == "onpeak" else "bfo_12"
        target_col = "sp_onpeak" if ct == "onpeak" else "sp_offpeak"

        # Train on per-ctype dormant population with per-ctype target
        train = df.filter(
            pl.col("py").is_in(train_pys)
            & (pl.col(dormant_col) == 0)
        )
        groups = train.group_by(["py", "aq"], maintain_order=True).len()["len"].to_numpy()
        y = assign_tiers_per_group(train[target_col].to_numpy(), groups)
        # ... train LambdaRank model ...
        nb_models[(eval_py, ct)] = model
```

#### Phase 3: Build eval data (class-specific builder) + load V4.4

```python
from ml.phase6.features import build_class_model_table

for eval_py in eval_pys:
    for aq in aqs:
        for ct in ["onpeak", "offpeak"]:
            # Class-specific table: da_rank_value, shadow_price_da, target, total_da_sp
            ct_table = build_class_model_table(eval_py, aq, ct)
            # V4.4 per-ctype
            v44 = load_v44(eval_py, aq, ct)
```

#### Phase 4: Eval loop

For each `(eval_py, aq, class_type)`:

1. Load class-specific table → class-specific `da_rank_value`, class BF, target SP, `total_da_sp_quarter`
2. Compute class-specific v0c: `compute_v0c(class_da_rank, rt_max, class_bf)`
3. Load per-ctype V4.4 → `v44_score = 1.0 - v44_rank` (branches without V4.4 get -inf)
4. Look up per-ctype NB model scores (from Phase 2, keyed by branch name)
5. Build two masks (NOT the same thing):
   - `is_dormant = (class_bf == 0)` — the full candidate pool for NB reserved slots
   - `is_nb_binder = is_dormant & (target_sp > 0)` — dormant branches that actually bind (for NB_bind, NB_SP, NB_VC, NB_Rec)
   - Do NOT reuse `is_nb_12` / `nb_onpeak_12` from the pipeline — those are binder-only flags that also check binding history window, not just `bf == 0`
6. For each config: `allocate_reserved_slots` with v0c backfill → compute metrics

**`allocate_reserved_slots` with backfill:**
```python
def allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c, n_nb):
    """Select n_v0c by v0c + n_nb by NB from dormant. Backfill with v0c if NB underfills.

    Returns (selected: set[int], nb_filled: int) where nb_filled is how many of the
    n_nb slots were genuinely filled by the NB scorer (vs v0c backfill).
    """
    K = n_v0c + n_nb
    selected = set()
    v0c_order = np.argsort(v0c_scores)[::-1]
    # v0c picks
    for idx in v0c_order:
        if len(selected) >= n_v0c:
            break
        selected.add(int(idx))
    # NB picks from dormant not already selected
    nb_filled = 0
    if n_nb > 0:
        candidates = [(i, nb_scores[i]) for i in range(len(v0c_scores))
                      if is_dormant[i] and i not in selected and np.isfinite(nb_scores[i])]
        candidates.sort(key=lambda x: -x[1])
        for idx, _ in candidates[:n_nb]:
            selected.add(idx)
            nb_filled += 1
    # Backfill: if NB couldn't fill all slots, add next-best v0c
    if len(selected) < K:
        for idx in v0c_order:
            if len(selected) >= K:
                break
            selected.add(int(idx))
    return selected, nb_filled
```

**Metrics computed per eval group:**
- VC, Abs_SP (per-ctype denominator), Rec, Prec, Bind, SP_cap, NDCG (using class-specific `label_tier` from class builder)
- NB_in (`is_dormant` in top-K), NB_bind (`is_nb_binder` in top-K), NB_SP (SP from `is_nb_binder` in top-K)
- NB_VC (NB_SP / total dormant-binder SP), NB_Rec (NB_bind / total dormant binders)
- D20 (per-ctype SP > $20K), D50 (per-ctype SP > $50K)
- **Fill count (ALL reserved-slot configs)**: how many of N_nb slots were genuinely NB/V4.4 picks vs v0c backfill. This applies to R*_nb AND R*_v44 — if the dormant candidate pool after top-N_v0c is smaller than N_nb, backfill silently inflates the result with v0c picks.

#### Phase 5: Case studies (per-ctype)

For each `(eval_py, aq, class_type)`, find branches with per-ctype SP > $20K. Report:
- Branch name, per-ctype SP, per-ctype NB flag
- v0c rank (class-specific — from `build_class_model_table` features, full universe)
- V4.4 rank (class-specific, full universe — -inf if no V4.4 coverage)
- ML_nb rank (per-ctype model, full universe)
- In/out for each config at K=200 and K=400

**All ranks on full universe for the given class type.**

#### Phase 6: Report + registry save

**Report structure:**
- Part 1: NB-only metrics per (eval_py, class_type, K) — scorers: ML_nb, V4.4 (with -inf for missing), v0c
  - NB universe = full per-ctype dormant. V4.4 missing branches scored -inf = rank last.
- Part 2: Full universe — 8 year-ctype-K tables: (2024, 2025) × (onpeak, offpeak) × (K=200, K=400)
  - Configs: pure_v0c, pure_v44, R30_nb, R30_v44, R50_nb, R50_v44
- Part 3: 4 aggregate tables: (onpeak, offpeak) × (K=200, K=400)
- Part 4: Delta vs pure_v0c per (class_type, K)
- Part 5: Case studies per (eval_py, aq, class_type) — onpeak and offpeak separately
- Part 6: Fill rate analysis per (config, class_type, K) — for ALL reserved-slot configs (R*_nb AND R*_v44): genuine NB picks vs v0c backfill

**Registry save:**
```python
import json
for ct in ["onpeak", "offpeak"]:
    path = f"registry/{ct}/nb_v2"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/config.json", "w") as f:
        json.dump(config[ct], f, indent=2)  # includes: features, hyperparams, dormant_col, target_col
    with open(f"{path}/metrics.json", "w") as f:
        json.dump(metrics[ct], f, indent=2)
```

- [ ] **Step 4: Write `scripts/nb_experiment_v2.py`**

Key invariants to assert:
```python
# Per-ctype target sanity (from combined builder)
assert (df["sp_onpeak"] + df["sp_offpeak"] - df["sp_combined"]).abs().max() < 0.01

# Per-ctype total_da_sp is constant within quarter
assert ct_table["total_da_sp_quarter"].n_unique() == 1

# NB scores mapped correctly
assert j == len(nb_pred), f"NB score mapping mismatch: {j} != {len(nb_pred)}"

# Fixed K
assert len(selected) == n_v0c + n_nb, f"K not fixed: {len(selected)} != {n_v0c + n_nb}"
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `PYTHONPATH=. python -m pytest tests/test_nb_experiment_v2.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add scripts/nb_experiment_v2.py tests/test_nb_experiment_v2.py
git commit -m "feat: NB experiment V2 — 2 per-ctype models, V4.4 benchmark, per-group labels"
```

### Task 3: Run experiment and save registry artifacts

**Files:**
- Create: `registry/onpeak/nb_v2/metrics.json`, `registry/onpeak/nb_v2/config.json`
- Create: `registry/offpeak/nb_v2/metrics.json`, `registry/offpeak/nb_v2/config.json`

- [ ] **Step 7: Run the experiment**

```bash
source /home/xyz/workspace/pmodel/.venv/bin/activate
RAY_ADDRESS=ray://10.8.0.36:10001 PYTHONPATH=. uv run python scripts/nb_experiment_v2.py
```
Expected: ~400s (builds combined tables for all PYs + class-specific tables for eval years + V4.4 loading).

- [ ] **Step 8: Verify registry artifacts**

```bash
cat registry/onpeak/nb_v2/config.json | python -m json.tool
cat registry/offpeak/nb_v2/config.json | python -m json.tool
# Verify config has dormant_col, target_col per ctype
# Verify no V4.4 features in feature list
grep -i "dev_max\|dev_sum\|pct100\|shadow_rank_v44" registry/*/nb_v2/config.json
# Check metrics
cat registry/onpeak/nb_v2/metrics.json | python -m json.tool | head -40
```

- [ ] **Step 9: Commit**

```bash
git add registry/onpeak/nb_v2/ registry/offpeak/nb_v2/
git commit -m "results: NB experiment V2 — onpeak + offpeak registry artifacts"
```

### Task 4: Write results report and update README

**Files:**
- Create: `docs/2026-03-23-nb-v2-experiment-report.md`
- Modify: `README.md`

- [ ] **Step 10: Write experiment report**

Structure:
1. Motivation + V1 bugs fixed (the "What changed from V1" table)
2. Why 2 models (per-ctype dormant universes differ)
3. Experimental setup: features, CV, configs, two builders, backfill rule
4. NB-only metrics per (eval_py, class_type) — ML_nb vs V4.4 vs v0c
5. 8 full-metric tables: (2024, 2025) × (onpeak, offpeak) × (K=200, K=400)
   - Configs: pure_v0c, pure_v44, R30_nb, R30_v44, R50_nb, R50_v44
   - Columns: VC, Abs, Rec, Prec, Bind, SP_cap, NB_in, NB_b, NB_SP, NB_VC, NB_Rec, D20, D50
6. 4 aggregate tables: (onpeak, offpeak) × (K=200, K=400)
7. Delta vs pure_v0c per (class_type, K)
8. Fill rate analysis: genuine NB/V4.4 picks vs v0c backfill for ALL reserved-slot configs
9. Per-ctype comparison: does NB help equally for onpeak vs offpeak?
10. Case study highlights per ctype
11. Conclusions and recommended config per ctype

- [ ] **Step 11: Update README**

Replace NB V1 section. Note V1 superseded.

- [ ] **Step 12: Commit**

```bash
git add docs/2026-03-23-nb-v2-experiment-report.md README.md
git commit -m "docs: NB experiment V2 report + README update"
```

### Task 5: Archive V1 scripts

- [ ] **Step 13: Archive V1 artifacts**

```bash
git mv scripts/nb_model_yearly.py scripts/archive/nb_model_yearly_v1.py
mkdir -p docs/archive
git mv docs/2026-03-23-nb-model-experiment-report.md docs/archive/2026-03-23-nb-v1-experiment-report.md
git add -A && git commit -m "archive: move NB V1 scripts and docs (superseded by V2)"
```

---

## Verification checklist

After all tasks complete, verify:

- [ ] `PYTHONPATH=. python -m pytest tests/test_nb_experiment_v2.py -v` — all pass
- [ ] **2 NB models trained per eval year** (grep for `nb_models[(eval_py, ct)]` or equivalent)
- [ ] Onpeak NB trains on `bf_12 == 0`, target = `onpeak_sp` (grep script)
- [ ] Offpeak NB trains on `bfo_12 == 0`, target = `offpeak_sp` (grep script)
- [ ] Eval uses `build_class_model_table` (grep for `build_class_model_table`)
- [ ] v0c uses class-specific `da_rank_value` from class builder (not combined)
- [ ] Target is class-specific `realized_shadow_price` from class builder
- [ ] Abs_SP uses class-specific `total_da_sp_quarter` from class builder
- [ ] V4.4 loaded per-ctype (grep for `load_v44.*onpeak` and `load_v44.*offpeak`)
- [ ] V4.4 NOT used as NB model feature
- [ ] V4.4 underfill → v0c backfill → K always fixed (grep for `backfill` or `len(selected) < K`)
- [ ] V4.4 fill rate reported
- [ ] NB-only comparison uses full dormant universe with V4.4 missing scored -inf
- [ ] Case studies per-ctype with per-ctype ranks
- [ ] Report has 8 year-ctype-K tables + 4 aggregates
- [ ] Registry has per-ctype config with `dormant_col`, `target_col`
- [ ] NDCG uses class-specific `label_tier` from `build_class_model_table` (not combined)
- [ ] Fill count reported for ALL reserved-slot configs (R*_nb AND R*_v44), not just V4.4
- [ ] `is_dormant` and `is_nb_binder` are separate masks; pipeline's `is_nb_12` / `nb_onpeak_12` NOT reused
