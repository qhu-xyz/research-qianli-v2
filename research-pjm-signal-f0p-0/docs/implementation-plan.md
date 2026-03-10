# PJM V7.0b Constraint-Tier ML Signal — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PJM V7.0b constraint-tier signal that replaces V6.2B formula scoring with ML-scored tiers for f0/f1 across 3 class types, with V6.2B passthrough for f2-f11.

**Architecture:** Fork-and-adapt the proven MISO stage5 ML pipeline (`research-stage5-tier/ml/`, 14 modules) for PJM. The critical PJM-specific adaptation is the branch-level constraint→DA target join (naive join captures only 46% of binding value; branch-level captures 96-99%). The pipeline follows the same walk-forward evaluation with LightGBM LambdaRank, 9 features, tiered labels, and row-percentile tiering. After research validation, deploy as a signal writer analogous to `research-miso-signal7/`.

**Tech Stack:** Python, polars, LightGBM (LambdaRank), NumPy, pbase (PjmApTools for DA data), Ray (for DA fetch only).

---

## Independent Review Protocol

Each task ends with a **Review** block. A reviewer with zero prior context should be able to:

1. **Run the verification commands** listed in the review block
2. **Check the acceptance criteria** — every item must pass
3. **Flag any issues** before the implementer moves to the next task

### Review severity levels
- **BLOCK**: Must fix before proceeding. Data correctness, leakage, or schema violations.
- **WARN**: Should fix but can proceed. Style, naming, minor inefficiency.
- **INFO**: Optional improvement. Nice-to-have.

### How to review
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
# Then run the verification commands from each review block
```

### Global invariants (check after EVERY task)
- [ ] No file imports from `pbase.data.pjm.ap_tools` (does not exist — use `pbase.analysis.tools.all_positions.PjmApTools`)
- [ ] No use of `constraint_id` as DA join key (must use `branch_name`)
- [ ] No `multiprocessing`, `concurrent.futures`, `joblib`, `dask`, or `threading` imports
- [ ] All LightGBM usage includes `num_threads=4`
- [ ] No leaky columns used as features: `rank`, `rank_ori`, `tier`, `shadow_sign`, `shadow_price`, `actual_shadow_price`, `actual_binding`, `error`, `abs_error`
- [ ] `mem_mb()` printed at key stages in any script that loads data
- [ ] All parquet reads use `pl.read_parquet()` or `pl.scan_parquet()` (polars, not pandas)

---

## File Structure

```
research-pjm-signal-f0p-0/
├── ml/
│   ├── __init__.py
│   ├── config.py              # PJM paths, auction schedule, LTRConfig, PipelineConfig
│   ├── branch_mapping.py      # constraint_id → branch_name mapping via constraint_info
│   ├── realized_da.py         # Fetch + cache realized DA by branch_name
│   ├── spice6_loader.py       # Load spice6 density features (PJM paths)
│   ├── data_loader.py         # Load V6.2B + spice6 + realized DA (branch-level join)
│   ├── features.py            # Feature prep + query groups (reuse from MISO)
│   ├── train.py               # LightGBM LambdaRank training (copy from MISO)
│   ├── evaluate.py            # VC@20, Recall, NDCG metrics (copy from MISO)
│   ├── v62b_formula.py        # V6.2B formula reproduction (copy from MISO)
│   ├── pipeline.py            # load → train → predict → evaluate (adapted)
│   ├── compare.py             # Gate comparison system (copy from MISO)
│   ├── registry_paths.py      # Registry path helpers (copy from MISO)
│   └── tests/
│       ├── __init__.py
│       ├── test_branch_mapping.py
│       ├── test_realized_da.py
│       └── test_config.py
├── scripts/
│   ├── cache_realized_da.py      # Preflight: cache all needed realized DA months
│   ├── run_v0_formula_baseline.py # V0 baseline for all 6 slices
│   ├── run_v2_ml.py              # V2 ML model for all 6 slices
│   ├── run_blend_search.py       # Blend search for all 6 slices
│   └── run_holdout.py            # Holdout evaluation
├── v70/
│   ├── __init__.py
│   ├── cache.py               # Realized DA cache management for deployment
│   ├── inference.py           # ML inference (train on history, score target)
│   └── signal_writer.py       # Rank/tier computation, signal assembly
├── registry/                  # Dev evaluation results
│   ├── f0/
│   │   ├── onpeak/
│   │   ├── dailyoffpeak/
│   │   └── wkndonpeak/
│   └── f1/
│       ├── onpeak/
│       ├── dailyoffpeak/
│       └── wkndonpeak/
├── holdout/                   # Holdout evaluation results (same structure)
├── data/
│   └── realized_da/           # Local realized DA cache (branch_name keyed)
└── docs/
    └── implementation-plan.md # This file
```

### Key difference from MISO

| Module | MISO | PJM change |
|--------|------|------------|
| `config.py` | MISO paths, `MISO_AUCTION_SCHEDULE` | PJM paths, `PJM_AUCTION_SCHEDULE`, 3 class types |
| `branch_mapping.py` | N/A (MISO joins on constraint_id) | **NEW**: constraint_info → branch_name mapping |
| `realized_da.py` | Join on `constraint_id` | Join on `branch_name` via branch mapping |
| `data_loader.py` | Join realized DA on `constraint_id` | Join realized DA on `branch_name` |
| `spice6_loader.py` | MISO density path | PJM density path |
| `train.py` | — | Copy verbatim |
| `evaluate.py` | — | Copy verbatim |
| `v62b_formula.py` | — | Copy verbatim |
| `features.py` | — | Copy verbatim (add `v7_formula_score` derived feature) |

---

## Chunk 1: Core ML Modules (Config, Branch Mapping, Realized DA)

### Task 1: Create `ml/__init__.py` and copy unchanged modules

Copy modules that need zero PJM-specific changes from MISO stage5.

**Files:**
- Create: `ml/__init__.py`
- Create: `ml/train.py` (copy from `research-stage5-tier/ml/train.py`)
- Create: `ml/evaluate.py` (copy from `research-stage5-tier/ml/evaluate.py`)
- Create: `ml/v62b_formula.py` (copy from `research-stage5-tier/ml/v62b_formula.py`)
- Create: `ml/registry_paths.py` (copy from `research-stage5-tier/ml/registry_paths.py`)
- Create: `ml/compare.py` (copy from `research-stage5-tier/ml/compare.py`)
- Create: `ml/tests/__init__.py`

- [ ] **Step 1: Copy unchanged modules**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
mkdir -p ml/tests
touch ml/__init__.py ml/tests/__init__.py

# These modules have NO MISO-specific logic — copy verbatim
cp ../research-stage5-tier/ml/train.py ml/train.py
cp ../research-stage5-tier/ml/evaluate.py ml/evaluate.py
cp ../research-stage5-tier/ml/v62b_formula.py ml/v62b_formula.py
cp ../research-stage5-tier/ml/registry_paths.py ml/registry_paths.py
cp ../research-stage5-tier/ml/compare.py ml/compare.py
```

- [ ] **Step 2: Verify files exist (imports deferred to after Task 2)**

Note: `train.py` imports `LTRConfig` from `ml.config`, which doesn't exist yet (created in Task 2).
Full import verification is deferred to Task 2's review. For now, just confirm files are present.

```bash
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
ls -1 ml/train.py ml/evaluate.py ml/v62b_formula.py ml/registry_paths.py ml/compare.py
# Verify modules that DON'T depend on ml.config:
python -c "from ml.v62b_formula import v62b_score, dense_rank_normalized; print('v62b OK')"
python -c "from ml.evaluate import evaluate_ltr, aggregate_months; print('evaluate OK')"
```

Expected: files exist, v62b and evaluate import OK. train/registry_paths/compare will work after Task 2.

- [ ] **Step 3: Commit**

```bash
git add ml/
git commit -m "copy unchanged MISO modules: train, evaluate, v62b_formula, registry_paths, compare"
```

#### Review: Task 1

**Verify commands:**
```bash
# 1. All 5 modules exist
ls -1 ml/train.py ml/evaluate.py ml/v62b_formula.py ml/registry_paths.py ml/compare.py

# 2. Imports that DON'T depend on ml.config work now
python -c "from ml.evaluate import evaluate_ltr, aggregate_months, value_capture_at_k; print('evaluate OK')"
python -c "from ml.v62b_formula import v62b_score, dense_rank_normalized; print('v62b OK')"
# train.py, registry_paths.py, compare.py import ml.config — verified in Task 2 review

# 3. No MISO-specific hardcoded paths leaked in
grep -rn "MISO\|miso" ml/train.py ml/evaluate.py ml/v62b_formula.py ml/registry_paths.py ml/compare.py || echo "No MISO references (OK)"

# 4. num_threads=4 present in train.py
grep -n "num_threads" ml/train.py
```

**Acceptance criteria:**
- [ ] All 5 files exist
- [ ] `ml/__init__.py` and `ml/tests/__init__.py` exist
- [ ] v62b_formula and evaluate import successfully
- [ ] `train.py` contains `"num_threads": 4` (or equivalent)
- [ ] No hardcoded MISO paths in any copied module (paths come from config, which is Task 2)
- [ ] Commit exists with the 5 files + 2 `__init__.py`
- [ ] train/registry_paths/compare imports deferred to Task 2 review (they depend on `ml.config`)

---

### Task 2: Create `ml/config.py` with PJM paths and auction schedule

This is the central configuration module. Key PJM differences:
- All data paths point to PJM locations
- PJM auction schedule: f0-f11, varies by month (May: f0 only, June: all 12)
- 3 class types: onpeak, dailyoffpeak, wkndonpeak
- Leaky features include PJM-specific columns

**Files:**
- Create: `ml/config.py`
- Test: `ml/tests/test_config.py`

- [ ] **Step 1: Write config test**

```python
# ml/tests/test_config.py
"""Tests for PJM config module."""
import pytest
from ml.config import (
    delivery_month,
    has_period_type,
    period_offset,
    collect_usable_months,
    PJM_CLASS_TYPES,
    V62B_SIGNAL_BASE,
    SPICE6_DENSITY_BASE,
)


def test_period_offset():
    assert period_offset("f0") == 0
    assert period_offset("f1") == 1
    assert period_offset("f11") == 11


def test_delivery_month():
    assert delivery_month("2025-01", "f0") == "2025-01"
    assert delivery_month("2025-01", "f1") == "2025-02"
    assert delivery_month("2025-06", "f11") == "2026-05"


def test_has_period_type_may_f0_only():
    """May auctions have only f0."""
    assert has_period_type("2025-05", "f0") is True
    assert has_period_type("2025-05", "f1") is False


def test_has_period_type_june_all():
    """June auctions have f0-f11."""
    for i in range(12):
        assert has_period_type("2025-06", f"f{i}") is True


def test_class_types():
    assert PJM_CLASS_TYPES == ["onpeak", "dailyoffpeak", "wkndonpeak"]


def test_pjm_paths_exist():
    from pathlib import Path
    assert Path(V62B_SIGNAL_BASE).exists(), f"V6.2B path missing: {V62B_SIGNAL_BASE}"
    assert Path(SPICE6_DENSITY_BASE).exists(), f"Spice6 density path missing: {SPICE6_DENSITY_BASE}"


def test_collect_usable_months_f0():
    """f0 should return 8 contiguous months for a well-covered eval month."""
    months = collect_usable_months("2023-06", "f0", n_months=8)
    assert len(months) == 8
    # Most recent should be 2 months before target (lag built into collect_usable_months)
    assert months[0] <= "2023-04"


def test_collect_usable_months_f1_skips_gaps():
    """f1 should skip months where f1 doesn't exist (May)."""
    months = collect_usable_months("2023-06", "f1", n_months=8)
    assert len(months) >= 6  # may be < 8 due to gaps
    for m in months:
        assert has_period_type(m, "f1"), f"{m} should have f1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python -m pytest ml/tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ml.config'`

- [ ] **Step 3: Write `ml/config.py`**

```python
# ml/config.py
"""PJM LTR pipeline configuration.

Adapted from research-stage5-tier/ml/config.py for PJM market structure.
Key PJM differences:
  - 3 class types: onpeak, dailyoffpeak, wkndonpeak
  - f0 through f11 period types (vs MISO f0-f3)
  - PJM-specific data paths
  - Branch-level DA join (not constraint_id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── PJM class types ──
PJM_CLASS_TYPES = ["onpeak", "dailyoffpeak", "wkndonpeak"]

# ── Leakage guard ──
_LEAKY_FEATURES: set[str] = {
    "rank", "rank_ori", "tier",
    "shadow_sign", "shadow_price",
}

# ── V10E features (9, same as MISO champion) ──
V10E_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "prob_exceed_110", "constraint_limit",
    "da_rank_value",
]
V10E_MONOTONE = [1, 1, 1, 1, 1, -1, 1, 0, -1]

# ── Eval months (PJM-specific, based on V6.2B availability from 2017-06) ──
# Dev: 36 months (2020-06 to 2023-05), same range as MISO
_FULL_EVAL_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}"
    for y in range(2020, 2024)
    for m in range(1, 13)
    if (y, m) >= (2020, 6) and (y, m) <= (2023, 5)
]

# Screen: 4 fast months
_SCREEN_EVAL_MONTHS: list[str] = [
    "2020-12", "2021-09", "2022-06", "2023-03",
]

# Default: 12 representative months
_DEFAULT_EVAL_MONTHS: list[str] = [
    "2020-09", "2020-12", "2021-03", "2021-06",
    "2021-09", "2021-12", "2022-03", "2022-06",
    "2022-09", "2022-12", "2023-03", "2023-05",
]

# Holdout: 2024-2025
HOLDOUT_MONTHS: list[str] = [
    f"{y:04d}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)
]

# ── Data paths ──
V62B_SIGNAL_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1"
SPICE6_DENSITY_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density"
SPICE6_MLPRED_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred"
SPICE6_CI_BASE = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info"
REALIZED_DA_CACHE = str(Path(__file__).resolve().parent.parent / "data" / "realized_da")

# ── PJM Auction Schedule ──
# Determined from actual V6.2B data on disk. PJM has f0-f11.
# June exposes all 12 monthly periods; schedule shrinks as planning year progresses.
# Source: pbase/src/pbase/data/const/period/pjm.py + V6.2B directory enumeration.
PJM_AUCTION_SCHEDULE: dict[int, list[str]] = {
    1: ["f0", "f1", "f2", "f3", "f4"],
    2: ["f0", "f1", "f2", "f3"],
    3: ["f0", "f1", "f2"],
    4: ["f0", "f1"],
    5: ["f0"],
    6: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11"],
    7: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
    8: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
    9: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
    10: ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
    11: ["f0", "f1", "f2", "f3", "f4", "f5", "f6"],
    12: ["f0", "f1", "f2", "f3", "f4", "f5"],
}


def period_offset(period_type: str) -> int:
    """f0→0, f1→1, ..., f11→11."""
    if not period_type.startswith("f"):
        raise ValueError(f"period_offset only supports fN types, got '{period_type}'")
    return int(period_type[1:])


def delivery_month(auction_month: str, period_type: str) -> str:
    """Compute delivery month = auction_month + period_offset."""
    import pandas as pd
    offset = period_offset(period_type)
    if offset == 0:
        return auction_month
    dt = pd.Timestamp(auction_month) + pd.DateOffset(months=offset)
    return dt.strftime("%Y-%m")


def has_period_type(auction_month: str, period_type: str) -> bool:
    """Check if period type exists for a given auction month."""
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return period_type in PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])


def collect_usable_months(
    target_auction_month: str,
    period_type: str,
    n_months: int = 8,
    min_months: int = 6,
    max_lookback: int = 24,
) -> list[str]:
    """Walk backward from target, collect n_months usable training auction months.

    A month is "usable" if:
      1. The period_type exists for that month (per PJM auction schedule)
      2. delivery_month(month, period_type) <= last_full_known

    last_full_known = target_auction_month - 2 (last complete realized DA at decision time).
    Returns most-recent-first order. Caller should reverse for chronological.
    """
    import pandas as pd

    target_ts = pd.Timestamp(target_auction_month)
    last_full_known = (target_ts - pd.DateOffset(months=2)).strftime("%Y-%m")

    usable = []
    for i in range(1, max_lookback + 1):
        candidate = (target_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        if not has_period_type(candidate, period_type):
            continue
        dm = delivery_month(candidate, period_type)
        if dm > last_full_known:
            continue
        usable.append(candidate)
        if len(usable) >= n_months:
            break

    if len(usable) < min_months:
        return []
    return usable


@dataclass
class LTRConfig:
    """Learning-to-rank configuration."""

    features: list[str] = field(default_factory=lambda: list(V10E_FEATURES))
    monotone_constraints: list[int] = field(default_factory=lambda: list(V10E_MONOTONE))
    backend: str = "lightgbm"
    label_mode: str = "tiered"
    n_estimators: int = 100
    learning_rate: float = 0.05
    min_child_weight: int = 25
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    max_depth: int = 5
    early_stopping_rounds: int = 20

    def __post_init__(self) -> None:
        if len(self.monotone_constraints) != len(self.features):
            raise ValueError(
                f"len(monotone_constraints) must match len(features) "
                f"({len(self.monotone_constraints)} != {len(self.features)})"
            )
        filtered_features: list[str] = []
        filtered_mono: list[int] = []
        for feat, mono in zip(self.features, self.monotone_constraints):
            if feat in _LEAKY_FEATURES:
                print(f"[config] WARNING: dropped leaky feature: {feat}")
                continue
            filtered_features.append(feat)
            filtered_mono.append(mono)
        self.features = filtered_features
        self.monotone_constraints = filtered_mono

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LTRConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    ltr: LTRConfig = field(default_factory=LTRConfig)
    train_months: int = 8
    val_months: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"ltr": self.ltr.to_dict(), "train_months": self.train_months, "val_months": self.val_months}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        return cls(ltr=LTRConfig.from_dict(d["ltr"]), train_months=d["train_months"], val_months=d["val_months"])


@dataclass
class GateConfig:
    """Quality gates loaded from JSON."""
    gates: dict[str, Any] = field(default_factory=dict)
    noise_tolerance: float = 0.0
    tail_max_failures: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> GateConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(
            gates=data.get("gates", {}),
            noise_tolerance=data.get("noise_tolerance", 0.0),
            tail_max_failures=data.get("tail_max_failures", 0),
        )
```

**IMPORTANT**: The `PJM_AUCTION_SCHEDULE` above is a best-guess from the handoff docs. The implementing agent MUST verify it against the actual V6.2B directory structure on disk by running:

```python
from pathlib import Path
base = Path("/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1")
schedule = {}
for month_dir in sorted(base.iterdir()):
    ptypes = sorted([p.name for p in month_dir.iterdir() if p.is_dir()])
    month_num = int(month_dir.name.split("-")[1])
    if month_num not in schedule:
        schedule[month_num] = ptypes
    else:
        # Take the union across years
        schedule[month_num] = sorted(set(schedule[month_num]) | set(ptypes))
print(schedule)
```

Update `PJM_AUCTION_SCHEDULE` with the verified values before proceeding.

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_config.py -v
```

Expected: All tests PASS. If `PJM_AUCTION_SCHEDULE` is wrong, fix it first.

- [ ] **Step 5: Commit**

```bash
git add ml/config.py ml/tests/test_config.py
git commit -m "feat: add PJM config with auction schedule, paths, and LTRConfig"
```

#### Review: Task 2

**Verify commands:**
```bash
# 0. Deferred from Task 1: verify config-dependent modules now import
python -c "from ml.train import train_ltr_model, predict_scores; print('train OK')"
python -c "from ml.registry_paths import registry_root, holdout_root; print('registry OK')"
python -c "from ml.compare import compare_versions; print('compare OK')"

# 1. Tests pass
python -m pytest ml/tests/test_config.py -v

# 2. Auction schedule was verified against real data (not just the handoff guess)
python -c "
from pathlib import Path
base = Path('/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1')
real = {}
for month_dir in sorted(base.iterdir()):
    if not month_dir.is_dir(): continue
    ptypes = sorted([p.name for p in month_dir.iterdir() if p.is_dir()])
    mn = int(month_dir.name.split('-')[1])
    if mn not in real:
        real[mn] = set(ptypes)
    else:
        real[mn] |= set(ptypes)

from ml.config import PJM_AUCTION_SCHEDULE
for mn in sorted(real):
    code_set = set(PJM_AUCTION_SCHEDULE.get(mn, ['f0']))
    disk_set = real[mn]
    if code_set != disk_set:
        print(f'MISMATCH month={mn}: code={sorted(code_set)} disk={sorted(disk_set)}')
    else:
        print(f'OK month={mn}: {sorted(code_set)}')
"

# 3. Data paths exist
python -c "
from pathlib import Path
from ml.config import V62B_SIGNAL_BASE, SPICE6_DENSITY_BASE, SPICE6_MLPRED_BASE, SPICE6_CI_BASE
for name, p in [('V62B', V62B_SIGNAL_BASE), ('density', SPICE6_DENSITY_BASE),
                ('mlpred', SPICE6_MLPRED_BASE), ('ci', SPICE6_CI_BASE)]:
    exists = Path(p).exists()
    print(f'{name}: {\"OK\" if exists else \"MISSING\"} -> {p}')
"

# 4. Temporal leakage guard: collect_usable_months respects lag
python -c "
from ml.config import collect_usable_months, delivery_month
months = collect_usable_months('2023-06', 'f0', n_months=8)
print(f'Training months for eval 2023-06/f0: {months}')
# The latest training month's delivery month must be < 2023-05 (M-1 for f0)
latest = months[0]
dm = delivery_month(latest, 'f0')
print(f'Latest train month={latest}, delivery_month={dm}')
assert dm < '2023-05', f'LEAKAGE: delivery_month {dm} >= 2023-05'
print('Leakage check PASSED')
"

# 5. Leaky features blocked
python -c "
from ml.config import LTRConfig
cfg = LTRConfig(features=['rank', 'da_rank_value', 'rank_ori'], monotone_constraints=[0, -1, 0])
print(f'Features after filter: {cfg.features}')
assert 'rank' not in cfg.features, 'rank not filtered'
assert 'rank_ori' not in cfg.features, 'rank_ori not filtered'
assert 'da_rank_value' in cfg.features, 'da_rank_value wrongly filtered'
print('Leaky feature guard PASSED')
"
```

**Acceptance criteria:**
- [ ] All tests pass
- [ ] **BLOCK**: `PJM_AUCTION_SCHEDULE` matches actual V6.2B directories on disk (no MISMATCH lines)
- [ ] **BLOCK**: All 4 data paths exist on disk
- [ ] **BLOCK**: Temporal leakage check passes — latest training delivery month < eval_month - 1
- [ ] **BLOCK**: Leaky features (`rank`, `rank_ori`, `tier`, `shadow_sign`, `shadow_price`) are filtered out by `LTRConfig.__post_init__`
- [ ] `PJM_CLASS_TYPES` is `["onpeak", "dailyoffpeak", "wkndonpeak"]` (3 types, not 2)
- [ ] `V10E_FEATURES` has exactly 9 features matching MISO champion
- [ ] `HOLDOUT_MONTHS` covers 2024-01 through 2025-12

---

### Task 3: Create `ml/branch_mapping.py` — constraint_id → branch_name mapping

This is the **most critical PJM-specific module**. It maps V6.2B `constraint_id` to `branch_name` via `constraint_info`, then maps DA `monitored_facility` to `branch_name`. Without this, the target join captures only 46% of binding value.

Reference: `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805`

**Files:**
- Create: `ml/branch_mapping.py`
- Test: `ml/tests/test_branch_mapping.py`

- [ ] **Step 1: Write branch mapping test**

```python
# ml/tests/test_branch_mapping.py
"""Tests for PJM branch mapping module.

NOTE: This test must NOT import from ml.realized_da (Task 4).
Task 3 must be independently completable.
"""
import polars as pl
import pytest
from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches


def test_load_constraint_info_returns_dataframe():
    """constraint_info should load for a known month."""
    ci = load_constraint_info("2025-01", period_type="f0")
    assert isinstance(ci, pl.DataFrame)
    assert len(ci) > 0
    assert "constraint_id" in ci.columns
    assert "branch_name" in ci.columns


def test_build_branch_map_has_match_str():
    """Branch map should have match_str for DA joining."""
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)
    assert "match_str" in bmap.columns
    assert "branch_name" in bmap.columns
    # match_str should be uppercase
    for s in bmap["match_str"].head(5).to_list():
        assert s == s.upper(), f"match_str not uppercase: {s}"


def test_map_da_to_branches_captures_most_value():
    """Branch-level join should capture >90% of DA value.

    NOTE: This test uses PjmApTools directly (not ml.realized_da which is Task 4).
    This keeps Task 3 independently completable.
    """
    ci = load_constraint_info("2025-01", period_type="f0")
    bmap = build_branch_map(ci)

    # Fetch DA directly (not via ml.realized_da — that module doesn't exist yet)
    try:
        import pandas as pd
        from pbase.analysis.tools.all_positions import PjmApTools
        # polars already imported at module level

        st = pd.Timestamp("2025-01-01")
        et = st + pd.offsets.MonthBegin(1)
        aptools = PjmApTools()
        da_shadow = aptools.tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type="onpeak")
        if da_shadow is None or len(da_shadow) == 0:
            pytest.skip("No DA data for 2025-01")
        da_df = pl.from_pandas(da_shadow.reset_index()).select([
            pl.col("monitored_facility").cast(pl.String),
            pl.col("shadow_price").cast(pl.Float64),
        ])
    except Exception as e:
        pytest.skip(f"Cannot fetch DA data: {e}")

    result = map_da_to_branches(da_df, bmap)
    assert "branch_name" in result.columns
    assert "realized_sp" in result.columns

    # Check coverage: matched value / total value
    total_value = da_df["shadow_price"].abs().sum()
    matched_value = result["realized_sp"].sum()
    coverage = matched_value / total_value if total_value > 0 else 0
    print(f"DA value coverage: {coverage:.1%}")
    assert coverage > 0.90, f"Branch mapping coverage too low: {coverage:.1%}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest ml/tests/test_branch_mapping.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `ml/branch_mapping.py`**

```python
# ml/branch_mapping.py
"""PJM constraint_id → branch_name mapping via constraint_info.

The naive join (constraint_id.split(":")[0] → DA monitored_facility) captures
only ~46% of DA binding value. The branch-level join via constraint_info
captures 96-99%.

Reference: research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805

How it works:
  1. constraint_info maps each constraint_id to a branch_name
  2. constraint_id.split(":")[0] → monitored_facility → .upper() → match_str
  3. DA monitored_facility (uppercased) matches match_str → branch_name
  4. Interface fallback: prefix-match for interface contingencies
  5. Aggregate realized_sp by branch_name
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_CI_BASE


def load_constraint_info(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load constraint_info for an auction month.

    constraint_info is stored only under class_type=onpeak (by design —
    it's physical topology, class-invariant).
    """
    path = (
        Path(SPICE6_CI_BASE)
        / f"auction_month={auction_month}"
        / "market_round=1"
        / f"period_type={period_type}"
        / "class_type=onpeak"
    )
    if not path.exists():
        return pl.DataFrame()

    # Read all parquet files in the directory
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame()

    dfs = [pl.read_parquet(str(f)) for f in parquet_files]
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

    # Ensure constraint_id is string
    if "constraint_id" in df.columns:
        df = df.with_columns(pl.col("constraint_id").cast(pl.String))

    return df


def build_branch_map(ci: pl.DataFrame) -> pl.DataFrame:
    """Build branch mapping from constraint_info.

    Returns DataFrame with columns: constraint_id, branch_name, match_str, type.
    match_str = monitored_facility.upper() (extracted from constraint_id).
    """
    if len(ci) == 0:
        return pl.DataFrame(schema={
            "constraint_id": pl.String,
            "branch_name": pl.String,
            "match_str": pl.String,
            "type": pl.String,
        })

    # Extract monitored_facility from constraint_id (split on ":")
    result = ci.select([
        pl.col("constraint_id"),
        pl.col("branch_name"),
        pl.col("type") if "type" in ci.columns else pl.lit("branch_constraint").alias("type"),
    ]).unique()

    # Build match_str: monitored_facility = constraint_id.split(":")[0], uppercased
    result = result.with_columns(
        pl.col("constraint_id")
        .str.split(":")
        .list.first()
        .str.to_uppercase()
        .alias("match_str")
    )

    return result


def map_da_to_branches(
    da_df: pl.DataFrame,
    branch_map: pl.DataFrame,
) -> pl.DataFrame:
    """Map DA shadow prices to branch_names and aggregate.

    Parameters
    ----------
    da_df : pl.DataFrame
        Raw DA data with columns: monitored_facility, shadow_price.
    branch_map : pl.DataFrame
        From build_branch_map(), with columns: branch_name, match_str, type.

    Returns
    -------
    pl.DataFrame
        Columns: branch_name, realized_sp (sum of |shadow_price| per branch).
    """
    if len(da_df) == 0 or len(branch_map) == 0:
        return pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })

    # Uppercase DA monitored_facility for matching
    da = da_df.with_columns(
        pl.col("monitored_facility").str.to_uppercase().alias("match_str")
    )

    # Deduplicate branch_map on match_str (take first branch_name per match_str)
    # Separate interface and non-interface entries
    non_interface = branch_map.filter(pl.col("type") != "interface")
    interface_map = branch_map.filter(pl.col("type") == "interface")

    # Step 1: Direct match on match_str (non-interface)
    direct_map = non_interface.select(["match_str", "branch_name"]).unique(subset=["match_str"])
    da = da.join(direct_map, on="match_str", how="left")

    # Step 2: Interface fallback — prefix match for unmatched DA rows
    if len(interface_map) > 0:
        unmatched = da.filter(pl.col("branch_name").is_null())
        if len(unmatched) > 0:
            # For each unmatched DA row, try to prefix-match against interface match_strs
            interface_strs = interface_map.select(["match_str", "branch_name"]).unique(subset=["match_str"])
            interface_prefixes = interface_strs["match_str"].to_list()
            interface_branch_names = interface_strs["branch_name"].to_list()

            # Build a mapping: for each unmatched match_str, find interface prefix
            unmatched_strs = unmatched["match_str"].unique().to_list()
            prefix_map = {}
            for ums in unmatched_strs:
                # Try splitting on space and matching the first word
                first_word = ums.split(" ")[0] if " " in ums else ums
                for ip, ibn in zip(interface_prefixes, interface_branch_names):
                    if first_word == ip or ums.startswith(ip + " "):
                        prefix_map[ums] = ibn
                        break

            if prefix_map:
                prefix_df = pl.DataFrame({
                    "match_str": list(prefix_map.keys()),
                    "interface_branch": list(prefix_map.values()),
                })
                da = da.join(prefix_df, on="match_str", how="left")
                da = da.with_columns(
                    pl.col("branch_name").fill_null(pl.col("interface_branch"))
                )
                if "interface_branch" in da.columns:
                    da = da.drop("interface_branch")

    # Aggregate: sum |shadow_price| by branch_name
    result = (
        da.filter(pl.col("branch_name").is_not_null())
        .group_by("branch_name")
        .agg(pl.col("shadow_price").abs().sum().alias("realized_sp"))
    )

    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_branch_mapping.py -v
```

Expected: All PASS. The coverage test should show >90%.

- [ ] **Step 5: Commit**

```bash
git add ml/branch_mapping.py ml/tests/test_branch_mapping.py
git commit -m "feat: add PJM branch mapping (constraint_id → branch_name via constraint_info)"
```

#### Review: Task 3

**Verify commands:**
```bash
# 1. Tests pass
python -m pytest ml/tests/test_branch_mapping.py -v

# 2. Coverage test: branch mapping captures >90% DA value
# (test_map_da_to_branches_captures_most_value does this, but run explicitly)
python -c "
from ml.branch_mapping import load_constraint_info, build_branch_map
ci = load_constraint_info('2025-01', period_type='f0')
bmap = build_branch_map(ci)
print(f'constraint_info rows: {len(ci)}')
print(f'branch_map rows: {len(bmap)}')
print(f'Unique branch_names: {bmap[\"branch_name\"].n_unique()}')
print(f'Unique match_strs: {bmap[\"match_str\"].n_unique()}')
print(f'Sample match_strs: {bmap[\"match_str\"].head(5).to_list()}')
# Verify all match_str are uppercase
import polars as pl
non_upper = bmap.filter(pl.col('match_str') != pl.col('match_str').str.to_uppercase())
print(f'Non-uppercase match_strs: {len(non_upper)} (should be 0)')
"

# 3. Verify branch mapping uses the reference approach (not the naive join)
grep -n "monitored_facility" ml/branch_mapping.py
grep -n "interface" ml/branch_mapping.py
# Should see: interface prefix matching logic present
```

**Acceptance criteria:**
- [ ] All tests pass
- [ ] **BLOCK**: DA value coverage >90% (printed by `test_map_da_to_branches_captures_most_value`)
- [ ] **BLOCK**: `match_str` values are ALL uppercase (case normalization working)
- [ ] Interface prefix matching is implemented (not just direct matching)
- [ ] `load_constraint_info` uses `class_type=onpeak` path (constraint_info is class-invariant)
- [ ] `build_branch_map` extracts `constraint_id.split(":")[0]` for match_str
- [ ] Empty DataFrames handled gracefully (no crashes on missing data)
- [ ] **BLOCK**: Test file does NOT import from `ml.realized_da` (Task 4 doesn't exist yet — Task 3 must be independently completable)

**Reference check**: Compare logic against `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805`. Key elements:
1. `constraint_id.split(":")[0]` → uppercase → `match_str` ✓
2. Direct match on `match_str` ✓
3. Interface prefix fallback ✓
4. Aggregate by `branch_name` ✓

---

### Task 4: Create `ml/realized_da.py` — Fetch and cache realized DA by branch_name

PJM realized DA is fetched via `PjmApTools`, aggregated by `branch_name` (not `constraint_id`), and cached as parquet.

**Files:**
- Create: `ml/realized_da.py`
- Test: `ml/tests/test_realized_da.py`

- [ ] **Step 1: Write realized DA test**

```python
# ml/tests/test_realized_da.py
"""Tests for PJM realized DA loader."""
import polars as pl
import pytest
from pathlib import Path
from ml.realized_da import load_realized_da, fetch_and_cache_month, _fetch_raw_da


def test_fetch_raw_da_returns_data():
    """Raw DA fetch should return monitored_facility + shadow_price."""
    # This requires Ray — skip if not available
    try:
        df = _fetch_raw_da("2024-06", "onpeak")
    except Exception:
        pytest.skip("Ray not available or DA fetch failed")
    assert isinstance(df, pl.DataFrame)
    assert "monitored_facility" in df.columns
    assert "shadow_price" in df.columns
    assert len(df) > 0


def test_load_realized_da_has_branch_name():
    """Cached realized DA should have branch_name and realized_sp columns."""
    cache_dir = str(Path(__file__).resolve().parent.parent.parent / "data" / "realized_da")
    try:
        df = load_realized_da("2024-06", peak_type="onpeak", cache_dir=cache_dir)
    except FileNotFoundError:
        pytest.skip("No cached data for 2024-06")
    assert "branch_name" in df.columns
    assert "realized_sp" in df.columns
    assert df["realized_sp"].dtype == pl.Float64
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

Expected: FAIL

- [ ] **Step 3: Write `ml/realized_da.py`**

```python
# ml/realized_da.py
"""PJM realized DA shadow price loader and fetcher.

Key difference from MISO: PJM realized DA is aggregated by branch_name
(via branch mapping from constraint_info), not by constraint_id.

load_realized_da    -- read a cached month from parquet
fetch_and_cache_month -- fetch from PJM API via Ray, map to branches, cache
_fetch_raw_da       -- raw DA fetch (no branch mapping)

REQUIRES RAY for fetch_and_cache_month.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import REALIZED_DA_CACHE


def _cache_path(month: str, peak_type: str, cache_dir: str) -> Path:
    """Return cache file path."""
    if peak_type == "onpeak":
        return Path(cache_dir) / f"{month}.parquet"
    return Path(cache_dir) / f"{month}_{peak_type}.parquet"


def load_realized_da(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Read cached realized DA shadow prices for a month.

    Returns
    -------
    pl.DataFrame
        Columns: [branch_name (String), realized_sp (Float64)]
    """
    p = _cache_path(month, peak_type, cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"No cached realized DA for {month}/{peak_type}: {p}")
    df = pl.read_parquet(str(p))
    return df.select(
        pl.col("branch_name").cast(pl.String),
        pl.col("realized_sp").cast(pl.Float64),
    )


def _fetch_raw_da(month: str, peak_type: str) -> pl.DataFrame:
    """Fetch raw PJM DA shadow prices for one month.

    PJM uses US/Eastern timezone (unlike MISO which uses US/Central).

    Returns polars DataFrame with columns: monitored_facility, shadow_price.
    """
    from pbase.analysis.tools.all_positions import PjmApTools

    st = pd.Timestamp(f"{month}-01", tz="US/Eastern")
    et = st + pd.offsets.MonthBegin(1)

    aptools = PjmApTools()
    da_shadow = aptools.tools.get_da_shadow_by_peaktype(
        st=st, et_ex=et, peak_type=peak_type,
    )

    if da_shadow is None or len(da_shadow) == 0:
        return pl.DataFrame(schema={
            "monitored_facility": pl.String,
            "shadow_price": pl.Float64,
        })

    return pl.from_pandas(da_shadow.reset_index()).select([
        pl.col("monitored_facility").cast(pl.String),
        pl.col("shadow_price").cast(pl.Float64),
    ])


def fetch_and_cache_month(
    month: str,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
    period_type: str = "f0",
) -> Path:
    """Fetch realized DA, map to branches via constraint_info, and cache.

    The branch mapping uses constraint_info for the given month. If
    constraint_info is unavailable, falls back to a nearby month.
    If no constraint_info found at all, raises ValueError (fail closed).

    NOTE on period_type: constraint_info is stored under period_type={P}/class_type=onpeak.
    Per CLAUDE.md, constraint_info is "physical topology, class-invariant" — stored only
    under class_type=onpeak by design. The implementer MUST verify that the branch mapping
    is also period-type-invariant by running the verification in Task 4 Step 3a (below).
    If f0 and f1 branch mappings differ materially, the cache must be keyed by period_type.

    Returns path to cached parquet with columns: branch_name, realized_sp.
    """
    from ml.branch_mapping import load_constraint_info, build_branch_map, map_da_to_branches

    out_path = _cache_path(month, peak_type, cache_dir)
    if out_path.exists():
        return out_path

    # Fetch raw DA
    raw_da = _fetch_raw_da(month, peak_type)

    if len(raw_da) == 0:
        df = pl.DataFrame(schema={
            "branch_name": pl.String,
            "realized_sp": pl.Float64,
        })
    else:
        # Load constraint_info for branch mapping
        # Try the target month first, then fall back to nearby months
        ci = load_constraint_info(month, period_type=period_type)
        if len(ci) == 0:
            # Try adjacent months
            for offset in [1, -1, 2, -2]:
                alt = (pd.Timestamp(month) + pd.DateOffset(months=offset)).strftime("%Y-%m")
                ci = load_constraint_info(alt, period_type=period_type)
                if len(ci) > 0:
                    print(f"[realized_da] Using constraint_info from {alt} for {month}")
                    break

        if len(ci) == 0:
            raise ValueError(
                f"[realized_da] FATAL: no constraint_info for {month} or neighbors. "
                f"Cannot build branch mapping — refusing to fall back to naive join "
                f"(captures only ~46% of DA value). Fix constraint_info availability "
                f"or manually populate the cache for this month."
            )
        else:
            bmap = build_branch_map(ci)
            df = map_da_to_branches(raw_da, bmap)

    # Write atomically
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.write_parquet(str(tmp))
    tmp.rename(out_path)

    print(f"[realized_da] Cached {month}/{peak_type}: {len(df)} branches")
    return out_path
```

- [ ] **Step 3a: Verify branch mapping is period-type-invariant**

This is CRITICAL. The cache uses `period_type="f0"` by default for constraint_info lookup.
If f1 has different branch mappings, f1 targets will be silently incomplete.

```python
# Run this to compare f0 vs f1 branch mappings for a month that has both
python -c "
from ml.branch_mapping import load_constraint_info, build_branch_map
import polars as pl

month = '2025-01'  # pick a month with both f0 and f1
ci_f0 = load_constraint_info(month, period_type='f0')
ci_f1 = load_constraint_info(month, period_type='f1')
print(f'constraint_info rows: f0={len(ci_f0)}, f1={len(ci_f1)}')

if len(ci_f1) == 0:
    print('f1 has no constraint_info — likely uses same as f0 (class-invariant)')
else:
    bmap_f0 = build_branch_map(ci_f0)
    bmap_f1 = build_branch_map(ci_f1)
    f0_branches = set(bmap_f0['branch_name'].to_list())
    f1_branches = set(bmap_f1['branch_name'].to_list())
    overlap = f0_branches & f1_branches
    f1_only = f1_branches - f0_branches
    print(f'f0 branches: {len(f0_branches)}, f1 branches: {len(f1_branches)}')
    print(f'Overlap: {len(overlap)}, f1-only: {len(f1_only)}')
    if f1_only:
        print(f'WARNING: {len(f1_only)} branches in f1 not in f0!')
        print(f'ACTION NEEDED: cache must be keyed by period_type, not shared')
    else:
        print('SAFE: f0 branch map is a superset — period_type=f0 default is fine')
"
```

If f1 has branches not in f0, the implementer must:
1. Add `period_type` to the cache key (file name)
2. Update `cache_realized_da.py` to pass explicit `period_type`

- [ ] **Step 4: Run tests**

```bash
python -m pytest ml/tests/test_realized_da.py -v
```

Expected: PASS (may skip tests if Ray is not available or data not cached yet).

- [ ] **Step 5: Commit**

```bash
git add ml/realized_da.py ml/tests/test_realized_da.py
git commit -m "feat: add PJM realized DA with branch-level aggregation"
```

#### Review: Task 4

**Verify commands:**
```bash
# 1. Tests pass (some may skip if Ray unavailable or data not cached)
python -m pytest ml/tests/test_realized_da.py -v

# 2. Verify load_realized_da returns branch_name (NOT constraint_id)
python -c "
import polars as pl
from ml.realized_da import load_realized_da
# This only works if cache exists — skip if not
try:
    df = load_realized_da('2024-06', 'onpeak')
    print(f'Columns: {df.columns}')
    assert 'branch_name' in df.columns, 'MISSING branch_name column'
    assert 'constraint_id' not in df.columns, 'UNEXPECTED constraint_id column'
    assert df['realized_sp'].dtype == pl.Float64
    print(f'Rows: {len(df)}, non-zero: {len(df.filter(pl.col(\"realized_sp\") > 0))}')
    print('Schema check PASSED')
except FileNotFoundError:
    print('SKIP: no cached data yet (will work after Task 9)')
"

# 3. Verify PjmApTools import path (not pbase.data.pjm.ap_tools)
grep -n "PjmApTools\|pjm.*tools\|ap_tools" ml/realized_da.py
# Should see: from pbase.analysis.tools.all_positions import PjmApTools

# 4. Verify atomic write pattern
grep -n "tmp\|rename" ml/realized_da.py
# Should see: write to .tmp then rename

# 5. Verify branch_mapping is used (not direct constraint_id aggregation)
grep -n "branch_mapping\|map_da_to_branches\|build_branch_map" ml/realized_da.py
```

**Acceptance criteria:**
- [ ] **BLOCK**: `load_realized_da` returns columns `[branch_name, realized_sp]` — NOT `constraint_id`
- [ ] **BLOCK**: `_fetch_raw_da` uses `PjmApTools` from `pbase.analysis.tools.all_positions`
- [ ] **BLOCK**: `fetch_and_cache_month` uses `branch_mapping.map_da_to_branches()` for the join
- [ ] **BLOCK**: No naive `monitored_facility` fallback — if constraint_info is missing, `fetch_and_cache_month` must raise `ValueError` (fail closed), NOT silently fall back to the 46%-coverage naive join
- [ ] **BLOCK**: Period-type invariance verified (Step 3a) — if f1 has branches not in f0, cache must be keyed by period_type
- [ ] Atomic write: writes to `.tmp` then renames (no partial files on crash)
- [ ] Cache-hit short circuit: skips fetch if file already exists
- [ ] Adjacent-month constraint_info fallback tries offsets [1, -1, 2, -2] before failing

---

### Task 5: Create `ml/spice6_loader.py` — PJM spice6 density features

Adapted from MISO's `spice6_loader.py`. Only the base path changes.

**Files:**
- Create: `ml/spice6_loader.py`

- [ ] **Step 1: Write `ml/spice6_loader.py`**

Copy from `research-stage5-tier/ml/spice6_loader.py` and change the import path:

```python
# ml/spice6_loader.py
"""Load spice6 density features for PJM.

Adapted from MISO spice6_loader.py. Key PJM differences:
  - Base path points to PJM density directory
  - PJM always uses new schema (score.parquet with single 'score' column)
  - No legacy score_df.parquet files exist for PJM (legacy branch is dead code)
  - PJM uses US/Eastern timezone (not US/Central)
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import SPICE6_DENSITY_BASE, delivery_month as _delivery_month

logger = logging.getLogger(__name__)


def load_spice6_density(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for one month.

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, prob_exceed_110, prob_exceed_100,
        prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit.
    """
    market_month = _delivery_month(auction_month, period_type)
    logger.info("spice6 density: auction=%s ptype=%s market_month=%s",
                auction_month, period_type, market_month)
    market_round = "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    score_dfs = []
    limit_dfs = []

    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        # PJM always uses new schema: score.parquet with single 'score' column
        # (no legacy score_df.parquet exists — verified in preflight)
        score_path = od_dir / "score.parquet"
        limit_path = od_dir / "limit.parquet"
        if score_path.exists():
            score_dfs.append(pl.read_parquet(score_path))
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not score_dfs:
        return pl.DataFrame()

    all_scores = pl.concat(score_dfs)

    # New schema: 'score' column = prob(exceed 110%)
    density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
        pl.col("score").mean().alias("prob_exceed_110"),
    ])
    for col in ["prob_exceed_100", "prob_exceed_90", "prob_exceed_85", "prob_exceed_80"]:
        density = density.with_columns(pl.lit(0.0).alias(col))

    if limit_dfs:
        all_limits = pl.concat(limit_dfs)
        limits = all_limits.group_by("constraint_id").agg(
            pl.col("limit").mean().alias("constraint_limit")
        )
        density = density.join(limits, on="constraint_id", how="left")
    else:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density
```

- [ ] **Step 2: Verify spice6 loads for a known month**

```bash
python -c "
from ml.spice6_loader import load_spice6_density
df = load_spice6_density('2025-01', 'f0')
print(f'Loaded {len(df)} rows, columns: {df.columns}')
print(df.head(3))
"
```

Expected: prints rows with prob_exceed_110, constraint_limit, etc.

- [ ] **Step 3: Commit**

```bash
git add ml/spice6_loader.py
git commit -m "feat: add PJM spice6 density loader"
```

#### Review: Task 5

**Verify commands:**
```bash
# 1. Smoke test: load density for a known month
python -c "
from ml.spice6_loader import load_spice6_density
df = load_spice6_density('2025-01', 'f0')
print(f'Rows: {len(df)}, Columns: {df.columns}')
required = ['constraint_id', 'flow_direction', 'prob_exceed_110', 'constraint_limit']
for c in required:
    assert c in df.columns, f'MISSING column: {c}'
print('Schema check PASSED')
"

# 2. Verify PJM density path (not MISO)
grep -n "SPICE6_DENSITY_BASE\|prod_f0p_model" ml/spice6_loader.py
# Should reference ml.config.SPICE6_DENSITY_BASE (PJM path)

# 3. Verify delivery_month is used for market_month
grep -n "delivery_month\|market_month" ml/spice6_loader.py
```

**Acceptance criteria:**
- [ ] Returns DataFrame with `constraint_id`, `flow_direction`, `prob_exceed_110`, `constraint_limit`
- [ ] Uses `SPICE6_DENSITY_BASE` from `ml.config` (PJM path, not MISO)
- [ ] Uses `delivery_month()` to compute `market_month` from `auction_month + period_type`
- [ ] Uses new schema only (`score.parquet` with `score` column) — PJM has no legacy `score_df.parquet` (verified in preflight). Remove the legacy branch or guard it behind a "never reached for PJM" assertion.
- [ ] Returns empty DataFrame gracefully if path doesn't exist

---

### Task 6: Create `ml/features.py` — Feature preparation

Copy from MISO with the `v7_formula_score` derived feature. The binding frequency enrichment will be done in the script layer (same as MISO's `run_v10e_lagged.py`), not in this module.

**Files:**
- Create: `ml/features.py`

- [ ] **Step 1: Write `ml/features.py`**

```python
# ml/features.py
"""Feature preparation for PJM LTR pipeline.

Identical to MISO features.py. Binding frequency enrichment happens in
the script layer (run_v2_ml.py), not here.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ml.config import LTRConfig


def _add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add v7_formula_score if not already present.

    IMPORTANT: This column MUST be named 'v7_formula_score' to match
    V10E_FEATURES in ml/config.py. The v2 script (run_v2_ml.py) computes
    it with per-slice blend weights; this fallback uses the V6.2B formula.
    """
    if "v7_formula_score" not in df.columns:
        has_cols = all(c in df.columns for c in
                       ["da_rank_value", "density_mix_rank_value", "density_ori_rank_value"])
        if has_cols:
            df = df.with_columns(
                (0.60 * pl.col("da_rank_value")
                 + 0.30 * pl.col("density_mix_rank_value")
                 + 0.10 * pl.col("density_ori_rank_value")
                ).alias("v7_formula_score")
            )
    return df


def prepare_features(
    df: pl.DataFrame,
    cfg: LTRConfig,
) -> tuple[np.ndarray, list[int]]:
    """Extract feature matrix from df, fill nulls with 0."""
    df = _add_derived_features(df)
    cols = list(cfg.features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[features] WARNING: {len(missing)} features missing, filling with 0: {missing}")

    X_parts = []
    for c in cols:
        if c in df.columns:
            X_parts.append(df[c].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            X_parts.append(np.zeros(len(df), dtype=np.float64))

    X = np.column_stack(X_parts) if X_parts else np.zeros((len(df), 0))
    return X, list(cfg.monotone_constraints)


def compute_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Compute query group sizes from query_month column."""
    months = df["query_month"].to_list()
    groups = []
    if not months:
        return np.array([], dtype=np.int32)

    current = months[0]
    count = 0
    for m in months:
        if m == current:
            count += 1
        else:
            groups.append(count)
            current = m
            count = 1
    groups.append(count)
    return np.array(groups, dtype=np.int32)
```

- [ ] **Step 2: Commit**

```bash
git add ml/features.py
git commit -m "feat: add PJM feature preparation module"
```

#### Review: Task 6

**Verify commands:**
```bash
# 1. Import check
python -c "
from ml.features import prepare_features, compute_query_groups, _add_derived_features
print('features import OK')
"

# 2. CRITICAL: Verify derived feature is named v7_formula_score (NOT v62b_formula_score)
grep -n "v7_formula_score\|v62b_formula_score" ml/features.py
# MUST see: v7_formula_score
# MUST NOT see: v62b_formula_score (this was a name mismatch bug in the original plan)

# 3. Verify name matches V10E_FEATURES in config
python -c "
from ml.config import V10E_FEATURES
assert 'v7_formula_score' in V10E_FEATURES, f'v7_formula_score not in V10E_FEATURES: {V10E_FEATURES}'
assert 'v62b_formula_score' not in V10E_FEATURES, 'Wrong name in V10E_FEATURES'
print(f'V10E_FEATURES: {V10E_FEATURES}')
print('Feature name consistency PASSED')
"

# 4. Verify the derived feature actually gets created
python -c "
import polars as pl
from ml.features import _add_derived_features
df = pl.DataFrame({
    'da_rank_value': [0.1, 0.5],
    'density_mix_rank_value': [0.2, 0.6],
    'density_ori_rank_value': [0.3, 0.7],
})
df = _add_derived_features(df)
assert 'v7_formula_score' in df.columns, 'v7_formula_score not created!'
expected = 0.60 * 0.1 + 0.30 * 0.2 + 0.10 * 0.3  # = 0.15
actual = df['v7_formula_score'][0]
assert abs(actual - expected) < 1e-10, f'Wrong value: {actual} vs {expected}'
print(f'v7_formula_score = {actual} (expected {expected}) — PASSED')
"

# 5. Verify no leaky features can sneak through
python -c "
from ml.config import LTRConfig
cfg = LTRConfig(features=['da_rank_value', 'rank'], monotone_constraints=[-1, 0])
assert 'rank' not in cfg.features
print('Leakage guard OK')
"
```

**Acceptance criteria:**
- [ ] **BLOCK**: Derived feature is named `v7_formula_score`, NOT `v62b_formula_score` — must match `V10E_FEATURES` in config
- [ ] `prepare_features` fills missing features with 0 (no crash on partial data)
- [ ] `_add_derived_features` uses exact V6.2B formula: `0.60*da + 0.30*dmix + 0.10*dori`
- [ ] `compute_query_groups` handles empty input and single-group input
- [ ] Feature matrix dtype is `np.float64`

---

### Task 7: Create `ml/data_loader.py` — V6.2B + spice6 + realized DA

The data loader joins V6.2B signal data with spice6 density features and realized DA ground truth. **Key PJM change**: realized DA is joined on `branch_name` (V6.2B already has this column), not `constraint_id`.

**Files:**
- Create: `ml/data_loader.py`

- [ ] **Step 1: Write `ml/data_loader.py`**

```python
# ml/data_loader.py
"""Data loading for PJM LTR ranking pipeline.

Loads V6.2B signal data, enriches with spice6 density features,
and joins realized DA shadow prices as ground truth.

KEY PJM DIFFERENCE: ground truth joins on branch_name (not constraint_id).
V6.2B parquet already has a branch_name column.
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.realized_da import load_realized_da
from ml.spice6_loader import load_spice6_density


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


_MONTH_CACHE: dict[tuple[str, str, str, str | None], pl.DataFrame] = {}


def clear_month_cache() -> None:
    _MONTH_CACHE.clear()


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density + realized DA.

    PJM-specific: realized DA is joined on branch_name, not constraint_id.
    V6.2B parquet already contains a branch_name column.
    """
    cache_key = (auction_month, period_type, class_type, cache_dir)
    if cache_key in _MONTH_CACHE:
        return _MONTH_CACHE[cache_key]

    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Ensure constraint_id and branch_name are strings
    df = df.with_columns(
        pl.col("constraint_id").cast(pl.String),
        pl.col("branch_name").cast(pl.String),
    )

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(spice6, on=["constraint_id", "flow_direction"], how="left")
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Join realized DA ground truth on branch_name
    from ml.config import delivery_month as _delivery_month
    gt_month = _delivery_month(auction_month, period_type)

    # Map class_type to peak_type for DA fetch
    peak_type = class_type  # PJM uses same names: onpeak, dailyoffpeak, wkndonpeak
    da_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        realized = load_realized_da(gt_month, peak_type=peak_type, **da_kwargs)
        # Join on branch_name (PJM-specific)
        df = df.join(realized, on="branch_name", how="left")
        df = df.with_columns(pl.col("realized_sp").fill_null(0.0))
        n_binding = len(df.filter(pl.col("realized_sp") > 0))
        print(f"[data_loader] realized DA: {n_binding}/{len(df)} binding for {auction_month} "
              f"(gt_month={gt_month})")
    except FileNotFoundError:
        print(f"[data_loader] WARNING: no realized DA for {gt_month}/{peak_type}")
        df = df.with_columns(pl.lit(0.0).alias("realized_sp"))

    _MONTH_CACHE[cache_key] = df
    return df
```

- [ ] **Step 2: Smoke-test the data loader**

```bash
python -c "
from ml.data_loader import load_v62b_month
df = load_v62b_month('2023-06', 'f0', 'onpeak')
print(f'Loaded: {len(df)} rows, {df.columns}')
n_binding = len(df.filter(df['realized_sp'] > 0))
print(f'Binding: {n_binding}/{len(df)}')
"
```

Expected: prints row count, columns including `realized_sp`, and binding count.
Note: This will only work after realized DA is cached (Task 9). If not cached, it falls back to realized_sp=0.

- [ ] **Step 3: Commit**

```bash
git add ml/data_loader.py
git commit -m "feat: add PJM data loader with branch-level DA join"
```

#### Review: Task 7

**Verify commands:**
```bash
# 1. Verify branch_name join (THE critical PJM check)
grep -n "branch_name\|constraint_id" ml/data_loader.py
# MUST see: join on branch_name for realized DA
# MUST NOT see: join realized DA on constraint_id

# 2. Verify V6.2B path structure matches disk
python -c "
from pathlib import Path
from ml.config import V62B_SIGNAL_BASE
p = Path(V62B_SIGNAL_BASE) / '2023-06' / 'f0' / 'onpeak'
print(f'Path exists: {p.exists()} -> {p}')
files = list(p.glob('*.parquet'))
print(f'Parquet files: {len(files)}')
"

# 3. Verify branch_name column exists in V6.2B data
python -c "
import polars as pl
from pathlib import Path
from ml.config import V62B_SIGNAL_BASE
p = Path(V62B_SIGNAL_BASE) / '2023-06' / 'f0' / 'onpeak'
df = pl.read_parquet(str(p))
print(f'Columns: {df.columns}')
assert 'branch_name' in df.columns, 'V6.2B MISSING branch_name — data_loader join will fail!'
assert 'constraint_id' in df.columns
assert 'da_rank_value' in df.columns
print('V6.2B schema check PASSED')
"

# 4. Verify delivery_month is used for ground truth lookup (not auction_month)
grep -n "delivery_month\|gt_month" ml/data_loader.py

# 5. Verify class_type maps to peak_type for DA fetch
grep -n "peak_type" ml/data_loader.py
```

**Acceptance criteria:**
- [ ] **BLOCK**: Realized DA is joined on `branch_name`, NOT `constraint_id`
- [ ] **BLOCK**: Ground truth month = `delivery_month(auction_month, period_type)`, not `auction_month`
- [ ] **BLOCK**: V6.2B parquet on disk actually has a `branch_name` column (verify with command above)
- [ ] Spice6 density joined on `[constraint_id, flow_direction]` (these are spice6 features, not DA)
- [ ] Month cache (`_MONTH_CACHE`) prevents redundant reads
- [ ] Missing realized DA gracefully falls back to `realized_sp=0` with WARNING
- [ ] `mem_mb()` available for memory tracking

---

## Chunk 2: Scripts (Cache, V0 Baseline, V2 ML, Blend, Holdout)

### Task 8: Create `ml/pipeline.py` — single-month pipeline

Adapted from MISO's pipeline.py. Provides a simple pipeline for formula-based evaluation (v0) and basic single-month ML.

**IMPORTANT**: This pipeline does NOT include binding-frequency enrichment or `collect_usable_months()` — those are in `run_v2_ml.py` (Task 11). This module is used for the v0 formula baseline and as a building block. The v2 ML script has its own enrichment logic that supersedes this pipeline.

**Files:**
- Create: `ml/pipeline.py`

- [ ] **Step 1: Write `ml/pipeline.py`**

Copy from `research-stage5-tier/ml/pipeline.py` — the only change is the import paths (now `from ml.config` instead of cross-repo). The code is identical since `data_loader.py` already handles the PJM-specific branch join.

```python
# ml/pipeline.py
"""LTR pipeline: load -> features -> train -> predict -> evaluate."""
from __future__ import annotations

import gc
import resource
from typing import Any

import numpy as np
import polars as pl

from ml.config import PipelineConfig
from ml.data_loader import load_v62b_month
from ml.evaluate import evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.train import predict_scores, train_ltr_model


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_pipeline(
    config: PipelineConfig,
    version_id: str,
    eval_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> dict[str, Any]:
    """Run the LTR pipeline for a single evaluation month."""
    print(f"[pipeline] version={version_id} eval_month={eval_month}")

    import pandas as pd
    eval_ts = pd.Timestamp(eval_month)
    total_lookback = config.train_months + config.val_months

    train_month_strs = []
    for i in range(total_lookback, config.val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    dfs = []
    for m in train_month_strs:
        try:
            df = load_v62b_month(m, period_type, class_type)
            df = df.with_columns(pl.lit(m).alias("query_month"))
            dfs.append(df)
        except FileNotFoundError:
            print(f"[pipeline] WARNING: skipping {m}")
    if not dfs:
        return {"metrics": {}}
    train_df = pl.concat(dfs)

    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    train_df = train_df.sort("query_month")
    X_train, _ = prepare_features(train_df, config.ltr)
    y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
    groups_train = compute_query_groups(train_df)

    del dfs
    gc.collect()

    model = train_ltr_model(X_train, y_train, groups_train, config.ltr)
    del X_train, y_train, groups_train
    gc.collect()

    X_test, _ = prepare_features(test_df, config.ltr)
    scores = predict_scores(model, X_test)
    actual_sp = test_df["realized_sp"].to_numpy().astype(np.float64)

    metrics = evaluate_ltr(actual_sp, scores)

    feat_names = config.ltr.features
    if hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        importance = model.feature_importances_
    metrics["_feature_importance"] = {
        name: float(imp)
        for name, imp in sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    }

    del X_test, scores, actual_sp, test_df, model
    gc.collect()

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")

    return {"metrics": metrics}
```

- [ ] **Step 2: Commit**

```bash
git add ml/pipeline.py
git commit -m "feat: add PJM pipeline module"
```

#### Review: Task 8

**Verify commands:**
```bash
# 1. Import check
python -c "from ml.pipeline import run_pipeline; print('pipeline OK')"

# 2. Verify train data ordering (must be sorted by query_month for LTR)
grep -n "sort.*query_month" ml/pipeline.py

# 3. Verify gc.collect() after each phase
grep -cn "gc.collect" ml/pipeline.py
# Should be >= 2 (after training, after evaluation)

# 4. Verify feature importance extraction
grep -n "feature_importance\|feature_importances" ml/pipeline.py
```

**Acceptance criteria:**
- [ ] Training data sorted by `query_month` before `compute_query_groups`
- [ ] `gc.collect()` called after training data freed and after evaluation
- [ ] Feature importance extracted from model and included in metrics
- [ ] Uses `load_v62b_month` (which handles branch-level join internally)
- [ ] **NOTE**: This pipeline does NOT include BF enrichment or `collect_usable_months()` — those are in `run_v2_ml.py` (Task 11). This is intentional: pipeline.py is a building block, not the full v2 model.

---

### Task 9: Create `scripts/cache_realized_da.py` — Preflight DA cache

This script must be run FIRST (requires Ray) to cache all realized DA months needed for evaluation. It fetches PJM DA via `PjmApTools`, maps to branches, and caches.

**Files:**
- Create: `scripts/cache_realized_da.py`

- [ ] **Step 1: Write the cache script**

```python
#!/usr/bin/env python
"""Cache PJM realized DA shadow prices for all months needed by the ML pipeline.

Must be run before any experiment scripts. Requires Ray connection.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python scripts/cache_realized_da.py --peak-types onpeak dailyoffpeak wkndonpeak
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import REALIZED_DA_CACHE, _FULL_EVAL_MONTHS, HOLDOUT_MONTHS
from ml.realized_da import fetch_and_cache_month, _cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-types", nargs="+",
                        default=["onpeak", "dailyoffpeak", "wkndonpeak"])
    parser.add_argument("--start", default="2019-01",
                        help="First month to cache (YYYY-MM)")
    parser.add_argument("--end", default="2026-02",
                        help="Last month to cache (YYYY-MM)")
    parser.add_argument("--cache-dir", default=REALIZED_DA_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Init Ray
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import pandas as pd

    # Generate all months in range
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += pd.DateOffset(months=1)

    print(f"[cache] Caching {len(months)} months × {len(args.peak_types)} peak types")
    print(f"[cache] Cache dir: {args.cache_dir}")

    if args.dry_run:
        # Count existing vs missing
        existing = sum(
            1 for m in months for pt in args.peak_types
            if _cache_path(m, pt, args.cache_dir).exists()
        )
        total = len(months) * len(args.peak_types)
        print(f"[cache] DRY RUN: {existing}/{total} already cached, {total - existing} to fetch")
        return

    t0 = time.time()
    fetched = 0
    skipped = 0
    already_cached = 0

    for pt in args.peak_types:
        for m in months:
            p = _cache_path(m, pt, args.cache_dir)
            if p.exists():
                already_cached += 1
                continue
            try:
                fetch_and_cache_month(m, pt, cache_dir=args.cache_dir)
                fetched += 1
            except Exception as e:
                print(f"[cache] SKIP {m}/{pt}: {e}")
                skipped += 1

    elapsed = time.time() - t0
    print(f"\n[cache] Done in {elapsed:.0f}s: {fetched} fetched, "
          f"{already_cached} cached, {skipped} skipped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the cache script**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python scripts/cache_realized_da.py --dry-run
```

Expected: prints count of existing vs missing files. Then run without `--dry-run` to actually fetch.

- [ ] **Step 3: Commit**

```bash
git add scripts/cache_realized_da.py
git commit -m "feat: add realized DA cache script for PJM (branch-level)"
```

#### Review: Task 9

**Verify commands:**
```bash
# 1. Dry run works
python scripts/cache_realized_da.py --dry-run

# 2. After full run: verify cache contents
python -c "
from pathlib import Path
from ml.config import REALIZED_DA_CACHE
import polars as pl

cache = Path(REALIZED_DA_CACHE)
files = sorted(cache.glob('*.parquet'))
print(f'Cached files: {len(files)}')
if files:
    # Spot-check one file
    df = pl.read_parquet(str(files[len(files)//2]))
    print(f'Sample file: {files[len(files)//2].name}')
    print(f'Columns: {df.columns}')
    print(f'Rows: {len(df)}')
    assert 'branch_name' in df.columns, 'MISSING branch_name in cache!'
    assert 'realized_sp' in df.columns, 'MISSING realized_sp in cache!'
    print('Cache schema PASSED')
"

# 3. Verify Ray init pattern
grep -n "RAY_ADDRESS\|init_ray" scripts/cache_realized_da.py
# Must see RAY_ADDRESS set BEFORE init_ray()

# 4. Check for all 3 peak types
grep -n "peak.type\|onpeak\|dailyoffpeak\|wkndonpeak" scripts/cache_realized_da.py
```

**Acceptance criteria:**
- [ ] **BLOCK**: Cached parquet files have columns `[branch_name, realized_sp]` (NOT `constraint_id`)
- [ ] **BLOCK**: Ray initialized with correct address before any data fetch
- [ ] All 3 peak types cached: onpeak, dailyoffpeak, wkndonpeak
- [ ] Month range covers at least 2019-01 to 2026-02 (training lookback + holdout)
- [ ] Dry-run flag works (no actual fetching)
- [ ] Already-cached files are skipped (idempotent)
- [ ] Print total time, fetched count, skipped count

---

### Task 10: Create `scripts/run_v0_formula_baseline.py`

Run V6.2B formula baseline for all 6 ML slices. Establishes gates and champion.

**Files:**
- Create: `scripts/run_v0_formula_baseline.py`

- [ ] **Step 1: Write the v0 baseline script**

Adapt from `research-stage5-tier/scripts/run_v0_formula_baseline.py`. Key changes:
- Loop over 3 class types instead of 2
- Use `branch_name` for realized DA join (already handled by `data_loader.py`)
- PJM auction schedule for f1 filtering

```python
#!/usr/bin/env python
"""V0 formula baseline for PJM: evaluate V6.2B formula against realized DA.

Runs all 6 ML slices: f0×{onpeak,dailyoffpeak,wkndonpeak} + f1×{same}.
Saves to registry/{ptype}/{ctype}/v0/ and calibrates gates.json + champion.json.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v0_formula_baseline.py
    python scripts/run_v0_formula_baseline.py --ptype f0 --class-type onpeak  # single slice
    python scripts/run_v0_formula_baseline.py --holdout  # include holdout
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    V62B_SIGNAL_BASE, PJM_CLASS_TYPES, _FULL_EVAL_MONTHS,
    HOLDOUT_MONTHS, has_period_type,
)
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.realized_da import load_realized_da
from ml.registry_paths import registry_root, holdout_root
from ml.v62b_formula import v62b_score

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def evaluate_month(month: str, class_type: str, period_type: str) -> dict:
    """Evaluate V6.2B formula on one month against realized DA."""
    from ml.config import delivery_month as _delivery_month

    path = Path(V62B_SIGNAL_BASE) / month / period_type / class_type
    df = pl.read_parquet(str(path))
    df = df.with_columns(pl.col("branch_name").cast(pl.String))

    gt_month = _delivery_month(month, period_type)
    realized = load_realized_da(gt_month, peak_type=class_type)
    df = df.join(realized, on="branch_name", how="left")
    df = df.with_columns(pl.col("realized_sp").fill_null(0.0))

    actual = df["realized_sp"].to_numpy().astype(np.float64)

    # Negate because lower rank_value = more binding, but eval expects higher = better
    scores = -v62b_score(
        da_rank_value=df["da_rank_value"].to_numpy(),
        density_mix_rank_value=df["density_mix_rank_value"].to_numpy(),
        density_ori_rank_value=df["density_ori_rank_value"].to_numpy(),
    )

    metrics = evaluate_ltr(actual, scores)
    n_binding = int((actual > 0).sum())
    print(f"  {month}: n={len(df)}, binding={n_binding}, "
          f"VC@20={metrics['VC@20']:.4f}, VC@100={metrics['VC@100']:.4f}")

    del df, realized, actual, scores
    gc.collect()
    return metrics


def build_gates(agg: dict) -> dict:
    group_a = ["VC@20", "VC@100", "Recall@20", "Recall@50", "Recall@100", "NDCG"]
    gates = {}
    means = agg["mean"]
    mins = agg["min"]
    for metric in group_a:
        if metric in means:
            gates[metric] = {
                "floor": round(0.9 * means[metric], 4),
                "tail_floor": round(mins[metric], 4),
                "direction": "higher",
                "group": "A",
            }
    return {
        "gates": gates,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "calibrated_from": "v0",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_slice(
    period_type: str,
    class_type: str,
    eval_months: list[str],
    holdout: bool = False,
):
    """Run v0 for one (ptype, ctype) slice."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[v0] {period_type}/{class_type} ({len(eval_months)} eval months)")
    print(f"{'='*60}")

    per_month: dict[str, dict] = {}
    for month in eval_months:
        try:
            per_month[month] = evaluate_month(month, class_type, period_type)
        except FileNotFoundError as e:
            print(f"  {month}: SKIP ({e})")

    if not per_month:
        print(f"[v0] No valid months for {period_type}/{class_type}")
        return

    agg = aggregate_months(per_month)
    means = agg["mean"]
    print(f"\n[v0] Aggregate ({len(per_month)} months):")
    for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG", "Spearman"]:
        print(f"  {metric:<12} {means.get(metric, 0):.4f}")

    # Save to registry
    v0_dir = registry_root(period_type, class_type, base_dir=REGISTRY) / "v0"
    v0_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "eval_config": {"eval_months": sorted(per_month.keys()), "class_type": class_type,
                        "period_type": period_type, "mode": "eval"},
        "per_month": per_month, "aggregate": agg,
        "n_months": len(per_month), "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(per_month.keys())),
    }
    with open(v0_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    config_out = {
        "method": "v62b_formula",
        "formula": "-(0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value)",
        "ground_truth": f"realized_da (branch-level sum, {class_type})",
    }
    with open(v0_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # Gates + champion
    slice_dir = registry_root(period_type, class_type, base_dir=REGISTRY)
    gates_data = build_gates(agg)
    with open(slice_dir / "gates.json", "w") as f:
        json.dump(gates_data, f, indent=2)

    champion_data = {
        "version": "v0",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "reason": f"initial baseline ({period_type}/{class_type})",
    }
    with open(slice_dir / "champion.json", "w") as f:
        json.dump(champion_data, f, indent=2)

    print(f"[v0] Saved to {v0_dir} ({time.time()-t0:.1f}s)")

    # Holdout
    if holdout:
        ho_months = [m for m in HOLDOUT_MONTHS if has_period_type(m, period_type)]
        print(f"\n[v0] Holdout ({len(ho_months)} months)...")
        ho_pm: dict[str, dict] = {}
        for month in ho_months:
            try:
                ho_pm[month] = evaluate_month(month, class_type, period_type)
            except FileNotFoundError:
                print(f"  {month}: SKIP")

        if ho_pm:
            ho_agg = aggregate_months(ho_pm)
            ho_dir = holdout_root(period_type, class_type, base_dir=HOLDOUT_DIR) / "v0"
            ho_dir.mkdir(parents=True, exist_ok=True)
            ho_out = {
                "eval_config": {"eval_months": sorted(ho_pm.keys()), "class_type": class_type,
                                "period_type": period_type, "mode": "holdout"},
                "per_month": ho_pm, "aggregate": ho_agg,
                "n_months": len(ho_pm), "n_months_requested": len(ho_months),
                "skipped_months": sorted(set(ho_months) - set(ho_pm.keys())),
            }
            with open(ho_dir / "metrics.json", "w") as f:
                json.dump(ho_out, f, indent=2)
            ho_means = ho_agg["mean"]
            print(f"[v0] Holdout aggregate ({len(ho_pm)} months):")
            for metric in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                print(f"  {metric:<12} {ho_means.get(metric, 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None, help="Period type (f0 or f1). Default: both.")
    parser.add_argument("--class-type", default=None, help="Class type. Default: all 3.")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--full", action="store_true", help="Use 36 eval months")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES

    for ptype in ptypes:
        eval_months = _FULL_EVAL_MONTHS if args.full else _FULL_EVAL_MONTHS
        eval_months = [m for m in eval_months if has_period_type(m, ptype)]
        for ctype in ctypes:
            run_slice(ptype, ctype, eval_months, holdout=args.holdout)

    print("\n[v0] All slices complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run v0 for a single slice first**

```bash
python scripts/run_v0_formula_baseline.py --ptype f0 --class-type onpeak
```

Expected: prints per-month metrics, aggregate, saves to `registry/f0/onpeak/v0/metrics.json`.

- [ ] **Step 3: Run v0 for all 6 slices**

```bash
python scripts/run_v0_formula_baseline.py --holdout
```

Expected: saves v0 results for all 6 slices to registry/ and holdout/.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v0_formula_baseline.py registry/ holdout/
git commit -m "feat: v0 formula baseline for all 6 PJM slices"
```

#### Review: Task 10

**Verify commands:**
```bash
# 1. Registry structure: all 6 slices have v0
python -c "
from pathlib import Path
import json
root = Path('registry')
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        v0 = root / ptype / ctype / 'v0' / 'metrics.json'
        gates = root / ptype / ctype / 'gates.json'
        champ = root / ptype / ctype / 'champion.json'
        m = json.load(open(v0)) if v0.exists() else None
        print(f'{ptype}/{ctype}: metrics={v0.exists()} gates={gates.exists()} champion={champ.exists()}', end='')
        if m:
            agg = m['aggregate']['mean']
            print(f'  VC@20={agg[\"VC@20\"]:.4f}  n_months={m[\"n_months\"]}')
        else:
            print('  MISSING')
"

# 2. Sanity check: VC@20 in reasonable range
python -c "
import json
from pathlib import Path
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        p = Path(f'registry/{ptype}/{ctype}/v0/metrics.json')
        if not p.exists(): continue
        vc20 = json.load(open(p))['aggregate']['mean']['VC@20']
        status = 'OK' if 0.10 <= vc20 <= 0.45 else 'SUSPICIOUS'
        print(f'{ptype}/{ctype} VC@20={vc20:.4f} [{status}]')
"

# 3. Verify branch_name join is used (not constraint_id)
grep -n "branch_name\|constraint_id" scripts/run_v0_formula_baseline.py
# Should see: join on branch_name

# 4. Verify formula negation (lower rank_value = more binding, eval expects higher = better)
grep -n "negate\|\-v62b" scripts/run_v0_formula_baseline.py

# 5. Gates calibration sanity
python -c "
import json
from pathlib import Path
g = json.load(open('registry/f0/onpeak/gates.json'))
print(json.dumps(g, indent=2))
# floor should be ~0.9 * mean for each metric
"
```

**Acceptance criteria:**
- [ ] **BLOCK**: All 6 slices have `v0/metrics.json`, `gates.json`, `champion.json`
- [ ] **BLOCK**: VC@20 in range 0.10-0.45 for all slices (if 0.0 or >0.5, target join is broken)
- [ ] **BLOCK**: Realized DA joined on `branch_name` in evaluate_month()
- [ ] Formula scores negated before evaluation (lower rank = better, but evaluate_ltr expects higher = better)
- [ ] Gates floor = 0.9 × mean for each Group A metric
- [ ] `champion.json` version is "v0" for all slices
- [ ] Holdout results saved to `holdout/` if `--holdout` was run

---

### Task 11: Create `scripts/run_v2_ml.py` — ML model with binding frequency

The v2 ML script trains LightGBM LambdaRank with 9 features (5 binding freq + v7_formula_score + prob_exceed_110 + constraint_limit + da_rank_value).

**Key**: Binding frequency uses `branch_name` keys (not constraint_id). BF_LAG=1 always.

**Files:**
- Create: `scripts/run_v2_ml.py`

- [ ] **Step 1: Write the v2 ML script**

Adapt from `research-stage5-tier/scripts/run_v10e_lagged.py`. Key PJM changes:
- `load_all_binding_sets()` reads branch_name from cached parquet (not constraint_id)
- `compute_bf()` operates on branch_name
- Loop over 3 class types
- Per-slice blend weights (start with MISO defaults, will tune in blend search)

```python
#!/usr/bin/env python
"""V2 ML model for PJM: LightGBM LambdaRank with 9 features.

Runs dev (36mo) and holdout (24mo) for all 6 ML slices.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/run_v2_ml.py --ptype f0 --class-type onpeak
    python scripts/run_v2_ml.py  # all 6 slices
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    REALIZED_DA_CACHE, LTRConfig, PipelineConfig, V10E_FEATURES, V10E_MONOTONE,
    _FULL_EVAL_MONTHS, HOLDOUT_MONTHS, PJM_CLASS_TYPES,
    has_period_type, collect_usable_months,
)
from ml.data_loader import load_v62b_month
from ml.evaluate import aggregate_months, evaluate_ltr
from ml.features import compute_query_groups, prepare_features
from ml.registry_paths import registry_root, holdout_root
from ml.train import predict_scores, train_ltr_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry"
HOLDOUT_DIR = ROOT / "holdout"

# Per-(period_type, class_type) blend weights for v7_formula_score.
# Start with MISO defaults. Will be tuned by blend search script.
BLEND_WEIGHTS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("f0", "onpeak"): (0.85, 0.00, 0.15),
    ("f0", "dailyoffpeak"): (0.85, 0.00, 0.15),
    ("f0", "wkndonpeak"): (0.85, 0.00, 0.15),
    ("f1", "onpeak"): (0.70, 0.00, 0.30),
    ("f1", "dailyoffpeak"): (0.80, 0.00, 0.20),
    ("f1", "wkndonpeak"): (0.80, 0.00, 0.20),
}
_DEFAULT_BLEND = (0.85, 0.00, 0.15)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_all_binding_sets(
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, set[str]]:
    """Load all cached DA into {month: set(branch_names)}.

    PJM-specific: keys are branch_name, not constraint_id.
    """
    binding_sets: dict[str, set[str]] = {}
    if peak_type == "onpeak":
        pattern = "[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"
    else:
        pattern = f"*_{peak_type}.parquet"
    for f in sorted(Path(cache_dir).glob(pattern)):
        df = pl.read_parquet(str(f))
        month = f.stem.replace(f"_{peak_type}", "")
        binding_sets[month] = set(
            df.filter(pl.col("realized_sp") > 0)["branch_name"].to_list()
        )
    print(f"[bf] Loaded {len(binding_sets)} months of {peak_type} binding sets")
    return binding_sets


def compute_bf(
    branch_names: list[str], month: str,
    bs: dict[str, set[str]], lookback: int,
) -> np.ndarray:
    """Compute binding frequency for branch_names."""
    prior = [m for m in sorted(bs.keys()) if m < month][-lookback:]
    n = len(prior)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)
    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in prior:
        s = bs[m]
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def prev_month(m: str) -> str:
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def enrich_df(
    df: pl.DataFrame, month: str, bs: dict[str, set[str]],
    blend_weights: tuple[float, float, float],
) -> pl.DataFrame:
    """Add binding_freq and formula score features. BF_LAG=1 always."""
    cutoff = prev_month(month)  # BF sees months strictly before M-1
    w_da, w_dmix, w_dori = blend_weights

    # PJM-specific: use branch_name for BF, not constraint_id
    branch_names = df["branch_name"].to_list()

    df = df.with_columns(
        (w_da * pl.col("da_rank_value")
         + w_dmix * pl.col("density_mix_rank_value")
         + w_dori * pl.col("density_ori_rank_value")
        ).alias("v7_formula_score")
    )
    for lb in [1, 3, 6, 12, 15]:
        col_name = f"binding_freq_{lb}"
        if col_name not in df.columns:
            freq = compute_bf(branch_names, cutoff, bs, lb)
            df = df.with_columns(pl.Series(col_name, freq))
    return df


def run_variant(
    label: str,
    eval_months: list[str],
    bs: dict[str, set[str]],
    class_type: str,
    period_type: str,
) -> dict[str, dict]:
    blend = BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND)
    print(f"\n[{label}] 9f, {len(eval_months)} months, ptype={period_type}, "
          f"class_type={class_type}, blend={blend}")

    cfg = PipelineConfig(
        ltr=LTRConfig(
            features=list(V10E_FEATURES),
            monotone_constraints=list(V10E_MONOTONE),
            backend="lightgbm",
            label_mode="tiered",
        ),
        train_months=8,
        val_months=0,
    )

    per_month: dict[str, dict] = {}

    for m in eval_months:
        t0 = time.time()
        train_month_strs = collect_usable_months(m, period_type, n_months=8)
        if not train_month_strs:
            print(f"  {m}: SKIP (insufficient history)")
            continue
        train_month_strs = list(reversed(train_month_strs))

        parts = []
        for tm in train_month_strs:
            try:
                part = load_v62b_month(tm, period_type, class_type)
                part = part.with_columns(pl.lit(tm).alias("query_month"))
                part = enrich_df(part, tm, bs, blend)
                parts.append(part)
            except FileNotFoundError:
                pass
        if not parts:
            continue
        train_df = pl.concat(parts)

        try:
            test_df = load_v62b_month(m, period_type, class_type)
        except FileNotFoundError:
            continue
        test_df = test_df.with_columns(pl.lit(m).alias("query_month"))
        test_df = enrich_df(test_df, m, bs, blend)

        train_df = train_df.sort("query_month")
        X_train, _ = prepare_features(train_df, cfg.ltr)
        y_train = train_df["realized_sp"].to_numpy().astype(np.float64)
        groups = compute_query_groups(train_df)

        model = train_ltr_model(X_train, y_train, groups, cfg.ltr)
        X_test, _ = prepare_features(test_df, cfg.ltr)
        scores = predict_scores(model, X_test)
        actual = test_df["realized_sp"].to_numpy().astype(np.float64)

        metrics = evaluate_ltr(actual, scores)
        per_month[m] = metrics

        if hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type="gain")
            metrics["_fi"] = dict(zip(cfg.ltr.features, [float(x) for x in imp]))

        elapsed = time.time() - t0
        n_bind = int((actual > 0).sum())
        print(f"  {m}: VC@20={metrics['VC@20']:.4f} binding={n_bind} ({elapsed:.1f}s)")

        del train_df, test_df, parts, X_train, y_train, groups, model, X_test, actual, scores
        gc.collect()

    if per_month:
        clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                 for m, met in per_month.items()}
        agg = aggregate_months(clean)
        means = agg["mean"]
        print(f"  => VC@20={means['VC@20']:.4f} VC@100={means['VC@100']:.4f} "
              f"NDCG={means['NDCG']:.4f} Spearman={means['Spearman']:.4f}")

    return per_month


def save_results(label, per_month, eval_months, dest_dir, class_type, period_type):
    clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
             for m, met in per_month.items()}
    agg = aggregate_months(clean)
    d = dest_dir / label
    d.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_config": {"eval_months": sorted(clean.keys()), "class_type": class_type,
                        "period_type": period_type, "lag": 1},
        "per_month": clean, "aggregate": agg, "n_months": len(clean),
        "n_months_requested": len(eval_months),
        "skipped_months": sorted(set(eval_months) - set(clean.keys())),
    }
    with open(d / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    config = {
        "method": "lightgbm_lambdarank_tiered",
        "features": list(V10E_FEATURES),
        "lag": 1, "period_type": period_type, "class_type": class_type,
        "blend_weights": dict(zip(["da", "dmix", "dori"],
                                  BLEND_WEIGHTS.get((period_type, class_type), _DEFAULT_BLEND))),
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved to {d}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptype", default=None)
    parser.add_argument("--class-type", default=None)
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()

    ptypes = [args.ptype] if args.ptype else ["f0", "f1"]
    ctypes = [args.class_type] if args.class_type else PJM_CLASS_TYPES
    t_start = time.time()

    for ptype in ptypes:
        for ctype in ctypes:
            print(f"\n{'='*70}")
            print(f"V2 ML: {ptype}/{ctype}")
            print(f"{'='*70}")

            bs = load_all_binding_sets(peak_type=ctype)

            dev_eval = [m for m in _FULL_EVAL_MONTHS if has_period_type(m, ptype)]
            reg_slice = registry_root(ptype, ctype, base_dir=REGISTRY)

            dev_pm = run_variant("v2", dev_eval, bs, class_type=ctype, period_type=ptype)
            if dev_pm:
                save_results("v2", dev_pm, dev_eval, reg_slice,
                             class_type=ctype, period_type=ptype)

            if not args.dev_only:
                ho_eval = [m for m in HOLDOUT_MONTHS if has_period_type(m, ptype)]
                ho_slice = holdout_root(ptype, ctype, base_dir=HOLDOUT_DIR)
                ho_pm = run_variant("v2-holdout", ho_eval, bs,
                                    class_type=ctype, period_type=ptype)
                if ho_pm:
                    save_results("v2", ho_pm, ho_eval, ho_slice,
                                 class_type=ctype, period_type=ptype)

            # Print comparison vs v0
            v0_path = reg_slice / "v0" / "metrics.json"
            if v0_path.exists() and dev_pm:
                v0_agg = json.load(open(v0_path))["aggregate"]["mean"]
                clean = {m: {k: v for k, v in met.items() if not k.startswith("_")}
                         for m, met in dev_pm.items()}
                v2_agg = aggregate_months(clean)["mean"]
                print(f"\n  {'Metric':<12} {'v0':>8} {'v2':>8} {'delta':>8}")
                for met in ["VC@20", "VC@50", "VC@100", "Recall@20", "NDCG"]:
                    v0_v = v0_agg.get(met, 0)
                    v2_v = v2_agg.get(met, 0)
                    delta = (v2_v / v0_v - 1) * 100 if v0_v > 0 else 0
                    print(f"  {met:<12} {v0_v:>8.4f} {v2_v:>8.4f} {delta:>+7.1f}%")

    print(f"\n[main] All done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run v2 for a single slice first**

```bash
python scripts/run_v2_ml.py --ptype f0 --class-type onpeak --dev-only
```

Expected: prints per-month VC@20, aggregate, saves to `registry/f0/onpeak/v2/`.

- [ ] **Step 3: Run v2 for all slices**

```bash
python scripts/run_v2_ml.py
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_v2_ml.py registry/ holdout/
git commit -m "feat: v2 ML model for all 6 PJM slices"
```

#### Review: Task 11

**Verify commands:**
```bash
# 1. v2 results exist for all 6 slices
python -c "
import json
from pathlib import Path
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        v2 = Path(f'registry/{ptype}/{ctype}/v2/metrics.json')
        v0 = Path(f'registry/{ptype}/{ctype}/v0/metrics.json')
        if not v2.exists():
            print(f'{ptype}/{ctype}: v2 MISSING')
            continue
        v2_vc = json.load(open(v2))['aggregate']['mean']['VC@20']
        v0_vc = json.load(open(v0))['aggregate']['mean']['VC@20']
        delta = (v2_vc / v0_vc - 1) * 100 if v0_vc > 0 else 0
        status = 'OK' if delta > 0 else 'REGRESSION'
        print(f'{ptype}/{ctype}: v0={v0_vc:.4f} v2={v2_vc:.4f} delta={delta:+.1f}% [{status}]')
"

# 2. LEAKAGE CHECK: binding freq uses correct lag
grep -n "prev_month\|cutoff\|BF_LAG\|months.*before\|< M-1" scripts/run_v2_ml.py
# cutoff should be prev_month(auction_month), i.e., BF sees months < M-1

# 3. LEAKAGE CHECK: training window uses collect_usable_months
grep -n "collect_usable_months" scripts/run_v2_ml.py
# Must be present — this enforces the delivery month lag

# 4. Binding freq uses branch_name (not constraint_id)
grep -n "branch_name\|constraint_id" scripts/run_v2_ml.py
# BF should use branch_name from cached DA (which has branch_name columns)

# 5. Feature count = 9
python -c "
from ml.config import V10E_FEATURES
print(f'Features ({len(V10E_FEATURES)}): {V10E_FEATURES}')
assert len(V10E_FEATURES) == 9, f'Expected 9 features, got {len(V10E_FEATURES)}'
"

# 6. Feature importance check (top features should make sense)
python -c "
import json
from pathlib import Path
p = Path('registry/f0/onpeak/v2/metrics.json')
if p.exists():
    data = json.load(open(p))
    # Check first month's feature importance
    first_month = sorted(data['per_month'].keys())[0]
    fi = data['per_month'][first_month].get('_fi', {})
    if fi:
        print('Feature importance (f0/onpeak, first month):')
        for k, v in sorted(fi.items(), key=lambda x: -x[1]):
            print(f'  {k}: {v:.1f}')
    else:
        print('No feature importance saved (check _fi key)')
"

# 7. Memory: check script doesn't leak memory
grep -n "gc.collect\|del " scripts/run_v2_ml.py
```

**Acceptance criteria:**
- [ ] **BLOCK**: v2 VC@20 > v0 VC@20 for at least 5 of 6 slices (ML must beat formula)
- [ ] **BLOCK**: If v2 is NOT better than v0 for any slice, investigate target join — branch mapping may be wrong
- [ ] **BLOCK**: BF cutoff = `prev_month(auction_month)` — months strictly < M-1 (temporal leakage guard)
- [ ] **BLOCK**: Training uses `collect_usable_months()` which respects delivery month lag
- [ ] **BLOCK**: BF keys are `branch_name` (from cached DA parquet), not `constraint_id`
- [ ] Exactly 9 features used
- [ ] Feature importance is saved in per-month metrics
- [ ] `gc.collect()` called after each eval month
- [ ] Walltime printed at end
- [ ] **HIGH**: f1 snapshot provenance verified (see check below)

**f1 snapshot provenance check** (Finding 7 from review):
V6.2B and Spice6 snapshots for f1 must have been produced AT or shortly after
the auction date — not bulk-regenerated later with data that wasn't available
at auction time. If all files share a recent mtime (e.g., 2026), the snapshots
were regenerated and may embed forward-looking information = leakage.

```bash
python -c "
from pathlib import Path
import os, pandas as pd

base = Path('/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1')

for month in ['2023-06', '2024-01', '2025-01']:
    f1_path = base / month / 'f1' / 'onpeak'
    if not f1_path.exists():
        print(f'{month}/f1: NOT FOUND')
        continue
    files = list(f1_path.glob('*.parquet'))
    if files:
        mtime = os.path.getmtime(str(files[0]))
        mtime_ts = pd.Timestamp(mtime, unit='s')
        # Auction for month M happens ~mid(M-1).
        # File should have been written around auction time.
        auction_approx = pd.Timestamp(month) - pd.Timedelta(days=15)
        lag_days = (mtime_ts - auction_approx).days
        if lag_days > 90:
            status = 'SUSPECT — written >90d after auction, likely regenerated'
        elif lag_days < -30:
            status = 'SUSPECT — written >30d BEFORE auction'
        else:
            status = 'OK — written near auction time'
        print(f'{month}/f1: mtime={mtime_ts.date()}, auction~{auction_approx.date()}, lag={lag_days}d [{status}]')
"
```

**How to interpret**:
- `OK` (mtime within ~90 days of auction): snapshot was produced around auction time, features are point-in-time.
- `SUSPECT — written >90d after auction`: all files may have been bulk-regenerated with future data.
  If ALL sampled months show the same recent mtime (e.g., all 2026), escalate to data owner.
  The features cannot be trusted as point-in-time without confirmation.
- `SUSPECT — written >30d BEFORE auction`: file predates the auction — unusual, investigate.

---

### Task 12: Create `scripts/run_blend_search.py`

Search for optimal `(w_da, w_dmix, w_dori)` blend per slice on the full simplex.

**Files:**
- Create: `scripts/run_blend_search.py`

- [ ] **Step 1: Write blend search script**

This runs the v2 ML model with different blend weights for `v7_formula_score` and evaluates which produces the best dev VC@20.

**Artifact contract**: The blend search updates `BLEND_WEIGHTS` in `run_v2_ml.py` and saves the winning blend to `registry/{ptype}/{ctype}/v2/config.json` under `"blend_weights"`. There is NO separate "v1" version — the blend search refines v2's formula-score feature. After blend search, re-run `run_v2_ml.py` with the optimized weights and commit updated v2 results.

The blend weights control how `v7_formula_score` (a feature) is computed:
`v7_formula_score = w_da * da_rank_value + w_dmix * density_mix_rank_value + w_dori * density_ori_rank_value`

Search grid: Explore the full `(w_da, w_dmix, w_dori)` simplex (all three weights, summing to 1.0) in 0.05 increments. Do NOT hardcode `w_dmix = 0` without PJM-specific evidence that `density_mix_rank_value` adds no value.

This script can be adapted from `research-stage5-tier/scripts/run_f1_blend_search.py`.

- [ ] **Step 2: Run blend search for f0/onpeak**

```bash
python scripts/run_blend_search.py --ptype f0 --class-type onpeak
```

- [ ] **Step 3: Run for all 6 slices and commit**

```bash
python scripts/run_blend_search.py
git add scripts/run_blend_search.py registry/
git commit -m "feat: blend search for all 6 PJM slices"
```

#### Review: Task 12

**Verify commands:**
```bash
# 1. Optimized blend weights stored in v2/config.json for all slices
python -c "
import json
from pathlib import Path
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        cfg_path = Path(f'registry/{ptype}/{ctype}/v2/config.json')
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            bw = cfg.get('blend_weights', {})
            print(f'{ptype}/{ctype}: blend_weights={bw}')
        else:
            print(f'{ptype}/{ctype}: v2/config.json MISSING')
"

# 2. BLEND_WEIGHTS in run_v2_ml.py match config.json
grep -A 8 "BLEND_WEIGHTS" scripts/run_v2_ml.py

# 3. Blend search explored full simplex (not just w_dmix=0 edge)
grep -n "w_dmix\|density_mix\|simplex" scripts/run_blend_search.py
# Should see: w_dmix explored (not hardcoded to 0)

# 4. v2 results were re-run with optimized blend (check updated metrics)
python -c "
import json
from pathlib import Path
# v2 metrics should reflect optimized blend, not defaults
p = Path('registry/f0/onpeak/v2/config.json')
if p.exists():
    cfg = json.load(open(p))
    bw = cfg.get('blend_weights', {})
    print(f'f0/onpeak blend: {bw}')
    # If w_da=0.85, w_dmix=0.0, w_dori=0.15 — these are defaults; blend search may not have changed them
    # But the search should have at least TRIED other values
"
```

**Acceptance criteria:**
- [ ] Blend search explored the full `(w_da, w_dmix, w_dori)` simplex (at least 20 combinations per slice)
- [ ] Best blend stored in `registry/{ptype}/{ctype}/v2/config.json` under `"blend_weights"`
- [ ] `BLEND_WEIGHTS` in `run_v2_ml.py` updated to match search results
- [ ] v2 metrics re-generated with optimized blend (not stale from pre-search run)
- [ ] Best blend VC@20 >= v2 with default blend (blend search should not regress)

---

### Task 13: Create `scripts/run_holdout.py` — Final holdout evaluation

Re-run v2 (with optimized blend from Task 12) on held-out months (2024-2025).

**Files:**
- Create: `scripts/run_holdout.py`

- [ ] **Step 1: Write holdout script**

This is a thin wrapper that runs `run_variant()` from `run_v2_ml.py` on holdout months. It reads `BLEND_WEIGHTS` from `run_v2_ml.py` (already updated by blend search in Task 12), ensuring holdout uses the same blend as the dev evaluation.

- [ ] **Step 2: Run holdout and commit**

```bash
python scripts/run_holdout.py
git add scripts/run_holdout.py holdout/
git commit -m "feat: holdout evaluation for all 6 PJM slices"
```

#### Review: Task 13

**Verify commands:**
```bash
# 1. Holdout results for all 6 slices
python -c "
import json
from pathlib import Path
print(f'{\"Slice\":<25} {\"Dev VC@20\":>10} {\"HO VC@20\":>10} {\"HO vs Dev\":>10}')
print('-' * 60)
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        dev = Path(f'registry/{ptype}/{ctype}/v2/metrics.json')
        ho = Path(f'holdout/{ptype}/{ctype}/v2/metrics.json')
        dev_vc = json.load(open(dev))['aggregate']['mean']['VC@20'] if dev.exists() else 0
        ho_vc = json.load(open(ho))['aggregate']['mean']['VC@20'] if ho.exists() else 0
        ratio = ho_vc / dev_vc if dev_vc > 0 else 0
        print(f'{ptype}/{ctype:<20} {dev_vc:>10.4f} {ho_vc:>10.4f} {ratio:>9.1%}')
"

# 2. Holdout vs v0 holdout (ML should beat formula on holdout too)
python -c "
import json
from pathlib import Path
print(f'{\"Slice\":<25} {\"v0 HO\":>10} {\"v2 HO\":>10} {\"Delta\":>10}')
print('-' * 60)
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        v0_ho = Path(f'holdout/{ptype}/{ctype}/v0/metrics.json')
        v2_ho = Path(f'holdout/{ptype}/{ctype}/v2/metrics.json')
        v0_vc = json.load(open(v0_ho))['aggregate']['mean']['VC@20'] if v0_ho.exists() else 0
        v2_vc = json.load(open(v2_ho))['aggregate']['mean']['VC@20'] if v2_ho.exists() else 0
        delta = (v2_vc / v0_vc - 1) * 100 if v0_vc > 0 else 0
        status = 'OK' if delta > 0 else 'FAIL'
        print(f'{ptype}/{ctype:<20} {v0_vc:>10.4f} {v2_vc:>10.4f} {delta:>+9.1f}% [{status}]')
"

# 3. Holdout months cover 2024-2025
python -c "
import json
from pathlib import Path
p = Path('holdout/f0/onpeak/v2/metrics.json')
if p.exists():
    data = json.load(open(p))
    months = sorted(data['per_month'].keys())
    print(f'Holdout months ({len(months)}): {months[0]} .. {months[-1]}')
    assert months[0] >= '2024-01', f'Holdout starts too early: {months[0]}'
    print('Holdout range OK')
"
```

**Acceptance criteria:**
- [ ] **BLOCK**: All 6 slices have holdout results in `holdout/{ptype}/{ctype}/v2/metrics.json`
- [ ] **BLOCK**: v2 holdout VC@20 > v0 holdout VC@20 for at least 5 of 6 slices
- [ ] Holdout months are from 2024-2025 (out-of-sample, not overlapping dev)
- [ ] Holdout VC@20 is within 0.5-1.2× of dev VC@20 (large divergence = overfitting concern)
- [ ] Summary table printed comparing all slices

---

## Chunk 3: Deployment Signal Writer

### Task 14: Create `v70/` deployment modules

Adapt from `research-miso-signal7/v70/` for PJM.

**Files:**
- Create: `v70/__init__.py`
- Create: `v70/cache.py`
- Create: `v70/inference.py`
- Create: `v70/signal_writer.py`
- Create: `scripts/generate_v70_signal.py`

- [ ] **Step 1: Create `v70/cache.py`**

Same as MISO `cache.py` but with PJM paths and 3 class types.

```python
# v70/cache.py
"""Realized DA cache management for PJM V7.0 signal generation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import collect_usable_months, delivery_month, has_period_type

REALIZED_DA_CACHE = os.environ.get(
    "PJM_REALIZED_DA_CACHE",
    str(Path(__file__).resolve().parent.parent / "data" / "realized_da"),
)

_MAX_BF_LOOKBACK = 15


def _cache_path(month: str, peak_type: str) -> Path:
    if peak_type == "onpeak":
        return Path(REALIZED_DA_CACHE) / f"{month}.parquet"
    return Path(REALIZED_DA_CACHE) / f"{month}_{peak_type}.parquet"


def _prev_month(m: str) -> str:
    import pandas as pd
    return (pd.Timestamp(m) - pd.DateOffset(months=1)).strftime("%Y-%m")


def _months_before(month: str, n: int) -> list[str]:
    import pandas as pd
    ts = pd.Timestamp(month)
    return [(ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, n + 1)]


def required_realized_da_months(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> set[tuple[str, str]]:
    needed: set[tuple[str, str]] = set()
    for ptype in ptypes:
        if not has_period_type(auction_month, ptype):
            continue
        train_months = collect_usable_months(auction_month, ptype, n_months=8)
        for tm in train_months:
            dm = delivery_month(tm, ptype)
            for ct in ctypes:
                needed.add((dm, ct))
            bf_cutoff = _prev_month(tm)
            for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
                for ct in ctypes:
                    needed.add((bf_month, ct))
        bf_cutoff = _prev_month(auction_month)
        for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
            for ct in ctypes:
                needed.add((bf_month, ct))
    return needed


def ensure_realized_da_cache(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> None:
    needed = required_realized_da_months(auction_month, ptypes, ctypes)
    missing = [(m, pt) for m, pt in sorted(needed) if not _cache_path(m, pt).exists()]

    if not missing:
        print(f"[cache] All {len(needed)} required DA files present")
        return

    print(f"[cache] {len(missing)}/{len(needed)} missing, fetching...")
    from ml.realized_da import fetch_and_cache_month

    fetched, skipped = 0, 0
    for month, peak_type in missing:
        try:
            out = fetch_and_cache_month(month, peak_type, cache_dir=REALIZED_DA_CACHE)
            fetched += 1 if out.exists() else 0
        except Exception as e:
            skipped += 1
            print(f"[cache]   SKIP {month}/{peak_type}: {e}")

    print(f"[cache] Fetched {fetched}, skipped {skipped}")
```

- [ ] **Step 2: Create `v70/signal_writer.py`**

```python
# v70/signal_writer.py
"""Rank/tier computation and signal assembly for PJM V7.0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PJM_AUCTION_SCHEDULE


def compute_rank_tier(
    scores: np.ndarray,
    v62b_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-percentile rank with V6.2B tie-breaking.

    1. ML score descending (higher = more binding = lower rank)
    2. V6.2B rank_ori ascending (lower = more binding, tie-break)
    3. Original index ascending (final tie-break)
    """
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([], dtype=int)
    order = np.lexsort((np.arange(n), v62b_rank, -scores))
    rank = np.empty(n, dtype=np.float64)
    rank[order] = (np.arange(n) + 1) / n
    tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    return rank, tier


def available_ptypes(auction_month: str) -> list[str]:
    import pandas as pd
    month_num = pd.Timestamp(auction_month).month
    return PJM_AUCTION_SCHEDULE.get(month_num, ["f0"])
```

- [ ] **Step 3: Create `v70/inference.py`**

Adapt from MISO's `v70/inference.py`. Key PJM changes:
- binding sets keyed on `branch_name`
- 3 class types
- PJM paths

- [ ] **Step 4: Create `scripts/generate_v70_signal.py`**

Adapt from MISO's `scripts/generate_v70_signal.py`. Key changes:
- `"pjm"` instead of `"miso"` in `ConstraintsSignal` calls
- Concrete PJM signal names: input=`TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1`, output=`TEST.TEST.Signal.PJM.SPICE_F0P_V7.0B.R1` (verify these with the downstream PJM consumer or `ConstraintsSignal` API before finalizing)
- SF signal: input/output from `/opt/data/xyz-dataset/signal_data/pjm/sf/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/` (passthrough, no modification)
- 3 class types: onpeak, dailyoffpeak, wkndonpeak
- No `SO_MW_Transfer` exception (MISO-specific)

- [ ] **Step 5: Commit**

```bash
git add v70/ scripts/generate_v70_signal.py
git commit -m "feat: PJM V7.0 signal deployment pipeline"
```

#### Review: Task 14

**Verify commands:**
```bash
# 1. All deployment modules exist
ls -1 v70/__init__.py v70/cache.py v70/inference.py v70/signal_writer.py scripts/generate_v70_signal.py

# 2. Import check
python -c "
from v70.cache import ensure_realized_da_cache, required_realized_da_months
from v70.signal_writer import compute_rank_tier, available_ptypes
print('v70 imports OK')
"

# 3. Verify compute_rank_tier produces valid tiers (0-4)
python -c "
import numpy as np
from v70.signal_writer import compute_rank_tier
scores = np.array([10.0, 5.0, 1.0, 8.0, 3.0, 7.0, 2.0, 6.0, 4.0, 9.0])
v62b = np.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.15])
rank, tier = compute_rank_tier(scores, v62b)
print(f'Ranks: {rank}')
print(f'Tiers: {tier}')
assert tier.min() >= 0 and tier.max() <= 4, f'Tier out of range: {tier.min()}-{tier.max()}'
assert len(set(tier)) > 1, 'All same tier — ranking broken'
# Highest score should get lowest rank (tier 0)
best_idx = np.argmax(scores)
print(f'Best score idx={best_idx}, tier={tier[best_idx]}')
assert tier[best_idx] == 0, f'Best score should be tier 0, got {tier[best_idx]}'
print('Rank/tier check PASSED')
"

# 4. PJM-specific: no SO_MW_Transfer exception
grep -rn "SO_MW\|so_mw" v70/ scripts/generate_v70_signal.py || echo "No SO_MW_Transfer (correct for PJM)"

# 5. Verify 3 class types in signal writer
grep -n "onpeak\|dailyoffpeak\|wkndonpeak\|class_type\|PJM_CLASS_TYPES" v70/ scripts/generate_v70_signal.py

# 6. Verify signal uses PJM naming (not MISO)
grep -rn "miso\|MISO" v70/ scripts/generate_v70_signal.py || echo "No MISO references (correct)"
grep -n "pjm\|PJM" scripts/generate_v70_signal.py
```

**Acceptance criteria:**
- [ ] All 4 v70 modules + generation script exist and import cleanly
- [ ] **BLOCK**: `compute_rank_tier` produces tiers 0-4 with highest ML score → tier 0
- [ ] **BLOCK**: Lexsort order: ML score descending, V6.2B rank ascending, index ascending
- [ ] **BLOCK**: No `SO_MW_Transfer` exception (MISO-specific, not applicable to PJM)
- [ ] Signal uses PJM naming, not MISO
- [ ] 3 class types handled (onpeak, dailyoffpeak, wkndonpeak)
- [ ] `available_ptypes()` uses `PJM_AUCTION_SCHEDULE`
- [ ] `ensure_realized_da_cache` covers BF lookback months (15 months before cutoff)
- [ ] f2-f11 passthrough: no ML scoring, V6.2B rank preserved as-is

**Output integrity checks** (Finding 8 from review):

The implementer MUST add a `--verify` flag to `generate_v70_signal.py` that runs this
comparison automatically after signal generation. The flag produces a PASS/FAIL report.

```bash
# 7. Run signal generation with --verify for a test month
python scripts/generate_v70_signal.py --auction-month 2025-01 --verify

# The --verify flag must perform these checks internally:
#   a. Row universe: v70 output has same constraint_ids and row count as V6.2B input per (ptype, ctype)
#   b. Immutable columns: all columns EXCEPT rank, rank_ori, tier are bit-identical to V6.2B
#   c. SF passthrough: shift factor output directory is byte-identical to V6.2B SF input
#   d. ML-only changes: for f0/f1 slices, rank/rank_ori/tier differ; for f2+ passthrough, all columns identical
#   e. Prints per-slice summary: "f0/onpeak: 1234 rows, 3 cols changed, 18 cols identical — PASS"

# If --verify is not yet implemented, run this standalone check:
python3 << 'PYEOF'
import polars as pl
from pathlib import Path

v62b_base = Path("/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1")
# V70 output path — set by generate_v70_signal.py's --output-dir flag
v70_base = Path("output/v70")  # adjust to actual output location

month = "2025-01"
changed_cols = {"rank", "rank_ori", "tier"}

for ptype in ["f0", "f1"]:
    for ctype in ["onpeak", "dailyoffpeak", "wkndonpeak"]:
        v62b_path = v62b_base / month / ptype / ctype
        v70_path = v70_base / month / ptype / ctype
        if not v70_path.exists():
            print(f"{ptype}/{ctype}: v70 output NOT FOUND — SKIP")
            continue

        old = pl.read_parquet(str(v62b_path)).sort("constraint_id")
        new = pl.read_parquet(str(v70_path)).sort("constraint_id")

        # a. Row universe
        assert len(old) == len(new), f"Row count mismatch: {len(old)} vs {len(new)}"
        assert old["constraint_id"].to_list() == new["constraint_id"].to_list(), "constraint_id mismatch"

        # b. Immutable columns
        immutable = [c for c in old.columns if c not in changed_cols and c in new.columns]
        for col in immutable:
            if old[col].to_list() != new[col].to_list():
                print(f"  FAIL: {ptype}/{ctype} column '{col}' changed but should be immutable")
                break
        else:
            print(f"{ptype}/{ctype}: {len(new)} rows, {len(immutable)} immutable cols verified — PASS")

# c. Check a passthrough period type
for ptype in ["f2", "f3"]:
    for ctype in ["onpeak"]:
        v62b_path = v62b_base / month / ptype / ctype
        v70_path = v70_base / month / ptype / ctype
        if not v70_path.exists():
            continue
        old = pl.read_parquet(str(v62b_path))
        new = pl.read_parquet(str(v70_path))
        if old.frame_equal(new):
            print(f"{ptype}/{ctype}: passthrough — bit-identical PASS")
        else:
            print(f"{ptype}/{ctype}: passthrough — FAIL (should be identical)")
PYEOF

# 8. Signal naming: must use concrete PJM signal names
grep -n "signal_name\|Signal.*PJM\|SPICE_F0P" scripts/generate_v70_signal.py
# Must see concrete PJM signal names (e.g., TEST.TEST.Signal.PJM.SPICE_F0P_V7.0B.R1)
# NOT generic "PJM signal names" placeholder
```

**Acceptance criteria (continued):**
- [ ] **BLOCK**: For ML slices (f0, f1), only `rank`, `rank_ori`, and `tier` columns change vs V6.2B; all other constraint columns are bit-identical
- [ ] **BLOCK**: Shift factors (SF) are completely unchanged (passthrough from V6.2B)
- [ ] **BLOCK**: Row universe (set of constraint_ids per month/ptype/ctype) is identical to V6.2B
- [ ] **HIGH**: Concrete PJM signal names specified in the generation script (not placeholder text)
- [ ] **HIGH**: Signal naming verified against downstream PJM consumer expectations (check with `ConstraintsSignal` API)

---

## Execution Order Summary

The tasks must be executed in this order:

1. **Task 1**: Copy unchanged modules (train, evaluate, v62b_formula, registry_paths, compare)
2. **Task 2**: Create `ml/config.py` — **verify PJM_AUCTION_SCHEDULE against real data**
3. **Task 3**: Create `ml/branch_mapping.py` — **most critical PJM adaptation**
4. **Task 4**: Create `ml/realized_da.py` — fetch + cache DA by branch_name
5. **Task 5**: Create `ml/spice6_loader.py` — PJM density features
6. **Task 6**: Create `ml/features.py` — feature preparation
7. **Task 7**: Create `ml/data_loader.py` — V6.2B + spice6 + realized DA
8. **Task 8**: Create `ml/pipeline.py` — single-month pipeline
9. **Task 9**: Create `scripts/cache_realized_da.py` + **run it** (requires Ray)
10. **Task 10**: Create `scripts/run_v0_formula_baseline.py` + **run it**
11. **Task 11**: Create `scripts/run_v2_ml.py` + **run it**
12. **Task 12**: Create `scripts/run_blend_search.py` + **run it**
13. **Task 13**: Create `scripts/run_holdout.py` + **run it**
14. **Task 14**: Create `v70/` deployment modules

Tasks 1-8 can be done without Ray. Task 9 requires Ray and must complete before Tasks 10-13. Task 14 can be done after all research is validated.

---

## Preflight Findings (Verified 2026-03-10)

Facts confirmed by probing actual data on disk:

1. **V6.2B has `branch_name` column** — confirmed in 2023-06/f0/onpeak. 21 columns total.
2. **PJM_AUCTION_SCHEDULE corrected**: Jan has f0-f4 (not f0-f2), Feb has f0-f3 (not f0-f1). Fixed in plan.
3. **constraint_info is period-type invariant**: f0 and f1 have identical rows (32297 each, 4835 branches) for every sampled month. Safe to use `period_type="f0"` default.
4. **PJM density uses new schema only**: always `score.parquet` with single `score` column. No legacy `score_df.parquet` exists. Legacy branch in spice6_loader is dead code.
5. **PjmApTools DA output**: index=`datetime_beginning_utc`, columns include `monitored_facility`, `shadow_price`, `day`, `contingency_facility`, `constraint_full`, `monitored_line`. Timezone is US/Eastern (not US/Central like MISO).
6. **V6.2B spans 105 months**: 2017-06 to 2026-03.
7. **Spice6 ml_pred starts 2018-06** (92 auction months). Pre-2018-06 features fill with 0.
8. **MISO modules to copy**: `train.py` imports `ml.config`, `compare.py` imports `ml.registry_paths`. Others (`evaluate`, `v62b_formula`, `registry_paths`) have no ml-internal imports.

## Critical Invariants to Verify

After each major phase, verify these:

1. **Branch mapping coverage**: `map_da_to_branches()` captures >90% of DA value for test months
2. **No temporal leakage**: BF features use months strictly before `prev_month(auction_month)`, training window uses `collect_usable_months()` which respects delivery month lag
3. **V0 baseline sanity**: VC@20 should be in 0.15-0.35 range (if 0.0 or >0.5, something is wrong)
4. **V2 vs V0 improvement**: ML should beat formula by >20% on VC@20 (if not, check target join)
5. **Registry schema**: every slice has `metrics.json`, `config.json`, `gates.json`, `champion.json`
6. **Memory**: never exceed 40 GiB for research scripts (`mem_mb()` at each stage)
7. **LightGBM threads**: always `num_threads=4` (already hardcoded in `train.py`)
8. **PJM timezone**: all DA fetch timestamps must use `tz="US/Eastern"` (not `US/Central`)

---

## Final End-to-End Review

Run this after ALL tasks are complete. This is the go/no-go gate for declaring the pipeline ready.

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
```

### 1. Codebase health
```bash
# No MISO references in PJM code (except comments explaining the adaptation)
grep -rn "from.*miso\|import.*miso" ml/ scripts/ v70/ --include="*.py" || echo "Clean (no MISO imports)"

# No pandas used for data loading (should be polars)
grep -rn "pd.read_parquet\|pd.read_csv" ml/ scripts/ --include="*.py" || echo "Clean (polars only)"

# All scripts have sys.path setup
for f in scripts/*.py; do grep -l "sys.path" "$f" > /dev/null || echo "MISSING sys.path: $f"; done

# No dangling imports
python -c "
import ml.config, ml.branch_mapping, ml.realized_da, ml.spice6_loader
import ml.features, ml.data_loader, ml.pipeline
import ml.train, ml.evaluate, ml.v62b_formula, ml.registry_paths, ml.compare
print('All ml modules import OK')
"
```

### 2. Data correctness (THE most important check)
```bash
python -c "
import json
from pathlib import Path

print('=== FULL RESULTS SUMMARY ===')
print()
print(f'{\"Slice\":<25} {\"v0 Dev\":>8} {\"v2 Dev\":>8} {\"v0 HO\":>8} {\"v2 HO\":>8} {\"Dev Δ\":>8} {\"HO Δ\":>8}')
print('-' * 75)

all_ok = True
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        def get_vc20(path):
            if not Path(path).exists(): return None
            return json.load(open(path))['aggregate']['mean']['VC@20']

        v0d = get_vc20(f'registry/{ptype}/{ctype}/v0/metrics.json')
        v2d = get_vc20(f'registry/{ptype}/{ctype}/v2/metrics.json')
        v0h = get_vc20(f'holdout/{ptype}/{ctype}/v0/metrics.json')
        v2h = get_vc20(f'holdout/{ptype}/{ctype}/v2/metrics.json')

        dev_delta = f'{(v2d/v0d-1)*100:+.0f}%' if v0d and v2d else 'N/A'
        ho_delta = f'{(v2h/v0h-1)*100:+.0f}%' if v0h and v2h else 'N/A'

        line = f'{ptype}/{ctype:<20} {v0d or 0:>8.4f} {v2d or 0:>8.4f} {v0h or 0:>8.4f} {v2h or 0:>8.4f} {dev_delta:>8} {ho_delta:>8}'
        print(line)

        # Checks
        if v2d and v0d and v2d <= v0d:
            print(f'  ⚠ WARN: v2 dev <= v0 dev for {ptype}/{ctype}')
        if v2h and v0h and v2h <= v0h:
            print(f'  ⚠ WARN: v2 holdout <= v0 holdout for {ptype}/{ctype}')
            all_ok = False

print()
print('VERDICT:', 'PASS — ML beats formula on holdout' if all_ok else 'NEEDS INVESTIGATION')
"
```

### 3. Leakage audit
```bash
python -c "
# Check every module for potential temporal leakage patterns
import ast, sys
from pathlib import Path

suspect_patterns = [
    'actual_shadow_price', 'actual_binding', 'abs_error', 'error',
    'shadow_sign', 'shadow_price',  # as feature (not in fetch context)
]

for py in sorted(Path('ml').glob('*.py')):
    if py.name == '__init__.py': continue
    code = py.read_text()
    for pat in suspect_patterns:
        # Skip if it's in a comment or in fetch/cache context
        for i, line in enumerate(code.split('\n'), 1):
            stripped = line.strip()
            if stripped.startswith('#'): continue
            if pat in stripped and 'feature' in stripped.lower():
                print(f'SUSPECT {py}:{i}: {stripped}')

print('Leakage audit complete')
"
```

### 4. Registry completeness
```bash
python -c "
from pathlib import Path
import json

required_files = ['metrics.json', 'config.json']
slice_files = ['gates.json', 'champion.json']

missing = []
for ptype in ['f0', 'f1']:
    for ctype in ['onpeak', 'dailyoffpeak', 'wkndonpeak']:
        base = Path(f'registry/{ptype}/{ctype}')
        for sf in slice_files:
            if not (base / sf).exists():
                missing.append(f'{base / sf}')
        for v in ['v0', 'v2']:
            for rf in required_files:
                if not (base / v / rf).exists():
                    missing.append(f'{base / v / rf}')

if missing:
    print(f'MISSING {len(missing)} files:')
    for m in missing:
        print(f'  {m}')
else:
    print('Registry complete: all 6 slices × 2 versions × required files')
"
```

### Final acceptance criteria
- [ ] **BLOCK**: ML (v2) beats formula (v0) on holdout VC@20 for at least 5/6 slices
- [ ] **BLOCK**: No temporal leakage suspects found in audit
- [ ] **BLOCK**: Registry complete — all files present for all 6 slices
- [ ] **BLOCK**: All modules import cleanly
- [ ] All holdout VC@20 values in range 0.10-0.50
- [ ] Dev-to-holdout ratio between 0.5-1.2 (no severe overfitting)
- [ ] No MISO imports in production code
- [ ] Deployment modules (v70/) produce valid tiers 0-4
