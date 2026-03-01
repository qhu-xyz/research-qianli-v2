# Real-Data Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-month synthetic evaluation with rolling-window multi-month real-data evaluation using cascade stages, three-layer gates, and Ray parallelism.

**Architecture:** Reuse source repo's `MisoDataLoader` for data loading. New `ml/benchmark.py` orchestrates Ray-parallel evaluation across 12 months. `ml/compare.py` gains three-layer gate checks (mean + tail safety + tail non-regression). `gates.json` v2 schema adds cascade stages and tail floors.

**Tech Stack:** Python (polars, XGBoost, scikit-learn, Ray), source repo's `MisoDataLoader` + `PredictionConfig`

**Design doc:** `docs/plans/2026-03-01-real-data-evaluation-redesign.md`

**Source repo:** `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/`

---

## Phase A: Gate System Upgrade (Tasks 1-3)

### Task 1: Update `gates.json` to v2 schema

**Files:**
- Modify: `registry/gates.json`
- Modify: `ml/config.py` (GateConfig class)
- Modify: `ml/tests/test_config.py`

**Step 1: Write failing test for new GateConfig fields**

Add to `ml/tests/test_config.py`:
```python
def test_gate_config_v2_fields(tmp_path):
    """GateConfig v2 must expose cascade_stages, tail fields, and eval_months."""
    import json
    gates_data = {
        "version": 2,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": ["2020-09"], "stress": ["2021-02"]},
        "cascade_stages": [
            {"stage": 1, "ptype": "f0", "blocking": True}
        ],
        "gates": {
            "S1-AUC": {
                "floor": 0.65, "tail_floor": 0.55,
                "direction": "higher", "group": "A",
                "pending_v0": False
            }
        }
    }
    p = tmp_path / "gates.json"
    p.write_text(json.dumps(gates_data))
    from ml.config import GateConfig
    gc = GateConfig(str(p))
    assert gc.data["version"] == 2
    assert gc.tail_max_failures == 1
    assert gc.eval_months["primary"] == ["2020-09"]
    assert gc.cascade_stages[0]["ptype"] == "f0"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate && cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow && SMOKE_TEST=true python -m pytest ml/tests/test_config.py::test_gate_config_v2_fields -v`
Expected: FAIL (AttributeError on `tail_max_failures`, `eval_months`, `cascade_stages`)

**Step 3: Update GateConfig in `ml/config.py`**

Add these properties to the `GateConfig` class (after existing properties):
```python
@property
def tail_max_failures(self) -> int:
    """Max months allowed below tail_floor per gate."""
    return self.data.get("tail_max_failures", 1)

@property
def eval_months(self) -> dict:
    """Primary and stress evaluation months."""
    return self.data.get("eval_months", {"primary": [], "stress": []})

@property
def cascade_stages(self) -> list[dict]:
    """Cascade stage definitions."""
    return self.data.get("cascade_stages", [])
```

**Step 4: Run test to verify it passes**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_config.py -v`
Expected: all PASS

**Step 5: Write new `registry/gates.json`**

Replace contents with the v2 schema from design doc §3 (all floors null, pending_v0=true, 4 hard gates, 6 monitor gates, cascade stages, eval months). Keep `S1-REC` floor at 0.10 with `tail_floor: 0.0` (not pending since it's a fixed policy floor).

**Step 6: Commit**

```bash
git add registry/gates.json ml/config.py ml/tests/test_config.py
git commit -m "gates: upgrade to v2 schema with cascade stages, tail floors, eval months"
```

---

### Task 2: Three-layer gate check in `ml/compare.py`

**Files:**
- Modify: `ml/compare.py`
- Modify: `ml/tests/test_compare.py`

**Step 1: Write failing tests for three-layer gate checking**

Add to `ml/tests/test_compare.py`:
```python
def test_check_gates_multi_month_mean():
    """Layer 1: mean(metric) >= floor."""
    from ml.compare import check_gates_multi_month
    gates = {
        "S1-AUC": {"floor": 0.65, "tail_floor": 0.55, "direction": "higher", "group": "A"},
    }
    # 3 months: mean = (0.70 + 0.68 + 0.72) / 3 = 0.70 >= 0.65
    per_month = {"2020-09": {"S1-AUC": 0.70}, "2020-11": {"S1-AUC": 0.68}, "2021-01": {"S1-AUC": 0.72}}
    results = check_gates_multi_month(per_month, gates, tail_max_failures=1)
    assert results["S1-AUC"]["mean_passed"] is True


def test_check_gates_multi_month_tail_safety():
    """Layer 2: count(metric < tail_floor) <= tail_max_failures."""
    from ml.compare import check_gates_multi_month
    gates = {
        "S1-AUC": {"floor": 0.65, "tail_floor": 0.55, "direction": "higher", "group": "A"},
    }
    # 1 month below tail_floor (0.50 < 0.55), tail_max_failures=1 => pass
    per_month = {"m1": {"S1-AUC": 0.70}, "m2": {"S1-AUC": 0.50}, "m3": {"S1-AUC": 0.72}}
    results = check_gates_multi_month(per_month, gates, tail_max_failures=1)
    assert results["S1-AUC"]["tail_passed"] is True

    # 2 months below => fail
    per_month2 = {"m1": {"S1-AUC": 0.70}, "m2": {"S1-AUC": 0.50}, "m3": {"S1-AUC": 0.40}}
    results2 = check_gates_multi_month(per_month2, gates, tail_max_failures=1)
    assert results2["S1-AUC"]["tail_passed"] is False


def test_check_gates_multi_month_tail_regression():
    """Layer 3: mean_bottom_2(new) >= mean_bottom_2(champion) - noise_tol."""
    from ml.compare import check_gates_multi_month
    gates = {
        "S1-AUC": {"floor": 0.65, "tail_floor": 0.55, "direction": "higher", "group": "A"},
    }
    per_month = {"m1": {"S1-AUC": 0.70}, "m2": {"S1-AUC": 0.60}, "m3": {"S1-AUC": 0.72}}
    champ_per_month = {"m1": {"S1-AUC": 0.72}, "m2": {"S1-AUC": 0.65}, "m3": {"S1-AUC": 0.74}}
    # new bottom_2 mean = (0.60 + 0.70) / 2 = 0.65
    # champ bottom_2 mean = (0.65 + 0.72) / 2 = 0.685
    # 0.65 < 0.685 - 0.02 = 0.665 => FAIL (tail regressed)
    results = check_gates_multi_month(per_month, gates, tail_max_failures=1,
                                       champion_per_month=champ_per_month, noise_tolerance=0.02)
    assert results["S1-AUC"]["tail_regression_passed"] is False


def test_check_gates_multi_month_lower_direction():
    """Lower-is-better (BRIER): directions inverted for all 3 layers."""
    from ml.compare import check_gates_multi_month
    gates = {
        "S1-BRIER": {"floor": 0.12, "tail_floor": 0.18, "direction": "lower", "group": "B"},
    }
    # mean = 0.10 <= 0.12 => pass; no month > 0.18 => tail pass
    per_month = {"m1": {"S1-BRIER": 0.09}, "m2": {"S1-BRIER": 0.11}, "m3": {"S1-BRIER": 0.10}}
    results = check_gates_multi_month(per_month, gates, tail_max_failures=1)
    assert results["S1-BRIER"]["mean_passed"] is True
    assert results["S1-BRIER"]["tail_passed"] is True


def test_evaluate_overall_pass_multi_month():
    """Overall pass requires all Group A gates to pass all 3 layers."""
    from ml.compare import evaluate_overall_pass_multi_month
    results = {
        "S1-AUC": {"group": "A", "mean_passed": True, "tail_passed": True, "tail_regression_passed": True},
        "S1-BRIER": {"group": "B", "mean_passed": False, "tail_passed": True, "tail_regression_passed": True},
    }
    ga, gb = evaluate_overall_pass_multi_month(results)
    assert ga is True   # Group A all pass
    assert gb is False  # Group B mean failed
```

**Step 2: Run tests to verify they fail**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_compare.py::test_check_gates_multi_month_mean -v`
Expected: FAIL (ImportError: `check_gates_multi_month` does not exist)

**Step 3: Implement `check_gates_multi_month` and `evaluate_overall_pass_multi_month` in `ml/compare.py`**

Add after `evaluate_overall_pass()`:

```python
def check_gates_multi_month(
    per_month: dict[str, dict],
    gates: dict,
    tail_max_failures: int = 1,
    champion_per_month: dict[str, dict] | None = None,
    noise_tolerance: float = 0.02,
) -> dict[str, dict]:
    """Three-layer gate check across multiple evaluation months.

    Layer 1: mean(metric) >= floor (or <= for lower-is-better)
    Layer 2: count(metric violating tail_floor) <= tail_max_failures
    Layer 3: mean_bottom_2(metric) >= mean_bottom_2(champion) - noise_tolerance

    Parameters
    ----------
    per_month : dict
        {month_id: {gate_name: value, ...}, ...}
    gates : dict
        Gate definitions with floor, tail_floor, direction, group.
    tail_max_failures : int
        Max months allowed below tail_floor per gate.
    champion_per_month : dict or None
        Champion's per-month metrics for tail regression check.
    noise_tolerance : float
        Tolerance for tail regression check.

    Returns
    -------
    results : dict
        {gate_name: {mean_value, mean_passed, tail_failures, tail_passed,
                     bottom_2_mean, tail_regression_passed, group, overall_passed}}
    """
    months = sorted(per_month.keys())
    results = {}

    for gate_name, gate_def in gates.items():
        floor = gate_def.get("floor")
        tail_floor = gate_def.get("tail_floor")
        direction = gate_def["direction"]
        group = gate_def.get("group", "A")

        values = [per_month[m].get(gate_name) for m in months]
        values = [v for v in values if v is not None and (not isinstance(v, float) or v == v)]

        if not values:
            results[gate_name] = {
                "mean_value": None, "mean_passed": None,
                "tail_failures": None, "tail_passed": None,
                "bottom_2_mean": None, "tail_regression_passed": None,
                "group": group, "overall_passed": None,
            }
            continue

        mean_val = sum(values) / len(values)

        # Layer 1: mean check
        if floor is not None:
            if direction == "higher":
                mean_passed = mean_val >= floor
            else:
                mean_passed = mean_val <= floor
        else:
            mean_passed = None

        # Layer 2: tail safety
        tail_failures = 0
        if tail_floor is not None:
            for v in values:
                if direction == "higher" and v < tail_floor:
                    tail_failures += 1
                elif direction == "lower" and v > tail_floor:
                    tail_failures += 1
            tail_passed = tail_failures <= tail_max_failures
        else:
            tail_passed = None

        # Layer 3: tail non-regression (mean of bottom 2)
        sorted_vals = sorted(values) if direction == "higher" else sorted(values, reverse=True)
        n_bottom = min(2, len(sorted_vals))
        bottom_2_mean = sum(sorted_vals[:n_bottom]) / n_bottom

        tail_regression_passed = True  # default if no champion
        if champion_per_month is not None:
            champ_values = [champion_per_month[m].get(gate_name)
                            for m in months if m in champion_per_month]
            champ_values = [v for v in champ_values if v is not None and (not isinstance(v, float) or v == v)]
            if champ_values:
                champ_sorted = sorted(champ_values) if direction == "higher" else sorted(champ_values, reverse=True)
                cn = min(2, len(champ_sorted))
                champ_bottom_2 = sum(champ_sorted[:cn]) / cn
                if direction == "higher":
                    tail_regression_passed = bottom_2_mean >= champ_bottom_2 - noise_tolerance
                else:
                    tail_regression_passed = bottom_2_mean <= champ_bottom_2 + noise_tolerance

        overall = all(x is not False for x in [mean_passed, tail_passed, tail_regression_passed])

        results[gate_name] = {
            "mean_value": round(mean_val, 4),
            "mean_passed": mean_passed,
            "tail_failures": tail_failures,
            "tail_passed": tail_passed,
            "bottom_2_mean": round(bottom_2_mean, 4),
            "tail_regression_passed": tail_regression_passed,
            "group": group,
            "overall_passed": overall if mean_passed is not None else None,
        }

    return results


def evaluate_overall_pass_multi_month(gate_results: dict[str, dict]) -> tuple[bool, bool]:
    """Evaluate overall pass from multi-month gate results.

    Returns (group_a_passed, group_b_passed).
    """
    group_a_passed = True
    group_b_passed = True
    for result in gate_results.values():
        passed = result.get("overall_passed")
        group = result.get("group", "A")
        if passed is not True:
            if group == "A":
                group_a_passed = False
            else:
                group_b_passed = False
    return group_a_passed, group_b_passed
```

**Step 4: Run all compare tests**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_compare.py -v`
Expected: all PASS (new + existing)

**Step 5: Commit**

```bash
git add ml/compare.py ml/tests/test_compare.py
git commit -m "compare: add three-layer multi-month gate check (mean + tail safety + tail regression)"
```

---

### Task 3: Update `ml/populate_v0_gates.py` for v2 schema

**Files:**
- Modify: `ml/populate_v0_gates.py`
- Modify: `ml/tests/test_registry.py`

**Step 1: Write failing test**

Add to `ml/tests/test_registry.py`:
```python
def test_populate_v0_gates_v2_tail_floor(tmp_path):
    """v2 schema: populate both floor and tail_floor."""
    import json
    from ml.populate_v0_gates import populate_v0_gates

    v0 = tmp_path / "registry" / "v0"
    v0.mkdir(parents=True)
    # Per-month metrics for v0 (new schema)
    (v0 / "metrics.json").write_text(json.dumps({
        "aggregate": {
            "mean": {"S1-AUC": 0.72},
            "min": {"S1-AUC": 0.65},
            "max": {"S1-BRIER": 0.12},
            "bottom_2_mean": {"S1-AUC": 0.66},
        },
        "per_month": {
            "2020-09": {"S1-AUC": 0.72, "S1-BRIER": 0.09},
            "2020-11": {"S1-AUC": 0.65, "S1-BRIER": 0.12},
            "2021-01": {"S1-AUC": 0.71, "S1-BRIER": 0.10},
        }
    }))

    gates_path = tmp_path / "registry" / "gates.json"
    gates_path.write_text(json.dumps({
        "version": 2,
        "noise_tolerance": 0.02,
        "tail_max_failures": 1,
        "eval_months": {"primary": ["2020-09", "2020-11", "2021-01"], "stress": []},
        "cascade_stages": [{"stage": 1, "ptype": "f0", "blocking": True}],
        "gates": {
            "S1-AUC": {
                "floor": None, "tail_floor": None,
                "direction": "higher", "group": "A",
                "pending_v0": True, "v0_offset": 0.05, "v0_tail_offset": 0.10
            }
        }
    }))

    result = populate_v0_gates(
        registry_dir=str(tmp_path / "registry"),
        gates_path=str(gates_path),
    )
    gate = result["gates"]["S1-AUC"]
    assert gate["pending_v0"] is False
    assert gate["floor"] == round(0.72 - 0.05, 6)       # mean - offset
    assert gate["tail_floor"] == round(0.65 - 0.10, 6)   # min - tail_offset
```

**Step 2: Run test to verify it fails**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_registry.py::test_populate_v0_gates_v2_tail_floor -v`
Expected: FAIL (populate reads flat metrics, not aggregate)

**Step 3: Update `ml/populate_v0_gates.py`**

Replace the main loop logic to handle both v1 (flat metrics) and v2 (aggregate metrics) schemas:

```python
def populate_v0_gates(
    registry_dir: str = "registry",
    gates_path: str = "registry/gates.json",
) -> dict:
    registry_dir = Path(registry_dir)
    gates_path = Path(gates_path)

    v0_metrics_path = registry_dir / "v0" / "metrics.json"
    if not v0_metrics_path.exists():
        raise FileNotFoundError(f"v0 metrics not found at {v0_metrics_path}")
    with open(v0_metrics_path) as f:
        v0_data = json.load(f)

    with open(gates_path) as f:
        gates_data = json.load(f)

    # Detect schema: v2 has "aggregate", v1 has flat metrics
    is_v2 = "aggregate" in v0_data
    if is_v2:
        v0_mean = v0_data["aggregate"]["mean"]
        v0_min = v0_data["aggregate"].get("min", {})
        v0_max = v0_data["aggregate"].get("max", {})
    else:
        v0_mean = v0_data  # flat metrics ARE the mean (single month)
        v0_min = v0_data
        v0_max = v0_data

    modified = False
    for gate_name, gate_def in gates_data["gates"].items():
        if not gate_def.get("pending_v0", False):
            continue

        mean_val = v0_mean.get(gate_name)
        if mean_val is None:
            print(f"[populate_v0] WARNING: {gate_name} not in v0 metrics, skipping")
            continue

        v0_offset = gate_def.get("v0_offset", 0.0)
        v0_tail_offset = gate_def.get("v0_tail_offset", v0_offset)
        direction = gate_def["direction"]

        if direction == "higher":
            gate_def["floor"] = round(mean_val - v0_offset, 6)
            extreme = v0_min.get(gate_name, mean_val)
            gate_def["tail_floor"] = round(extreme - v0_tail_offset, 6)
        else:
            gate_def["floor"] = round(mean_val + v0_offset, 6)
            extreme = v0_max.get(gate_name, mean_val)
            gate_def["tail_floor"] = round(extreme + v0_tail_offset, 6)

        gate_def["pending_v0"] = False
        modified = True
        print(f"[populate_v0] {gate_name}: floor={gate_def['floor']}, "
              f"tail_floor={gate_def['tail_floor']} "
              f"(mean={mean_val}, extreme={extreme}, direction={direction})")

    if modified:
        with open(gates_path, "w") as f:
            json.dump(gates_data, f, indent=2)
        print(f"[populate_v0] Updated {gates_path}")
    else:
        print("[populate_v0] No pending gates to populate")

    return gates_data
```

**Step 4: Run all registry tests**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_registry.py -v`
Expected: all PASS (new + existing; existing v1 test still passes via flat-metrics fallback)

**Step 5: Commit**

```bash
git add ml/populate_v0_gates.py ml/tests/test_registry.py
git commit -m "populate_v0: support v2 schema (mean floor + tail_floor from per-month extremes)"
```

---

## Phase B: Real Data Loading (Tasks 4-5)

### Task 4: Wire up `MisoDataLoader` in `ml/data_loader.py`

**Files:**
- Modify: `ml/data_loader.py`

**Step 1: Implement `_load_real()` using source repo's `MisoDataLoader`**

Replace the `NotImplementedError` in `_load_real()`:

```python
def _load_real(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load real data via source repo's MisoDataLoader + Ray."""
    import sys
    import gc
    import pandas as pd
    print(f"[data_loader] mem before imports: {mem_mb():.0f} MB")

    # Import source repo's loader
    src_path = "/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from shadow_price_prediction.data_loader import MisoDataLoader
    from shadow_price_prediction.config import PredictionConfig

    # Init Ray
    from pbase.config.ray import init_ray
    import pmodel
    import ml as shadow_ml
    init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel, shadow_ml])
    print(f"[data_loader] mem after Ray init: {mem_mb():.0f} MB")

    # Create source config
    pred_config = PredictionConfig()
    pred_config.class_type = config.class_type
    loader = MisoDataLoader(pred_config)

    # Compute training window
    auction_month = pd.Timestamp(config.auction_month)
    lookback = config.train_months + config.val_months
    train_start = auction_month - pd.DateOffset(months=lookback)
    train_end = auction_month - pd.DateOffset(months=0)  # exclusive

    # Determine required period types
    required_ptypes = {config.period_type}

    print(f"[data_loader] loading {train_start} to {train_end}, ptypes={required_ptypes}")
    train_data_pd = loader.load_training_data(
        train_start=train_start,
        train_end=train_end,
        required_period_types=required_ptypes,
    )
    print(f"[data_loader] loaded {len(train_data_pd)} rows, mem: {mem_mb():.0f} MB")

    # Convert to polars
    train_data = pl.from_pandas(train_data_pd)
    del train_data_pd; gc.collect()

    # Split: first train_months for fit, last val_months for val
    val_boundary = train_start + pd.DateOffset(months=config.train_months)
    val_boundary_str = val_boundary.strftime("%Y-%m")

    # auction_month column may be Timestamp or string; normalize
    if train_data["auction_month"].dtype == pl.Utf8:
        fit_df = train_data.filter(pl.col("auction_month") < val_boundary_str)
        val_df = train_data.filter(pl.col("auction_month") >= val_boundary_str)
    else:
        fit_df = train_data.filter(pl.col("auction_month") < val_boundary)
        val_df = train_data.filter(pl.col("auction_month") >= val_boundary)

    del train_data; gc.collect()
    print(f"[data_loader] fit: {fit_df.shape}, val: {val_df.shape}, mem: {mem_mb():.0f} MB")

    import ray
    ray.shutdown()
    print(f"[data_loader] Ray shutdown, mem: {mem_mb():.0f} MB")

    return fit_df, val_df
```

**Step 2: Verify smoke mode still works**

Run: `SMOKE_TEST=true python -c "from ml.data_loader import load_data; from ml.config import PipelineConfig; t, v = load_data(PipelineConfig()); print(t.shape, v.shape)"`
Expected: shapes printed

**Step 3: Commit**

```bash
git add ml/data_loader.py
git commit -m "data_loader: implement _load_real() using source repo MisoDataLoader + Ray"
```

---

### Task 5: Add multi-month aggregation to `ml/evaluate.py`

**Files:**
- Modify: `ml/evaluate.py`
- Modify: `ml/tests/test_evaluate.py`

Note: `evaluate.py` is HUMAN-WRITE-ONLY during pipeline runtime. This is a bootstrap modification (same as initial creation in Task 10 of the original plan).

**Step 1: Write failing test**

Add to `ml/tests/test_evaluate.py`:
```python
def test_aggregate_months():
    """aggregate_months computes mean, std, min, max, bottom_2_mean."""
    from ml.evaluate import aggregate_months
    per_month = {
        "2020-09": {"S1-AUC": 0.72, "S1-BRIER": 0.09},
        "2020-11": {"S1-AUC": 0.65, "S1-BRIER": 0.12},
        "2021-01": {"S1-AUC": 0.71, "S1-BRIER": 0.10},
    }
    agg = aggregate_months(per_month)
    assert abs(agg["mean"]["S1-AUC"] - 0.6933) < 0.001
    assert agg["min"]["S1-AUC"] == 0.65
    assert agg["max"]["S1-AUC"] == 0.72
    # bottom_2_mean = (0.65 + 0.71) / 2 = 0.68
    assert abs(agg["bottom_2_mean"]["S1-AUC"] - 0.68) < 0.001
    # For BRIER (lower is better), bottom_2 should be worst 2 (highest values)
    # bottom_2_mean = (0.12 + 0.10) / 2 = 0.11
    assert abs(agg["bottom_2_mean"]["S1-BRIER"] - 0.11) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_evaluate.py::test_aggregate_months -v`
Expected: FAIL (ImportError)

**Step 3: Add `aggregate_months()` to `ml/evaluate.py`**

Append after `evaluate_classifier()`:

```python
# Gate metrics that are "lower is better" (worst = highest values)
_LOWER_IS_BETTER = {"S1-BRIER"}


def aggregate_months(per_month: dict[str, dict]) -> dict:
    """Aggregate per-month metrics into mean, std, min, max, bottom_2_mean.

    Parameters
    ----------
    per_month : dict
        {month_id: {metric_name: value, ...}, ...}

    Returns
    -------
    aggregate : dict
        {"mean": {...}, "std": {...}, "min": {...}, "max": {...}, "bottom_2_mean": {...}}
    """
    if not per_month:
        return {"mean": {}, "std": {}, "min": {}, "max": {}, "bottom_2_mean": {}}

    months = sorted(per_month.keys())
    all_keys = set()
    for m in months:
        all_keys.update(per_month[m].keys())

    mean_d, std_d, min_d, max_d, b2_d = {}, {}, {}, {}, {}

    for key in sorted(all_keys):
        vals = []
        for m in months:
            v = per_month[m].get(key)
            if v is not None and isinstance(v, (int, float)) and (not isinstance(v, float) or v == v):
                vals.append(v)
        if not vals:
            continue

        mean_d[key] = round(sum(vals) / len(vals), 4)
        min_d[key] = round(min(vals), 4)
        max_d[key] = round(max(vals), 4)

        if len(vals) > 1:
            m = sum(vals) / len(vals)
            std_d[key] = round((sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5, 4)
        else:
            std_d[key] = 0.0

        # bottom_2_mean: worst 2 values
        # For "lower is better" metrics, worst = highest values
        if key in _LOWER_IS_BETTER:
            sorted_vals = sorted(vals, reverse=True)  # worst first (highest)
        else:
            sorted_vals = sorted(vals)  # worst first (lowest)
        n_bottom = min(2, len(sorted_vals))
        b2_d[key] = round(sum(sorted_vals[:n_bottom]) / n_bottom, 4)

    return {"mean": mean_d, "std": std_d, "min": min_d, "max": max_d, "bottom_2_mean": b2_d}
```

**Step 4: Run all evaluate tests**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_evaluate.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add ml/evaluate.py ml/tests/test_evaluate.py
git commit -m "evaluate: add aggregate_months() for multi-month metric aggregation"
```

---

## Phase C: Benchmark Runner + Pipeline Update (Tasks 6-7)

### Task 6: Create `ml/benchmark.py` (multi-month Ray-parallel eval)

**Files:**
- Create: `ml/benchmark.py`
- Create: `ml/tests/test_benchmark.py`

**Step 1: Write failing test**

Create `ml/tests/test_benchmark.py`:
```python
import json
import pytest


@pytest.fixture(autouse=True)
def smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "true")


def test_benchmark_smoke_single_month(tmp_path):
    """Benchmark in smoke mode evaluates a single month and writes metrics."""
    from ml.benchmark import run_benchmark

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    result = run_benchmark(
        version_id="v0",
        eval_months=["2021-07"],
        class_type="onpeak",
        ptype="f0",
        registry_dir=str(reg),
    )

    assert "per_month" in result
    assert "2021-07" in result["per_month"]
    assert "aggregate" in result
    assert "S1-AUC" in result["aggregate"]["mean"]
    assert (reg / "v0" / "metrics.json").exists()


def test_benchmark_smoke_multi_month(tmp_path):
    """Benchmark with multiple months aggregates correctly."""
    from ml.benchmark import run_benchmark

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "version_counter.json").write_text('{"next_id": 1}')

    result = run_benchmark(
        version_id="v0",
        eval_months=["2021-07", "2021-08"],
        class_type="onpeak",
        ptype="f0",
        registry_dir=str(reg),
    )

    assert len(result["per_month"]) == 2
    assert result["n_months"] == 2
    assert "bottom_2_mean" in result["aggregate"]
```

**Step 2: Run test to verify it fails**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_benchmark.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement `ml/benchmark.py`**

```python
"""Multi-month benchmark evaluation for shadow price classification.

Runs the full pipeline (load → train → threshold → evaluate) for each
evaluation month independently. In real mode, uses Ray to parallelize
across months.

CLI: python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak
"""

import argparse
import gc
import json
import os
import resource
from pathlib import Path

from ml.config import FeatureConfig, HyperparamConfig, PipelineConfig
from ml.data_loader import load_data
from ml.evaluate import aggregate_months, evaluate_classifier
from ml.features import compute_binary_labels, prepare_features
from ml.registry import register_version
from ml.threshold import apply_threshold, find_optimal_threshold
from ml.train import predict_proba, train_classifier


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _eval_single_month(
    auction_month: str,
    class_type: str,
    ptype: str,
    hyperparam_config: HyperparamConfig,
    feature_config: FeatureConfig,
    threshold_beta: float = 0.7,
) -> dict:
    """Train + evaluate on a single auction month. Returns metrics dict."""
    print(f"[benchmark] Evaluating {auction_month} (ptype={ptype}), mem: {mem_mb():.0f} MB")

    config = PipelineConfig(
        auction_month=auction_month,
        class_type=class_type,
        period_type=ptype,
        threshold_beta=threshold_beta,
    )

    # Load data (train + val)
    train_df, val_df = load_data(config)
    print(f"[benchmark]   train={train_df.shape}, val={val_df.shape}")

    # Prepare features
    X_train, _ = prepare_features(train_df, feature_config)
    y_train = compute_binary_labels(train_df)
    X_val, _ = prepare_features(val_df, feature_config)
    y_val = compute_binary_labels(val_df)

    # Train
    model = train_classifier(X_train, y_train, hyperparam_config, feature_config)

    # Threshold
    val_proba = predict_proba(model, X_val)
    threshold, max_fbeta = find_optimal_threshold(y_val, val_proba, beta=threshold_beta)

    # Evaluate on val set
    val_pred = apply_threshold(val_proba, threshold)
    val_sp = val_df["actual_shadow_price"].to_numpy()
    metrics = evaluate_classifier(y_val, val_proba, val_pred, val_sp, threshold)

    # Cleanup
    del train_df, val_df, X_train, X_val, y_train, y_val, model, val_proba, val_pred, val_sp
    gc.collect()

    print(f"[benchmark]   AUC={metrics['S1-AUC']}, AP={metrics['S1-AP']}, "
          f"BRIER={metrics['S1-BRIER']}, mem: {mem_mb():.0f} MB")
    return metrics


def run_benchmark(
    version_id: str,
    eval_months: list[str],
    class_type: str = "onpeak",
    ptype: str = "f0",
    registry_dir: str = "registry",
    hyperparam_config: HyperparamConfig | None = None,
    feature_config: FeatureConfig | None = None,
    threshold_beta: float = 0.7,
    overrides: dict | None = None,
) -> dict:
    """Run benchmark across multiple evaluation months.

    Parameters
    ----------
    version_id : str
        Version ID to register results under (e.g. "v0").
    eval_months : list[str]
        List of auction months to evaluate (e.g. ["2020-09", "2020-11"]).
    class_type : str
        "onpeak" or "offpeak".
    ptype : str
        Period type ("f0", "f1", etc.).
    registry_dir : str
        Path to registry directory.
    hyperparam_config : HyperparamConfig or None
        Override hyperparameters.
    feature_config : FeatureConfig or None
        Override feature config.
    threshold_beta : float
        F-beta parameter for threshold optimization.
    overrides : dict or None
        Config overrides (applied to hyperparam + pipeline configs).

    Returns
    -------
    result : dict
        {"per_month": {...}, "aggregate": {...}, "eval_config": {...}, ...}
    """
    if hyperparam_config is None:
        hyperparam_config = HyperparamConfig()
    if feature_config is None:
        feature_config = FeatureConfig()

    if overrides:
        from ml.pipeline import _apply_overrides
        pc_dummy = PipelineConfig(threshold_beta=threshold_beta)
        hyperparam_config, pc_dummy = _apply_overrides(hyperparam_config, pc_dummy, overrides)
        threshold_beta = pc_dummy.threshold_beta

    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    per_month = {}

    if smoke or len(eval_months) == 1:
        # Sequential (smoke test or single month)
        for month in eval_months:
            metrics = _eval_single_month(
                month, class_type, ptype, hyperparam_config, feature_config, threshold_beta
            )
            per_month[month] = metrics
    else:
        # Ray-parallel for real data
        import ray
        if not ray.is_initialized():
            from pbase.config.ray import init_ray
            import pmodel
            import ml as shadow_ml
            init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel, shadow_ml])

        @ray.remote
        def _eval_remote(month, ct, pt, hc, fc, tb):
            return month, _eval_single_month(month, ct, pt, hc, fc, tb)

        futures = [
            _eval_remote.remote(m, class_type, ptype, hyperparam_config, feature_config, threshold_beta)
            for m in eval_months
        ]
        for month, metrics in ray.get(futures):
            per_month[month] = metrics

        ray.shutdown()

    # Aggregate
    agg = aggregate_months(per_month)

    result = {
        "eval_config": {
            "eval_months": eval_months,
            "class_type": class_type,
            "ptype": ptype,
            "train_months": 10,
            "val_months": 2,
            "threshold_beta": threshold_beta,
        },
        "per_month": per_month,
        "aggregate": agg,
        "n_months": len(per_month),
        "threshold_per_month": {m: per_month[m].get("threshold") for m in per_month},
    }

    # Register in registry
    registry_path = Path(registry_dir)
    version_dir = registry_path / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Wrote metrics to {version_dir / 'metrics.json'}")

    config_out = {
        "hyperparams": hyperparam_config.to_dict(),
        "features": feature_config.features,
        "eval_config": result["eval_config"],
    }
    with open(version_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    meta = {"n_months": len(per_month), "version_id": version_id}
    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[benchmark] Benchmark complete: {len(per_month)} months evaluated")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run multi-month benchmark evaluation")
    parser.add_argument("--version-id", required=True, help="Version ID (e.g. v0)")
    parser.add_argument("--ptype", default="f0", help="Period type")
    parser.add_argument("--class-type", default="onpeak", help="Class type")
    parser.add_argument("--eval-months", nargs="+", default=None,
                        help="Eval months (default: read from gates.json)")
    parser.add_argument("--registry-dir", default="registry", help="Registry directory")
    parser.add_argument("--gates-path", default="registry/gates.json", help="Gates JSON")
    parser.add_argument("--overrides", default=None, help="JSON config overrides")
    args = parser.parse_args()

    eval_months = args.eval_months
    if eval_months is None:
        with open(args.gates_path) as f:
            gates = json.load(f)
        eval_months = gates.get("eval_months", {}).get("primary", [])
        if not eval_months:
            raise ValueError("No eval_months in gates.json and none provided via --eval-months")

    overrides = json.loads(args.overrides) if args.overrides else None

    run_benchmark(
        version_id=args.version_id,
        eval_months=eval_months,
        class_type=args.class_type,
        ptype=args.ptype,
        registry_dir=args.registry_dir,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `SMOKE_TEST=true python -m pytest ml/tests/test_benchmark.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add ml/benchmark.py ml/tests/test_benchmark.py
git commit -m "benchmark: add multi-month Ray-parallel evaluation runner"
```

---

### Task 7: Clean registry and run v0 baseline

**Files:**
- Delete: `registry/v0001/` (stale smoke artifact)
- Reset: `registry/version_counter.json`
- Reset: `registry/champion.json`
- Rebuild: `registry/v0/` from real data

**Step 1: Clean stale artifacts**

```bash
rm -rf registry/v0001/
echo '{"next_id": 1}' > registry/version_counter.json
echo '{"version": null, "promoted_at": null}' > registry/champion.json
rm -rf registry/v0/
```

**Step 2: Run v0 baseline (SMOKE_TEST first to verify pipeline)**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage1-shadow
export PYTHONPATH="${PWD}"
SMOKE_TEST=true python ml/benchmark.py --version-id v0 --ptype f0 --eval-months 2021-07
```

Verify: `jq '.aggregate.mean."S1-AUC"' registry/v0/metrics.json` produces a number.

**Step 3: Run v0 on real data (requires Ray cluster)**

```bash
SMOKE_TEST=false python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak
```

This reads eval months from `gates.json` (12 primary months), runs Ray-parallel.

**Step 4: Populate gate floors from v0**

```bash
python ml/populate_v0_gates.py
```

Verify: `jq '.gates | to_entries[] | select(.value.pending_v0 == true)' registry/gates.json` returns empty.

**Step 5: Commit**

```bash
git add registry/
git commit -m "registry: v0 baseline on real data, gate floors populated"
```

---

## Phase D: Agent Prompt Updates (Task 8)

### Task 8: Update agent prompts for new gate system

**Files:**
- Modify: `agents/prompts/orchestrator_plan.md`
- Modify: `agents/prompts/orchestrator_synthesize.md`
- Modify: `agents/prompts/worker.md`
- Modify: `agents/prompts/reviewer_claude.md`
- Modify: `agents/prompts/reviewer_codex.md`
- Modify: `runbook.md`

**Step 1: Update orchestrator_plan.md**

Add after the gate-reading section:
```
### Gate System (v2)
- Gates use THREE-LAYER checks: mean quality, tail safety, tail non-regression
- 4 hard gates (Group A): S1-AUC, S1-AP, S1-VCAP@100, S1-NDCG
- 6 monitor gates (Group B): S1-BRIER, S1-VCAP@500, S1-VCAP@1000, S1-REC, S1-CAP@100, S1-CAP@500
- Cascade stages: f0 must pass → f1 must pass → f2+ monitor only
- metrics.json now contains per_month breakdown and aggregate stats
- When analyzing gates, check per-month performance for tail risk, not just averages
```

**Step 2: Update orchestrator_synthesize.md**

Update the promotion decision section:
```
### Promotion Decision (v2 gate system)
- Read metrics.json for the new version: it contains per_month and aggregate sections
- THREE checks per Group A gate:
  1. mean(metric) >= floor
  2. count(months below tail_floor) <= 1
  3. mean_bottom_2(metric) >= mean_bottom_2(champion) - noise_tolerance
- Cascade: f0 gates must pass before checking f1, f1 before f2+
- Promote ONLY if all blocking cascade stages pass all 3 layers
- Set decisions.promote_version to version_id if promoting, null otherwise
```

**Step 3: Update worker.md**

Update the pipeline command section:
```
### Running the pipeline
In SMOKE_TEST mode (single month):
  SMOKE_TEST=true python ml/pipeline.py --version-id ${VERSION_ID} --auction-month 2021-07 ...

In real mode (multi-month benchmark):
  python ml/benchmark.py --version-id ${VERSION_ID} --ptype f0 --class-type onpeak

The benchmark reads eval months from gates.json and runs Ray-parallel.
metrics.json will contain per_month breakdown + aggregate stats.
```

**Step 4: Update reviewer_claude.md and reviewer_codex.md**

Add to both:
```
### Gate Analysis (v2 — three-layer system)
When reviewing gate performance, analyze all three layers:
1. **Mean quality**: Is the average metric across 12 months above floor?
2. **Tail safety**: Are there any catastrophic months? (check per_month in metrics.json)
3. **Tail regression**: Are the worst 2 months better or worse than champion's worst 2?

Also check: which specific months are weakest? Is there a seasonal pattern?
A model that improves summer but degrades winter may not be worth promoting.
```

**Step 5: Update `runbook.md`**

Add a section describing the new gate system after the existing "Agent Roles & Constraints" section.

**Step 6: Commit**

```bash
git add agents/prompts/ runbook.md
git commit -m "prompts: update all agent prompts for v2 gate system (three-layer, cascade, per-month)"
```

---

## Verification

After all tasks complete:

1. `SMOKE_TEST=true python -m pytest ml/tests/ -v` — all pass
2. `jq '.version' registry/gates.json` — returns `2`
3. `jq '.cascade_stages | length' registry/gates.json` — returns `3`
4. `jq '.gates."S1-AUC".tail_floor' registry/gates.json` — not null (after v0 baseline)
5. `jq '.gates | to_entries[] | select(.value.pending_v0 == true)' registry/gates.json` — empty
6. `jq '.aggregate.mean."S1-AUC"' registry/v0/metrics.json` — real number
7. `jq '.per_month | keys | length' registry/v0/metrics.json` — 12 (after real data run)
8. Agent prompts mention "three-layer", "cascade", "per_month"
