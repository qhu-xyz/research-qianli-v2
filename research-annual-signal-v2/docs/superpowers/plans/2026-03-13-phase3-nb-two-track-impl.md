# Phase 3: NB Two-Track Reranking — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-track scoring infrastructure that reserves R slots in the top-50 for NB candidate branches, enabling the model to capture the ~20% of binding SP currently invisible to the shortlist.

**Architecture:** Split the branch universe into Track A (established, bf_combined_12 > 0) and Track B (NB candidates: dormant + zero-history cohorts). Train a separate binary classifier on Track B using density features. Merge with reserved slots via `top_k_override` parameter threading through `evaluate_group()` and `cohort_contribution()`. Gate on top-50 metrics only (VC@50, Recall@50, Abs_SP@50).

**Tech Stack:** Python 3.11, Polars, LightGBM, NumPy, scikit-learn (LogisticRegression for Track B baseline), pytest.

**Spec:** `docs/superpowers/plans/2026-03-13-phase3-nb-two-track-design.md`

---

## File Structure

| File | Responsibility | Phase |
|------|---------------|-------|
| `ml/config.py` | Gate metric constants (remove NB12_Recall@50, add TWO_TRACK_GATE_METRICS) | 3.0.1 |
| `ml/evaluate.py` | New NB metrics, `top_k_override` param, `check_nb_threshold()`, `check_gates()` gate_metrics param | 3.0.2 |
| `ml/registry.py` | `nb_gate_results` persistence | 3.0.3 |
| `registry/baseline_contract.json` | NB baseline documentation | 3.0.4 |
| `ml/merge.py` | NEW — `merge_tracks()` returning top_k_indices | 3.3.1 |
| `scripts/run_nb_analysis.py` | NEW — Phase 3.1 NB population analysis | 3.1 |
| `scripts/run_track_b_experiment.py` | NEW — Phase 3.2 Track B model dev | 3.2 |
| `scripts/run_two_track_experiment.py` | NEW — Phase 3.3-3.4 merge sweep + holdout | 3.3-3.4 |
| `tests/test_evaluate.py` | Tests for new metrics, top_k_override, check_nb_threshold | 3.0.2 |
| `tests/test_registry.py` | Test for nb_gate_results persistence | 3.0.3 |
| `tests/test_merge.py` | NEW — Tests for merge_tracks | 3.3.1 |

---

## Chunk 1: Phase 3.0 — Infrastructure

### Task 1: Remove NB12_Recall@50 from TIER1_GATE_METRICS

**Files:**
- Modify: `ml/config.py:170-173`
- Modify: `tests/test_evaluate.py:122-143` (test_gates_restrict_to_tier1_metrics)

- [ ] **Step 1: Update the gate test to expect no NB12_Recall@50**

In `tests/test_evaluate.py`, the test `test_gates_restrict_to_tier1_metrics` asserts that gate keys are a subset of `TIER1_GATE_METRICS`. After our change, `NB12_Recall@50` will no longer be in `TIER1_GATE_METRICS`, so the existing test will still pass (it only checks inclusion). Add a new explicit assertion:

```python
def test_nb12_recall_not_in_tier1_gates():
    """NB12_Recall@50 was removed from TIER1_GATE_METRICS (Phase 3.0.1).
    It was permanently inert (0.0 vs 0.0) and blocked all models equally.
    """
    from ml.config import TIER1_GATE_METRICS
    assert "NB12_Recall@50" not in TIER1_GATE_METRICS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_nb12_recall_not_in_tier1_gates -v`
Expected: FAIL — `NB12_Recall@50` is currently in `TIER1_GATE_METRICS`

- [ ] **Step 3: Remove NB12_Recall@50 from TIER1_GATE_METRICS**

In `ml/config.py:170-173`, change:

```python
# Before:
TIER1_GATE_METRICS: list[str] = [
    "VC@50", "VC@100", "Recall@50", "Recall@100",
    "NDCG", "Abs_SP@50", "NB12_Recall@50",
]

# After:
TIER1_GATE_METRICS: list[str] = [
    "VC@50", "VC@100", "Recall@50", "Recall@100",
    "NDCG", "Abs_SP@50",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_nb12_recall_not_in_tier1_gates -v`
Expected: PASS

- [ ] **Step 5: Run full evaluate test suite to check no regressions**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add ml/config.py tests/test_evaluate.py
git commit -m "phase3.0.1: remove NB12_Recall@50 from TIER1_GATE_METRICS

Permanently inert gate (0.0 vs 0.0) that blocked all models equally.
NB performance will be gated via check_nb_threshold() instead."
```

---

### Task 2: Add TWO_TRACK_GATE_METRICS and gate_metrics parameter

**Files:**
- Modify: `ml/config.py:170+` (add constant)
- Modify: `ml/evaluate.py:195-255` (add `gate_metrics` param to `check_gates`)
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for TWO_TRACK_GATE_METRICS constant**

In `tests/test_evaluate.py`:

```python
def test_two_track_gate_metrics():
    """Phase 3.0.1: TWO_TRACK_GATE_METRICS restricts gating to top-50 only."""
    from ml.config import TWO_TRACK_GATE_METRICS
    assert TWO_TRACK_GATE_METRICS == ["VC@50", "Recall@50", "Abs_SP@50"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_two_track_gate_metrics -v`
Expected: FAIL — `TWO_TRACK_GATE_METRICS` not yet defined

- [ ] **Step 3: Add TWO_TRACK_GATE_METRICS to config.py**

In `ml/config.py`, after `TIER1_GATE_METRICS`, add:

```python
# Two-track gate metrics — top-50 only (K>50 metrics are monitoring, not gated)
TWO_TRACK_GATE_METRICS: list[str] = ["VC@50", "Recall@50", "Abs_SP@50"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_two_track_gate_metrics -v`
Expected: PASS

- [ ] **Step 5: Write test for check_gates with gate_metrics override**

```python
def test_check_gates_with_custom_gate_metrics():
    """Phase 3.0.1: check_gates accepts gate_metrics param to override TIER1_GATE_METRICS."""
    from ml.evaluate import check_gates
    candidate = {
        "2025-06/aq1": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
        "2025-06/aq2": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
        "2025-06/aq3": {"VC@50": 0.30, "VC@100": 0.40, "Recall@50": 0.25},
    }
    baseline = {
        "2025-06/aq1": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
        "2025-06/aq2": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
        "2025-06/aq3": {"VC@50": 0.25, "VC@100": 0.50, "Recall@50": 0.20},
    }
    # Only gate on VC@50 and Recall@50 — VC@100 should NOT appear
    gates = check_gates(
        candidate, baseline, "v0c",
        ["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"],
        gate_metrics=["VC@50", "Recall@50"],
    )
    assert "VC@50" in gates
    assert "Recall@50" in gates
    assert "VC@100" not in gates, "VC@100 should be excluded by gate_metrics override"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_check_gates_with_custom_gate_metrics -v`
Expected: FAIL — `check_gates` doesn't accept `gate_metrics` yet

- [ ] **Step 7: Add gate_metrics parameter to check_gates**

In `ml/evaluate.py`, modify `check_gates` signature and body:

```python
def check_gates(
    candidate: dict,
    baseline: dict,
    baseline_name: str,
    holdout_groups: list[str],
    gate_metrics: list[str] | None = None,
) -> dict:
    """Check gate metrics: candidate vs baseline.
    ...
    Args:
        ...
        gate_metrics: override metric list. Defaults to TIER1_GATE_METRICS.
    """
    metrics_to_check = gate_metrics if gate_metrics is not None else TIER1_GATE_METRICS
    gates: dict = {}

    for metric in metrics_to_check:
        # ... rest unchanged, just replace TIER1_GATE_METRICS with metrics_to_check
```

Change line 220 from `for metric in TIER1_GATE_METRICS:` to `for metric in metrics_to_check:`.

- [ ] **Step 8: Run tests to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py -v`
Expected: All PASS (including both new tests and existing tests)

- [ ] **Step 9: Commit**

```bash
git add ml/config.py ml/evaluate.py tests/test_evaluate.py
git commit -m "phase3.0.1: add TWO_TRACK_GATE_METRICS and gate_metrics param to check_gates

- TWO_TRACK_GATE_METRICS = [VC@50, Recall@50, Abs_SP@50] for two-track gating
- check_gates() accepts optional gate_metrics to override TIER1_GATE_METRICS
- Existing callers unchanged (default=TIER1_GATE_METRICS)"
```

---

### Task 3: Add top_k_override to evaluate_group and cohort_contribution

**Files:**
- Modify: `ml/evaluate.py:93-125` (evaluate_group), `ml/evaluate.py:169-189` (cohort_contribution)
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for evaluate_group with top_k_override**

In `tests/test_evaluate.py`:

```python
import polars as pl

def _make_group_df(n=10, nb_indices=None, binding_indices=None, scores=None):
    """Helper to create a minimal group_df for evaluate_group tests."""
    if binding_indices is None:
        binding_indices = [0, 1, 2, 3]
    if nb_indices is None:
        nb_indices = []
    if scores is None:
        scores = list(range(n, 0, -1))  # descending: n, n-1, ..., 1

    sp = [0.0] * n
    for i in binding_indices:
        sp[i] = 100.0 - i * 10  # decreasing SP
    label_tier = [0] * n
    for i in binding_indices:
        label_tier[i] = 2
    is_nb_12 = [False] * n
    for i in nb_indices:
        is_nb_12[i] = True
    is_nb_6 = is_nb_12[:]
    is_nb_24 = is_nb_12[:]
    cohort = ["established"] * n
    for i in nb_indices:
        cohort[i] = "history_dormant"

    return pl.DataFrame({
        "score": scores,
        "realized_shadow_price": sp,
        "label_tier": label_tier,
        "total_da_sp_quarter": [1000.0] * n,
        "is_nb_12": is_nb_12,
        "is_nb_6": is_nb_6,
        "is_nb_24": is_nb_24,
        "cohort": cohort,
        "onpeak_sp": [s * 0.6 for s in sp],
        "offpeak_sp": [s * 0.4 for s in sp],
    })


def test_evaluate_group_top_k_override():
    """Phase 3.0.2: top_k_override forces specific indices into top-K."""
    import numpy as np
    from ml.evaluate import evaluate_group

    # 10 branches: 0-3 are binders (established), 8-9 are NB binders
    # Default scores rank 0-9 in order (0 is highest score)
    gdf = _make_group_df(
        n=10,
        binding_indices=[0, 1, 2, 3, 8, 9],
        nb_indices=[8, 9],
        scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    # Without override: top-5 = [0,1,2,3,4] — indices 8,9 excluded
    m_default = evaluate_group(gdf, k=5)
    assert m_default["NB12_Count@5"] == 0  # no NB in top-5 by score

    # With override: force indices [0,1,2,8,9] into top-5
    override = np.array([0, 1, 2, 8, 9])
    m_override = evaluate_group(gdf, k=5, top_k_override=override)
    assert m_override["NB12_Count@5"] == 2  # both NB binders included
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_evaluate_group_top_k_override -v`
Expected: FAIL — `evaluate_group` doesn't accept `k` or `top_k_override` params

- [ ] **Step 3: Refactor evaluate_group to accept top_k_override and k**

In `ml/evaluate.py`, modify `evaluate_group`:

```python
def evaluate_group(
    group_df: pl.DataFrame,
    k: int = 50,
    top_k_override: np.ndarray | None = None,
) -> dict:
    """Compute all metrics for one (PY, quarter) group.

    Args:
        group_df: must have score, realized_shadow_price, total_da_sp_quarter,
            is_nb_12, is_nb_6, is_nb_24, cohort, onpeak_sp, offpeak_sp, label_tier.
        k: top-K cutoff for metrics (default 50, used for two-track merge).
        top_k_override: pre-computed ordered indices for the merged top-K.
            When provided, ALL K-level metrics use these indices instead of
            argsort(scores)[::-1][:k].
    """
    actual = group_df["realized_shadow_price"].to_numpy().astype(np.float64)
    label_tier = group_df["label_tier"].to_numpy().astype(np.float64)
    scores = group_df["score"].to_numpy().astype(np.float64)
    total_da_sp = float(group_df["total_da_sp_quarter"][0])
    is_nb_12 = group_df["is_nb_12"].to_numpy()
    is_nb_6 = group_df["is_nb_6"].to_numpy() if "is_nb_6" in group_df.columns else None
    is_nb_24 = group_df["is_nb_24"].to_numpy() if "is_nb_24" in group_df.columns else None

    n = len(actual)
    metrics: dict = {"n_branches": n, "n_binding": int((actual > 0).sum())}

    # Compute top-K indices: override or score-based
    if top_k_override is not None:
        assert len(top_k_override) <= k, (
            f"top_k_override length {len(top_k_override)} > k={k}"
        )
        top_k_idx = top_k_override
        k_actual = len(top_k_idx)  # may be < k when populations are small
    else:
        if k <= n:
            top_k_idx = np.argsort(scores)[::-1][:k]
        else:
            top_k_idx = np.argsort(scores)[::-1]  # all branches when n < k
        k_actual = len(top_k_idx)

    # Top-K mask for set-based metrics
    top_k_mask = np.zeros(n, dtype=bool)
    top_k_mask[top_k_idx] = True

    # VC@K, Recall@K, Abs_SP@K at the policy K (uses k for metric key, k_actual for computation)
    total_sp = actual.sum()
    if total_sp > 0:
        metrics[f"VC@{k}"] = float(actual[top_k_idx].sum() / total_sp)
    else:
        metrics[f"VC@{k}"] = 0.0

    n_binding = (actual > 0).sum()
    if n_binding > 0:
        metrics[f"Recall@{k}"] = float((actual[top_k_idx] > 0).sum() / n_binding)
    else:
        metrics[f"Recall@{k}"] = 0.0

    if total_da_sp > 0:
        metrics[f"Abs_SP@{k}"] = float(actual[top_k_idx].sum() / total_da_sp)
    else:
        metrics[f"Abs_SP@{k}"] = 0.0

    # NB12_Recall@K
    nb12_binders = (actual > 0) & is_nb_12
    n_nb12 = nb12_binders.sum()
    metrics[f"NB12_Recall@{k}"] = float((nb12_binders & top_k_mask).sum() / n_nb12) if n_nb12 > 0 else 0.0

    # --- New NB metrics (Phase 3.0.2) ---
    # NB12_Count@K: count of NB12 binders in top-K
    metrics[f"NB12_Count@{k}"] = int((nb12_binders & top_k_mask).sum())

    # NB12_SP@K: SP from NB12 binders in top-K / total NB12 SP
    total_nb12_sp = actual[is_nb_12].sum()
    if total_nb12_sp > 0:
        metrics[f"NB12_SP@{k}"] = float(actual[nb12_binders & top_k_mask].sum() / total_nb12_sp)
    else:
        metrics[f"NB12_SP@{k}"] = 0.0

    # NB6_Recall@K
    if is_nb_6 is not None:
        nb6_binders = (actual > 0) & is_nb_6
        n_nb6 = nb6_binders.sum()
        metrics[f"NB6_Recall@{k}"] = float((nb6_binders & top_k_mask).sum() / n_nb6) if n_nb6 > 0 else 0.0

    # NB24_Recall@K
    if is_nb_24 is not None:
        nb24_binders = (actual > 0) & is_nb_24
        n_nb24 = nb24_binders.sum()
        metrics[f"NB24_Recall@{k}"] = float((nb24_binders & top_k_mask).sum() / n_nb24) if n_nb24 > 0 else 0.0

    # Note: metric keys always use `k` (the requested K), NOT `k_actual`.
    # When k_actual < k (small populations), the metric reflects whatever was available.

    # Additional K levels (score-based, NOT overridden — monitoring only)
    # NB12_Count and NB12_SP are only computed at policy K, not extra K levels.
    for extra_k in [20, 50, 100]:
        if extra_k == k:
            continue  # already computed above
        if extra_k <= n:
            extra_idx = np.argsort(scores)[::-1][:extra_k]
            sp_sum = actual.sum()
            metrics[f"VC@{extra_k}"] = float(actual[extra_idx].sum() / sp_sum) if sp_sum > 0 else 0.0
            metrics[f"Recall@{extra_k}"] = float((actual[extra_idx] > 0).sum() / n_binding) if n_binding > 0 else 0.0
            metrics[f"Abs_SP@{extra_k}"] = float(actual[extra_idx].sum() / total_da_sp) if total_da_sp > 0 else 0.0
            # NB12_Recall at extra K levels (score-based)
            extra_mask = np.zeros(n, dtype=bool)
            extra_mask[extra_idx] = True
            metrics[f"NB12_Recall@{extra_k}"] = float((nb12_binders & extra_mask).sum() / n_nb12) if n_nb12 > 0 else 0.0

    # NDCG and Spearman (always score-based — full ranking metrics)
    metrics["NDCG"] = ndcg(label_tier, scores)
    metrics["Spearman"] = spearman_corr(actual, scores)

    # Cohort contribution at the policy K (uses override if provided)
    if k <= n:
        metrics["cohort_contribution"] = cohort_contribution(
            group_df, k=k, top_k_override=top_k_override,
        )

    return metrics
```

Also update `cohort_contribution`:

```python
def cohort_contribution(
    group_df: pl.DataFrame,
    k: int,
    top_k_override: np.ndarray | None = None,
) -> dict:
    """Tier 3: cohort breakdown of top-K branches.

    Args:
        top_k_override: pre-computed indices. When provided, uses these
            instead of score-based argsort.
    """
    if top_k_override is not None:
        top_k_idx = top_k_override
    else:
        scores = group_df["score"].to_numpy()
        top_k_idx = np.argsort(scores)[::-1][:k]

    top_k_mask = np.zeros(len(group_df), dtype=bool)
    top_k_mask[top_k_idx] = True

    cohorts = group_df["cohort"].to_list()
    actual = group_df["realized_shadow_price"].to_numpy()

    result: dict = {}
    for cohort_name in ["established", "history_dormant", "history_zero"]:
        cohort_mask = np.array([c == cohort_name for c in cohorts])
        in_top_k = (cohort_mask & top_k_mask).sum()
        sp_captured = actual[cohort_mask & top_k_mask].sum()
        result[cohort_name] = {
            "count_in_top_k": int(in_top_k),
            "sp_captured": float(sp_captured),
        }

    return result
```

**Important**: Update `evaluate_all` to pass through any extra kwargs:

```python
def evaluate_all(model_table: pl.DataFrame) -> dict:
    # ... unchanged, calls evaluate_group(group_df) with defaults
```

`evaluate_all` continues calling `evaluate_group(group_df)` with default k=50 and no override — the override is only used by the two-track experiment script that calls `evaluate_group` directly per group.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_evaluate_group_top_k_override -v`
Expected: PASS

- [ ] **Step 5: Write backward-compatibility test for evaluate_group defaults**

```python
def test_evaluate_group_backward_compat():
    """evaluate_group with default args produces same keys as before refactor."""
    from ml.evaluate import evaluate_group

    gdf = _make_group_df(
        n=100,
        binding_indices=list(range(20)),
        nb_indices=[15, 16, 17, 18, 19],
        scores=list(range(100, 0, -1)),
    )

    m = evaluate_group(gdf)

    # Must contain all original metric keys
    for key in ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "Recall@100",
                 "Abs_SP@20", "Abs_SP@50", "Abs_SP@100",
                 "NB12_Recall@20", "NB12_Recall@50", "NB12_Recall@100",
                 "NDCG", "Spearman", "n_branches", "n_binding", "cohort_contribution"]:
        assert key in m, f"Missing metric key: {key}"

    # New NB metrics also present at default k=50
    for key in ["NB12_Count@50", "NB12_SP@50", "NB6_Recall@50", "NB24_Recall@50"]:
        assert key in m, f"Missing new NB metric key: {key}"

    # VC@50 should equal score-based argsort (no override)
    import numpy as np
    actual = gdf["realized_shadow_price"].to_numpy().astype(np.float64)
    scores = gdf["score"].to_numpy().astype(np.float64)
    top_50 = np.argsort(scores)[::-1][:50]
    expected_vc50 = actual[top_50].sum() / actual.sum()
    assert abs(m["VC@50"] - expected_vc50) < 1e-10
```

- [ ] **Step 6: Run backward-compat test**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_evaluate_group_backward_compat -v`
Expected: PASS (after implementing evaluate_group refactor)

- [ ] **Step 7: Write test for cohort_contribution with top_k_override**

```python
def test_cohort_contribution_with_override():
    """Phase 3.0.2: cohort_contribution uses top_k_override when provided."""
    import numpy as np
    from ml.evaluate import cohort_contribution

    gdf = _make_group_df(
        n=10,
        binding_indices=[0, 1, 8, 9],
        nb_indices=[8, 9],
        scores=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    # With override including NB indices
    override = np.array([0, 1, 8, 9, 4])  # 2 established, 2 NB, 1 non-binding
    cc = cohort_contribution(gdf, k=5, top_k_override=override)
    assert cc["history_dormant"]["count_in_top_k"] == 2
    assert cc["established"]["count_in_top_k"] == 3  # indices 0, 1, 4
```

- [ ] **Step 8: Run to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_cohort_contribution_with_override -v`
Expected: PASS

- [ ] **Step 9: Write tests for NB12_SP and NB6/NB24_Recall with known values**

```python
def test_nb12_sp_calculation():
    """NB12_SP@K = SP from NB12 binders in top-K / total NB12 SP."""
    import numpy as np
    from ml.evaluate import evaluate_group

    # 8 branches: indices 6,7 are NB12 binders with SP=50,30
    # total NB12 SP = 80
    gdf = _make_group_df(
        n=8,
        binding_indices=[0, 1, 6, 7],
        nb_indices=[6, 7],
        scores=[8, 7, 6, 5, 4, 3, 2, 1],
    )
    # Override to include NB index 6 (SP=40) but not 7 (SP=30)
    override = np.array([0, 1, 2, 3, 6])
    m = evaluate_group(gdf, k=5, top_k_override=override)

    # NB12 binders in top-5: index 6 only (SP=40.0)
    # total NB12 SP: indices 6 (SP=40) + 7 (SP=30) = 70
    assert m["NB12_SP@5"] > 0  # should capture some NB12 SP


def test_nb6_nb24_recall():
    """NB6/NB24_Recall use their respective is_nb_N flags."""
    import numpy as np
    from ml.evaluate import evaluate_group

    gdf = _make_group_df(
        n=8,
        binding_indices=[0, 1, 6, 7],
        nb_indices=[6, 7],
        scores=[8, 7, 6, 5, 4, 3, 2, 1],
    )

    # Default: top-5 by score = [0,1,2,3,4] — no NB in top-5
    m = evaluate_group(gdf, k=5)
    assert m["NB6_Recall@5"] == 0.0
    assert m["NB24_Recall@5"] == 0.0

    # Override to include NB
    override = np.array([0, 1, 2, 6, 7])
    m2 = evaluate_group(gdf, k=5, top_k_override=override)
    assert m2["NB6_Recall@5"] > 0.0  # both NB binders now in top-5
    assert m2["NB24_Recall@5"] > 0.0
```

- [ ] **Step 10: Run NB metric tests to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_nb12_sp_calculation tests/test_evaluate.py::test_nb6_nb24_recall -v`
Expected: PASS

- [ ] **Step 11: Update existing evaluate_group call in evaluate_all**

In `evaluate_all`, the call `evaluate_group(group_df)` uses defaults (k=50, no override). Verify that `evaluate_all` still works by running:

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py -v`
Expected: All PASS

- [ ] **Step 12: Commit**

```bash
git add ml/evaluate.py tests/test_evaluate.py
git commit -m "phase3.0.2: add top_k_override to evaluate_group and cohort_contribution

- evaluate_group() accepts k and top_k_override params
- When top_k_override provided, all K-level metrics use override indices
- cohort_contribution() also accepts top_k_override
- Additional K levels (20, 100) remain score-based (monitoring)
- New NB metrics: NB12_Count@K, NB12_SP@K, NB6_Recall@K, NB24_Recall@K"
```

---

### Task 4: Add check_nb_threshold

**Files:**
- Modify: `ml/evaluate.py` (add function)
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for check_nb_threshold**

```python
def test_check_nb_threshold_passes():
    """Phase 3.0.2: cross-group NB gate passes when total NB12_Count >= threshold."""
    from ml.evaluate import check_nb_threshold
    per_group = {
        "2025-06/aq1": {"NB12_Count@50": 2},
        "2025-06/aq2": {"NB12_Count@50": 0},
        "2025-06/aq3": {"NB12_Count@50": 1},
    }
    result = check_nb_threshold(
        per_group,
        holdout_groups=["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"],
        min_total_count=3,
    )
    assert result["passed"] is True
    assert result["total_count"] == 3


def test_check_nb_threshold_fails():
    """Phase 3.0.2: cross-group NB gate fails when total < threshold."""
    from ml.evaluate import check_nb_threshold
    per_group = {
        "2025-06/aq1": {"NB12_Count@50": 1},
        "2025-06/aq2": {"NB12_Count@50": 0},
        "2025-06/aq3": {"NB12_Count@50": 1},
    }
    result = check_nb_threshold(
        per_group,
        holdout_groups=["2025-06/aq1", "2025-06/aq2", "2025-06/aq3"],
        min_total_count=3,
    )
    assert result["passed"] is False
    assert result["total_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py::test_check_nb_threshold_passes tests/test_evaluate.py::test_check_nb_threshold_fails -v`
Expected: FAIL — function doesn't exist yet

- [ ] **Step 3: Implement check_nb_threshold**

In `ml/evaluate.py`, add after `check_gates`:

```python
def check_nb_threshold(
    per_group: dict,
    holdout_groups: list[str],
    min_total_count: int = 3,
) -> dict:
    """Cross-group NB gate: sum NB12_Count@50 across holdout groups >= min_total_count.

    Separate from check_gates() — only applies to two-track candidates.

    Returns:
        dict with passed, total_count, per_group_counts.
    """
    per_group_counts: dict[str, int] = {}
    total = 0
    for g in holdout_groups:
        count = per_group.get(g, {}).get("NB12_Count@50", 0)
        per_group_counts[g] = count
        total += count

    passed = total >= min_total_count
    logger.info(
        "NB threshold: total=%d (min=%d) -> %s | per-group: %s",
        total, min_total_count, "PASS" if passed else "FAIL", per_group_counts,
    )

    return {
        "passed": passed,
        "total_count": total,
        "min_total_count": min_total_count,
        "per_group_counts": per_group_counts,
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_evaluate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ml/evaluate.py tests/test_evaluate.py
git commit -m "phase3.0.2: add check_nb_threshold for cross-group NB gate

Sums NB12_Count@50 across holdout groups, passes if >= min_total_count (default 3).
Separate from check_gates() — only used by two-track candidates."
```

---

### Task 5: Extend registry.py with nb_gate_results persistence

**Files:**
- Modify: `ml/registry.py:13-52`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Write test for nb_gate_results persistence**

In `tests/test_registry.py`:

```python
def test_save_nb_gate_results(tmp_path, monkeypatch):
    """Phase 3.0.3: nb_gate_results saved as separate JSON file."""
    import ml.registry as registry
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)

    config = {"version": "test_nb", "features": ["bin_80_cid_max"]}
    metrics = {"per_group": {"2024-06/aq1": {"VC@50": 0.3}}}
    nb_results = {
        "passed": True,
        "total_count": 4,
        "min_total_count": 3,
        "per_group_counts": {"2025-06/aq1": 2, "2025-06/aq2": 1, "2025-06/aq3": 1},
    }

    path = registry.save_experiment("test_nb_v1", config, metrics, nb_gate_results=nb_results)
    nb_path = path / "nb_gate_results.json"
    assert nb_path.exists()

    import json
    with open(nb_path) as f:
        loaded = json.load(f)
    assert loaded["passed"] is True
    assert loaded["total_count"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_registry.py::test_save_nb_gate_results -v`
Expected: FAIL — `save_experiment` doesn't accept `nb_gate_results`

- [ ] **Step 3: Add nb_gate_results parameter to save_experiment**

In `ml/registry.py`, modify `save_experiment`:

```python
def save_experiment(
    version_id: str,
    config: dict,
    metrics: dict,
    gate_results: dict | None = None,
    baseline_version: str | None = None,
    nb_gate_results: dict | None = None,
) -> Path:
    """Save experiment results to registry/{version_id}/.

    Creates:
      - registry/{version_id}/config.json
      - registry/{version_id}/metrics.json
      - registry/{version_id}/gate_results.json (if gate_results provided)
      - registry/{version_id}/nb_gate_results.json (if nb_gate_results provided)

    Returns path to version directory.
    """
    # ... existing code unchanged ...

    if nb_gate_results is not None:
        nb_path = version_dir / "nb_gate_results.json"
        # Note: unlike gate_results which wraps in {"baseline_version":..., "gates":...},
        # nb_gate_results is written directly since check_nb_threshold() already returns
        # a self-contained dict with passed, total_count, min_total_count, per_group_counts.
        with open(nb_path, "w") as f:
            json.dump(nb_gate_results, f, indent=2, default=str)
        logger.info("NB gate results saved to %s", nb_path)

    logger.info("Saved experiment %s to %s", version_id, version_dir)
    return version_dir
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_registry.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ml/registry.py tests/test_registry.py
git commit -m "phase3.0.3: add nb_gate_results persistence to save_experiment

Writes registry/{version}/nb_gate_results.json when provided.
Same pattern as existing gate_results parameter."
```

---

### Task 6: Update baseline_contract.json

**Files:**
- Modify: `registry/baseline_contract.json`

- [ ] **Step 1: Read current baseline_contract.json** (already read above)

- [ ] **Step 2: Add NB documentation and gate changes**

Add the following keys to `registry/baseline_contract.json`:

```json
{
  "authoritative_baseline": "v0c",
  "rationale": "...(existing)...",
  "formula": { "...(existing)..." },
  "implementation": "scripts/run_v0c_full_blend.py",
  "baselines_evaluated": { "...(existing)..." },
  "frozen_date": "2026-03-12",
  "nb_baseline": {
    "NB12_Recall@50": 0.0,
    "NB12_Count@50": 0,
    "NB12_SP@50": 0.0,
    "NB6_Recall@50": 0.0,
    "NB24_Recall@50": 0.0,
    "note": "All NB metrics are 0.0 for v0c — single-model global ranking never selects NB candidates into top-50."
  },
  "gate_changes": {
    "phase3.0.1": {
      "removed": "NB12_Recall@50 from TIER1_GATE_METRICS — permanently inert (0.0 vs 0.0)",
      "added": "TWO_TRACK_GATE_METRICS = [VC@50, Recall@50, Abs_SP@50] for two-track candidates",
      "nb_gate": "check_nb_threshold() — cross-group NB gate separate from check_gates()"
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add registry/baseline_contract.json
git commit -m "phase3.0.4: update baseline_contract with NB baselines and gate changes

- NB metrics all 0.0 for v0c (expected — single-model never selects NB into top-50)
- Document NB12_Recall@50 removal from TIER1_GATE_METRICS
- Document TWO_TRACK_GATE_METRICS and check_nb_threshold()"
```

---

### Task 7: Compute NB supplement metrics for v0c and v3a

**Files:**
- Create: `scripts/run_nb_supplement.py`
- Create: `registry/v0c/nb_metrics_supplement.json`
- Create: `registry/v3a/nb_metrics_supplement.json`

This task creates a one-shot script that recomputes metrics for v0c and v3a with the new NB metrics, saving as supplementary artifacts.

- [ ] **Step 1: Write the supplement computation script**

Create `scripts/run_nb_supplement.py`:

```python
"""Compute NB supplement metrics for existing registry entries.

Phase 3.0.5: adds NB12_Count@50, NB12_SP@50, NB6_Recall@50, NB24_Recall@50
to existing versions WITHOUT overwriting frozen metrics.json.

Usage:
    PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v0c
    PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v3a
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import polars as pl

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS,
    AQ_QUARTERS, REGISTRY_DIR,
)
from ml.features import build_model_table_all
from ml.evaluate import evaluate_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_v0c_scores(model_table: pl.DataFrame) -> pl.DataFrame:
    """Replicate v0c formula scoring. See baseline_contract.json."""
    def minmax_norm(s: pl.Series) -> pl.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pl.Series([0.5] * len(s))
        return (s - mn) / (mx - mn)

    frames = []
    for (py, aq), gdf in model_table.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        da_norm = 1.0 - minmax_norm(gdf["da_rank_value"])
        rt_max = gdf.select(
            pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                              "bin_100_cid_max", "bin_110_cid_max")
        ).to_series()
        rt_norm = minmax_norm(rt_max)
        bf_norm = minmax_norm(gdf["bf_combined_12"])
        score = 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf_norm
        frames.append(gdf.with_columns(score.alias("score")))

    return pl.concat(frames, how="diagonal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=["v0c", "v3a"])
    args = parser.parse_args()

    t0 = time.time()

    # Build all groups needed for eval
    all_groups = []
    all_pys: set[str] = set()
    for split_info in EVAL_SPLITS.values():
        all_pys.update(split_info["train_pys"])
        all_pys.update(split_info["eval_pys"])
    for py in sorted(all_pys):
        for aq in AQ_QUARTERS:
            key = f"{py}/{aq}"
            if key == "2025-06/aq4":
                continue
            all_groups.append(key)

    eval_groups = DEV_GROUPS + HOLDOUT_GROUPS
    model_table = build_model_table_all(all_groups)

    if args.version == "v0c":
        scored = compute_v0c_scores(model_table)
        scored = scored.filter(
            (pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(eval_groups)
        )
    elif args.version == "v3a":
        # For v3a, we need to retrain — but since this is a one-shot script,
        # load the config to get feature list and retrain
        from ml.registry import load_config
        from ml.train import train_and_predict

        config = load_config("v3a")
        feature_cols = config["features"]

        scored_frames = []
        for eval_key, split_info in EVAL_SPLITS.items():
            scored, _ = train_and_predict(
                model_table=model_table,
                train_pys=split_info["train_pys"],
                eval_pys=split_info["eval_pys"],
                feature_cols=feature_cols,
            )
            valid_groups = set(eval_groups)
            scored = scored.filter(
                (pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(valid_groups)
            )
            scored_frames.append(scored)
        scored = pl.concat(scored_frames, how="diagonal")

    # Evaluate each group with new NB metrics
    nb_supplement: dict = {"per_group": {}}
    for (py, aq), gdf in scored.group_by(
        ["planning_year", "aq_quarter"], maintain_order=True
    ):
        key = f"{py}/{aq}"
        m = evaluate_group(gdf)
        # Extract only NB-related metrics
        nb_metrics = {
            k: v for k, v in m.items()
            if k.startswith("NB") or k == "cohort_contribution"
        }
        nb_supplement["per_group"][key] = nb_metrics

    # Aggregate
    for split_name, split_groups in [("dev_mean", DEV_GROUPS), ("holdout_mean", HOLDOUT_GROUPS)]:
        present = [g for g in split_groups if g in nb_supplement["per_group"]]
        if present:
            agg = {}
            all_keys = set()
            for g in present:
                all_keys.update(nb_supplement["per_group"][g].keys())
            for mk in all_keys:
                vals = [nb_supplement["per_group"][g][mk] for g in present
                        if mk in nb_supplement["per_group"][g]
                        and isinstance(nb_supplement["per_group"][g][mk], (int, float))]
                if vals:
                    agg[mk] = sum(vals) / len(vals)
            nb_supplement[split_name] = agg

    # Save
    out_path = REGISTRY_DIR / args.version / "nb_metrics_supplement.json"
    with open(out_path, "w") as f:
        json.dump(nb_supplement, f, indent=2, default=str)

    logger.info("Saved NB supplement to %s (%.1fs)", out_path, time.time() - t0)

    # Print summary
    for split_name in ["dev_mean", "holdout_mean"]:
        if split_name in nb_supplement:
            print(f"\n{split_name}:")
            for k, v in sorted(nb_supplement[split_name].items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for v0c**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v0c`
Expected: Creates `registry/v0c/nb_metrics_supplement.json` with NB12_Count@50 = 0 everywhere.

- [ ] **Step 3: Run for v3a**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_nb_supplement.py --version v3a`
Expected: Creates `registry/v3a/nb_metrics_supplement.json` with NB12_Count@50 = 0 everywhere.

- [ ] **Step 4: Verify both supplement files exist and contain expected metrics**

```bash
cat registry/v0c/nb_metrics_supplement.json | python -m json.tool | head -30
cat registry/v3a/nb_metrics_supplement.json | python -m json.tool | head -30
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_nb_supplement.py registry/v0c/nb_metrics_supplement.json registry/v3a/nb_metrics_supplement.json
git commit -m "phase3.0.5: compute NB supplement metrics for v0c and v3a

NB12_Count@50 = 0 for all groups (expected — confirms structural problem).
Saved as nb_metrics_supplement.json, not overwriting frozen metrics.json."
```

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/ -v`
Expected: All PASS

---

## Chunk 2: Phase 3.1 — NB Population Analysis

### Task 8: NB population analysis script

**Files:**
- Create: `scripts/run_nb_analysis.py`

This is an analysis script (Phase 3.1.1 + 3.1.2 + 3.1.3). It computes NB populations at multiple windows, profiles Track B features, computes per-feature AUC, and correlation matrices. Output is printed + saved to `registry/nb_analysis/`.

- [ ] **Step 1: Create the analysis script**

Create `scripts/run_nb_analysis.py`:

```python
"""Phase 3.1: NB population analysis and Track B feature profiling.

3.1.1 - NB populations at multiple windows (6, 12, 24)
3.1.2 - Track B feature profiling: base rates, feature distributions, per-feature AUC
3.1.3 - Feature correlation within Track B, prune |r| > 0.85

Usage:
    PYTHONPATH=. uv run python scripts/run_nb_analysis.py
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, REGISTRY_DIR,
    DENSITY_MAX_FEATURES, DENSITY_MIN_FEATURES,
    LIMIT_FEATURES, METADATA_FEATURES,
)
from ml.features import build_model_table_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Track B candidate features (from spec — density + limits + metadata)
TRACK_B_FEATURES: list[str] = (
    DENSITY_MAX_FEATURES + DENSITY_MIN_FEATURES
    + LIMIT_FEATURES + METADATA_FEATURES
)


def analyze_nb_populations(model_table: pl.DataFrame, groups: list[str]) -> dict:
    """3.1.1: NB populations at multiple windows."""
    results: dict = {}
    for g in groups:
        py, aq = g.split("/")
        gdf = model_table.filter(
            (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
        )
        n_total = len(gdf)
        n_binding = gdf.filter(pl.col("realized_shadow_price") > 0).height
        total_sp = gdf["realized_shadow_price"].sum()

        group_result: dict = {"n_total": n_total, "n_binding": n_binding, "total_sp": float(total_sp)}
        for window in [6, 12, 24]:
            col = f"is_nb_{window}"
            nb_mask = gdf[col].to_numpy()
            sp = gdf["realized_shadow_price"].to_numpy()
            n_nb = int(nb_mask.sum())
            nb_sp = float(sp[nb_mask].sum())
            group_result[f"nb{window}_count"] = n_nb
            group_result[f"nb{window}_sp"] = nb_sp
            group_result[f"nb{window}_sp_share"] = nb_sp / total_sp if total_sp > 0 else 0.0

        # Cohort breakdown
        for cohort in ["established", "history_dormant", "history_zero"]:
            c_mask = gdf["cohort"] == cohort
            group_result[f"cohort_{cohort}_count"] = int(c_mask.sum())

        results[g] = group_result

    return results


def profile_track_b(model_table: pl.DataFrame, groups: list[str]) -> dict:
    """3.1.2: Track B feature profiling."""
    # Filter to Track B population across dev groups
    track_b = model_table.filter(
        (pl.col("cohort").is_in(["history_dormant", "history_zero"]))
        & ((pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(groups))
    )

    n_total = len(track_b)
    target = (track_b["realized_shadow_price"].to_numpy() > 0).astype(int)
    n_binders = int(target.sum())
    base_rate = n_binders / n_total if n_total > 0 else 0.0

    # Per-feature AUC
    feature_auc: dict[str, float] = {}
    for feat in TRACK_B_FEATURES:
        if feat not in track_b.columns:
            continue
        vals = track_b[feat].to_numpy().astype(np.float64)
        # Skip if no variance
        if np.std(vals) == 0 or n_binders == 0 or n_binders == n_total:
            feature_auc[feat] = 0.5
            continue
        try:
            auc = roc_auc_score(target, vals)
            feature_auc[feat] = float(auc)
        except ValueError:
            feature_auc[feat] = 0.5

    return {
        "n_track_b": n_total,
        "n_binders": n_binders,
        "base_rate": base_rate,
        "feature_auc": feature_auc,
    }


def compute_correlation_matrix(
    model_table: pl.DataFrame,
    groups: list[str],
    features: list[str],
) -> tuple[dict, list[str]]:
    """3.1.3: Feature correlation within Track B, prune |r| > 0.85."""
    track_b = model_table.filter(
        (pl.col("cohort").is_in(["history_dormant", "history_zero"]))
        & ((pl.col("planning_year") + "/" + pl.col("aq_quarter")).is_in(groups))
    )

    available = [f for f in features if f in track_b.columns]
    X = track_b.select(available).to_numpy().astype(np.float64)
    corr = np.corrcoef(X, rowvar=False)

    # Find pairs with |r| > 0.85
    high_corr_pairs: list[dict] = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if abs(corr[i, j]) > 0.85:
                high_corr_pairs.append({
                    "feat_a": available[i],
                    "feat_b": available[j],
                    "correlation": float(corr[i, j]),
                })

    return {"high_corr_pairs": high_corr_pairs, "n_features": len(available)}, available


def main():
    t0 = time.time()

    # Build model tables for dev groups only
    all_groups = []
    for split_info in EVAL_SPLITS.values():
        if split_info["split"] == "dev":
            for py in split_info["eval_pys"]:
                from ml.config import AQ_QUARTERS
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    all_groups.append(key)

    # Also need training PYs for model table assembly
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"]:
            from ml.config import AQ_QUARTERS
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
        for py in split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")

    logger.info("Building model tables for %d groups...", len(all_needed))
    model_table = build_model_table_all(sorted(all_needed))

    # 3.1.1: NB populations
    logger.info("=== Phase 3.1.1: NB Population Analysis ===")
    pop_results = analyze_nb_populations(model_table, DEV_GROUPS)

    print("\n" + "=" * 90)
    print("  Phase 3.1.1: NB Populations at Multiple Windows (Dev Only)")
    print("=" * 90)
    header = f"{'Group':<16} {'N':>6} {'Bind':>6} {'NB6':>5} {'NB12':>5} {'NB24':>5} {'NB12_SP%':>8} {'Dormant':>8} {'Zero':>8}"
    print(header)
    print("-" * len(header))
    for g in DEV_GROUPS:
        r = pop_results[g]
        print(
            f"{g:<16} {r['n_total']:>6} {r['n_binding']:>6} "
            f"{r['nb6_count']:>5} {r['nb12_count']:>5} {r['nb24_count']:>5} "
            f"{r['nb12_sp_share']:>7.1%} "
            f"{r['cohort_history_dormant_count']:>8} {r['cohort_history_zero_count']:>8}"
        )

    # 3.1.2: Track B profiling
    logger.info("=== Phase 3.1.2: Track B Feature Profiling ===")
    profile = profile_track_b(model_table, DEV_GROUPS)

    print(f"\n{'=' * 70}")
    print(f"  Phase 3.1.2: Track B Profiling (Dev)")
    print(f"{'=' * 70}")
    print(f"  Track B total: {profile['n_track_b']}")
    print(f"  Track B binders (NB12): {profile['n_binders']}")
    print(f"  Base rate: {profile['base_rate']:.2%}")
    print(f"\n  Per-Feature AUC (descending):")
    sorted_auc = sorted(profile["feature_auc"].items(), key=lambda x: x[1], reverse=True)
    for feat, auc in sorted_auc:
        direction = "^" if auc > 0.5 else "v" if auc < 0.5 else "="
        print(f"    {feat:<30} AUC={auc:.4f} {direction}")

    # 3.1.3: Correlation
    logger.info("=== Phase 3.1.3: Feature Correlation ===")
    corr_results, available_feats = compute_correlation_matrix(
        model_table, DEV_GROUPS, TRACK_B_FEATURES,
    )

    print(f"\n{'=' * 70}")
    print(f"  Phase 3.1.3: High Correlation Pairs (|r| > 0.85)")
    print(f"{'=' * 70}")
    for pair in corr_results["high_corr_pairs"]:
        print(f"  {pair['feat_a']:<30} x {pair['feat_b']:<30} r={pair['correlation']:.3f}")
    if not corr_results["high_corr_pairs"]:
        print("  (none)")

    # Save all results
    out_dir = REGISTRY_DIR / "nb_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "population.json", "w") as f:
        json.dump(pop_results, f, indent=2, default=str)
    with open(out_dir / "track_b_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    with open(out_dir / "correlation.json", "w") as f:
        json.dump(corr_results, f, indent=2, default=str)

    logger.info("Results saved to %s (%.1fs)", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_nb_analysis.py`
Expected: Prints NB population tables, Track B AUC table, correlation pairs. Creates `registry/nb_analysis/*.json`.

- [ ] **Step 3: Review AUC results — decide Track B feature set**

Based on the AUC results:
- Features with AUC > 0.55: include in Track B model
- Features with AUC < 0.52: exclude (no discriminative power)
- Prune one from each pair with |r| > 0.85 (keep higher AUC)

Record the selected feature set in `registry/nb_analysis/selected_features.json`:

```json
{
    "track_b_features": ["<selected features>"],
    "selection_criteria": "AUC > 0.55 on Track B dev, pruned |r| > 0.85 pairs"
}
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_nb_analysis.py registry/nb_analysis/
git commit -m "phase3.1: NB population analysis and Track B feature profiling

3.1.1: NB populations at windows 6/12/24 on dev groups
3.1.2: Track B per-feature AUC for binary classification
3.1.3: Feature correlation pruning (|r| > 0.85)"
```

---

## Chunk 3: Phase 3.2 — Track B Model Development

### Task 9: Track B binary classifier and logistic baseline

**Files:**
- Create: `scripts/run_track_b_experiment.py`

- [ ] **Step 1: Create Track B experiment script**

Create `scripts/run_track_b_experiment.py`:

```python
"""Phase 3.2: Track B model development — binary classifier for NB candidates.

Trains on Track B population (cohort in {history_dormant, history_zero}) only.
Target: realized_shadow_price > 0 (any binding = positive).

Models:
  3.2.1 - LightGBM binary classifier with class weights
  3.2.2 - Logistic regression baseline

Usage:
    PYTHONPATH=. uv run python scripts/run_track_b_experiment.py
"""
from __future__ import annotations

import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, AQ_QUARTERS, REGISTRY_DIR,
)
from ml.features import build_model_table_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_track_b_features() -> list[str]:
    """Load selected Track B features from Phase 3.1 analysis."""
    path = REGISTRY_DIR / "nb_analysis" / "selected_features.json"
    assert path.exists(), f"Run Phase 3.1 first: {path}"
    with open(path) as f:
        data = json.load(f)
    return data["track_b_features"]


def train_lgbm_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Train LightGBM binary classifier with scale_pos_weight."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 15,  # smaller than Track A — less data
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 5,
        "scale_pos_weight": scale,
        "num_threads": 4,
        "verbose": -1,
    }

    n_est = params.pop("n_estimators")
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    eval_ds = lgb.Dataset(X_eval, feature_name=feature_names, free_raw_data=False)

    model = lgb.train(params, train_ds, num_boost_round=n_est)
    scores = model.predict(X_eval)

    # Feature importance
    raw_imp = model.feature_importance(importance_type="gain")
    total = raw_imp.sum()
    fi = dict(zip(feature_names, (raw_imp / total).tolist())) if total > 0 else {}

    return scores, {"model": "lgbm_binary", "feature_importance": fi}


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict]:
    """Train L2-regularized logistic regression."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    scores = lr.predict_proba(X_eval)[:, 1]

    coefs = dict(zip(feature_names, lr.coef_[0].tolist()))
    return scores, {"model": "logistic", "coefficients": coefs}


def evaluate_track_b(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> dict:
    """Evaluate Track B model: AUC, precision@K, recall@K."""
    auc = roc_auc_score(y_true, scores) if y_true.sum() > 0 and y_true.sum() < len(y_true) else 0.5
    top_k = np.argsort(scores)[::-1][:k]
    pred_at_k = np.zeros(len(y_true), dtype=int)
    pred_at_k[top_k] = 1

    prec_k = precision_score(y_true, pred_at_k, zero_division=0)
    rec_k = recall_score(y_true, pred_at_k, zero_division=0)

    return {"AUC": float(auc), f"Precision@{k}": float(prec_k), f"Recall@{k}": float(rec_k)}


def main():
    t0 = time.time()

    features = load_track_b_features()
    logger.info("Track B features (%d): %s", len(features), features)

    # Build model tables
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")

    model_table = build_model_table_all(sorted(all_needed))

    # Filter to Track B only
    track_b_all = model_table.filter(
        pl.col("cohort").is_in(["history_dormant", "history_zero"])
    )

    # Binary target
    track_b_all = track_b_all.with_columns(
        (pl.col("realized_shadow_price") > 0).cast(pl.Int32).alias("target_bind")
    )

    # Expanding-window train/eval on dev splits only
    results: dict = {"lgbm": {}, "logistic": {}}

    for eval_key, split_info in EVAL_SPLITS.items():
        if split_info["split"] != "dev":
            continue

        train_pys = split_info["train_pys"]
        eval_pys = split_info["eval_pys"]

        train_df = track_b_all.filter(pl.col("planning_year").is_in(train_pys))
        eval_df = track_b_all.filter(pl.col("planning_year").is_in(eval_pys))

        if len(train_df) == 0 or len(eval_df) == 0:
            continue

        X_train = train_df.select(features).to_numpy().astype(np.float64)
        y_train = train_df["target_bind"].to_numpy()
        X_eval = eval_df.select(features).to_numpy().astype(np.float64)
        y_eval = eval_df["target_bind"].to_numpy()

        logger.info(
            "Split %s: train=%d (pos=%.1f%%), eval=%d (pos=%.1f%%)",
            eval_key, len(train_df), y_train.mean() * 100,
            len(eval_df), y_eval.mean() * 100,
        )

        # LightGBM binary
        lgbm_scores, lgbm_info = train_lgbm_binary(X_train, y_train, X_eval, features)
        lgbm_metrics = evaluate_track_b(y_eval, lgbm_scores)
        results["lgbm"][eval_key] = {**lgbm_metrics, **lgbm_info}

        # Logistic regression
        lr_scores, lr_info = train_logistic(X_train, y_train, X_eval, features)
        lr_metrics = evaluate_track_b(y_eval, lr_scores)
        results["logistic"][eval_key] = {**lr_metrics, **lr_info}

    # Print comparison
    print(f"\n{'='*80}")
    print("  Phase 3.2: Track B Model Comparison (Dev Only)")
    print(f"{'='*80}\n")

    header = f"{'Split':<12} {'Model':<12} {'AUC':>8} {'P@10':>8} {'R@10':>8}"
    print(header)
    print("-" * len(header))
    for eval_key in sorted(results["lgbm"].keys()):
        for model_name in ["lgbm", "logistic"]:
            m = results[model_name][eval_key]
            print(
                f"{eval_key:<12} {model_name:<12} "
                f"{m['AUC']:>8.4f} {m.get('Precision@10', 0):>8.4f} "
                f"{m.get('Recall@10', 0):>8.4f}"
            )

    # Means
    for model_name in ["lgbm", "logistic"]:
        aucs = [results[model_name][k]["AUC"] for k in results[model_name]]
        print(f"\n  {model_name} mean AUC: {sum(aucs)/len(aucs):.4f}")

    # Save results
    out_dir = REGISTRY_DIR / "track_b_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved to %s (%.1fs)", out_dir, time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Track B experiment**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_track_b_experiment.py`
Expected: Prints AUC comparison table. LightGBM should beat logistic if there's meaningful non-linear signal.

- [ ] **Step 3: Evaluate results against spec criteria**

Check:
- If LightGBM AUC > 0.60 → proceed with LightGBM as Track B model
- If LightGBM AUC < 0.55 → fall back to density-composite heuristic (Risk 1)
- If LightGBM beats logistic by < 3% AUC → use logistic (simpler, spec criterion)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_track_b_experiment.py registry/track_b_experiment/
git commit -m "phase3.2: Track B model development — binary classifier vs logistic

LightGBM binary + L2 logistic regression on NB candidate population.
Expanding-window training on dev groups only."
```

---

## Chunk 4: Phase 3.3-3.4 — Merge Policy and Holdout Validation

### Task 10: Create merge module

**Files:**
- Create: `ml/merge.py`
- Create: `tests/test_merge.py`

- [ ] **Step 1: Write tests for merge_tracks**

Create `tests/test_merge.py`:

```python
"""Tests for ml/merge.py — two-track merge logic."""
import numpy as np
import polars as pl
import pytest


def _make_track_dfs():
    """Create minimal Track A and Track B DataFrames."""
    track_a = pl.DataFrame({
        "branch_name": [f"est_{i}" for i in range(8)],
        "score": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "cohort": ["established"] * 8,
        "realized_shadow_price": [100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0, 0.0],
    })
    track_b = pl.DataFrame({
        "branch_name": [f"nb_{i}" for i in range(4)],
        "score": [0.9, 0.7, 0.3, 0.1],
        "cohort": ["history_dormant"] * 4,
        "realized_shadow_price": [50.0, 30.0, 0.0, 0.0],
    })
    return track_a, track_b


def test_merge_tracks_basic():
    """merge_tracks returns correct top_k_indices for K=5, R=2."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=2)

    # merged should have 12 rows total
    assert len(merged) == 12

    # top_k_idx has 5 entries: 3 from Track A + 2 from Track B
    assert len(top_k_idx) == 5

    # First 3 indices should be Track A branches (est_0, est_1, est_2)
    merged_names = merged["branch_name"].to_list()
    for idx in top_k_idx[:3]:
        assert merged_names[idx].startswith("est_")

    # Last 2 indices should be Track B branches (nb_0, nb_1 — top 2 by Track B score)
    for idx in top_k_idx[3:]:
        assert merged_names[idx].startswith("nb_")


def test_merge_tracks_r_zero():
    """R=0 means all slots from Track A."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=0)

    assert len(top_k_idx) == 5
    merged_names = merged["branch_name"].to_list()
    # All 5 should be Track A
    for idx in top_k_idx:
        assert merged_names[idx].startswith("est_")


def test_merge_tracks_r_exceeds_track_b():
    """If R > Track B population, use all Track B branches."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=10)

    # Only 4 Track B branches available, so top-K = 1 Track A + 4 Track B
    merged_names = merged["branch_name"].to_list()
    track_b_in_top = sum(1 for idx in top_k_idx if merged_names[idx].startswith("nb_"))
    assert track_b_in_top == 4


def test_merge_tracks_small_populations():
    """When total population < K, top_k_indices has len < K."""
    from ml.merge import merge_tracks
    track_a = pl.DataFrame({
        "branch_name": ["a0", "a1"],
        "score": [2.0, 1.0],
        "cohort": ["established"] * 2,
        "realized_shadow_price": [100.0, 50.0],
    })
    track_b = pl.DataFrame({
        "branch_name": ["b0"],
        "score": [0.5],
        "cohort": ["history_dormant"],
        "realized_shadow_price": [30.0],
    })
    merged, top_k_idx = merge_tracks(track_a, track_b, k=50, r=10)
    # Only 3 branches total — top_k_indices should have length 3, not 50
    assert len(top_k_idx) == 3
    assert len(merged) == 3


def test_merge_tracks_provenance():
    """Merged DataFrame has a 'track' column."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, _ = merge_tracks(track_a, track_b, k=5, r=2)

    assert "track" in merged.columns
    tracks = merged["track"].to_list()
    assert tracks.count("A") == 8
    assert tracks.count("B") == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_merge.py -v`
Expected: FAIL — `ml/merge.py` doesn't exist

- [ ] **Step 3: Implement merge_tracks**

Create `ml/merge.py`:

```python
"""Two-track merge logic — combines Track A and Track B rankings.

Track A: established branches scored by existing model (v0c or v3a).
Track B: NB candidates scored by Track B binary classifier.

merge_tracks() returns a pre-computed top_k_indices array for use with
evaluate_group(top_k_override=...).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def merge_tracks(
    track_a: pl.DataFrame,
    track_b: pl.DataFrame,
    k: int,
    r: int,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Merge Track A and Track B into a single DataFrame with top-K indices.

    Args:
        track_a: established branches with 'score' column (Track A model scores)
        track_b: NB candidates with 'score' column (Track B model scores)
        k: total top-K slots (e.g. 50)
        r: reserved slots for Track B

    Returns:
        (merged_df, top_k_indices):
        - merged_df: vertical concat of track_a + track_b with 'track' provenance column
        - top_k_indices: np.ndarray of length min(k, len(merged_df)),
          first (k - r_actual) indices from Track A, last r_actual from Track B
          (all in merged_df index space)
    """
    # Add provenance
    a = track_a.with_columns(pl.lit("A").alias("track"))
    b = track_b.with_columns(pl.lit("B").alias("track"))

    merged = pl.concat([a, b], how="diagonal")

    n_a = len(track_a)
    n_b = len(track_b)

    # Actual R (capped by Track B population)
    r_actual = min(r, n_b)
    n_a_slots = min(k - r_actual, n_a)

    # Top Track A indices (in merged_df space: 0..n_a-1)
    a_scores = track_a["score"].to_numpy().astype(np.float64)
    a_order = np.argsort(a_scores)[::-1][:n_a_slots]

    # Top Track B indices (in merged_df space: n_a..n_a+n_b-1)
    b_scores = track_b["score"].to_numpy().astype(np.float64)
    b_order = np.argsort(b_scores)[::-1][:r_actual]
    b_order_merged = b_order + n_a  # offset into merged index space

    top_k_indices = np.concatenate([a_order, b_order_merged])

    return merged, top_k_indices
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/test_merge.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ml/merge.py tests/test_merge.py
git commit -m "phase3.3.1: add merge_tracks for two-track top-K construction

Returns merged DataFrame + pre-computed top_k_indices array.
Allocates (K-R) slots from Track A, R from Track B."
```

---

### Task 11: Two-track sweep and holdout validation script

**Files:**
- Create: `scripts/run_two_track_experiment.py`

- [ ] **Step 1: Create the two-track experiment script**

Create `scripts/run_two_track_experiment.py`:

```python
"""Phase 3.3-3.4: Two-track merge sweep + holdout validation.

Sweeps R in {0, 5, 10, 15} for K=50 on dev, validates best R on holdout.
Tests both v0c and v3a as Track A models.

Usage:
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v3a
    PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c --holdout --r 10
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from ml.config import (
    EVAL_SPLITS, DEV_GROUPS, HOLDOUT_GROUPS, AQ_QUARTERS,
    REGISTRY_DIR, TWO_TRACK_GATE_METRICS,
)
from ml.features import build_model_table_all
from ml.evaluate import evaluate_group, check_gates, check_nb_threshold
from ml.registry import save_experiment, load_metrics
from ml.merge import merge_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


R_VALUES = [0, 5, 10, 15]
K = 50


def load_track_b_features() -> list[str]:
    """Load selected Track B features from Phase 3.1."""
    path = REGISTRY_DIR / "nb_analysis" / "selected_features.json"
    with open(path) as f:
        return json.load(f)["track_b_features"]


def load_track_b_model_choice() -> str:
    """Load Track B model choice from Phase 3.2."""
    path = REGISTRY_DIR / "track_b_experiment" / "results.json"
    with open(path) as f:
        results = json.load(f)
    # Pick model with higher mean AUC
    lgbm_aucs = [v["AUC"] for v in results["lgbm"].values()]
    lr_aucs = [v["AUC"] for v in results["logistic"].values()]
    lgbm_mean = sum(lgbm_aucs) / len(lgbm_aucs) if lgbm_aucs else 0
    lr_mean = sum(lr_aucs) / len(lr_aucs) if lr_aucs else 0
    # Use logistic unless LightGBM beats it by > 3%
    if lgbm_mean > lr_mean + 0.03:
        return "lgbm"
    return "logistic"


def compute_v0c_scores(group_df: pl.DataFrame) -> np.ndarray:
    """Compute v0c formula scores for a single group."""
    def _minmax(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    da_rank = group_df["da_rank_value"].to_numpy().astype(np.float64)
    da_norm = 1.0 - _minmax(da_rank)

    rt_max = group_df.select(
        pl.max_horizontal("bin_80_cid_max", "bin_90_cid_max",
                          "bin_100_cid_max", "bin_110_cid_max")
    ).to_series().to_numpy().astype(np.float64)
    rt_norm = _minmax(rt_max)

    bf = group_df["bf_combined_12"].to_numpy().astype(np.float64)
    bf_norm = _minmax(bf)

    return 0.40 * da_norm + 0.30 * rt_norm + 0.30 * bf_norm


def train_track_b_model(
    train_df: pl.DataFrame,
    features: list[str],
    model_type: str,
) -> object:
    """Train Track B model on training data."""
    X = train_df.select(features).to_numpy().astype(np.float64)
    y = (train_df["realized_shadow_price"].to_numpy() > 0).astype(int)

    if model_type == "lgbm":
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 5,
            "scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0,
            "num_threads": 4, "verbose": -1,
        }
        ds = lgb.Dataset(X, label=y, feature_name=features, free_raw_data=False)
        model = lgb.train(params, ds, num_boost_round=200)
    else:
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
        model.fit(X, y)

    return model


def predict_track_b(model, df: pl.DataFrame, features: list[str], model_type: str) -> np.ndarray:
    """Score Track B candidates."""
    X = df.select(features).to_numpy().astype(np.float64)
    if model_type == "lgbm":
        return model.predict(X)
    else:
        return model.predict_proba(X)[:, 1]


def run_two_track_group(
    group_df: pl.DataFrame,
    track_a_model: str,
    track_b_model,
    track_b_features: list[str],
    track_b_model_type: str,
    r: int,
) -> dict:
    """Run two-track merge + evaluation for one (PY, quarter) group."""
    # Split into Track A and Track B
    track_a_df = group_df.filter(pl.col("cohort") == "established")
    track_b_df = group_df.filter(pl.col("cohort").is_in(["history_dormant", "history_zero"]))

    # Score Track A
    if track_a_model == "v0c":
        a_scores = compute_v0c_scores(track_a_df)
    elif track_a_model == "v3a":
        # v3a requires a pre-trained model passed in — caller must train per split
        # and pass the LightGBM model via track_b_model arg overload or separate param
        raise NotImplementedError(
            "v3a Track A scoring requires per-split LightGBM training. "
            "Implement after v0c baseline confirms approach viability (spec Phase 3.3.3)."
        )

    track_a_scored = track_a_df.with_columns(pl.Series("score", a_scores))

    # Score Track B
    b_scores = predict_track_b(track_b_model, track_b_df, track_b_features, track_b_model_type)
    track_b_scored = track_b_df.with_columns(pl.Series("score", b_scores))

    # Merge
    merged, top_k_idx = merge_tracks(track_a_scored, track_b_scored, k=K, r=r)

    # Evaluate
    metrics = evaluate_group(merged, k=K, top_k_override=top_k_idx)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-a", default="v0c", choices=["v0c", "v3a"])
    parser.add_argument("--holdout", action="store_true", help="Run on holdout groups")
    parser.add_argument("--r", type=int, default=None, help="Fixed R value (skip sweep)")
    parser.add_argument("--version", default=None, help="Version ID for registry save")
    args = parser.parse_args()

    t0 = time.time()

    track_b_features = load_track_b_features()
    track_b_model_type = load_track_b_model_choice()
    logger.info("Track B features: %s", track_b_features)
    logger.info("Track B model: %s", track_b_model_type)

    # Build model tables
    all_needed = set()
    for split_info in EVAL_SPLITS.values():
        for py in split_info["train_pys"] + split_info["eval_pys"]:
            for aq in AQ_QUARTERS:
                all_needed.add(f"{py}/{aq}")
    all_needed.discard("2025-06/aq4")
    model_table = build_model_table_all(sorted(all_needed))

    # Determine eval groups
    eval_groups = HOLDOUT_GROUPS if args.holdout else DEV_GROUPS
    r_values = [args.r] if args.r is not None else R_VALUES

    # Results: {R: {group: metrics}}
    all_results: dict[int, dict] = {}

    for r in r_values:
        all_results[r] = {}

        for eval_key, split_info in EVAL_SPLITS.items():
            target_split = "holdout" if args.holdout else "dev"
            if split_info["split"] != target_split:
                continue

            # Train Track B on training PYs (Track B population only)
            train_df = model_table.filter(
                pl.col("planning_year").is_in(split_info["train_pys"])
                & pl.col("cohort").is_in(["history_dormant", "history_zero"])
            )
            tb_model = train_track_b_model(train_df, track_b_features, track_b_model_type)

            # Eval on each quarter
            for py in split_info["eval_pys"]:
                for aq in AQ_QUARTERS:
                    key = f"{py}/{aq}"
                    if key not in eval_groups:
                        continue

                    gdf = model_table.filter(
                        (pl.col("planning_year") == py) & (pl.col("aq_quarter") == aq)
                    )
                    metrics = run_two_track_group(
                        gdf, args.track_a, tb_model,
                        track_b_features, track_b_model_type, r,
                    )
                    all_results[r][key] = metrics

    # Print results
    print(f"\n{'='*100}")
    print(f"  Two-Track Sweep: Track A={args.track_a}, Track B={track_b_model_type}")
    split_label = "HOLDOUT" if args.holdout else "DEV"
    print(f"  Split: {split_label}")
    print(f"{'='*100}\n")

    for r in r_values:
        print(f"\n--- R={r} ---")
        header = f"{'Group':<16} {'VC@50':>8} {'Recall@50':>10} {'Abs_SP@50':>10} {'NB12_Cnt':>9} {'NB12_SP':>8} {'NB12_R':>8}"
        print(header)
        print("-" * len(header))

        per_group = all_results[r]
        for g in eval_groups:
            if g not in per_group:
                continue
            m = per_group[g]
            print(
                f"{g:<16} {m.get(f'VC@{K}', 0):>8.4f} {m.get(f'Recall@{K}', 0):>10.4f} "
                f"{m.get(f'Abs_SP@{K}', 0):>10.4f} {m.get(f'NB12_Count@{K}', 0):>9d} "
                f"{m.get(f'NB12_SP@{K}', 0):>8.4f} {m.get(f'NB12_Recall@{K}', 0):>8.4f}"
            )

        # Mean
        vals = list(per_group.values())
        if vals:
            mean_vc = sum(m.get(f"VC@{K}", 0) for m in vals) / len(vals)
            mean_nb_cnt = sum(m.get(f"NB12_Count@{K}", 0) for m in vals) / len(vals)
            print(f"\n  Mean VC@50={mean_vc:.4f}, Mean NB12_Count@50={mean_nb_cnt:.1f}")

    # If holdout mode with specific R, do gate checks and save
    if args.holdout and args.r is not None and args.version:
        r = args.r
        per_group = all_results[r]

        # Gate check vs v0c
        baseline_metrics = load_metrics("v0c")
        gate_results = check_gates(
            candidate=per_group,
            baseline=baseline_metrics["per_group"],
            baseline_name="v0c",
            holdout_groups=HOLDOUT_GROUPS,
            gate_metrics=TWO_TRACK_GATE_METRICS,
        )

        # NB threshold check
        nb_results = check_nb_threshold(per_group, HOLDOUT_GROUPS)

        # Print gate results
        print(f"\n{'='*60}")
        print(f"  Gate Check vs v0c (TWO_TRACK_GATE_METRICS)")
        print(f"{'='*60}")
        for metric, gate in gate_results.items():
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"  {metric:<20} {status:>4}  wins={gate['wins']}/{gate['n_groups']}")

        print(f"\n  NB Threshold: {'PASS' if nb_results['passed'] else 'FAIL'} "
              f"(total={nb_results['total_count']}, min={nb_results['min_total_count']})")

        # Save to registry
        config = {
            "version": args.version,
            "track_a_model": args.track_a,
            "track_b_model": track_b_model_type,
            "track_b_features": track_b_features,
            "r": r, "k": K,
            "gate_metrics": TWO_TRACK_GATE_METRICS,
        }
        metrics_out = {"per_group": per_group}
        save_experiment(
            args.version, config, metrics_out,
            gate_results=gate_results,
            baseline_version="v0c",
            nb_gate_results=nb_results,
        )
        logger.info("Saved to registry/%s/", args.version)

    logger.info("Done (%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dev sweep**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c`
Expected: Prints R=0,5,10,15 results. Identify R* with best NB12_Count@50 vs VC@50 tradeoff.

- [ ] **Step 3: Analyze Pareto frontier**

From the sweep output:
- R=0 should match v0c baseline (NB12_Count=0)
- Higher R should increase NB12_Count but may decrease VC@50
- Select R* where gates still pass and NB12_Count > 0

- [ ] **Step 4: Run holdout validation with best R**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run python scripts/run_two_track_experiment.py --track-a v0c --holdout --r <R*> --version tt_v0c_r<R*>`

Expected: Gate results printed. Check:
1. VC@50, Recall@50, Abs_SP@50 gates pass vs v0c
2. NB threshold passes (NB12_Count@50 >= 3 across holdout)
3. NB12_SP@50 > 0 on at least 2/3 holdout quarters

- [ ] **Step 5: Commit**

```bash
git add scripts/run_two_track_experiment.py registry/tt_*/
git commit -m "phase3.3-3.4: two-track merge sweep and holdout validation

Sweeps R in {0,5,10,15} on dev, validates best R on holdout.
Gates on TWO_TRACK_GATE_METRICS + check_nb_threshold().
Track A: v0c, Track B: <model_type> binary classifier."
```

---

### Task 12: Final test suite verification and cleanup

**Files:**
- All test files

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 && PYTHONPATH=. uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify no regressions in existing evaluate_all behavior**

The key contract: `evaluate_all` calls `evaluate_group(group_df)` with defaults (k=50, no override). Verify that calling `evaluate_all` produces the same output shape as before by checking that existing registry entries can be reproduced.

- [ ] **Step 3: Final commit with all artifacts**

```bash
git add -A
git commit -m "phase3: NB Two-Track implementation complete

Phase 3.0: Infrastructure (gate cleanup, NB metrics, top_k_override, registry)
Phase 3.1: NB population analysis and Track B feature profiling
Phase 3.2: Track B binary classifier development
Phase 3.3: Merge policy sweep (R=0,5,10,15)
Phase 3.4: Holdout validation with gate checks"
```

---

## Execution Order and Dependencies

```
Task 1 (remove NB12_Recall@50)
  └─> Task 2 (TWO_TRACK_GATE_METRICS + gate_metrics param)
      └─> Task 3 (top_k_override in evaluate_group)
          └─> Task 4 (check_nb_threshold)
              └─> Task 5 (registry nb_gate_results)
                  └─> Task 6 (baseline_contract.json)
                      └─> Task 7 (NB supplement for v0c/v3a)
                          └─> Task 8 (NB analysis script) — Phase 3.1
                              └─> Task 9 (Track B experiment) — Phase 3.2
                                  └─> Task 10 (merge module) — Phase 3.3.1
                                      └─> Task 11 (two-track sweep + holdout) — Phase 3.3-3.4
                                          └─> Task 12 (final verification)
```

Tasks 1-7 are strictly sequential (infrastructure). Tasks 8 and 10 could run in parallel (analysis vs merge module), but 9 depends on 8's output, and 11 depends on both 9 and 10.

## Decision Points

After these tasks require human judgment:

1. **After Task 8** (Phase 3.1): Review AUC results, select Track B feature set. If all AUCs < 0.55, fall back to density-composite heuristic (spec Risk 1).

2. **After Task 9** (Phase 3.2): Choose LightGBM vs logistic. If LightGBM doesn't beat logistic by > 3% AUC, use logistic.

3. **After Task 11 Step 3** (Phase 3.3): Select R* from Pareto frontier. If even R=5 causes > 5% VC@50 regression, consider expanding K (spec Risk 2).

4. **After Task 11 Step 4** (Phase 3.4): Final pass/fail decision. If all gates pass + NB threshold met, the two-track candidate is the new champion.
