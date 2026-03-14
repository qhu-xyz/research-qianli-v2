# V4 Comprehensive Redesign: Balanced Metrics + New Constraint Detection

**Date**: 2026-03-11
**Status**: Revised (addressing review feedback)

## Motivation

Three findings from the v0-v3 investigation fundamentally change our approach:

1. **The V6.2B formula weights are wrong for PJM.** The default 0.60/0.30/0.10 (da/dmix/dori) underweights `da_rank_value`. Optimal is ~0.80/0.15/0.05, yielding +4.1% VC@20, +8.6% Recall@20, +11.4% NDCG on holdout — a free improvement from changing three numbers.

2. **We over-focused on VC@20.** All N-cutoffs matter equally (N=20, 50, 100). All metric families matter (VC, Recall, NDCG, Spearman). The gate system currently only has Group A gates on 6 metrics. We need consistent improvement across the board, not a single-metric optimization.

3. **New constraint detection is a blind spot.** 27% of binding constraints each month haven't bound in the prior 6 months ("new@6"). They account for 18% of total value. The formula ranks them at **median position ~190/400** — barely above random (new_Recall@20 = 0.057 vs random 0.049). ML features (density, binding_probability, predicted_shadow_price) don't depend on binding history and should dominate here.

## What Changes

### 1. New Baseline: v0b (Optimized Formula)

**File**: `scripts/run_v0_formula_baseline.py`

Replace V6.2B's 0.60/0.30/0.10 with 0.80/0.15/0.05 as the new `v0b` baseline. This becomes the bar every ML version must beat.

**Impact**: The old v0 (0.60/0.30/0.10) was a straw man. ML versions that appeared to "nearly match" it were actually losing to a weak baseline. v0b raises the bar by +4-11% on all metrics.

**Registry changes**:
- Run v0b on all 6 slices (dev + holdout)
- Save to `registry/{ptype}/{ctype}/v0b/` and `holdout/{ptype}/{ctype}/v0b/`
- Recalibrate gates from v0b (not v0)
- Promote v0b as new champion in all 6 slices

### 2. New Constraint Cohort Metrics

**Cohort taxonomy** (addressing review finding 2 — single `new@6` is too coarse):

| Cohort | Definition | Typical share of binders |
|--------|------------|:---:|
| **BF-zero** | `binding_freq_6 == 0` — not bound in prior 6 months | ~27% |
| **BF-positive** | `binding_freq_6 > 0` — has recent binding history | ~73% |
| **History-zero** | branch has NEVER appeared in ANY binding set | subset of BF-zero |

**BF-zero** is the operationally useful gate — it captures both truly novel and lapsed constraints. ML features (density, binding_probability, predicted_shadow_price) don't depend on binding history and should dominate here. History-zero is a diagnostic subset for deeper analysis but too small/noisy for gating.

**File**: `ml/evaluate.py`

Add 4 performance metrics + 3 context metrics to `evaluate_ltr()`:

Performance (gated):
- `NewBind_Recall@20`: Of BF-zero binders, what fraction appears in model's top-20?
- `NewBind_Recall@50`: Same at top-50.
- `NewBind_VC@20`: Value of BF-zero binders captured in top-20 / total BF-zero binder value.
- `NewBind_VC@50`: Same at top-50.

Context (reported, not gated — addressing review finding 5):
- `n_new`: count of BF-zero binders this month
- `new_value_share`: BF-zero binder value / total binder value
- `new_row_share`: BF-zero rows / total rows

These require passing a `new_mask` (boolean array) into `evaluate_ltr()`. The function signature changes:

```python
def evaluate_ltr(
    actual_shadow_price: np.ndarray,
    scores: np.ndarray,
    new_mask: np.ndarray | None = None,  # NEW: True for BF-zero constraints
) -> dict:
```

When `new_mask` is None, new-bind metrics are omitted (backward compatible).

### 3. Expanded Gate System

**Files**: `registry/{ptype}/{ctype}/gates.json` (all 6), `ml/compare.py`

Current gates (Group A, 6 metrics):
```
VC@20, VC@100, Recall@20, Recall@50, Recall@100, NDCG
```

New gates (Group A, 12 metrics — all recalibrated from v0b):
```
VC@20, VC@50, VC@100,
Recall@20, Recall@50, Recall@100,
NDCG, Spearman,
NewBind_Recall@20, NewBind_Recall@50, NewBind_VC@20, NewBind_VC@50
```

Changes:
- **Add VC@50 to Group A** (currently Group B) — N=50 matters as much as N=20 and N=100
- **Add Spearman to Group A** (currently Group B) — overall rank correlation is a quality signal
- **Add 4 NewBind gates to Group A** — new constraint detection is a first-class requirement (VC@50 included per review finding 3: "all cutoffs matter" must be consistent)
- **Recalibrate all floors from v0b** — floors are mean - 1 std of v0b dev performance
- **Recalibrate all tail floors from v0b** — tail floors are v0b dev p5

The gate check in `ml/compare.py` already handles arbitrary gate names — no code change needed there, only `gates.json` content changes.

### 4. New Constraint Mask Pipeline

**File**: `ml/data_loader.py` or new `ml/new_constraint.py`

Add a function to compute the BF-zero mask for a given evaluation month:

```python
def compute_new_mask(
    branch_names: list[str],
    eval_month: str,
    binding_sets: dict[str, set[str]],
    lookback: int = 6,
) -> np.ndarray:
    """Return boolean mask: True if branch has NOT bound in prior `lookback` months (BF-zero)."""
```

This is called in the evaluation loop of every run script. The binding_sets dict is already loaded by `load_all_binding_sets()` — no new data loading required.

**Important**: The new mask uses the same lag rules as BF computation. For eval month M with f0 lag=1, binding history is checked through M-2 (same cutoff as BF). This prevents leakage.

Also add a history-zero diagnostic function for deeper analysis:

```python
def compute_history_zero_mask(
    branch_names: list[str],
    eval_month: str,
    binding_sets: dict[str, set[str]],
) -> np.ndarray:
    """Return boolean mask: True if branch has NEVER appeared in any binding set before eval_month."""
```

History-zero is a subset of BF-zero. It's reported but not gated (too few per month for stable gate floors).

### 5. Run Scripts: Pass new_mask Through Evaluation

**Files**: `scripts/run_v0_formula_baseline.py`, `scripts/run_v2_ml.py`, `scripts/run_v3_ml.py`

Every script that calls `evaluate_ltr()` must now also:
1. Compute `new_mask` using `compute_new_mask()`
2. Pass it to `evaluate_ltr(actual, scores, new_mask=new_mask)`
3. The new metrics are automatically included in per-month results

### 6. V4 ML: Feature-Enriched Model on Stronger Baseline

**File**: `scripts/run_v4_ml.py` (new, replaces v3)

Uses the v3b feature set (14 features including shadow_price_da, binding_probability, predicted_shadow_price) with tiered labels (log_value showed no improvement over tiered). Blend weights remain 0.80/0.15/0.05 for v7_formula_score.

Key difference from v3b: evaluated against v0b (not v0), with new constraint gates.

The hypothesis: v3b already beats v0 on Recall, Spearman, and VC@100. With the stronger v0b baseline, v3b may still lose on VC@20 for known constraints — but it should dominate on NewBind metrics. If the signal is good enough on new constraints, the overall value proposition is clear even with a small VC@20 trade-off on known constraints.

### 7. Inference Pipeline Update

**File**: `v70/inference.py`

Update `BLEND_WEIGHTS` to use 0.80/0.15/0.05 (currently has the old smoothed blends from v2 era). The inference pipeline produces the signal that gets deployed — it must use the optimized formula weights regardless of whether ML beats v0b.

## File-by-File Change Summary

| File | Change | Size |
|------|--------|------|
| `ml/evaluate.py` | Add `new_mask` param + 4 NewBind perf metrics + 3 context metrics | ~40 lines |
| `ml/data_loader.py` | Add `compute_new_mask()` function | ~20 lines |
| `ml/config.py` | Already has V3_FEATURES/V3_MONOTONE (done) | 0 |
| `ml/spice6_loader.py` | Already has `load_spice6_mlpred()` (done) | 0 |
| `ml/train.py` | Already has `_log_value_labels()` (done) | 0 |
| `scripts/run_v0_formula_baseline.py` | Add v0b blend + new_mask eval | ~20 lines |
| `scripts/run_v4_ml.py` | New script (adapted from run_v3_ml.py) | ~300 lines |
| `registry/*/gates.json` (×6) | Recalibrate from v0b, add 6 new gates (12 total) | config |
| `registry/*/champion.json` (×6) | Promote v0b | config |
| `v70/inference.py` | Update BLEND_WEIGHTS | 6 lines |
| `docs/plans/` | This document | doc |

## Execution Order

1. **Modify `ml/evaluate.py`** — add new_mask + NewBind metrics
2. **Add `compute_new_mask()` to `ml/data_loader.py`**
3. **Modify `scripts/run_v0_formula_baseline.py`** — add `--blend` flag, integrate new_mask
4. **Run v0b baseline** (0.80/0.15/0.05) on all 6 slices, dev + holdout, with NewBind metrics
5. **Recalibrate gates** from v0b results (all 12 metrics)
6. **Promote v0b** as champion in all 6 slices
7. **Create `scripts/run_v4_ml.py`** — 14 features, tiered labels, new_mask eval, temporal reporting
8. **Run v4 ML** on all 6 slices, dev + holdout
9. **Compare v4 vs v0b** — must pass all 12 gates including NewBind; report 2024 vs 2025 separately
10. **Update `v70/inference.py`** with 0.80/0.15/0.05 blend weights
11. **Commit all changes**

## Success Criteria

A version is champion if it passes ALL 12 Group A gates on holdout (mean, tail, no-regression):

| Gate | Metric | Calibrated From |
|------|--------|:---:|
| G1 | VC@20 | v0b |
| G2 | VC@50 | v0b |
| G3 | VC@100 | v0b |
| G4 | Recall@20 | v0b |
| G5 | Recall@50 | v0b |
| G6 | Recall@100 | v0b |
| G7 | NDCG | v0b |
| G8 | Spearman | v0b |
| G9 | NewBind_Recall@20 | v0b |
| G10 | NewBind_Recall@50 | v0b |
| G11 | NewBind_VC@20 | v0b |
| G12 | NewBind_VC@50 | v0b |

If v4 ML beats v0b on NewBind gates but loses on some overall VC gates, this is a valid trade-off that the user decides — not an automatic promotion. The gates prevent silent regressions but don't auto-promote.

### Temporal Segmentation (addressing review finding 4)

The holdout spans 2024-01 to 2025-12. All comparison reports MUST show:
- **2024 subset** (months in 2024)
- **2025 subset** (months in 2025)
- **Full holdout** (aggregate)

This is reporting, not gating — too few months per year for stable gate floors. But if 2025 materially degrades vs 2024 on any metric, this is flagged for investigation before promotion.

Additionally, per-month context metrics (`n_new`, `new_value_share`, `new_row_share`) are reported to detect cohort-size drift across years.

## Risks

| Risk | Mitigation |
|------|-----------|
| v0b gate floors too high for ML | Floors are v0b mean - 1std, with 2% noise tolerance |
| NewBind metrics noisy (few new binders/month) | Use 6-month lookback (avg 14 new/month), aggregate across 24+ holdout months |
| new_mask computation leaks future binding | Same lag rules as BF: cutoff = prev_month(eval_month) |
| 14-feature model overfits | Monotone constraints, regularization, explicit comparison vs 9-feature |
| Inference blend change affects production | Test v0b formula on holdout first — already validated above |
