# Phase 4a Design: Value-Weighted Track B on Dormant Subset

**Date**: 2026-03-13
**Status**: Completed (Phase 4a executed, Track B champion)
**Depends on**: Phase 3 (two-track infrastructure, merge_tracks, evaluate_group top_k_override)

---

## 1. Problem Statement

Phase 3 proved that two-track merge works mechanically: reserved NB slots capture NB12 binders
that single-model scoring misses entirely (NB12_Count@50 went from 0 to 1-7 depending on R).
But the NB captures are almost entirely tiny-SP binders — NB12_SP@50 stays below 2% in nearly
every group, both dev and holdout. Reserving R slots costs 5-9% VC@50 while recovering almost
no value.

**Root cause**: Track B uses pure binary classification (`y = SP > 0`). A $0.02 binder and a
$500k binder are identical positives. The logistic regression ranks by P(bind), not by expected
binding value. So the top-R Track B candidates tend to be branches that are likely to bind at
all, not branches likely to bind with material SP.

**Secondary finding**: history_zero branches contribute 0 captured SP across all holdout runs.
They have identical density features (no constraint activity) and are indistinguishable to the
12-feature model. Mixing them into Track B training dilutes the signal from history_dormant
branches, which at least have some constraint activity pattern.

## 2. Scope

Phase 4a is narrowly scoped:

1. **Train Track B on history_dormant only** — exclude history_zero from training AND from
   reserved slot allocation. history_zero is reported separately as unsolved.
2. **Add value-aware sample weights** to the binary classification objective so that high-SP
   binders contribute more to the loss than low-SP binders.
3. **Compare two weighting schemes** on dev, validate winner on holdout.
4. **Evaluate at both K=50 and K=100** with independent R allocations. K=100 metrics are
   equally important as K=50 — the portfolio may use 100 constraints.

Phase 4a does NOT:
- Change Track A scoring (still v0c formula)
- Change the merge infrastructure (merge_tracks, evaluate_group top_k_override)
- Attempt to solve history_zero (that requires new feature engineering, not modeling tuning)
- Move to regression or LambdaRank on NB population

## 3. Weighting Schemes

### Scheme B: Tiered Weights (Reference)

Assign sample weights based on SP tier within the positive class:

```python
# Within each training group's positives:
#   SP == 0     -> weight = 1.0 (negatives)
#   SP > 0, bottom third  -> weight = 1.0
#   SP > 0, middle third  -> weight = 3.0
#   SP > 0, top third     -> weight = 10.0
```

Tier boundaries computed per training split (not per group) to avoid noise from small groups.
The 1/3/10 ratio is steep enough to make top-tier positives dominate the gradient without
completely ignoring low-tier ones. Rationale: this mirrors the tiered-label philosophy that
Track A uses successfully via LambdaRank.

### Scheme A: Continuous Weights (Challenger)

```python
weight = 1.0 + min(log1p(realized_shadow_price), 12.0)
# negatives: weight = 1.0
# positives: weight = 1.0 + log1p(SP), capped at 13.0
```

The `log1p(SP)` output is capped at 12.0, so the maximum weight is 13.0. `log1p(SP) = 12` at
SP ~$162k. This prevents a single $500k+ outlier from dominating training. The +1.0 base
ensures negatives still contribute to the loss.

### Class-Imbalance Correction

Value weights are applied ON TOP OF class-imbalance correction, not instead of it.

**Logistic**: `lr.fit(X, y, sample_weight=combined_weight)` with `class_weight="balanced"`.
sklearn multiplies `sample_weight` by the class-balanced correction internally.

**LightGBM**: Explicit per-sample class weight folded into the weight array. Do NOT use
`scale_pos_weight` param — it would double-count for positives:
```python
class_ratio = n_neg / n_pos
per_sample_class_w = np.where(y == 1, class_ratio, 1.0)
combined_weight = per_sample_class_w * value_weight
ds = lgb.Dataset(X, label=y, weight=combined_weight)
# params: do NOT set scale_pos_weight
```
This makes the per-sample gradient fully explicit and avoids interaction ambiguity between
`Dataset(weight=...)` and the `scale_pos_weight` training param.

## 4. Training Changes

### 4.1 Population: Dormant Only

```python
# Phase 3 (current):
train_df = model_table.filter(
    pl.col("cohort").is_in(["history_dormant", "history_zero"])
)

# Phase 4a:
train_df = model_table.filter(pl.col("cohort") == "history_dormant")
```

Track B scoring at inference time also applies only to history_dormant. history_zero branches
get score = 0 and are excluded from reserved slots.

### 4.2 Sample Weight Computation

New function in `scripts/run_phase4a_experiment.py`:

```python
def compute_sample_weights(
    sp: np.ndarray,
    scheme: str,  # "tiered" or "continuous"
) -> np.ndarray:
    """Compute value-aware sample weights for Track B training."""
    weights = np.ones(len(sp), dtype=np.float64)
    pos_mask = sp > 0

    if scheme == "tiered":
        if pos_mask.sum() > 0:
            pos_sp = sp[pos_mask]
            ranks = pos_sp.argsort().argsort()
            n = len(ranks)
            t1, t2 = n // 3, 2 * n // 3
            tier_w = np.where(ranks < t1, 1.0, np.where(ranks < t2, 3.0, 10.0))
            weights[pos_mask] = tier_w
    elif scheme == "continuous":
        weights[pos_mask] = 1.0 + np.minimum(np.log1p(sp[pos_mask]), 12.0)

    return weights
```

### 4.3 Model Training

Both logistic and LightGBM are tested (same model selection rule as Phase 3: use logistic
unless LightGBM beats it by > 3% AUC).

**Logistic**: `lr.fit(X, y, sample_weight=value_weights)` with `class_weight="balanced"`.
sklearn applies class_weight as a multiplier on sample_weight internally, so the combined
effect is `effective_weight = value_weight * (n_samples / (n_classes * n_per_class))`.

**LightGBM**: Per-sample class correction folded into weight array (see §3):
```python
class_ratio = n_neg / n_pos
per_sample_class_w = np.where(y == 1, class_ratio, 1.0)
combined_weight = per_sample_class_w * value_weights
ds = lgb.Dataset(X, label=y, weight=combined_weight)
```
Do NOT set `scale_pos_weight` in params — it is already encoded in `combined_weight`.

### 4.4 Features

Same 12 features as Phase 3 (from `registry/nb_analysis/selected_features.json`). No feature
changes in Phase 4a.

## 5. Evaluation Design

### 5.1 Dual-K Evaluation

Every experiment configuration is evaluated at both K=50 and K=100 with INDEPENDENT R
allocations:

- K=50: R sweep {0, 5, 10}
- K=100: R sweep {0, 10, 15, 20}

The script calls `merge_tracks()` separately for each K level and `evaluate_group()` with
the corresponding `top_k_override`. Metrics at each K level come from their own merge, not
from extrapolation.

### 5.2 Dev Sweep

For each weighting scheme (tiered, continuous) × model type (logistic, lgbm):
- Train on expanding-window dev splits (history_dormant only)
- Evaluate all (R50, R100) combos on dev groups
- Report at both K=50 and K=100

### 5.3 Comparison Metrics

Primary (must improve over Phase 3 binary baseline):
- **NB12_SP@50** — ratio of captured NB12 SP / total NB12 SP. This is the key bar.
- **NB12_SP@100** — same at K=100

Secondary (must not degrade materially):
- **VC@50**, **VC@100** — value capture at each K level
- **Abs_SP@50**, **Abs_SP@100**
- **NB12_Count@50**, **NB12_Count@100** — count should stay similar or improve

### 5.4 Holdout Validation

Winner from dev sweep (scheme × model) is validated on holdout with fixed (R50, R100).
Gate checks at both K levels vs v0c baseline.
NB threshold checked at both K=50 and K=100.

### 5.5 Cohort Reporting

Every table breaks out:
- **Dormant count in top-K**, **Dormant SP captured**
- **Zero-history count in top-K**, **Zero-history SP captured**
- history_zero is always 0 in Phase 4a reserved slots (excluded from Track B scoring)
  but may appear in Track A slots if v0c formula happens to rank them

### 5.6 Phase 3 Baseline

The Phase 3 binary (unweighted) result is the baseline. Registry versions:
- **`tt_v0c_r5_r15`**: holdout with R50=5, R100=15 — primary comparison target
- **`tt_v0c_r5_r20`**: holdout with R50=5, R100=20 — secondary

Dev baseline (from Phase 3 sweep, not saved to registry — re-run with `--track-a v0c`):
- R50=5: Mean NB12_SP@50 ≈ 0.011, Mean NB12_Count@50 = 1.1
- R100=15: Mean NB12_SP@100 ≈ 0.016, Mean NB12_Count@100 = 3.6

Phase 4a must show NB12_SP improvement at matched R values. The comparison script loads
Phase 3 holdout metrics via `load_metrics("tt_v0c_r5_r15")`.

## 6. Success Criteria

Phase 4a is successful if, on holdout at the selected (R50, R100):

1. **NB12_SP@50 > Phase 3 NB12_SP@50** at matched R50 — the weighted model captures more
   NB value, not just more NB count
2. **NB12_SP@100 > Phase 3 NB12_SP@100** at matched R100
3. **VC@50 does not degrade further** vs Phase 3 at same R50 (tolerance: < 0.01 absolute drop in mean VC@50)
4. **NB threshold passes** at both K=50 and K=100

If NB12_SP does not improve, the approach is not earning its complexity and Phase 4a is
declared negative.

## 6.1 Phase 4b Contingency: SP Regression on Dormant Subset

If Phase 4a is negative or shows only marginal NB12_SP improvement, Phase 4b explores
regression on the dormant subset. The Track B features (12 density bin features) describe
the forward-looking SP distribution — the same domain as the target (realized SP). This
makes a coherent regression problem: given simulated SP distribution, predict realized SP.

**Approach**: Two-stage model on dormant population:
1. **Stage 1**: Binary P(bind) filter (reuse Phase 4a model)
2. **Stage 2**: `log1p(realized_shadow_price)` regression on predicted-positive subset
3. **Ranking score**: `P(bind) × E[SP | bind]`

This avoids the zero-inflation problem (92% zeros dominating a single-stage regression)
while capturing binding strength. The two-stage approach lets each model focus on what it
does well: Stage 1 separates binders from non-binders, Stage 2 separates high-SP from
low-SP among predicted binders.

Phase 4b is out of scope for this implementation cycle.

## 7. Script Design

One script: `scripts/run_phase4a_experiment.py`

```
Usage:
    # Dev sweep (all schemes × models × R combos)
    PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c

    # Holdout validation
    PYTHONPATH=. uv run python scripts/run_phase4a_experiment.py --track-a v0c \
        --holdout --scheme tiered --r50 5 --r100 15 --version p4a_tiered_r5_r15
```

Dev sweep output: one table per (scheme, K) showing all R values, with dormant/zero breakdown.
Holdout output: gate checks at both K levels, NB threshold at both K levels, registry save.

### Diff from Phase 3 `run_two_track_experiment.py`:
1. `train_track_b_model()` gains `scheme` param, computes and passes sample weights
2. Training population: `pl.col("cohort") == "history_dormant"` (was `is_in(["history_dormant", "history_zero"])`)
3. `run_two_track_group()`: change `track_b_df` filter from `cohort.is_in(["history_dormant", "history_zero"])` to `cohort == "history_dormant"` — history_zero is excluded from Track B scoring at inference time
4. `run_two_track_group()` already handles dual-K from the Phase 3 update
5. Dev sweep iterates over `schemes = ["tiered", "continuous"]`
6. Phase 3 baseline results loaded from `registry/tt_v0c_r5_r15/` for side-by-side comparison

## 8. Files Changed

| File | Change |
|------|--------|
| `scripts/run_phase4a_experiment.py` | NEW — main experiment script |
| `ml/evaluate.py` | No changes (Phase 3 already has K-aware check_nb_threshold) |
| `ml/merge.py` | No changes |
| `ml/config.py` | No changes |
| `registry/phase4a_*/` | NEW — experiment results |
