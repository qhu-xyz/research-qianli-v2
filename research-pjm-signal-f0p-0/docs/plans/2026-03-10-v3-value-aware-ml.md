# V3 Value-Aware ML Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement v3 ML variants that beat v0 formula on ALL metrics by combining enriched value-predictive features with value-aware LambdaRank labels.

**Architecture:** Add spice6 ml_pred loader for new features (binding_probability, predicted_shadow_price, hist_da, prob_exceed_100). Add log-value label mode to train.py. Run 4 variants (v3a-v3d) across all 6 PJM slices, comparing against v0 on 8 metrics.

**Tech Stack:** polars, lightgbm, numpy. Existing PJM ML pipeline modules.

**Design Spec:** `docs/specs/2026-03-10-v3-value-aware-ml-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `ml/config.py` | Modify | Add V3_FEATURES, V3_MONOTONE constants |
| `ml/spice6_loader.py` | Modify | Add `load_spice6_mlpred()` for ml_pred features |
| `ml/data_loader.py` | Modify | Enrich with ml_pred features + shadow_price_da |
| `ml/train.py` | Modify | Add `_log_value_labels()` for value-aware labels |
| `scripts/run_v3_ml.py` | Create | Run v3a-v3d variants for all 6 slices |

---

## Chunk 1: Core ML Module Changes

### Task 1: Add V3 feature/label configs to ml/config.py

**Files:**
- Modify: `ml/config.py:28-34`

- [ ] **Step 1: Add V3 feature and monotone constants after V10E definitions (line 34)**

```python
# ── V3 features (14, enriched with value-predictive signals) ──
V3_FEATURES = [
    "binding_freq_1", "binding_freq_3", "binding_freq_6", "binding_freq_12",
    "binding_freq_15", "v7_formula_score", "da_rank_value",
    "shadow_price_da", "binding_probability", "predicted_shadow_price",
    "prob_exceed_110", "prob_exceed_100", "constraint_limit", "hist_da",
]
V3_MONOTONE = [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 0, 1]
```

- [ ] **Step 2: Verify** `len(V3_FEATURES) == len(V3_MONOTONE) == 14`

### Task 2: Add spice6 ml_pred loader

**Files:**
- Modify: `ml/spice6_loader.py` (add function at end)

- [ ] **Step 1: Add `load_spice6_mlpred()` function after `load_spice6_density()`**

```python
def load_spice6_mlpred(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load spice6 ml_pred features for one month.

    Returns DataFrame with columns:
        constraint_id, flow_direction, binding_probability,
        predicted_shadow_price, hist_da, prob_exceed_100
    """
    from ml.config import SPICE6_MLPRED_BASE

    market_month = _delivery_month(auction_month, period_type)
    path = (
        Path(SPICE6_MLPRED_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"class_type={class_type}"
        / "final_results.parquet"
    )

    if not path.exists():
        logger.warning("ml_pred not found: %s", path)
        return pl.DataFrame()

    # Safe columns only (per CLAUDE.md — never use actual_*, error, abs_error)
    safe_cols = [
        "constraint_id", "flow_direction",
        "binding_probability", "predicted_shadow_price",
        "hist_da", "prob_exceed_100",
    ]
    df = pl.read_parquet(str(path))
    # Keep only columns that exist and are safe
    keep = [c for c in safe_cols if c in df.columns]
    df = df.select(keep)

    return df
```

Note: ml_pred joins on `(constraint_id, flow_direction)` — same as spice6 density. The data_loader already joins density on these columns, so ml_pred follows the same pattern.

### Task 3: Enrich data_loader with ml_pred features

**Files:**
- Modify: `ml/data_loader.py:67-79` (after spice6 density join, before realized DA join)

- [ ] **Step 1: Add ml_pred import at top of file**

Add after line 25 (`from ml.spice6_loader import load_spice6_density`):
```python
from ml.spice6_loader import load_spice6_mlpred
```

- [ ] **Step 2: Add ml_pred enrichment after density join (after line 79, before the realized DA join)**

Insert after the density fallback block (line 79) and before the `# Join realized DA` comment (line 81):

```python
    # Enrich with spice6 ml_pred features
    mlpred = load_spice6_mlpred(auction_month, period_type, class_type)
    if len(mlpred) > 0:
        df = df.join(mlpred, on=["constraint_id", "flow_direction"], how="left")
        mlpred_cols = [c for c in mlpred.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in mlpred_cols])
        n_matched = len(df.filter(pl.col("binding_probability") > 0))
        print(f"[data_loader] ml_pred: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no ml_pred for {auction_month}/{class_type}")
        for col in ["binding_probability", "predicted_shadow_price", "hist_da", "prob_exceed_100"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))
```

Note: `shadow_price_da` is ALREADY in the V6.2B parquet (loaded by `load_v62b_month`). No loading changes needed — just include it in the feature list.

### Task 4: Add log-value label mode to train.py

**Files:**
- Modify: `ml/train.py:37-65` (add new function after `_tiered_labels`)

- [ ] **Step 1: Add `_log_value_labels()` function after `_tiered_labels()` (after line 65)**

```python
def _log_value_labels(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Convert continuous shadow prices to log-scaled integer relevance labels.

    Non-binding (y=0) → 0. Binding → log1p(y) scaled and discretized to integers.
    Winsorizes at p99 within each group to prevent outlier domination.
    Produces ~10-15 distinct label levels (vs 4 in tiered mode).
    """
    y_label = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g in groups:
        chunk = y[offset : offset + g]
        binding_mask = chunk > 0
        n_binding = binding_mask.sum()
        if n_binding > 0:
            binding_vals = chunk[binding_mask]
            # Winsorize at p99
            p99 = np.percentile(binding_vals, 99)
            clipped = np.minimum(binding_vals, p99)
            # Log transform
            log_vals = np.log1p(clipped)
            # Scale to 1-100 range, then discretize
            max_log = log_vals.max()
            if max_log > 0:
                scaled = (log_vals / max_log * 99).astype(np.int32) + 1  # 1-100
            else:
                scaled = np.ones(n_binding, dtype=np.int32)
            labels = np.zeros(g, dtype=np.int32)
            labels[binding_mask] = scaled
            y_label[offset : offset + g] = labels
        offset += g
    return y_label
```

- [ ] **Step 2: Add `"log_value"` branch in `_train_lightgbm()` (line 80-83)**

Change the label mode selection block at lines 80-83 from:
```python
    if cfg.label_mode == "tiered":
        y_rank = _tiered_labels(y_train, groups_train)
    else:
        y_rank = _rank_transform_labels(y_train, groups_train)
```

To:
```python
    if cfg.label_mode == "tiered":
        y_rank = _tiered_labels(y_train, groups_train)
    elif cfg.label_mode == "log_value":
        y_rank = _log_value_labels(y_train, groups_train)
    else:
        y_rank = _rank_transform_labels(y_train, groups_train)
```

- [ ] **Step 3: Same change for validation label transform (line 119-122)**

Change:
```python
        if cfg.label_mode == "tiered":
            y_val_rank = _tiered_labels(y_val, groups_val)
        else:
            y_val_rank = _rank_transform_labels(y_val, groups_val)
```

To:
```python
        if cfg.label_mode == "tiered":
            y_val_rank = _tiered_labels(y_val, groups_val)
        elif cfg.label_mode == "log_value":
            y_val_rank = _log_value_labels(y_val, groups_val)
        else:
            y_val_rank = _rank_transform_labels(y_val, groups_val)
```

- [ ] **Step 4: Commit core module changes**

```bash
git add ml/config.py ml/spice6_loader.py ml/data_loader.py ml/train.py
git commit -m "feat: v3 enriched features + log-value labels for PJM ML"
```

---

## Chunk 2: Run Script and Experiments

### Task 5: Create run_v3_ml.py

**Files:**
- Create: `scripts/run_v3_ml.py`

- [ ] **Step 1: Write the v3 run script**

This script is adapted from `scripts/run_v2_ml.py` with these changes:
- Accepts `--variant v3a|v3b|v3c|v3d` to select experiment variant
- v3a: 9 features (V10E) + log_value labels
- v3b: 14 features (V3) + tiered labels
- v3c: 14 features (V3) + log_value labels
- v3d: v3c + two-stage hybrid (formula top-K preserved)
- Saves to `registry/{ptype}/{ctype}/{variant}/` and `holdout/{ptype}/{ctype}/{variant}/`
- Prints comparison vs v0 on all 8 metrics

Key differences from run_v2_ml.py:
1. Feature/label selection based on `--variant`
2. `shadow_price_da` is already loaded by `load_v62b_month()` — just include in feature list
3. ml_pred features (`binding_probability`, `predicted_shadow_price`, `hist_da`, `prob_exceed_100`) are now loaded by `load_v62b_month()` via the data_loader changes in Task 3
4. v3d adds a two-stage post-processing step after scoring

The `enrich_df()` function is identical to run_v2_ml.py (binding_freq + formula score). The new features come from the data_loader, not the enrichment step.

For v3d two-stage logic (in the eval loop, after `scores = predict_scores(model, X_test)`):
```python
if variant == "v3d":
    # Two-stage: preserve formula ranking for top-K
    K = 30
    formula_scores = -test_df["v7_formula_score"].to_numpy()  # negate: lower rank = better
    formula_order = np.argsort(formula_scores)[::-1]
    top_k_idx = set(formula_order[:K].tolist())
    # For top-K: use formula score (scaled above ML range)
    # For rest: use ML score
    ml_max = scores.max()
    for idx in top_k_idx:
        scores[idx] = ml_max + 1.0 + formula_scores[idx]  # formula always on top
```

- [ ] **Step 2: Run v3a (label change only) — quick screen on f0/onpeak first**

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
python scripts/run_v3_ml.py --variant v3a --ptype f0 --class-type onpeak
```

Expected: ~2-3 minutes. Check if VC@20 improves vs v2 (should improve since log labels teach value magnitude).

- [ ] **Step 3: Run v3b (feature change only) — quick screen on f0/onpeak**

```bash
python scripts/run_v3_ml.py --variant v3b --ptype f0 --class-type onpeak
```

Expected: ~2-3 minutes. Check if enriched features improve VC@20.

- [ ] **Step 4: Evaluate screen results — decide which changes help**

Compare v3a, v3b, v3c against v0 and v2 on f0/onpeak:
- If v3a helps VC@20: log labels are valuable → keep
- If v3b helps VC@20: enriched features are valuable → keep
- If neither helps alone: try v3c (combined) before concluding

- [ ] **Step 5: Run winning variant(s) on all 6 slices**

```bash
python scripts/run_v3_ml.py --variant v3c  # all 6 slices, dev + holdout
```

Expected: ~15-20 minutes total (6 slices × ~2.5 min each).

- [ ] **Step 6: Generate full comparison table**

Write a quick comparison script (or inline in run_v3_ml.py) that prints all 8 metrics × 6 slices × dev+holdout for v0, v2, v3c. Same format as the dedup comparison.

- [ ] **Step 7: If v3c loses VC@20 — run v3d (two-stage fallback)**

```bash
python scripts/run_v3_ml.py --variant v3d  # two-stage hybrid
```

v3d guarantees VC@20 >= v0 (formula ranking preserved at top) while capturing ML's recall advantage.

- [ ] **Step 8: Commit results and update registry**

```bash
git add registry/ holdout/ scripts/run_v3_ml.py
git commit -m "results: v3 value-aware ML experiments for PJM (dev + holdout)"
```

### Task 6: Analyze and decide champion

- [ ] **Step 1: Print full multi-metric comparison table**

Must show: v0 vs v2 vs v3-best on all 8 metrics, 6 slices, dev + holdout.

- [ ] **Step 2: Check success criteria from design spec**

v3 must beat v0 on ALL metrics simultaneously (mean across 6 slices, holdout):
- VC@20 > 0.553
- Recall@20 > 0.255
- Spearman > 0.246
- NDCG > 0.525
- No individual slice regression > 10% on any metric

- [ ] **Step 3: If v3 passes gates → promote to champion**

Update `registry/{ptype}/{ctype}/champion.json` for winning variant.

- [ ] **Step 4: If no variant beats v0 on VC@20 → document findings and propose next direction**

Possible next steps: different blend weights on deduped data, feature interactions, alternative ML architectures.

---

## Notes

### Critical Rules (from CLAUDE.md)
- **LightGBM num_threads=4** — always. Container has 64 CPUs, causes deadlock.
- **Production lag**: f0 uses lag=1, f1 uses lag=2. `collect_usable_months()` handles this.
- **Branch-name dedup**: `data_loader.py` deduplicates by branch_name (keep lowest rank_ori). This is already in place.
- **Memory safety**: `del df; gc.collect()` between months. Print `mem_mb()` at stages.
- **No leaky features**: `shadow_price`, `shadow_sign`, `rank`, `rank_ori`, `tier` are in `_LEAKY_FEATURES`.
- **spice6 ml_pred safe columns**: `binding_probability`, `predicted_shadow_price`, `hist_da`, `prob_exceed_*`. Never use `actual_*`, `error`, `abs_error`.

### Virtual Environment
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
```
