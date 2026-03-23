# NB V3: Targeted Improvements Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the per-ctype NB model through 4 targeted changes: more training data (2020), class-specific training features, label/objective design, and feature refinement. Run as a controlled ablation against V2 baseline.

**Architecture:** Same V2 framework (2 per-ctype models, class-specific eval via `build_class_model_table`). Changes are additive — each variant builds on the previous winner.

---

## Current V2 Baseline (to beat)

| Metric | Onpeak K=400 R30 | Offpeak K=400 R30 |
|--------|------------------|-------------------|
| VC | 0.703 | 0.782 |
| dVC vs v0c | +0.4pp | -0.5pp |
| NB_SP | $56K | $50K |
| NB_bind | 13.2 | 14.8 |

## Changes to test (ordered by expected impact)

### Change 1: Add 2020-06 to training

**Current**: Train starts at 2021-06 (4 PYs max for 2025 holdout).
**Proposed**: Train starts at 2020-06 (5 PYs max).

**Why**: More training data is the single most reliable improvement for small-sample ranking models. The V2 NB model has only ~22K dormant training rows with 7% binder rate — adding 2020 gives ~25-30% more data.

**Implementation**: Add `"2020-06"` to `all_pys`. The combined builder `build_model_table("2020-06", aq)` must work — verify data exists.

**Pre-flight check (MUST pass before any ablation):**
```python
# Verify 2020-06 data builds for combined + both class-specific builders
from ml.features import build_model_table
from ml.phase6.features import build_class_model_table

for aq in ["aq1", "aq2", "aq3"]:
    t = build_model_table("2020-06", aq)
    assert len(t) > 1000, f"2020-06/{aq} combined too small: {len(t)}"
    for ct in ["onpeak", "offpeak"]:
        ct_t = build_class_model_table("2020-06", aq, ct)
        assert len(ct_t) > 1000, f"2020-06/{aq}/{ct} class-specific too small: {len(ct_t)}"
    print(f"2020-06/{aq}: combined={len(t)}, onpeak={len(ct_t)}")
```
If this fails, the experiment falls back to 2021+ only and this change is skipped.

**Eval splits after change:**
| Eval | Train PYs |
|------|-----------|
| 2023-06 | 2020, 2021, 2022 |
| 2024-06 | 2020, 2021, 2022, 2023 |
| 2025-06 | 2020, 2021, 2022, 2023, 2024 |

### Change 2: Class-specific training features

**Current**: NB model trains on combined `shadow_price_da` and `da_rank_value` from `build_model_table`.
**Proposed**: Train on class-specific versions from `build_class_model_table`.

**Why**: The eval already uses class-specific features. Training on combined creates a train/eval feature mismatch. For dormant branches with >12mo stale history, the combined and per-ctype values may differ (a branch could have old onpeak history but no offpeak history).

**Implementation**: Build class-specific tables for ALL PYs (not just eval years) and extract training features from them. This doubles the data build time but eliminates the feature mismatch.

**Pre-flight check**: Included in Change 1's check — `build_class_model_table` must succeed for all training PYs. If older PYs fail (incomplete NB detection or GT), fall back to combined features for those PYs only.

### Change 3: Label/objective design

**Current**: Per-group tertiles (0/1/2/3) of positive SP within each (PY, aq) group, LambdaRank objective.

Two distinct modeling families to test:

**Family A: Ranking with value-aware relevance (LambdaRank)**
| Variant | Label | Notes |
|---------|-------|-------|
| V2 (baseline) | Tertile 0/1/2/3 | Current |
| scaled_log1p | `min(255, round(log1p(SP) / log1p(max_SP_in_group) * 255))` | Bounded 0-255 continuous relevance per group. LambdaRank treats higher label = more relevant. Must be integer. |
| tiered_weighted | 0/1/2/3 with sample weights [1,1,3,10] | Same labels, disproportionate weight on top-tier binders |

**Family B: Binary detection with value weighting (Binary classification)**
| Variant | Label | Objective | Weight |
|---------|-------|-----------|--------|
| binary_sqrt | 0/1 | `binary` | `sqrt(SP)` for binders, 1.0 for non-binders × class imbalance correction |

**Why separate families**: Ranking (Family A) optimizes pairwise ordering — "rank the $100K binder above the $5K binder." Detection (Family B) optimizes "find binders vs non-binders" and uses SP as importance weight. These are different questions and should not be compared as if they're label tweaks of the same model.

### Change 4: Feature refinement

**Current**: `count_active_cids` is #1 feature (19% onpeak, 33% offpeak). 4 density bins use only `max` across CIDs.

| Variant | Change | Implementation |
|---------|--------|---------------|
| -count_active | Drop `count_active_cids` | Remove from feature list |
| +active_ratio | Replace with `count_active_cids / count_cids` | Compute in script: `active_ratio = count_active_cids / max(count_cids, 1)` |
| +top2_mean | Add `top2_mean` for bins 80/90/100/110 | See below |

**top2_mean implementation (exact)**:

Compute in the experiment script from raw density (Option B — no `ml/` changes):

```python
def compute_top2_mean(planning_year, aq_quarter, bins=["80","90","100","110"]):
    """For each branch and each bin, compute mean of top-2 CID values."""
    # Load raw density → Level 1 (mean across outage dates per CID)
    # Join bridge → get branch_name per CID
    # For each (branch, bin): sort CID values descending, take mean of top 2
    # If branch has only 1 CID, top2_mean = that CID's value

    # Returns DataFrame: branch_name, top2_bin_80, top2_bin_90, top2_bin_100, top2_bin_110
```

Requires loading raw density + bridge, same as `compute_density_features()` in `nb_feature_ablation.py`. Reuse that loading path.

---

## Experiment structure

### Phase 0: Pre-flight checks
- [ ] Verify `build_model_table("2020-06", aq)` works for aq1/2/3
- [ ] Verify `build_class_model_table("2020-06", aq, ct)` works for aq1/2/3 × on/offpeak
- [ ] If 2020 fails, document why and adjust matrix to start at 2021

### Phase 1: Run ablation matrix

Each row is one experiment. Rows 0-2 are infrastructure changes. Rows 3-5 are label/objective (pick winner). Rows 6-8 are feature refinement (apply to label winner).

| # | 2020? | Class-specific train? | Objective | Labels/Weights | Features | Name |
|---|-------|----------------------|-----------|---------------|----------|------|
| 0 | No | No | lambdarank | Tertile 0/1/2/3 | Baseline 8 | **V2_baseline** |
| 1 | **Yes** | No | lambdarank | Tertile 0/1/2/3 | Baseline 8 | +2020 |
| 2 | Yes | **Yes** | lambdarank | Tertile 0/1/2/3 | Baseline 8 | +class_train |
| 3 | Yes | Yes | lambdarank | **scaled_log1p 0-255** | Baseline 8 | +log1p |
| 4 | Yes | Yes | **binary** | **0/1, wt=sqrt(SP)** | Baseline 8 | +binary_sqrt |
| 5 | Yes | Yes | lambdarank | **Tertile, wt=[1,1,3,10]** | Baseline 8 | +tiered_wt |
| 6 | Yes | Yes | Best obj | **-count_active_cids** | 7 features | -count |
| 7 | Yes | Yes | Best obj | Best labels | **+active_ratio** | +ratio |
| 8 | Yes | Yes | Best obj | Best labels | **+top2_mean** | +top2 |

Each experiment trains 2 models (onpeak/offpeak) × 3 eval years = 6 models. Total: 54 models.

**"Best obj"**: Rows 6-8 use the winning objective/label from rows 3-5 (or V2 tertile if none improve).

### Phase 2: Report

Per (variant, ctype, year, K) with:
- NB-only: VC, Rec, NB_SP, NB_bind (K=50, K=100)
- Full universe: R30 at K=200 and K=400 — VC, NB_SP, NB_bind
- Delta vs V2_baseline for every cell

## Success criteria

A variant is a **win** if:
1. Improves aggregate NB_SP at same or better VC for at least 3 out of 4 (ctype × K) aggregate cells
2. No regression worse than -1pp VC on any aggregate cell
3. **Per-year guard**: No regression worse than -2pp VC on any individual (ctype × year × K) cell

## Artifact plan

**All 9 variants saved** (not just the winner):

| Path | Contents |
|------|----------|
| `registry/onpeak/nb_v3/{variant_name}/metrics.json` | Per-(eval_py, aq) combo metrics |
| `registry/offpeak/nb_v3/{variant_name}/metrics.json` | Same for offpeak |
| `registry/onpeak/nb_v3/{variant_name}/config.json` | Features, labels, objective, hyperparams |
| `registry/offpeak/nb_v3/{variant_name}/config.json` | Same |
| `registry/onpeak/nb_v3/comparison.json` | All variants' aggregate metrics side-by-side |
| `registry/offpeak/nb_v3/comparison.json` | Same |

Winner is symlinked or noted in `registry/{ct}/nb_v3/champion.json`.

## Files

| File | Action |
|------|--------|
| `scripts/nb_v3_ablation.py` | Create — full ablation script with all 9 variants |
| `tests/test_nb_v3.py` | Create — tests for: 2020 data availability, class-specific builder on older PYs, scaled_log1p label bounds, binary_sqrt weights, active_ratio computation, top2_mean computation |
| `registry/onpeak/nb_v3/` | Create — all 9 variant results |
| `registry/offpeak/nb_v3/` | Create — all 9 variant results |
| `docs/2026-03-24-nb-v3-ablation-report.md` | Create — full comparison report |

No `ml/` changes unless top2_mean proves valuable (then extend `_level2_collapse`).
