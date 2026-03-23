# NB V3: Targeted Improvements Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the per-ctype NB model through 4 targeted changes: more training data (2020), class-specific training features, label design, and feature refinement. Run as a controlled ablation against V2 baseline.

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

**Implementation**: Build class-specific tables for ALL PYs (not just eval years) and extract training features from them. This is slower (~2× data build time) but eliminates the feature mismatch.

**Risk**: `build_class_model_table` for 2020/2021 may fail if NB detection or GT data is incomplete for older PYs. Need to test first.

### Change 3: Label design variants

**Current**: Per-group tertiles (0/1/2/3) of positive SP within each (PY, aq) group.
**Proposed variants:**

| Variant | Label | LGB objective |
|---------|-------|--------------|
| V2 (baseline) | Tertile 0/1/2/3 | lambdarank |
| log1p | `log1p(SP)` continuous | lambdarank |
| binary_sqrt | 0/1 bind/nonbind | binary, weight=`sqrt(SP)` for binders |
| tiered_weighted | 0/1/2/3 | lambdarank, weight=[1,1,3,10] for tiers |

**Why**: Current tertiles treat a $5K binder and a $100K binder as only one tier apart. The business objective cares disproportionately about high-dollar dormant hits. `log1p` continuous relevance or tiered weights may better capture this.

### Change 4: Feature refinement

**Current**: `count_active_cids` is #1 feature (19% onpeak, 33% offpeak). 4 density bins use only `max` across CIDs.
**Proposed variants:**

| Variant | Change |
|---------|--------|
| -count_active | Drop `count_active_cids` |
| +active_ratio | Replace with `count_active_cids / count_cids` |
| +top2_mean | Add `top2_mean` for bins 80/90/100/110 (mean of top 2 CID values per branch) |

**Why**: `count_active_cids` may be too blunt (proxy for "big branch" rather than "binding-prone branch"). `active_ratio` normalizes by branch size. `top2_mean` is more robust than `max` for branches with many CIDs.

**Implementation for top2_mean**: Extend `_level2_collapse` in `ml/data_loader.py` or compute in experiment script from raw density.

---

## Experiment structure

**Phase 1**: Verify 2020 data availability. Build tables for 2020-06 (combined + class-specific).

**Phase 2**: Run ablation matrix. Each row is one experiment:

| # | 2020? | Class-specific train? | Labels | Features | Name |
|---|-------|----------------------|--------|----------|------|
| 0 | No | No | Tertile | Baseline 8 | V2_baseline |
| 1 | **Yes** | No | Tertile | Baseline 8 | +2020 |
| 2 | Yes | **Yes** | Tertile | Baseline 8 | +class_train |
| 3 | Yes | Yes | **log1p** | Baseline 8 | +log1p |
| 4 | Yes | Yes | **binary_sqrt** | Baseline 8 | +binary_sqrt |
| 5 | Yes | Yes | **tiered_weighted** | Baseline 8 | +tiered_wt |
| 6 | Yes | Yes | Best label | **-count_active** | -count |
| 7 | Yes | Yes | Best label | **+active_ratio** | +ratio |
| 8 | Yes | Yes | Best label | **+top2_mean** | +top2 |

Each experiment trains 2 models (onpeak/offpeak) × 3 eval years = 6 models. Total: 54 models.

**Phase 3**: Report per (ctype, year, K) with NB-only and R30 full-universe metrics. Compare each variant against V2_baseline.

## Success criteria

A variant is a **win** if it improves NB_SP at same or better VC for at least 3 out of 4 (ctype × K) aggregate cells, with no regression worse than -1pp VC on any cell.

## Files

| File | Action |
|------|--------|
| `scripts/nb_v3_ablation.py` | Create — full ablation script |
| `tests/test_nb_v3.py` | Create — test new label functions, top2_mean, active_ratio |
| `registry/onpeak/nb_v3/` | Create — winner's metrics |
| `registry/offpeak/nb_v3/` | Create — winner's metrics |
| `docs/2026-03-24-nb-v3-ablation-report.md` | Create — results |

No `ml/` changes unless top2_mean proves valuable (then extend `_level2_collapse`).
