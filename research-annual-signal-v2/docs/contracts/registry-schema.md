# Registry Schema Contract — MISO Annual

**Date**: 2026-03-24

Every promoted model, benchmark adapter, and deployment policy must have a registry entry with the required file shape defined here.

---

## 1. Required files

Every registry entry directory must contain:

| File | Required | Contents |
|------|----------|----------|
| `spec.json` | Yes | Model/benchmark/policy identity and provenance |
| `metrics.json` | Yes | Evaluation results at base grain |
| `summary.md` | No | Human-readable notes |
| `artifacts/` | No | Trained model files, predictions, feature importance |

---

## 2. spec.json schema

### 2.1 Model spec

```json
{
  "spec_type": "model",
  "model_id": "miso_annual_bucket_6_20_v1",
  "market": "miso",
  "product": "annual",
  "class_type": "onpeak",
  "market_round": 1,
  "universe_id": "miso_annual_branch_active_v1",
  "feature_recipe_id": "miso_annual_bucket_features_v1",
  "label_recipe_id": "miso_annual_bucket_5tier_v1",
  "objective": "lambdarank_ndcg",
  "train_window": {
    "type": "expanding",
    "train_pys": ["2018-06", "2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06"],
    "eval_pys": ["2025-06"]
  },
  "eval_quarters": ["aq1", "aq2", "aq3"],
  "round_sensitivity": "r1_only",
  "code_commit": "589d100",
  "cache_provenance": {
    "density_limits": "data/collapsed/_r1_ keys via load_collapsed(market_round=1)",
    "history": "data/realized_da/ (monthly) + /opt/tmp/qianli/realized_da_daily/ (daily)",
    "model_tables": "data/nb_cache/ (legacy R1-only, no round in key)"
  }
}
```

### 2.2 Benchmark spec

```json
{
  "spec_type": "benchmark",
  "benchmark_id": "miso_annual_v44_published_v1",
  "market": "miso",
  "product": "annual",
  "class_type": "onpeak",
  "market_round": 1,
  "universe_id": "miso_annual_v44_published_v1",
  "signal_path": "TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1",
  "rank_direction": "ascending",
  "round_sensitivity": "r1_only",
  "eval_quarters": ["aq1", "aq2", "aq3"]
}
```

### 2.3 Deployment policy spec

```json
{
  "spec_type": "policy",
  "policy_id": "miso_annual_r30_v1",
  "description": "170 v0c + 30 Bucket_6_20 dormant at K=200; 350+50 at K=400",
  "primary_model_id": "miso_annual_v0c_formula_v1",
  "secondary_model_id": "miso_annual_bucket_6_20_v1",
  "allocation": {
    "K_200": {"primary": 170, "secondary_dormant": 30},
    "K_400": {"primary": 350, "secondary_dormant": 50}
  },
  "dormant_definition": "class-specific BF_12 == 0",
  "market_round": 1,
  "round_sensitivity": "r1_only"
}
```

---

## 3. Required spec.json fields

| Field | Type | Required for | Description |
|-------|------|-------------|-------------|
| `spec_type` | string | all | `"model"`, `"benchmark"`, or `"policy"` |
| `model_id` / `benchmark_id` / `policy_id` | string | all | Unique identifier |
| `market` | string | all | `"miso"` or `"pjm"` |
| `product` | string | all | `"annual"` |
| `class_type` | string | all | `"onpeak"` or `"offpeak"` |
| `market_round` | int | all | Explicit round (1, 2, or 3) |
| `round_sensitivity` | string | all | `"r1_only"`, `"round_aware"`, or `"round_independent"` |
| `universe_id` | string | model, benchmark | From universe catalog |
| `feature_recipe_id` | string | model | From feature recipes |
| `label_recipe_id` | string | model | Label definition |
| `objective` | string | model | Training objective |
| `train_window` | object | model | Train/eval PY split |
| `eval_quarters` | list[string] | all | Quarters evaluated |
| `code_commit` | string | model | Git commit of training code |
| `cache_provenance` | object | model | Which caches were used |
| `signal_path` | string | benchmark | NFS signal path |
| `rank_direction` | string | benchmark | `"ascending"` or `"descending"` |

---

## 4. metrics.json schema

### 4.1 Base grain

Every metrics.json must store results at the annual base grain:

```
(planning_year, aq_quarter, class_type, market_round)
```

### 4.2 Structure

```json
{
  "base_grain": "planning_year/aq_quarter/class_type/market_round",
  "cells": [
    {
      "planning_year": "2025-06",
      "aq_quarter": "aq1",
      "class_type": "onpeak",
      "market_round": 1,
      "K": 200,
      "sp": 568528,
      "binders": 88,
      "precision": 0.440,
      "vc": 0.599,
      "recall": 0.264,
      "nb_in": 4,
      "nb_binders": 4,
      "nb_sp": 20516,
      "d20_hit": 9,
      "d20_total": 15,
      "d50_hit": 1,
      "d50_total": 1,
      "label_coverage": 200,
      "unlabeled": 0
    }
  ],
  "aggregates": {
    "description": "mean across aq1/aq2/aq3 for 2025-06 onpeak R1",
    "rule": "arithmetic mean per metric",
    "sp_mean": 690854,
    "vc_mean": 0.596
  }
}
```

### 4.3 Required cell fields

| Field | Type | Description |
|-------|------|-------------|
| `planning_year` | string | e.g. `"2025-06"` |
| `aq_quarter` | string | e.g. `"aq1"` |
| `class_type` | string | `"onpeak"` or `"offpeak"` |
| `market_round` | int | 1, 2, or 3 |
| `K` | int | Top-K selection size |
| `sp` | float | Total SP captured in top-K |
| `binders` | int | Count of binding branches in top-K |
| `precision` | float | binders / K |
| `vc` | float | SP captured / total in-universe SP |
| `recall` | float | binders in top-K / total binders |
| `nb_in` | int | Dormant branches in top-K |
| `nb_binders` | int | Dormant binders in top-K |
| `nb_sp` | float | SP from dormant binders in top-K |

### 4.4 Optional cell fields

| Field | Type | Description |
|-------|------|-------------|
| `d20_hit` / `d20_total` | int | Dangerous (SP > $20K) branch hit/total |
| `d50_hit` / `d50_total` | int | Very dangerous (SP > $50K) branch hit/total |
| `label_coverage` | int | Branches with GT labels in top-K |
| `unlabeled` | int | Benchmark picks outside our universe (benchmark only) |

### 4.5 Aggregates

The `aggregates` section is optional but if present must declare:
- which cells were pooled
- the aggregation rule (mean, sum, etc.)

Aggregates must never appear without per-cell results.

---

## 5. Registry layout

```
registry/
  miso/
    annual/
      models/
        v0c_formula_v1/
          onpeak/
            spec.json
            metrics.json
          offpeak/
            spec.json
            metrics.json
        bucket_6_20_v1/
          onpeak/
            spec.json
            metrics.json
            artifacts/
          offpeak/
            spec.json
            metrics.json
            artifacts/
      benchmarks/
        v44_published_v1/
          onpeak/
            spec.json
            metrics.json
          offpeak/
            spec.json
            metrics.json
      policies/
        r30_v1/
          spec.json
          metrics.json
      comparisons/
        champion_confirmation_r1/
          spec.json
          all_results.json
```

---

## 6. Migration from current registry

Current registry layout:
```
registry/
  onpeak/bucket_6_20/    -> move to registry/miso/annual/models/bucket_6_20_v1/onpeak/
  offpeak/bucket_6_20/   -> move to registry/miso/annual/models/bucket_6_20_v1/offpeak/
  champion_confirmation/ -> move to registry/miso/annual/comparisons/champion_confirmation_r1/
  archive/               -> keep as-is (legacy, not authoritative)
```

Migration rules:
- Do not retrofit missing metadata by guessing. Mark unknown fields as `null` with a comment.
- All migrated entries must have `market_round: 1` and `round_sensitivity: "r1_only"`.
- Old entries without `universe_id` or `feature_recipe_id` must add them from the contracts.
- Do not delete old entries until new layout is verified.

---

## 7. Validation rules

A registry entry is valid if:

1. `spec.json` exists and contains all required fields for its `spec_type`
2. `metrics.json` exists and every cell has the base grain fields
3. `market_round` in spec matches `market_round` in all metrics cells
4. `class_type` in spec matches `class_type` in all metrics cells
5. If `round_sensitivity` is `"r1_only"`, `market_round` must be 1
6. If `spec_type` is `"benchmark"`, `rank_direction` must be present
7. `universe_id` must exist in the universe catalog
8. `feature_recipe_id` must exist in the feature recipes contract (model only)
