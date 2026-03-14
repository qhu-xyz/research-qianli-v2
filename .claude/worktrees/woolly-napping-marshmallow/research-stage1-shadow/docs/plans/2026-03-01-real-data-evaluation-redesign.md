# Real-Data Evaluation Redesign

> **Status:** DESIGN — awaiting implementation plan

## 1. Purpose

Replace the single-month synthetic evaluation (n=20) with a rolling-window multi-month evaluation on real MISO data. Redesign gates with cascade stages, three-layer checks (mean + tail safety + tail non-regression), and monitor-vs-hard reclassification.

### Goals
- Run v0 baseline across 12 real eval months for f0, then cascade to f1, then f2+
- Establish per-ptype gate floors from real data
- Three-layer gate check: mean quality, tail safety, tail non-regression vs champion
- Reduce hard gates from 7 to 4 (AUC, AP, VCAP@100, NDCG)
- Parallel eval months via Ray
- Reuse source repo's `MisoDataLoader` for data loading (no reimplementation)

### Non-goals
- Branch-level models (source repo has these; we use single default classifier)
- Regression stage (Stage 2) — this is Stage 1 (binary classification) only
- Real-time prediction serving

---

## 2. Evaluation Architecture

### Rolling Window

For each eval month `M` with ptype of horizon `H`, train on a window of `10 + 2 + H` months ending at `M`. The extra `H` months compensate for the source loader's future-month guard: rows where `market_month >= M` have no labels at prediction time and are clipped. Without the horizon buffer, the last `H` val-window months produce empty validation sets (their market months land at or beyond `M`).

```
  f0 (H=0):  M-12 |--- fit (10mo) ---|-- val (2mo) --|- eval (M) -|
  f1 (H=1):  M-13 |--- fit (10mo) ---|-- val (2mo) -|- buffer -|- eval (M) -|
  f2 (H=2):  M-14 |--- fit (10mo) ---|-- val (2mo) -|-- buffer --|- eval (M) -|
```

**Why the buffer is needed:** Each training/val row predicts binding in `market_month = auction_month + H`. At prediction time (start of month `M`), labels only exist for market months that have fully elapsed (< `M`). The source loader enforces this via `if market_month >= train_end: skip`. For the last `H` auction months in the window, `market_month = auction_month + H >= M`, so they're skipped. The buffer shifts the val window earlier by `H` months so val rows have known labels.

**Implementation:** `ml/data_loader.py` computes `lookback = train_months + val_months + horizon` where `horizon = int(ptype[1:])` for f-series ptypes.

### Eval Month Selection (18 months total)

**Primary set (12 months)** — used for floor calibration:

| Month | Season | Rationale |
|-------|--------|-----------|
| 2020-09 | Summer tail | Post-COVID recovery |
| 2020-11 | Fall | Low binding, tests specificity |
| 2021-01 | Winter | Pre-polar-vortex baseline |
| 2021-04 | Spring | Shoulder season |
| 2021-06 | Early summer | Ramping load |
| 2021-08 | Peak summer | High binding rate |
| 2021-10 | Fall | Transition |
| 2021-12 | Winter | Heating season |
| 2022-03 | Spring | Shoulder |
| 2022-06 | Summer | High congestion |
| 2022-09 | Late summer | Derating season |
| 2022-12 | Winter | Year-end |

**Stress set (6 months)** — monitor-only, not used for floors:

| Month | Rationale |
|-------|-----------|
| 2020-04 | COVID demand collapse |
| 2020-07 | COVID recovery + summer |
| 2021-02 | Polar vortex / Winter Storm Uri |
| 2021-07 | Summer peak (current smoke month) |
| 2022-07 | Summer peak (inflation year) |
| 2022-01 | Winter Storm Elliott aftermath |

Data requirement: 2019-11 through 2022-12 (~37 months raw data).

### Cascade Stages (strict)

| Stage | ptype | Eval months | Hard gate? | Must pass to proceed? |
|-------|-------|-------------|------------|----------------------|
| Stage 1 | f0 | All 12 primary | Yes | Yes — blocks Stage 2 |
| Stage 2 | f1 | All 12 primary | Yes | Yes — blocks Stage 3 |
| Stage 3 | f2+ (q2, q3, q4) | Subset (months where available) | Monitor only | No |

Stage 3 uses only months where quarterly ptypes exist per MISO's auction schedule.

### Parallelism

Ray-parallel across eval months within each stage. Each eval month is an independent `@ray.remote` task that loads data, trains, evaluates, and returns metrics.

---

## 3. Three-Layer Gate System

### Per-metric gate check (for each hard gate)

```
Layer 1 (Mean quality):       mean(metric across 12 months) >= floor
Layer 2 (Tail safety):        count(metric < tail_floor) <= tail_max_failures
Layer 3 (Tail non-regression): mean_bottom_2(metric) >= mean_bottom_2(champion) - noise_tolerance
```

- **Layer 1** catches models that are generally worse
- **Layer 2** catches models with catastrophic single-month failures
- **Layer 3** catches models that regress in their worst months even if average improves

For "lower-is-better" metrics (BRIER), directions are inverted:
- Layer 1: `mean(metric) <= floor`
- Layer 2: `count(metric > tail_floor) <= tail_max_failures`
- Layer 3: `mean_top_2(metric) <= mean_top_2(champion) + noise_tolerance`

### Gate Reclassification

| Gate | Old Group | New Group | Rationale |
|------|-----------|-----------|-----------|
| **S1-AUC** | A (hard) | A (hard) | Core discriminative quality |
| **S1-AP** | A (hard) | A (hard) | Precision-recall tradeoff |
| **S1-VCAP@100** | A (hard) | A (hard) | Top-value capture = business metric |
| **S1-NDCG** | A (hard) | A (hard) | Ranking quality |
| **S1-BRIER** | A (hard) | **B (monitor)** | Calibration important but shouldn't block; recalibration is separate |
| **S1-VCAP@500** | A (hard) | **B (monitor)** | Redundant with @100 |
| **S1-VCAP@1000** | A (hard) | **B (monitor)** | Redundant with @100 |
| **S1-REC** | B (monitor) | B (monitor) | Threshold-dependent |
| **S1-CAP@100** | B (monitor) | B (monitor) | Threshold-dependent |
| **S1-CAP@500** | B (monitor) | B (monitor) | Threshold-dependent |

**4 hard gates** (AUC, AP, VCAP@100, NDCG) — all threshold-independent.

### New `gates.json` Schema

```json
{
  "version": 2,
  "effective_since": "2026-03-01",
  "noise_tolerance": 0.02,
  "tail_max_failures": 1,
  "eval_months": {
    "primary": ["2020-09", "2020-11", "2021-01", "2021-04", "2021-06", "2021-08",
                 "2021-10", "2021-12", "2022-03", "2022-06", "2022-09", "2022-12"],
    "stress": ["2020-04", "2020-07", "2021-02", "2021-07", "2022-07", "2022-01"]
  },
  "cascade_stages": [
    {"stage": 1, "ptype": "f0", "blocking": true},
    {"stage": 2, "ptype": "f1", "blocking": true},
    {"stage": 3, "ptype": "f2p", "blocking": false}
  ],
  "gates": {
    "S1-AUC": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "A",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-AP": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "A",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-VCAP@100": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "A",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-NDCG": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "A",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-BRIER": {
      "floor": null,
      "tail_floor": null,
      "direction": "lower",
      "group": "B",
      "pending_v0": true,
      "v0_offset": 0.02,
      "v0_tail_offset": 0.05
    },
    "S1-VCAP@500": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "B",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-VCAP@1000": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "B",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-REC": {
      "floor": 0.10,
      "tail_floor": 0.0,
      "direction": "higher",
      "group": "B"
    },
    "S1-CAP@100": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "B",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    },
    "S1-CAP@500": {
      "floor": null,
      "tail_floor": null,
      "direction": "higher",
      "group": "B",
      "pending_v0": true,
      "v0_offset": 0.05,
      "v0_tail_offset": 0.10
    }
  }
}
```

### Floor Calibration (from v0 baseline)

For each gate with `pending_v0: true`:
- `floor = mean(v0_metric across 12 months) - v0_offset` (higher-is-better)
- `floor = mean(v0_metric across 12 months) + v0_offset` (lower-is-better)
- `tail_floor = min(v0_metric across 12 months) - v0_tail_offset` (higher)
- `tail_floor = max(v0_metric across 12 months) + v0_tail_offset` (lower)

### Promotion Logic (updated)

```
For each cascade stage (f0 → f1 → f2+):
  For each Group A gate:
    Layer 1: mean(metric) >= floor                                    FAIL → reject
    Layer 2: count(metric < tail_floor) <= tail_max_failures          FAIL → reject
    Layer 3: mean_bottom_2(metric) >= mean_bottom_2(champ) - noise    FAIL → reject
  If stage is blocking and any gate fails → stop cascade, reject

All blocking stages pass → eligible for promotion
Orchestrator makes final call (may decline even if gates pass)
```

---

## 4. Data Loading Strategy

### Reuse Source Repo's MisoDataLoader

```python
import sys
sys.path.insert(0, "/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src")
from shadow_price_prediction.data_loader import MisoDataLoader
from shadow_price_prediction.config import PredictionConfig
```

**Why reuse:**
- 600+ lines of density parsing, feature engineering, label weighting
- Already handles MISO-specific period types, auction schedules, historical shadow prices
- Maintains consistency with source repo's results

**What our pipeline wraps:**
1. Create `PredictionConfig` matching our `PipelineConfig`
2. Call `MisoDataLoader.load_training_data()` for train+val data
3. Extract our 14 features from the full DataFrame
4. Compute binary labels from `actual_shadow_price > 0`

### Feature Extraction

Source loader returns ~30+ columns. Our Stage 1 classifier uses 14:

```python
STAGE1_FEATURES = [
    "prob_exceed_110", "prob_exceed_105", "prob_exceed_100",
    "prob_exceed_95", "prob_exceed_90",
    "prob_below_100", "prob_below_95", "prob_below_90",
    "expected_overload",
    "density_skewness", "density_kurtosis", "density_cv",
    "hist_da", "hist_da_trend",
]
```

These are selected from the full feature set and passed to XGBoost with monotone constraints.

---

## 5. Metrics Storage (per version)

### New `metrics.json` Schema

```json
{
  "eval_config": {
    "eval_months": ["2020-09", "2020-11", ...],
    "class_type": "onpeak",
    "ptype": "f0",
    "train_months": 10,
    "val_months": 2
  },
  "per_month": {
    "2020-09": {"S1-AUC": 0.72, "S1-AP": 0.45, ...},
    "2020-11": {"S1-AUC": 0.68, "S1-AP": 0.38, ...},
    ...
  },
  "aggregate": {
    "mean": {"S1-AUC": 0.71, "S1-AP": 0.42, ...},
    "std": {"S1-AUC": 0.03, "S1-AP": 0.05, ...},
    "min": {"S1-AUC": 0.65, "S1-AP": 0.33, ...},
    "max": {"S1-AUC": 0.78, "S1-AP": 0.52, ...},
    "bottom_2_mean": {"S1-AUC": 0.66, "S1-AP": 0.35, ...}
  },
  "n_months": 12,
  "n_samples_total": 240000,
  "threshold_per_month": {"2020-09": 0.42, "2020-11": 0.51, ...}
}
```

### Gate check reads `aggregate.mean` for Layer 1, `per_month` for Layer 2, `aggregate.bottom_2_mean` for Layer 3.

---

## 6. Files Changed

### Modified files
| File | Change |
|------|--------|
| `ml/data_loader.py` | Replace synthetic loader with `MisoDataLoader` wrapper; keep SMOKE_TEST path |
| `ml/pipeline.py` | Multi-month eval loop; Ray parallel; new metrics schema |
| `ml/evaluate.py` | Add multi-month aggregation (`aggregate_months()`); keep single-month `evaluate_classifier()` |
| `ml/compare.py` | Three-layer gate check; cascade stage logic; updated `check_gates()` |
| `ml/populate_v0_gates.py` | Populate `floor` AND `tail_floor` from v0 per-month metrics |
| `registry/gates.json` | New v2 schema with cascade stages, tail gates, reclassified groups |

### New files
| File | Purpose |
|------|---------|
| `ml/benchmark.py` | CLI: run v0 (or any version) across all eval months for a given ptype |

### Deleted files
| File | Reason |
|------|--------|
| `registry/v0001/` | Stale smoke artifact |

### Reset files
| File | Change |
|------|--------|
| `registry/version_counter.json` | Reset to `{"next_id": 1}` |
| `registry/champion.json` | Reset to `{"version": null}` |
| `registry/v0/` | Recreated from real data |

### Agent prompt updates
| File | Change |
|------|--------|
| `agents/prompts/orchestrator_plan.md` | Reference new gate schema (cascade stages, three layers, per-month breakdown) |
| `agents/prompts/orchestrator_synthesize.md` | Updated promotion logic (cascade + three-layer), new metrics.json schema |
| `agents/prompts/worker.md` | Updated pipeline invocation (multi-month), SMOKE_TEST still uses single month |
| `agents/prompts/reviewer_claude.md` | Reference three-layer gates, per-month analysis, tail risk assessment |
| `agents/prompts/reviewer_codex.md` | Same updates as Claude reviewer |

---

## 7. Execution Order (Steps 1-7)

1. **Rewrite `ml/data_loader.py`** — import `MisoDataLoader`, wrap for our interface
2. **Rewrite `ml/evaluate.py`** — add `aggregate_months()` for multi-month stats
3. **Rewrite `ml/compare.py`** — three-layer gate check, cascade logic
4. **Rewrite `registry/gates.json`** — new v2 schema
5. **Create `ml/benchmark.py`** — multi-month eval runner with Ray parallelism
6. **Update `ml/pipeline.py`** — integrate multi-month eval, new metrics schema
7. **Update `ml/populate_v0_gates.py`** — populate `floor` + `tail_floor`
8. **Delete `registry/v0001/`**, reset `version_counter.json` and `champion.json`
9. **Run new v0 baseline** — `python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak`
10. **Run `populate_v0_gates.py`** — set real floors from v0
11. **Update agent prompts** (5 files) — reference new gate system
12. **Update `runbook.md`** — document new evaluation architecture
