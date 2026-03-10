# Tier Classification Pipeline — Design Document

**Date:** 2026-03-04
**Status:** Draft — pending approval

## Goal

Build a new autonomous research pipeline (`research-stage3-tier`) that predicts **5 shadow-price tiers** for transmission constraints using a direct multi-class XGBoost classifier. Reuses the stage-2 autonomous infrastructure (3-iter batches, 2-hypothesis screening, 4-agent team) with new metrics and prompts tailored to ordinal tier classification.

## Motivation

Stage 2 predicts continuous shadow price magnitude (dollars). The downstream trading system (pmodel/ftr24/v1) ultimately bins constraints into tiers for path pool construction and capital allocation. A direct tier classifier:

1. Aligns the ML objective with the downstream decision boundary
2. Eliminates binning error from continuous prediction → tier conversion
3. Enables tier-specific precision/recall optimization
4. Produces probability distributions over tiers (richer signal than point estimates)

## Tier Definitions

Match the existing SPICE tier system exactly (from `pbase/analysis/tier_threshold_generator_1.py`):

| Tier | Shadow Price Range | Semantic | Ordinal Value |
|------|-------------------|----------|---------------|
| 0 | [3000, +inf) | Heavily binding | 4 (highest) |
| 1 | [1000, 3000) | Strongly binding | 3 |
| 2 | [100, 1000) | Moderately binding | 2 |
| 3 | [0, 100) | Lightly binding | 1 |
| 4 | (-inf, 0) | Not binding | 0 (lowest) |

**Bins:** `[-inf, 0, 100, 1000, 3000, inf]` with labels `[4, 3, 2, 1, 0]`.

No class_type scaling initially (onpeak bins only). Scaling can be a hypothesis in later iterations.

## Architecture

### Model

Single XGBoost multi-class classifier:
- `objective='multi:softprob'`, `num_class=5`
- Same 34 candidate features as stage-2 regressor (all mutable)
- Target: `pd.cut(actual_shadow_price, bins=[-inf, 0, 100, 1000, 3000, inf], labels=[4,3,2,1,0])`
- No frozen classifier gate — the entire model is the iterable component
- No separate regressor — one model, one output

### EV Score (for ranking)

The downstream system needs a ranking signal for capital allocation (top-100, top-500). A tier label alone is too coarse. We use the multi-class probabilities to construct a continuous EV score:

```
tier_ev_score = sum(P(tier=t) * midpoint[t] for t in [0,1,2,3,4])
```

Where midpoints are:
| Tier | Midpoint ($) |
|------|-------------|
| 0 | 4000 |
| 1 | 2000 |
| 2 | 550 |
| 3 | 50 |
| 4 | 0 |

This gives a probability-weighted expected shadow price that naturally ranks constraints by economic value. It's mathematically equivalent to `E[SP | features]` under the tier discretization.

### Pipeline Phases (6 phases, down from 7)

| Phase | Description |
|-------|-------------|
| 1 | Load train/val data (same data loader, same 6+2 month lookback) |
| 2 | Prepare features and tier labels (`pd.cut` on actual shadow price) |
| 3 | Train multi-class XGBoost (class weights for imbalance) |
| 4 | Load target-month test data |
| 5 | Evaluate: predict tier probabilities, compute tier_ev_score, compute all metrics |
| 6 | Return results |

## Metrics

### Group A — Blocking Gates

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **Tier-VC@100** | `sum(actual_sp[top_100_by_tier_ev]) / sum(actual_sp)` | Capital allocation quality — same concept as EV-VC@100, ranked by tier_ev_score |
| **Tier-VC@500** | Same, k=500 | Broader capture |
| **Tier-NDCG** | NDCG with actual_sp as relevance, tier_ev_score as ranking | Position-discounted ranking quality |
| **QWK** | Quadratic Weighted Kappa between actual_tier and pred_tier | Ordinal agreement — penalizes far-off tier errors more than adjacent |

### Group B — Monitor Only

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **Macro-F1** | Mean of per-tier F1 | Class-balanced classification quality |
| **Tier-Accuracy** | `correct_tier / total` | Overall accuracy |
| **Adjacent-Accuracy** | `(correct_tier + off_by_1) / total` | Allows 1-tier tolerance |
| **Tier-Recall@0** | Recall for tier 0 specifically | Catch rate for highest-value constraints |
| **Tier-Recall@1** | Recall for tier 1 | Catch rate for strongly binding |

### Design Rationale

- **Tier-VC@100/500 and Tier-NDCG** are retained because they directly measure the downstream business objective (capital allocation via ranking). Only the ranking signal changes from `P(bind)*SP` to `tier_ev_score`.
- **QWK replaces Spearman** as the ordinal consistency metric. Spearman on continuous SP doesn't apply; QWK naturally handles ordered categories and penalizes large tier mismatches quadratically.
- **Macro-F1** monitors class balance (tier 0 and 1 are rare; we need to catch them).
- **Tier-Recall@0/1** monitor the most valuable tiers specifically — missing a tier-0 constraint is catastrophic.
- **C-RMSE/C-MAE are dropped** — no continuous prediction to calibrate.

## Config Structure

Replace the two-config (ClassifierConfig + RegressorConfig) structure with a single `TierConfig`:

```python
@dataclass
class TierConfig:
    features: list[str]          # All 34 candidate features (mutable)
    monotone_constraints: list[int]
    bins: list[float]            # [-inf, 0, 100, 1000, 3000, inf]
    tier_midpoints: list[float]  # [4000, 2000, 550, 50, 0] for EV score
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 1.0
    min_child_weight: int = 25

@dataclass
class PipelineConfig:
    tier: TierConfig
    train_months: int = 6
    val_months: int = 2
```

## Agent Team — Changes from Stage 2

### What stays the same
- 4-agent structure: orchestrator (plan), worker, claude reviewer, codex reviewer, orchestrator (synthesize)
- 3-iteration loop with 2-hypothesis screening per iteration
- File-based handoff protocol
- Memory system (hot/warm/archive)
- `--overrides` JSON for screening without code changes
- Worker 10-step protocol structure
- Independent reviewer constraint
- 3-layer gate check logic in `compare.py`

### What changes

**Orchestrator prompts:**
- Remove all "frozen classifier" / "RegressorConfig" references
- Replace with single "TierConfig" — everything is mutable
- Update metric names: EV-VC → Tier-VC, Spearman → QWK, drop C-RMSE/C-MAE
- Add tier-specific analysis: per-tier precision/recall, class distribution shifts
- Hypothesis types now include: bin edge adjustment, class weighting, feature changes, hyperparams

**Worker prompt:**
- Remove two-stage clf+reg pipeline references
- Single model: multi-class XGBoost
- Allowed to modify: `ml/config.py` (TierConfig), `ml/train.py`, `ml/features.py`, `ml/pipeline.py`, `ml/data_loader.py`, `ml/benchmark.py`
- Still CANNOT modify: `ml/evaluate.py`, `registry/gates.json`
- Screening: same 2-hypothesis, 2-month protocol but uses Tier-VC@100 and QWK as winner criteria

**Reviewer prompts:**
- Replace regression-quality dimensions (C-RMSE, C-MAE, Spearman) with tier-quality dimensions (QWK, per-tier recall, confusion matrix analysis)
- Add confusion matrix review: are errors concentrated in adjacent tiers (acceptable) or distant tiers (problematic)?
- Add class imbalance analysis: are rare tiers (0, 1) being learned or ignored?

## Gate Calibration

Gates will be calibrated from v0 baseline (first run):
- `floor = 0.95 * v0_mean`
- `tail_floor = 0.90 * v0_worst_month`
- Same structure as stage 2

Initial gates.json will have `pending_baseline: true` for all gates until v0 runs.

## Directory Structure

```
research-stage3-tier/
├── agents/
│   ├── config.sh
│   ├── run_pipeline.sh          # identical to stage 2
│   ├── run_single_iter.sh       # identical to stage 2
│   ├── launch_orchestrator.sh   # identical
│   ├── launch_worker.sh         # identical
│   ├── launch_reviewer_claude.sh
│   ├── launch_reviewer_codex.sh
│   ├── audit_iter.sh            # identical
│   └── prompts/
│       ├── orchestrator_plan.md     # adapted for tier classification
│       ├── orchestrator_synthesize.md
│       ├── worker.md
│       ├── reviewer_claude.md
│       └── reviewer_codex.md
├── ml/
│   ├── config.py        # TierConfig replaces Classifier+Regressor
│   ├── pipeline.py      # 6-phase tier pipeline
│   ├── evaluate.py      # New tier metrics (HUMAN-WRITE-ONLY)
│   ├── train.py         # Multi-class XGBoost training
│   ├── features.py      # Same feature prep (minor changes)
│   ├── data_loader.py   # Identical to stage 2
│   ├── benchmark.py     # Same structure, new metric names
│   ├── compare.py       # Same 3-layer gate logic
│   ├── registry.py      # Identical
│   └── tests/
├── registry/
│   ├── gates.json       # New tier-specific gates
│   └── version_counter.json  # start at v0
├── human-input/
│   ├── mem.md
│   └── business_context.md  # adapted for tier classification
├── memory/
│   ├── hot/
│   ├── warm/
│   └── archive/
├── handoff/
└── state.json
```

## Class Imbalance Strategy

Tier distribution is highly skewed (most constraints are tier 4 = not binding). Strategy:

1. **XGBoost sample_weight**: weight training samples by tier economic value — tier 0 gets highest weight
2. **Configurable via TierConfig**: `class_weights: dict[int, float]` defaults to `{0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}`
3. Agents can adjust weights as a hypothesis axis

## Implementation Approach

1. Clone `research-stage2-shadow` → `research-stage3-tier`
2. Replace `config.py` with TierConfig
3. Replace `pipeline.py` with 6-phase tier pipeline
4. Replace `evaluate.py` with tier metrics
5. Adapt `train.py` for multi-class XGBoost
6. Adapt `features.py` (add `compute_tier_labels()`, remove `compute_regression_target()`)
7. Adapt all 5 agent prompts
8. Write `gates.json` with `pending_baseline: true`
9. Write `business_context.md` for tier classification
10. Run v0 baseline, calibrate gates, launch autonomous batch
