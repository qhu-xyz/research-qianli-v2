# Stage 2: Shadow Price Regression Pipeline — Design Doc

**Date**: 2026-03-03
**Status**: Approved
**Repo**: `research-stage2-shadow`

## Problem Statement

Predict shadow price **magnitude** (in dollars) for MISO transmission constraints. This is the second stage of a two-stage pipeline: stage 1 (in `research-stage1-shadow`) classifies whether a constraint binds; stage 2 predicts how much.

## Key Design Decisions

### 1. Classifier Handling: Hybrid Approach

Stage 1's promoted champion classifier config is **frozen** as infrastructure in stage 2. The agentic loop iterates **only on the regressor**.

- `ClassifierConfig` is a snapshot of stage 1's champion (v0006: 13 features, XGBoost 200 trees, threshold_beta=0.7)
- Updated at HUMAN_SYNC only, when stage 1 promotes a new champion
- Pipeline trains the classifier per eval month (self-contained), but its config never changes between versions
- Worker agents **cannot** modify `ClassifierConfig`

**Rationale**: Keeps stage 2 pure (regression research only) while being self-contained (no cross-repo model imports).

### 2. Regressor Architecture: Gated v0 Baseline

v0 baseline uses the **gated** approach (original repo v008-style):
- Regressor trains only on binding samples (label > 0)
- Target: `log1p(max(0, shadow_price))`
- 24 features (all 13 classifier features + 11 additional: tail stats, distribution moments, seasonal history)
- XGBoost 400 trees, max_depth=5

The **unified** approach (v012-style, regressor on all samples) is a natural iteration hypothesis (H1).

### 3. Metrics: EV-Based, Threshold-Independent

All blocking gates use **expected value scoring**: `score = P(binding) * expm1(regressor_pred)`. Constraints are ranked by EV score. No binary threshold gate in final ranking.

**Group A (blocking)**:
| Gate | Direction | 3-Layer |
|------|-----------|---------|
| EV-VC@100 | higher | Mean + Tail |
| EV-VC@500 | higher | Mean + Tail |
| EV-NDCG | higher | Mean + Tail |
| Spearman | higher | Mean only |

**Group B (monitor)**:
| Gate | Direction | Layer |
|------|-----------|-------|
| C-RMSE | lower | Mean only |
| C-MAE | lower | Mean only |
| EV-VC@1000 | higher | Mean only |
| R-REC@500 | higher | Mean only |

Floors bootstrapped from v0 baseline results.

### 4. Evaluation Scope

12 eval months (2020-09 through 2022-12), f0, onpeak — identical to stage 1. Cascade stages expand to f1/offpeak later.

### 5. Infrastructure: Selective Port from Stage 1

**Verbatim from stage 1**: `state_utils.sh`, `watchdog.sh`, `launch_*.sh`, `registry.py`, memory tier structure, handoff protocol, state machine flow.

**Fresh for stage 2**: All `ml/` code, all agent prompts, `gates.json`, `config.sh`.

## Architecture

```
research-stage2-shadow/
├── agents/                      # Ported from stage 1 (generic infra)
│   ├── run_pipeline.sh          # Master script (identical logic)
│   ├── run_single_iter.sh       # One iteration cycle (adapted)
│   ├── launch_orchestrator.sh   # Tmux launcher (identical)
│   ├── launch_worker.sh         # Git worktree + tmux (identical)
│   ├── launch_reviewer_claude.sh
│   ├── launch_reviewer_codex.sh
│   ├── state_utils.sh           # CAS transitions (verbatim)
│   ├── watchdog.sh              # Timeout detection (verbatim)
│   ├── config.sh                # Adapted paths
│   └── prompts/                 # FRESH — regression-specific
│       ├── orchestrator_plan.md
│       ├── orchestrator_synthesize.md
│       ├── worker.md
│       ├── reviewer_claude.md
│       └── reviewer_codex.md
├── ml/                          # FRESH — regression pipeline
│   ├── config.py                # ClassifierConfig (frozen) + RegressorConfig (mutable)
│   ├── data_loader.py           # Density, shadow prices, features
│   ├── features.py              # 13 clf + 24 reg features
│   ├── train.py                 # train_classifier() + train_regressor()
│   ├── evaluate.py              # HUMAN-WRITE-ONLY: EV-based metrics
│   ├── threshold.py             # F-beta threshold (frozen)
│   ├── pipeline.py              # load → clf → reg → EV-score → evaluate
│   ├── benchmark.py             # 12-month rolling eval via Ray
│   ├── compare.py               # Version comparison with gate checks
│   ├── registry.py              # Version allocation (ported)
│   ├── populate_v0_gates.py     # Bootstrap gates from v0
│   └── tests/
├── registry/
│   ├── v0/                      # Immutable baseline
│   ├── gates.json               # HUMAN-WRITE-ONLY
│   ├── champion.json
│   └── version_counter.json
├── memory/
│   ├── hot/                     # progress, champion, learning, gate_calibration, critique_summary, runbook
│   ├── warm/                    # experiment_log, hypothesis_log, decision_log
│   └── archive/
├── reports/
├── reviews/
├── handoff/                     # (gitignored)
├── human-input/
├── state.json                   # (gitignored)
└── runbook.md
```

## Pipeline Flow (per auction_month)

```
1. LOAD DATA
   └─ density files + shadow prices + historical context
   └─ Rolling window: 10 months train + 2 months val
   └─ Filter: f0, onpeak, market_round=1

2. CLASSIFIER (frozen config)
   └─ 13 step1 features, XGBoost 200 trees, monotone constraints
   └─ Optimize threshold on val_data (F-beta=0.7)
   └─ Output: P(binding) per constraint, threshold

3. REGRESSOR (mutable — iterated by agents)
   └─ 24 step2 features
   └─ Train on binding-only samples (gated mode for v0)
   └─ Target: log1p(max(0, shadow_price))
   └─ XGBoost 400 trees, max_depth=5

4. EV SCORING
   └─ score = P(binding) * expm1(regressor_pred)
   └─ Rank all constraints by EV score

5. EVALUATE
   └─ EV-VC@100, EV-VC@500 (value capture by EV rank)
   └─ EV-NDCG (ranking quality by EV rank)
   └─ Spearman (rank correlation, binding-only)
   └─ C-RMSE, C-MAE (calibration, binding-only)
```

## Config Dataclasses

```python
@dataclass
class ClassifierConfig:
    """Frozen from stage 1 champion. Updated at HUMAN_SYNC only."""
    step1_features: list[tuple[str, int]]  # 13 features + monotone constraints
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    threshold_beta: float = 0.7
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10

@dataclass
class RegressorConfig:
    """Mutable — iterated by agentic loop."""
    step2_features: list[tuple[str, int]]  # 24 features + constraints
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 10
    unified_regressor: bool = False  # v0: gated mode
    value_weighted: bool = False

@dataclass
class PipelineConfig:
    classifier: ClassifierConfig
    regressor: RegressorConfig
    train_months: int = 10
    val_months: int = 2
    ev_scoring: bool = True
```

## Agent Constraints

| Agent | Can Modify | Cannot Modify |
|-------|-----------|---------------|
| Orchestrator | `memory/`, handoff JSONs | `ml/`, `registry/`, `gates.json` |
| Worker | `ml/` (except `evaluate.py`), `registry/{own_version}/` | `registry/v0/`, `ml/evaluate.py`, `gates.json`, `ClassifierConfig` |
| Reviewers | `reviews/`, handoff JSONs | Everything else |

## Gate System

Same 3-layer checks as stage 1:
1. **Mean Quality**: `mean(metric across 12 months) >= floor`
2. **Tail Safety**: `count(months below tail_floor) <= 1`
3. **Tail Non-Regression**: `mean_bottom_2(new) >= mean_bottom_2(champion) - noise_tolerance`

Cascade stages: f0 (blocking) → f1 (blocking) → f2+ (monitor).

## Bootstrap Sequence

1. Port infrastructure scripts (state_utils.sh, launchers, watchdog)
2. Write `ml/` pipeline code
3. Write `config.sh` with stage 2 paths
4. Write agent prompts
5. Initialize memory stubs
6. Initialize registry (v0/, gates.json, champion.json, version_counter.json)
7. Run v0 baseline benchmark (12 months)
8. Bootstrap gates.json from v0 results
9. Verify full pipeline smoke test
10. Run first autonomous batch (3 iterations)

## Early Iteration Hypotheses

| ID | Hypothesis | Expected Impact |
|----|-----------|-----------------|
| H1 | Unified regressor (train on all samples) | Better ranking (eliminates gate bottleneck), worse calibration |
| H2 | Feature engineering (shift factors, constraint metadata) | New signal sources for magnitude prediction |
| H3 | Regressor HP tuning (trees, depth, learning rate) | Fine-tuning after H1/H2 |
| H4 | Value-weighted training (weight by shadow price magnitude) | Better predictions on high-value constraints |
| H5 | Log-transform vs raw target | Calibration improvement |
