# Should We Divide Into 2 Stages?

## The Problem

Stage 1 and Stage 2 were designed as separate repos with independent agentic loops:

- **Stage 1** (`research-stage1-shadow`): Binary classification — does constraint bind?
  - Champion: v0009, 29 features, XGBoost 200 trees
  - 10 experiments completed, feature-saturated plateau

- **Stage 2** (`research-stage2-shadow`): Shadow price regression — how much?
  - Current: v0 baseline, 24 features (13 from classifier + 11 additional), XGBoost 400 trees
  - Frozen classifier locked to stage 1's v0006 (13 features)

The **EV score** that drives FTR capital allocation is:

```
EV = P(binding) × predicted_shadow_price
```

Three problems have emerged:

### Problem 1: The Frozen Classifier Is Stale

Stage 2 locks its classifier to stage 1's v0006 (13 features). But stage 1 evolved to v0009 (29 features) through 10 iterations. The frozen classifier is now significantly weaker than the current champion:

| Metric | v0006 (frozen in S2) | v0009 (S1 champion) | Gap |
|--------|---------------------|---------------------|-----|
| AUC | 0.835 | 0.850 | +0.015 |
| AP | 0.394 | 0.445 | +0.051 |
| VCAP@100 | 0.015 | 0.027 | +0.012 |

Every EV score computation in stage 2 uses a P(binding) that is worse than necessary.

### Problem 2: Feature Overlap Is Now Severe

Stage 1 v0009 absorbed 6 of stage 2's 11 "additional" regressor features:

| Feature | In S1 v0006 | In S1 v0009 | In S2 regressor |
|---------|-------------|-------------|-----------------|
| tail_concentration | No | **Yes** | Yes |
| prob_band_95_100 | No | **Yes** | Yes |
| prob_band_100_105 | No | **Yes** | Yes |
| density_mean | No | **Yes** | Yes |
| density_variance | No | **Yes** | Yes |
| density_entropy | No | **Yes** | Yes |
| prob_exceed_85 | No | No | Yes |
| prob_exceed_80 | No | No | Yes |
| recent_hist_da | No | No | Yes |
| season_hist_da_1 | No | No | Yes |
| season_hist_da_2 | No | No | Yes |

After updating the frozen classifier to v0009, the regressor would only contribute **5 truly new features** beyond what the classifier already sees. The stage 2 "additional signal" story is weaker than originally planned.

### Problem 3: Two Independent Iteration Loops Can't Coordinate

- Stage 1 iterates on the classifier → pushes new champion → stage 2 must manually re-sync
- Stage 2 iterates on the regressor → but can't touch the classifier
- No mechanism to jointly optimize: "what if this feature helps regression but hurts classification?"
- Feature engineering in stage 1 is blind to stage 2's needs (stage 1 doesn't know which features matter for magnitude, only for binding probability)

---

## The Business Objective

Rank constraints by expected value to allocate limited FTR capital:

```
rank = P(constraint binds) × E[shadow_price | binds]
```

Key requirements:
1. **Precision over recall** — false positives waste capital
2. **Top-K ranking quality** — the top 100-500 constraints must capture real dollar value
3. **Magnitude calibration** — for position sizing, need reasonable dollar estimates
4. **Stability across months** — can't have catastrophic failures in any eval month

---

## Three Architectural Options

### Option A: Two Repos, Sync Classifier (Status Quo + Update)

Keep the current 2-repo structure. Periodically sync stage 1's champion into stage 2's frozen classifier.

**Changes needed:**
- Update `ClassifierConfig` from v0006 (13 features) to v0009 (29 features)
- Expand regressor features: 29 (from v0009) + 5 truly new = 34 features
- Add v0009's interaction features to regressor: `hist_physical_interaction`, `overload_exceedance_product`, `sf_max_abs`, etc.
- Rerun v0 benchmark with updated classifier
- Add sync protocol to CLAUDE.md

**Advantages:**
- Minimal code changes — infrastructure already built
- Each stage's iteration loop is focused on one model
- Stage 1 has proven its agentic loop works (10 experiments, clear progression)
- Clear separation of concerns: stage 1 = screening, stage 2 = pricing

**Disadvantages:**
- Manual sync friction — every stage 1 promotion requires stage 2 rebaseline
- Limited regressor innovation space (only 5 unique features remaining)
- Can't jointly optimize feature interactions across classifier and regressor
- Two benchmark runs needed (25+ minutes each) for any classifier change
- Frozen classifier means stage 2 agents can never discover "this feature helps P(binding) but you didn't include it in stage 1"

**Risk**: Sync drift. If stage 1 keeps evolving, stage 2 falls behind. If stage 2 finds a feature that helps regression, it can't propagate back to stage 1.

### Option B: Single Repo, Two-Part Model (Joint Iteration)

Merge both models into one repo. Both the classifier and regressor are iterable. One agentic loop optimizes both simultaneously.

```
Pipeline: [Data] → [Classifier (iterable)] → [Regressor (iterable)] → [EV Score]
```

**Changes needed:**
- Copy stage 1's feature engineering, training, and threshold logic into stage 2
- Remove `frozen=True` from `ClassifierConfig`, make it mutable
- Define joint gates: classifier metrics (AUC, AP, NDCG) + regressor metrics (EV-VC@K, Spearman, C-RMSE)
- Workers can modify both models within a single iteration
- Orchestrator plans can target classifier features, regressor features, or both

**Advantages:**
- No sync problem — one repo, one truth
- Joint feature engineering: discover features that help both classification and regression
- Can test "what if we trade classifier precision for better magnitude estimates?"
- Single benchmark run evaluates the full pipeline
- Larger feature exploration space
- Agent can discover non-obvious interactions (e.g., a feature that hurts AUC by 0.001 but improves EV-VC@100 by 0.01)

**Disadvantages:**
- Larger search space may slow convergence — agents must optimize 8+ gate metrics simultaneously
- Risk of regressing stage 1's hard-won gains (classifier might get worse)
- Need careful gate design to protect classifier quality while allowing regressor improvements
- More complex orchestrator prompts
- Loss of the "frozen classifier" safety rail

**Mitigation for disadvantages:**
- Hierarchical gates: classifier gates must ALL pass before regressor gates are checked
- Minimum classifier quality: AUC >= v0009 level (0.849), AP >= 0.440
- Staged iteration: first iterations focus on classifier-only, then regressor-only, then joint
- Gate ratchet: once classifier reaches a level, it can't regress

### Option C: Single Model, Zero-Inflated Regression

Replace both models with a single XGBoost that directly predicts E[shadow_price], including zeros.

**Approach options:**
1. **Tweedie loss** (`objective='reg:tweedie'`): Designed for zero-inflated continuous data. The Tweedie distribution (1 < p < 2) naturally handles exact zeros + continuous positives. Used extensively in insurance claims prediction (same pattern: many zeros, positive values with heavy tail).

2. **Log1p with squared error**: Target = log1p(shadow_price). Model predicts in log-space, back-transform with expm1. Zeros map to 0. But squared error treats zero and near-zero predictions equally.

3. **Quantile regression**: Predict conditional quantiles instead of means. More robust to outliers but harder to rank by expected value.

**Changes needed:**
- Remove classifier entirely
- Single XGBoost model with all features (29+ from v0009 + additional)
- Target: shadow_price directly (or log1p transformed)
- Rank by predicted value (no EV score computation needed)
- Redesign gates: no classifier metrics, pure regression/ranking metrics
- Rewrite evaluate.py for new metric set

**Advantages:**
- Simplest architecture — one model, one training, one prediction
- No threshold tuning, no gating logic, no P(binding) computation
- Model can learn the joint distribution of binding probability and magnitude
- Tweedie loss is mathematically equivalent to the two-part model under certain conditions
- Fewer hyperparameters to tune

**Disadvantages:**
- **Loses P(binding) interpretability**: Business wants to know "will it bind?" separately from "how much?"
- **Loses threshold control**: Current threshold (~0.83) controls precision. With single model, no equivalent lever.
- **XGBoost with 92.5% zeros**: Even Tweedie may struggle — the model may learn "predict near-zero for everything" as a low-loss strategy, especially for top-K ranking metrics.
- **Position sizing depends on separate P(binding)**: The trading desk uses P(binding) > threshold as a binary go/no-go decision. A single predicted value doesn't separate "unlikely to bind but if it does, it's huge" from "likely to bind but small."
- **Rewrite cost**: evaluate.py, pipeline.py, train.py, compare.py — everything changes. Gates need redesign. Previous v0 benchmark is incomparable.
- **Untested**: We've built and validated the two-part approach. Switching to single model means starting from scratch.

**Assessment**: Option C is architecturally clean but conflicts with business requirements. The trading desk needs separate P(binding) and magnitude estimates. The threshold is a business-critical lever. Abandoning the two-part model discards valuable structure.

---

## Analysis Matrix

| Criterion | A: Two Repos + Sync | B: Single Repo, Two-Part | C: Single Model |
|-----------|---------------------|--------------------------|-----------------|
| Sync friction | High (manual) | None | None |
| Feature innovation space | Small (5 new) | Large (joint) | Large |
| Classifier protection | Strong (frozen) | Medium (gated) | N/A |
| Business compatibility | Full | Full | **Partial** |
| Implementation cost | Low | Medium | High |
| Agent convergence speed | Fast (small space) | Medium | Unknown |
| Rebaseline cost | 2 benchmarks | 1 benchmark | Full rebuild |
| P(binding) available? | Yes | Yes | **No** |
| Threshold control? | Yes | Yes | **No** |

---

## Recommendation: Option B — Single Repo, Two-Part Model

**Why not A:** The sync friction is real and will worsen. Stage 1 is already at a plateau — there's nothing to protect by keeping it separate. And with only 5 unique regressor features, stage 2's innovation space is too cramped.

**Why not C:** Loses business-critical levers (P(binding), threshold, precision control). The two-part model structure is statistically well-motivated for zero-inflated data. Discarding it gains simplicity but loses interpretability and control.

**Why B:** Combines the best of both:
- Two-part model preserves P(binding) and threshold control
- Joint feature engineering unlocks cross-model optimization
- No sync friction
- Single benchmark run evaluates end-to-end
- Hierarchical gates protect classifier quality

### Implementation Sketch for Option B

**Gate Design (hierarchical):**

```
Tier 1 (Classifier, ALL must pass):
  - AUC >= 0.845 (floor, ~v0009 level)
  - AP >= 0.430

Tier 2 (Regressor, checked only if Tier 1 passes):
  - EV-VC@100, EV-VC@500 (blocking)
  - EV-NDCG, Spearman (blocking)

Tier 3 (Monitor):
  - C-RMSE, C-MAE, EV-VC@1000, R-REC@500
  - BRIER, REC, CAP@K (from stage 1)
```

**Feature Set (unified):**
- Start with v0009's 29 features for both classifier and regressor
- Regressor additionally gets: prob_exceed_85, prob_exceed_80, recent_hist_da, season_hist_da_1, season_hist_da_2 = 34 total
- Agent can add new features to either model
- Monotone constraints shared where applicable

**Iteration Strategy:**
- Batch 1: Rebaseline with v0009 classifier + 34-feature regressor
- Batch 2-3: Feature engineering targeting regressor metrics
- Batch 4+: Joint optimization (allow classifier feature changes with tier-1 gate protection)

**Agent Constraints:**
- Worker CAN modify both classifier and regressor features/HP
- Worker CANNOT modify evaluate.py or gates.json (same as current)
- Orchestrator sees both classifier and regressor metrics
- If tier-1 gate fails, the version is blocked regardless of regressor improvement

---

## Open Questions

1. **Should we port stage 1's full experiment history (v0-v0009) into this repo, or start fresh?**
   Starting fresh with v0009 as the baseline seems cleaner. The 10 experiments' learnings are captured in memory files.

2. **Should we expand eval to f1/f2+ now or stay f0-only?**
   Current cascade_stages define f1 as blocking, but no benchmark has ever run f1. If we're rebuilding, this is the time to decide. Recommendation: keep f0-only for now, add f1 in a later phase once f0 converges.

3. **How do we handle the regressor training on binding-only samples vs. all samples?**
   This is a tunable hypothesis (H1 in the design doc). With a joint repo, we can test unified vs. gated mode without worrying about classifier compatibility.

4. **Train_months: 10 (current S2) vs 14 (S1 v0009)?**
   v0009 gained stability from 14-month windows. Stage 2 should adopt this too. The longer window gives the regressor more binding samples to learn from.

5. **What new features can the regressor uniquely benefit from?**
   Now that most distribution-shape features are in the classifier, stage 2's unique value-add must come from features that predict magnitude but not binding:
   - `constraint_limit` (MW limit of the line — larger limits mean bigger shadow prices)
   - Shift factor features (`sf_max_abs`, `sf_mean_abs`, etc. — already in v0009 classifier though)
   - Regional features (ISO zone, voltage level)
   - Temporal features (month-of-year, day-of-week patterns)
   - Cross-constraint features (are nearby constraints also binding?)
