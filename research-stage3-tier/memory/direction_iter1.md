# Direction — Iteration 1

**Batch**: tier-v1-20260304-145001
**Champion**: v0 (baseline)
**Focus**: Fix catastrophically low Tier-Recall@1 and improve Tier-VC@100

## Current State Analysis

v0 baseline has two critical failures:
1. **Tier-Recall@1 = 0.098** — missing 90% of strongly binding constraints (tier 1: $1000-$3000)
2. **Tier-VC@100 = 0.075** — top-100 capital allocation captures only 7.5% of value

Root cause: class weights {0:10, 1:5} are too weak given class imbalance, and min_child_weight=25 prevents the model from learning fine-grained splits for rare tier 0/1 samples.

## Screen Months

- **Weak month: 2022-06** — worst VC@500 (0.047), worst NDCG (0.629), worst QWK (0.257), very low VC@100 (0.015). Changes that help here demonstrate ability to improve tail performance.
- **Strong month: 2021-09** — best VC@100 (0.246), best VC@500 (0.413), best NDCG (0.858). Changes must NOT regress here.

**Rationale**: 2022-06 tests whether class weight changes fix the worst-case scenario; 2021-09 confirms we don't destroy already-good performance.

## Hypothesis A (Primary): Moderate Class Weight Boost + Lower min_child_weight

**What**: Increase tier 1 weight from 5→15 (matching tier 0 at 15), increase tier 2 from 2→3, lower min_child_weight from 25→10.

**Why**: Tier 1 weight is currently 3x lower than tier 0 (5 vs 10), yet tier 1 recall is 4x worse (0.098 vs 0.374). Equalizing tier 0 and 1 weights should dramatically improve tier 1 detection. Lower min_child_weight allows finer splits for rare classes.

**Overrides**:
```json
{"tier": {"class_weights": {"0": 15, "1": 15, "2": 3, "3": 1, "4": 0.5}, "min_child_weight": 10}}
```

**Expected impact**:
- Tier-Recall@1: 0.098 → 0.20+ (most critical improvement)
- Tier-VC@100: 0.075 → 0.10+ (better rare-class detection → better EV ranking)
- QWK: may slightly improve from better ordinal consistency
- Tier-Accuracy: may dip slightly as more predictions shift to tier 0/1 (acceptable trade-off)

## Hypothesis B (Alternative): Aggressive Class Weight Boost + More Capacity

**What**: Push weights further (tier 0→20, tier 1→20), lower min_child_weight more aggressively (25→5), add more trees (400→600).

**Why**: If Hypothesis A doesn't push hard enough, this tests the upper bound. More trees compensate for the increased difficulty of learning with extreme class weights.

**Overrides**:
```json
{"tier": {"class_weights": {"0": 20, "1": 20, "2": 3, "3": 1, "4": 0.3}, "min_child_weight": 5, "n_estimators": 600}}
```

**Expected impact**:
- Tier-Recall@1: 0.098 → 0.25+ (aggressive target)
- Tier-VC@100: possible larger improvement if extreme weights help the EV ranking
- Risk: may over-predict tier 0/1, causing false positives and degrading NDCG/QWK
- Tier-Accuracy: likely to decrease as model becomes more aggressive on rare classes

## Winner Criteria

Pick the hypothesis with **higher mean Tier-VC@100 across the 2 screen months**, UNLESS:
- QWK drops below 0.15 on either screen month (catastrophic ordinal failure), OR
- Tier-NDCG drops below 0.55 on either screen month (ranking collapse)

If both hypotheses are close on VC@100 (within 0.01), prefer the one with higher QWK.

## Code Changes for Winner

No code changes needed — both hypotheses use only `--overrides` (hyperparameter changes). The winner config should be applied via overrides to the full 12-month benchmark run.

If the winner is promoted, the worker should update `ml/config.py` TierConfig defaults to match the winning overrides for future iterations.

## Risk Assessment

1. **Over-prediction risk**: Aggressive class weights may cause the model to over-predict tier 0/1 for tier 2/3 constraints, degrading QWK and potentially NDCG. The winner criteria guards against this.
2. **Accuracy trade-off**: Tier-Accuracy (currently 0.943) will likely drop as more predictions shift toward rare classes. This is an acceptable trade-off since Tier-Accuracy is Group B (non-blocking) and dominated by majority class.
3. **Variance risk**: min_child_weight=5 (Hyp B) may increase overfitting on some months. The 2-month screen catches this if it manifests.
