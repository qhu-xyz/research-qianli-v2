# Direction — Iteration 1

## Champion: v0 (baseline)

Mean Tier-VC@100=0.075, Tier-VC@500=0.217, Tier-NDCG=0.767, QWK=0.359, Tier-Recall@1=0.098

## Focus: Fix Tier-Recall@1 + Improve Tier-VC@100

The v0 baseline shows the model almost never predicts tier 1. This is the single biggest improvement opportunity.

---

## Hypothesis A (Primary): Aggressive Class Weight + Lower min_child_weight

**What**: Increase tier 1 class weight from 5→15, tier 0 from 10→15. Lower min_child_weight from 25→10.

**Why**: Tier-Recall@1 is 0.098 — the model barely learns tier 1 patterns. Higher class weight forces more attention to these samples. Lower min_child_weight allows finer-grained splits that can detect rare tier patterns.

**Overrides**:
```json
{"tier": {"class_weights": {"0": 15, "1": 15, "2": 2, "3": 1, "4": 0.5}, "min_child_weight": 10}}
```

---

## Hypothesis B (Alternative): Same Weights + More Trees

**What**: Same class weight changes as A, but keep min_child_weight=25 and increase n_estimators from 400→800.

**Why**: Tests whether capacity (more trees) is more effective than flexibility (smaller leaves) for rare class detection. More trees with high min_child_weight = more ensemble averaging without overfitting risk.

**Overrides**:
```json
{"tier": {"class_weights": {"0": 15, "1": 15, "2": 2, "3": 1, "4": 0.5}, "n_estimators": 800}}
```

---

## Screen Months

- **Weak month: 2022-06** — Worst NDCG (0.629), worst VC@500 (0.047). Tests whether weight changes help the hardest month.
- **Strong month: 2020-09** — Best Tier-Recall@0 (0.680) and Recall@1 (0.197). Regression sentinel.

---

## Winner Criteria

Pick the hypothesis with **higher mean Tier-VC@100 across the 2 screen months**, with:
1. Tiebreaker: If within ±5% on VC@100, prefer higher Tier-Recall@1.
2. **Veto**: If QWK drops > 0.03 on either screen month vs v0, disqualify.
3. If both pass, use Tier-VC@100 as primary selector.

---

## Expected Impact

| Metric | Hyp A (weights + mcw) | Hyp B (weights + trees) |
|--------|----------------------|------------------------|
| Tier-Recall@1 | +50-200% (from 0.098) | +50-200% |
| Tier-VC@100 | +20-100% | +20-80% |
| QWK | +0-10% (better ordinal) | +0-5% |
| Tier-Accuracy | -1-5% (more tier 0/1 predictions) | -1-3% |
