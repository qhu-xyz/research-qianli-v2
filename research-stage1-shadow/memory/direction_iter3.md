# Direction — Iteration 3 (feat-eng-20260303-060938)

## Hypothesis H8: Feature Pruning (17→13 features) + Revert Training Window to 14 Months

**Core question**: Does removing the 4 near-zero-importance features reduce noise and improve tail metrics (especially AP bottom-2), while reverting from the 18-month window (no benefit) to the optimal 14-month window?

**Rationale**: After 5 real-data experiments, two key facts are established:
1. **Feature importance data** (from iter 2) shows 4 features contribute <2% of total gain collectively: density_skewness (0.31%), exceed_severity_ratio (0.38%), density_cv (0.40%), density_kurtosis (0.58%). These consume model capacity without contributing signal.
2. **AP bottom-2 has been monotonically worsening** across window expansion experiments: v0002(-0.0017) → v0003(-0.0045) → v0004(-0.0040) → v0005(-0.0075). Removing noise features may help the model focus on informative features in the weakest months.

**Context — This is the final iteration.** The batch has 3 iterations total. Regardless of outcome, the orchestrator will produce an executive summary. The decision framework is:
- If v0006 shows meaningful improvement (AUC ≥ 0.837 or AP bot2 reversal): promote
- If v0006 is roughly equal to v0004 but AP bot2 improves: consider promoting v0004 or v0006 as modest improvement over v0
- If v0006 is neutral/negative: declare the feature set ceiling reached, recommend new signal sources at HUMAN_SYNC

## Specific Changes (Priority Order)

### Change 1 (PRIMARY): Remove 4 near-zero features from FeatureConfig

**File**: `ml/config.py`, lines 30-32 and 38
**What**: Remove these 4 features from `step1_features`:
- `("density_skewness", 0)` — line 31, gain=0.31%
- `("density_kurtosis", 0)` — line 32 (renumbered after deletion), gain=0.58%
- `("density_cv", 0)` — line 33 (renumbered), gain=0.40%
- `("exceed_severity_ratio", 1)` — line 38, gain=0.38%

**After edit**, the `step1_features` list should contain exactly 13 features:
```python
step1_features: list[tuple[str, int]] = field(
    default_factory=lambda: [
        # --- Density exceedance probabilities (core 5) ---
        ("prob_exceed_110", 1),
        ("prob_exceed_105", 1),
        ("prob_exceed_100", 1),
        ("prob_exceed_95", 1),
        ("prob_exceed_90", 1),
        # --- Density below-threshold probabilities ---
        ("prob_below_100", -1),
        ("prob_below_95", -1),
        ("prob_below_90", -1),
        # --- Severity signal ---
        ("expected_overload", 1),
        # --- Historical DA shadow price ---
        ("hist_da", 1),
        ("hist_da_trend", 1),
        # --- Interaction features ---
        ("hist_physical_interaction", 1),
        ("overload_exceedance_product", 1),
    ]
)
```

**Why these 4 and not others**: They are the bottom 4 by gain-based importance, each <0.6% of total gain, collectively 1.67%. The next feature above them (overload_exceedance_product at 0.90%) is 2.4x more important than the highest pruned feature (density_kurtosis at 0.58%), providing a natural cutoff. The 3 distribution shape features (skewness, kurtosis, CV) are also the only unconstrained features (monotone=0) — they may introduce fitting noise since the model can assign arbitrary direction to them.

### Change 2 (PRIMARY): Revert training window from 18 to 14 months

**File**: `ml/config.py`, line 95
**What**: Change `train_months: int = 18` → `train_months: int = 14`

v0004 (14-month window) is strictly better than v0005 (18-month window) on all Group A means. Reverting returns to the optimal configuration.

### Change 3: No other changes

- Keep all HPs at v0 defaults in `HyperparamConfig`. Do NOT modify.
- Keep `threshold_beta: float = 0.7` (precision-favoring). Do NOT modify.
- Keep `val_months: int = 2`. Do NOT modify.
- Keep feature importance extraction in benchmark.py (for comparison with iter 2 importance).

### Change 4: Update tests

- Update any test assertions for feature count: expect 13 features (was 17)
- Update `train_months` assertion back to 14 (was 18)
- Verify monotone constraints string has 13 values, not 17

## Expected Impact

| Metric | v0 Baseline | v0004 (best) | Expected v0006 (optimistic) | Expected v0006 (realistic) |
|--------|-------------|--------------|----------------------------|---------------------------|
| S1-AUC | 0.8348 | 0.8363 | 0.836–0.838 | 0.835–0.837 |
| S1-AP | 0.3936 | 0.3951 | 0.396–0.400 | 0.393–0.396 |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.020–0.025 | 0.015–0.022 |
| S1-NDCG | 0.7333 | 0.7371 | 0.737–0.740 | 0.734–0.738 |
| AP Bot2 | 0.3322 | 0.3282 | 0.330–0.340 (improvement) | 0.325–0.335 |
| BRIER | 0.1503 | 0.1516 | 0.148–0.152 (improvement) | 0.150–0.153 |

**Honest assessment**: The realistic outcome is neutral — the pruned features contribute so little (<2%) that removing them may not produce a measurable change. The optimistic scenario requires that these features were actively introducing noise that degraded tail months. The primary value is: (a) establishing whether feature pruning is a viable path, and (b) collecting feature importance data for the 13-feature model to compare against the 17-feature model.

## Risk Assessment

1. **Neutral result (HIGH probability)**: Pruning 4 features that contribute <2% may have zero measurable effect. Mitigation: even a null result is informative — it confirms the remaining 13 features define the model's ceiling, and the next improvement requires fundamentally new features.

2. **AP mean regression (LOW-MEDIUM probability)**: If the pruned features, despite low gain, provided useful regularization, AP could drop. Mitigation: if AP mean drops by >0.003 vs v0004, the pruning was too aggressive. Document and recommend restoring features at HUMAN_SYNC.

3. **VCAP@100 regression (LOW probability)**: The pruned features have near-zero importance — unlikely to affect top-100 ranking. But removing unconstrained features could subtly change the score distribution.

4. **Feature importance extraction compatibility (VERY LOW)**: With 13 instead of 17 features, the feature importance export should work identically (just fewer features). No code change needed.

## Success Criteria

| Outcome | Criteria | Action |
|---------|----------|--------|
| **Promotion-worthy** | AUC ≥ 0.837, AP > 0.396, AP bot2 > 0.330, ≥8/12 AUC wins vs v0 | Promote v0006 as new champion |
| **v0004 promotion** | v0006 ≈ v0004 on means, AP bot2 improved. v0004 remains best config. | Consider promoting v0004 as modest improvement over v0 (pending HUMAN_SYNC) |
| **Ceiling confirmed** | v0006 within ±0.002 of v0004 on all metrics | Declare ceiling reached. Recommend new signal sources at HUMAN_SYNC. |
| **Regression** | AUC < v0 (0.8348) or AP < 0.390 | Feature pruning harmful. Recommend reverting to v0004 config. |

## Worker Checklist

1. Read VERSION_ID from `${PROJECT_DIR}/state.json` (NOT the worktree copy)
2. Remove 4 features from `ml/config.py` FeatureConfig: `density_skewness`, `density_kurtosis`, `density_cv`, `exceed_severity_ratio` (remove the full tuple entries)
3. Change `train_months` from 18 to 14 in `ml/config.py` line 95
4. Update tests: `python -m pytest ml/tests/ -v` — fix any feature count or train_months assertions
5. Run benchmark: `python ml/benchmark.py --version-id ${VERSION_ID} --ptype f0 --class-type onpeak`
6. Verify `registry/${VERSION_ID}/metrics.json` has 12 months (none skipped)
7. Verify `registry/${VERSION_ID}/feature_importance.json` was created with 13 features
8. Run `python ml/validate.py --version-id ${VERSION_ID}` to confirm gate compliance
9. Run `python ml/compare.py --version-id ${VERSION_ID} --baseline v0` to generate comparison
10. Write `registry/${VERSION_ID}/changes_summary.md` with actual results
11. Commit all changes, then write handoff JSON

## What NOT To Do

- Do NOT change hyperparameters (confirmed dead end — 5 experiments)
- Do NOT change threshold_beta (keep 0.7, business requires precision > recall)
- Do NOT change val_months (keep 2)
- Do NOT modify gates.json or evaluate.py
- Do NOT touch registry/v0/ or any other registry/v*/ except the assigned VERSION_ID
- Do NOT remove hist_physical_interaction or overload_exceedance_product (they contribute meaningful gain)
- Do NOT expand or change the training window beyond 14 months
- Do NOT add new features — this iteration tests pruning only
