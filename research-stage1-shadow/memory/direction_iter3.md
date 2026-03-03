# Direction — Iteration 3 (feat-eng-20260303-060938)

## Hypothesis H8: Feature Pruning (17→13 features) + Revert Training Window to 14 Months

**Core question**: Does removing the 4 near-zero-importance features reduce noise and improve tail metrics (especially AP bottom-2), while reverting from the 18-month window (no benefit) to the optimal 14-month window?

**Rationale**: After 5 real-data experiments, two key facts are established:
1. **Feature importance data** (from iter 2) shows 4 features contribute <2% of total gain collectively: density_skewness (0.31%), exceed_severity_ratio (0.38%), density_cv (0.40%), density_kurtosis (0.58%). These consume model capacity without contributing signal.
2. **AP bottom-2 has been monotonically worsening** across window expansion experiments: v0002(-0.0017) → v0003(-0.0045) → v0004(-0.0040) → v0005(-0.0075). Removing noise features may help the model focus on informative features in the weakest months.

**Context — This is the final iteration.** The batch has 3 iterations total. Regardless of outcome, the orchestrator will produce an executive summary for HUMAN_SYNC. The decision framework is:
- If v0006 shows meaningful improvement (AUC ≥ 0.837 or AP bot2 reversal): promote
- If v0006 is roughly equal to v0004 but AP bot2 improves: consider promoting v0004 or v0006 as modest improvement over v0
- If v0006 is neutral/negative: declare the feature set ceiling reached, recommend new signal sources at HUMAN_SYNC

## Specific Changes (Priority Order)

### Change 1 (PRIMARY): Remove 4 near-zero features from FeatureConfig

**File**: `ml/config.py` → `FeatureConfig.step1_features`
**What**: Remove these 4 features:

| Feature | Rank (of 17) | % Gain | Monotone | Reason for removal |
|---------|-------------|--------|----------|-------------------|
| `density_skewness` | 17 | 0.31% | 0 (unconstrained) | Lowest gain, unconstrained adds fitting noise |
| `exceed_severity_ratio` | 16 | 0.38% | 1 | Weakest interaction feature, not earning its keep |
| `density_cv` | 15 | 0.40% | 0 (unconstrained) | Near-zero gain, distribution shape noise |
| `density_kurtosis` | 14 | 0.58% | 0 (unconstrained) | Near-zero gain, distribution shape noise |

**After edit**, the `step1_features` list should contain exactly **13 features**:
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
        # --- Interaction features (retained: top 2 of 3) ---
        ("hist_physical_interaction", 1),
        ("overload_exceedance_product", 1),
    ]
)
```

**Why these 4 and not others**: They are the bottom 4 by gain-based importance, each <0.6% of total gain, collectively 1.67%. The next feature above them (overload_exceedance_product at 0.90%) is 2.4x more important than the highest pruned feature (density_kurtosis at 0.58%), providing a natural cutoff. The 3 distribution shape features (skewness, kurtosis, CV) are ALL the unconstrained monotone features (monotone=0) — they may introduce fitting noise since the model can assign arbitrary direction to them.

### Change 2 (PRIMARY): Revert training window from 18 to 14 months

**File**: `ml/config.py` → `PipelineConfig`, line 95
**What**: Change `train_months: int = 18` → `train_months: int = 14`

v0004 (14-month window) is strictly better than v0005 (18-month window) on all Group A means. Reverting returns to the optimal configuration established in iter 1.

### Change 3: No other changes

- Keep all HPs at v0 defaults in `HyperparamConfig`. Do NOT modify.
- Keep `threshold_beta: float = 0.7` (precision-favoring). Do NOT modify.
- Keep `val_months: int = 2`. Do NOT modify.
- Keep feature importance extraction in benchmark.py (collect 13-feature importance for comparison).

### Change 4: Update tests

- Update any test assertions for feature count: expect **13** features (was 17)
- Update `train_months` assertion back to **14** (was 18)
- Verify monotone constraints string has 13 values, not 17
- Remove any references to the 4 dropped feature names in test fixtures

## Expected Impact

| Metric | v0 Baseline | v0004 (best) | v0005 (iter 2) | Expected v0006 | Rationale |
|--------|-------------|--------------|----------------|----------------|-----------|
| S1-AUC | 0.8348 | 0.8363 | 0.8361 | 0.835–0.837 | Neutral or slight positive. Removing noise features may marginally help. |
| S1-AP | 0.3936 | 0.3951 | 0.3929 | 0.393–0.397 | Slight positive expected if noise features were hurting AP in tail months. |
| S1-VCAP@100 | 0.0149 | 0.0205 | 0.0193 | 0.015–0.022 | Near v0004 level. Fewer features = less noise-driven top-K reordering. |
| S1-NDCG | 0.7333 | 0.7371 | 0.7365 | 0.734–0.738 | Neutral to slight positive. |
| AP Bot2 | 0.3322 | 0.3282 | 0.3247 | 0.325–0.335 | Key metric: reverting window + pruning noise should partially reverse trend. |
| BRIER | 0.1503 | 0.1516 | 0.1525 | 0.149–0.153 | May improve — simpler model, v0003-HP showed complexity↓ = BRIER↓. |

**Honest assessment**: The realistic outcome is neutral — the pruned features contribute so little (<2%) that removing them may not produce a measurable change. The optimistic scenario requires that these features were actively introducing noise that degraded tail months. The primary value is: (a) establishing whether feature pruning is a viable lever, and (b) completing the batch with a clean experimental conclusion.

## Risk Assessment

1. **Neutral result (HIGH probability)**: Pruning 4 features that contribute <2% may have zero measurable effect. Mitigation: even a null result is informative — it confirms the remaining 13 features define the model's ceiling, and the next improvement requires fundamentally new features.

2. **VCAP@500 bot2 risk (MEDIUM probability)**: Reverting to 14-month window may reproduce v0004's VCAP@500 bot2 of 0.0387 (below floor 0.0408). The 18-month window stabilized this in v0005 (bot2=0.0449). Mitigation: Group B (non-blocking). Document if it occurs.

3. **AP mean regression (LOW-MEDIUM probability)**: If the pruned features, despite low gain, provided useful regularization, AP could drop. Mitigation: if AP drops >0.003 vs v0004, document that pruning was too aggressive.

4. **Feature importance comparison (LOW risk)**: With 13 instead of 17 features, importance percentages will redistribute among retained features. This is expected and useful for analysis.

## Success Criteria

| Outcome | Criteria | Action |
|---------|----------|--------|
| **Promotion-worthy** | AUC ≥ 0.837, AP > 0.396, AP bot2 > 0.330, ≥8/12 AUC wins vs v0 | Promote v0006 as new champion |
| **Best-so-far** | AUC ≥ v0004 (0.8363) AND (AP > v0004 OR AP bot2 > v0004) | v0006 becomes top candidate. Recommend promotion at HUMAN_SYNC. |
| **Ceiling confirmed** | v0006 within ±0.002 of v0004 on all metrics | Declare ceiling reached. Recommend v0004 as best version. Summarize for HUMAN_SYNC. |
| **Regression** | AUC < v0 (0.8348) or AP < 0.390 | Feature pruning harmful. Recommend v0004 config at HUMAN_SYNC. |

## Worker Checklist

1. Read VERSION_ID from `${PROJECT_DIR}/state.json` (NOT the worktree copy)
2. Remove 4 features from `ml/config.py` FeatureConfig: `density_skewness`, `density_kurtosis`, `density_cv`, `exceed_severity_ratio` (remove the full tuple entries and their comments)
3. Change `train_months` from 18 to 14 in `ml/config.py` PipelineConfig
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
- Do NOT remove hist_physical_interaction or overload_exceedance_product (they contribute meaningful gain: 14.3% and 0.9% respectively)
- Do NOT expand or change the training window beyond 14 months
- Do NOT add new features — this iteration tests pruning only
