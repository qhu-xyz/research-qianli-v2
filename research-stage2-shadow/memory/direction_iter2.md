# Direction — Iteration 2 (batch ralph-v2-20260304-031811)

## Context

Champion is **v0** (baseline). The committed code defaults are now v0005's config (reg_lambda=5.0, mcw=25) — these are the new defaults in `ml/config.py`. Iteration 2 tests additional changes ON TOP of L2=5/mcw=25.

v0005 improved EV-VC@100 +6.5% and EV-VC@500 +5.9% over v0 but was blocked from promotion by Spearman L1 (mean 0.3920 < floor 0.3928, miss by 0.0008). The gate calibration is dysfunctional (floors at v0 exact mean), but we must work within it.

### v0005 Baseline (current committed defaults)

| Metric | Mean | Bot-2 |
|--------|------|-------|
| EV-VC@100 | 0.0735 | 0.0084 |
| EV-VC@500 | 0.2287 | 0.0689 |
| EV-NDCG | 0.7501 | 0.6458 |
| Spearman | 0.3920 | 0.2669 |

### v0 Champion (gate floors)

| Metric | Mean (=Floor) | Bot-2 |
|--------|--------------|-------|
| EV-VC@100 | 0.0690 | 0.0068 |
| EV-VC@500 | 0.2160 | 0.0558 |
| EV-NDCG | 0.7472 | 0.6476 |
| Spearman | 0.3928 | 0.2689 |

**To promote, the new version must beat ALL gate floors** (= v0 exact mean). The critical constraint is Spearman ≥ 0.3928 while maintaining EV-VC gains.

---

## Hypothesis A (primary) — Depth Reduction (depth=5→4)

**What**: Reduce max_depth from 5 to 4, keeping L2=5/mcw=25.

**Rationale**: L2=5/mcw=25 slightly reduced Spearman (-0.0008) because L2 compresses leaf weights, reducing prediction variance. Deeper trees (depth=5, 32 leaves) create more leaf-level variation that L2 must suppress. Reducing to depth=4 (16 leaves) reduces the number of interaction-driven leaf splits, giving L2 less to suppress. This should recover Spearman while maintaining most EV-VC gains.

Prior data supports this direction: depth=3 + L2 (v0004 in ralph-v1) doubled EV-VC@100 bottom-2 and improved Spearman (+0.0001 over v0) but sacrificed mean EV-VC. depth=4 is the untested middle ground.

**Overrides**:
```json
{"regressor": {"max_depth": 4}}
```
(reg_lambda=5.0 and min_child_weight=25 are already in the committed defaults)

**Expected impact**: Spearman recovery +0.001-0.003 (enough to pass L1). EV-VC@100 mean could be flat to slightly down (16 leaves captures fewer interactions than 32). Tail safety (bot-2) should improve.

**Risk**: Low — depth=3 and depth=5 are both tested. depth=4 interpolates between known outcomes.

---

## Hypothesis B (alternative) — L1 Regularization (reg_alpha=1.0)

**What**: Increase L1 penalty from 0.1 to 1.0, keeping L2=5/mcw=25.

**Rationale**: L1 (Lasso) penalty drives some feature weights to exactly zero, performing implicit feature selection. With 34 features (some with unclear value, e.g. `density_entropy`, `prob_band_95_100`), noisy features may be hurting Spearman by introducing random prediction noise. L1=1.0 could zero out the weakest features, improving rank correlation stability.

This is a DIFFERENT regularization axis from L2:
- L2 (Ridge) = shrinks all weights toward zero (variance reduction)
- L1 (Lasso) = zeros out some weights entirely (feature selection)

The two should complement rather than compete.

**Overrides**:
```json
{"regressor": {"reg_alpha": 1.0}}
```

**Expected impact**: Spearman improvement if noisy features are the cause of rank correlation instability. EV-VC potentially unchanged or slightly improved (fewer noise sources). If many features are zeroed, could hurt EV-VC on months that need those features.

**Risk**: Medium — reg_alpha=1.0 is a 10x increase from 0.1. Could be too aggressive if most features carry signal. The 34-feature set is heterogeneous (probability bands, historical signals, interaction features) — some are likely redundant.

---

## Screen Months

**Weak month: 2021-11**
- Worst Spearman for v0005 (0.2635, below tail_floor 0.2649)
- Moderate EV-VC@100 (0.0487) — not the worst, so improvement signal isn't dominated by a catastrophic baseline
- The Spearman weakness here is the primary diagnostic target

**Strong month: 2022-12**
- Best EV-VC@100 (0.1988), strong EV-NDCG (0.8226)
- Standard regression canary — if a config hurts this month, the regularization is too aggressive
- Used in iter 1 screening — enables cross-iteration comparison

---

## Winner Criteria

1. **Primary**: Higher mean Spearman across the 2 screen months (2021-11 and 2022-12)
   - Rationale: Spearman is the binding constraint for promotion. We need to find a config that passes Spearman L1 ≥ 0.3928.
2. **Override**: If the Spearman winner has EV-VC@100 drop > 0.01 below v0005 on either screen month, pick the other hypothesis instead (value capture regression is unacceptable)
3. **Tiebreak** (if Spearman difference < 0.002): Higher mean EV-VC@100 across screen months
4. **Default**: If both are within noise of each other, prefer A (simpler config, lower risk)
5. **Reject both**: If BOTH hypotheses degrade EV-VC@100 on 2022-12 by >10% vs v0005 screen AND fail to improve Spearman on 2021-11 by at least +0.003, flag for orchestrator — both changes harm value capture without meaningful Spearman recovery

---

## Code Changes for Winner

### If Hypothesis A wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `max_depth: int = 5` → `max_depth: int = 4`

Update `tests/test_config.py` — adjust any assertions on default values.

### If Hypothesis B wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `reg_alpha: float = 0.1` → `reg_alpha: float = 1.0`

Update `tests/test_config.py` — adjust any assertions on default values.

---

## DO NOT MODIFY

- Any file in `ml/` other than `config.py` (and `tests/test_config.py` for test assertions)
- `registry/gates.json`
- `ml/evaluate.py`
- Any classifier configuration (ClassifierConfig is FROZEN)
- `state.json` (orchestrator manages this)

---

## v0005 Screen Baselines (from iter 1)

For comparison, v0005's values on the screen months:

| Metric | 2021-11 | 2022-12 |
|--------|---------|---------|
| EV-VC@100 | 0.0487 | 0.1988 |
| EV-VC@500 | 0.1903 | 0.3552 |
| EV-NDCG | 0.7545 | 0.8226 |
| Spearman | 0.2635 | 0.3857 |
| C-RMSE | 4738 | 3015 |
