# Direction — Iteration 3 (FINAL) (batch ralph-v2-20260304-031811)

## Context

Champion is **v0** (baseline). The committed code defaults are v0005/v0006 config: reg_lambda=5.0, mcw=25, reg_alpha=1.0 (reg_alpha change had zero effect). Iteration 3 tests changes on top of these defaults.

**Iter 2 was wasted by a config bug**: v0006's full benchmark ran with reg_alpha=0.1 instead of 1.0, producing metrics identical to v0005. The only valid signal was the 2-month screen showing neither depth=4 nor L1=1.0 recovered Spearman. Regularization axis (L2, L1, depth, subsampling) is EXHAUSTED for Spearman recovery.

**The strategic question**: v0005 changed TWO parameters from v0: reg_lambda (1.0→5.0) AND min_child_weight (10→25). The Spearman compression (-0.0008) could be driven by either or both. This iteration decomposes the effect to find which axis preserves Spearman while retaining EV-VC gains.

### Current Committed Defaults (= v0005/v0006 config)

| Param | Value | v0 was |
|-------|-------|--------|
| reg_lambda | 5.0 | 1.0 |
| min_child_weight | 25 | 10 |
| reg_alpha | 1.0 | 0.1 |
| max_depth | 5 | 5 |
| learning_rate | 0.05 | 0.05 |
| n_estimators | 400 | 400 |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |

### v0005 Performance (= current committed defaults, confirmed stable)

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

**To promote, the new version must beat ALL gate floors** (= v0 exact mean). The critical constraint is Spearman ≥ 0.3928 while maintaining EV-VC@100 > 0.0690 and EV-NDCG > 0.7472.

---

## Hypothesis A (primary) — L2-Only: reg_lambda=3.0, mcw=10

**What**: Set reg_lambda=3.0 (moderate L2), revert mcw to v0's default of 10.

**Rationale**: v0005 changed BOTH reg_lambda (1→5) and mcw (10→25). The Spearman compression could be caused by either. This isolates the L2 effect:
- reg_lambda=3.0 provides moderate L2 penalty (between v0's 1.0 and v0005's 5.0)
- mcw=10 (v0's default) removes the leaf-size constraint that could compress predictions
- If Spearman stays near 0.3928, it means mcw was the Spearman problem, not L2
- If Spearman drops, L2 is intrinsically anti-Spearman

**Overrides**:
```json
{"regressor": {"reg_lambda": 3.0, "min_child_weight": 10}}
```

**Expected impact**: Spearman recovery toward v0's 0.3928 (mcw=10 allows smaller leaves → more prediction variance → better rank correlation). EV-VC@100 should be above v0 (L2=3 provides regularization benefit) but possibly below v0005 (less aggressive L2). Target: Spearman ≥ 0.392, EV-VC@100 ≈ 0.071-0.073.

**Risk**: Low — interpolates between two known points (v0 and v0005), with parameter decomposition.

---

## Hypothesis B (alternative) — MCW-Only: reg_lambda=1.0, mcw=25

**What**: Revert reg_lambda to v0's default of 1.0, keep mcw at 25.

**Rationale**: This isolates the mcw effect:
- reg_lambda=1.0 (v0's default) removes L2 leaf-weight penalty
- mcw=25 (v0005's value) retains the leaf-size constraint
- If Spearman stays near 0.3928, it means L2 was the Spearman problem, not mcw
- If Spearman drops, mcw is intrinsically anti-Spearman

**Overrides**:
```json
{"regressor": {"reg_lambda": 1.0}}
```
(mcw=25 is already the committed default, no override needed)

**Expected impact**: Spearman near v0 (no L2 compression). EV-VC@100 improved if mcw=25's leaf-size regularization provides independent value capture benefit. Target: Spearman ≈ 0.392-0.393, EV-VC@100 ≈ 0.069-0.073.

**Risk**: Low — close to v0 on the L2 axis, with mcw providing orthogonal regularization.

---

## Screen Months

**Weak month: 2021-11**
- Worst Spearman for v0005 (0.2635, below tail_floor 0.2649)
- v0 Spearman on this month: 0.2649 — recovery target
- Cross-iter comparable (used in iter 2 screen)

**Strong month: 2022-12**
- Best EV-VC@100 for v0005 (0.1988)
- v0 EV-VC@100 on this month: 0.1942
- Cross-iter comparable (used in iters 1 and 2)

---

## Winner Criteria

1. **Primary**: Higher mean Spearman across the 2 screen months (2021-11 and 2022-12)
   - Rationale: Spearman is the binding constraint for promotion. We need the config with best Spearman.
2. **Override**: If the Spearman winner has EV-VC@100 below 0.065 on EITHER screen month (more than 5% below v0 floor), pick the other hypothesis instead
3. **Tiebreak** (if Spearman difference < 0.002): Higher mean EV-VC@100 across screen months
4. **Default**: If both are within noise, prefer A (moderate L2 provides known EV-VC benefit)
5. **Reject both**: If BOTH hypotheses have Spearman below v0 on 2021-11 (below 0.2649) AND EV-VC@100 below v0 on 2022-12 (below 0.1942), reject both — the decomposition shows neither axis alone provides benefit

---

## v0005 Screen Baselines (from iter 2)

For comparison, v0005's values on the screen months:

| Metric | 2021-11 | 2022-12 |
|--------|---------|---------|
| EV-VC@100 | 0.0487 | 0.1988 |
| EV-VC@500 | 0.1903 | 0.3552 |
| EV-NDCG | 0.7545 | 0.8226 |
| Spearman | 0.2635 | 0.3857 |
| C-RMSE | 4738 | 3015 |

v0 values on the screen months:

| Metric | 2021-11 | 2022-12 |
|--------|---------|---------|
| EV-VC@100 | 0.0497 | 0.1942 |
| EV-VC@500 | 0.1819 | 0.3445 |
| EV-NDCG | 0.7545 | 0.8153 |
| Spearman | 0.2649 | 0.3847 |

---

## Code Changes for Winner

### If Hypothesis A wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `reg_lambda: float = 5.0` → `reg_lambda: float = 3.0`
- `min_child_weight: int = 25` → `min_child_weight: int = 10`

Update `ml/tests/test_config.py` — adjust any assertions on default values.

### If Hypothesis B wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `reg_lambda: float = 5.0` → `reg_lambda: float = 1.0`

Update `ml/tests/test_config.py` — adjust any assertions on default values.

---

## ⚠️ CONFIG BUG PREVENTION (CRITICAL)

**Iter 2 had a config bug** — the full benchmark ran with old defaults instead of the intended overrides, producing invalid results. To prevent this:

1. **VERIFY config.json AFTER benchmark**: After the full benchmark completes, check `registry/<version>/config.json` to confirm the intended config was used:
   - If Hyp A won: verify `reg_lambda=3.0` AND `min_child_weight=10` in config.json
   - If Hyp B won: verify `reg_lambda=1.0` in config.json
2. **If config.json doesn't match**: The benchmark ran with wrong config. Do NOT proceed with comparison/commit. Re-run the benchmark.
3. **Make code changes BEFORE running benchmark** so the defaults are in place when the benchmark reads config.

---

## DO NOT MODIFY

- Any file in `ml/` other than `config.py` (and `tests/test_config.py` for test assertions)
- `registry/gates.json`
- `ml/evaluate.py`
- Any classifier configuration (ClassifierConfig is FROZEN)
- `state.json` (orchestrator manages this)
- `pipeline.py` (even though `value_weighted` is unwired — that's a future pipeline fix)
