# Research Findings: Shadow Price Prediction Pipeline

> **Last updated**: 2026-02-25
> **For**: AI-to-AI handoff — provides full context, progress, concerns, and open questions
> **Project**: `research-spice-shadow-price-pred-qianli` (SPICE_F0P_V6.7B.R1)
> **Champion**: v008-focused-classifier (S1-AUC=0.69, C-RMSE=1109)
> **In Progress**: v012-unified-regressor (running, 9/32 parquets complete at last check)

---

## 1. What We Are Building

A two-stage ML pipeline predicting which MISO transmission constraints will bind in FTR auctions and estimating their shadow prices. Output drives the CIA Multistep trading workflow.

**Stage 1**: XGBClassifier → P(binding) per constraint per outage date
**Stage 2**: XGBRegressor → log1p(shadow_price), then expm1 back to $/MWh

Inputs: density flow probability parquets + historical DA shadow prices (no raw power flow data, no topology data).

---

## 2. Benchmark Scope

32 runs: PY20+PY21 × {onpeak, offpeak} × {f0, f1} × 8 auction months
Auction months: 2020-07, 2020-10, 2021-01, 2021-04, 2021-07, 2021-10, 2022-01, 2022-04

**4 evaluated segments**: onpeak/f0, onpeak/f1, offpeak/f0, offpeak/f1
Per-period gate enforcement: each segment must independently pass all gate floors.

---

## 3. Version History & Key Results

### Promotion Gate Values (threshold-independent metrics, 32-run average)

| Version | S1-AUC | S1-AP | R-REC@500 | C-VC@100 | C-VC@500 | C-VC@1000 | C-NDCG | C-RMSE | Promotable |
|---------|--------|-------|-----------|----------|----------|-----------|--------|--------|-----------|
| **v000-legacy** (baseline) | 0.6953 | 0.2175 | 0.4303 | 0.3661 | 0.7029 | 0.8484 | 0.3628 | 1328 | N/A |
| v007-enriched-features | 0.683 | — | — | — | — | 0.82 | — | 1057 | No (S2-SPR floors) |
| **v008-focused-classifier** | 0.6895 | 0.2033 | 0.3960 | 0.3377 | 0.7098 | 0.8429 | 0.3586 | 1109 | **Yes** |
| v009-single-threshold (0.7) | — | — | — | — | — | — | — | — | No (S1-REC fails onpeak/f1) |
| v010-value-weighted-ev | 0.6881 | 0.1916 | 0.3904 | 0.3255 | 0.7037 | 0.8421 | 0.3514 | 1105 | No (S1-REC fails onpeak/f1) |
| v011-reg-vw-ev | 0.6895 | 0.2033 | 0.3960 | 0.3377 | 0.7098 | 0.8429 | 0.3586 | 1098 | No (doesn't beat v000 on all gates) |
| **v012-unified-regressor** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (in progress) |

Floors: S1-AUC≥0.65, S1-AP≥0.12, R-REC@500≥0.30, C-VC@100≥0.20, C-VC@500≥0.45, C-VC@1000≥0.50, C-NDCG≥0.30, C-RMSE≤2000

### Monitoring Metrics (threshold-dependent, for reference only)

| Version | S1-REC | S1-PREC | S1-F1 | S2-SPR | C-CAP@20 | C-CAP@200 | C-CAP@1000 |
|---------|--------|---------|-------|--------|---------|---------|---------|
| v000-legacy | 0.39 | 0.24 | 0.30 | — | — | — | — |
| v008-focused-classifier | 0.29 | 0.48 | 0.36 | 0.36 | ~0.80 | ~0.55 | ~0.25 |
| v011-reg-vw-ev | 0.29 | 0.48 | 0.36 | 0.40 | ~0.80 | ~0.55 | ~0.25 |

---

## 4. Critical Finding: The Binary Gate Bottleneck

This is the most important architectural insight from our research.

### The Problem

The standard pipeline has a hard binary gate between Stage 1 and Stage 2:
```
Stage 1: classify → P(binding) ≥ threshold → predicted_binding = 1
Stage 2: regressor ONLY runs if predicted_binding = 1
         All other constraints → shadow_price = 0 (by definition)
```

This creates a **structural coverage problem**:
- v000 (threshold ~0.5): 46% of binding constraints get non-zero shadow price predictions
- v008/v011 (threshold ~0.9): 42% of binding constraints covered

The regressor is trained **only on binding samples** (~5% of data). For the 95% non-binding samples, the model has never learned what "close-to-binding but not binding" looks like, and those samples always get shadow_price=0.

### Why This Corrupts "Threshold-Independent" Metrics

Even ranking metrics like R-REC@500, C-VC@K, C-NDCG are affected by the threshold because:
1. Threshold determines which constraints get non-zero shadow price predictions
2. Constraints with predicted_shadow_price=0 sink to the bottom of any ranking
3. Ranking quality depends on HOW MANY binding constraints have been assigned non-zero predictions

This is why v000 (permissive threshold) beats v008/v011 on S1-AP, R-REC@500, C-VC@100 despite weaker model quality.

### The Gate Comparison Trap

We attempted to fix this by creating "threshold-independent" promotion gates. But the fix is only partial — as long as Stage 2 is gated by Stage 1's binary output, ALL metrics are influenced by the threshold.

### The Proposed Fix: Unified Regressor (v012)

Train the regressor on ALL samples with target=`log1p(max(0, shadow_price))`:
- Non-binding samples: target = 0
- Binding samples: target = log1p(shadow_price) > 0

At prediction time: run the regressor on ALL samples, no binary gate.

XGBoost with 400 trees + regularization should learn:
- "This type of flow distribution = near-zero shadow price"
- "This other distribution pattern = high shadow price"

One model handles both the binding/non-binding decision AND the severity estimation.

---

## 5. What Has Been Tried and Why It Didn't Solve the Problem

### Threshold Tuning (v009: threshold=0.7)
- **Result**: FAILS S1-REC on onpeak/f1 (0.228 < 0.25). Threshold=0.7 more permissive than v008's optimized ~0.9 but still not enough to compete with v000's ~0.5.
- **Why inadequate**: Threshold is a post-hoc adjustment. At 0.5, the model (v008 quality) is overconfident and assigns 0 to most constraints that should be non-zero.

### EV Scoring (v010, v011: expected_value_scoring=True)
- **Idea**: `predicted_shadow_price = P(binding) * regressor_pred`. Run regressor on all samples with P ≥ 0.05.
- **Result**: C-RMSE improved -16% (1098 vs 1308 for v000). S2-SPR improved from 0.36 to 0.40. But ranking metrics (R-REC@500, C-VC@K, C-NDCG) are **identical** to v008.
- **Why**: The regressor is still trained on binding-only data. Its predictions for non-binding samples are meaningless (it was never trained to predict them). Multiplying a meaningless prediction by a small P(binding) gives a tiny, effectively-random EV value. This doesn't help ranking — it just changes magnitudes.
- v010 also had a classifier recall collapse because value-weighting classifiers de-emphasizes low-value binding events.

### Value-Weighted Regressor (v011: value_weighted_reg=True)
- **Idea**: Weight regressor training samples by log1p(shadow_price) so high-value binding events matter more.
- **Result**: C-RMSE improvement maintained (-16%), S2-SPR improved. But ranking still identical to v008.
- **Why**: Same bottleneck — regressor only sees binding samples. Weighting within that subset helps calibration but doesn't expand coverage.

---

## 6. Pending Experiment: v012-Unified-Regressor

**Status**: Benchmark in progress — 9/32 parquets complete as of documentation.

**Config change**:
```python
TrainingConfig(unified_regressor=True)  # All other params same as v008
```

**Code changes made**:
- `models.py`: Branch regressor and default regressor both check `unified_regressor` flag. When True, trains on all samples (not binding-only).
- `prediction.py`: When `unified_regressor=True`, `regression_mask_local = np.ones(len(branch_indices), dtype=bool)` — no gate applied.
- `_experiment_worker.py`: Added `unified_regressor` override.

**Smoke test results** (2020-07/onpeak/f0):
- Default regressor trained on 1,349,036 samples (vs ~67K binding-only)
- Classifier unchanged: S1-AUC=0.712, Recall=0.342
- Regressor Spearman on TPs: 0.423 (good)
- Combined RMSE (all samples): 451.20 (excellent — much lower than v008's ~1109)
- **Concern**: Mean predicted = $468 vs mean actual = $1,302 on TPs → large negative bias. The regressor is underestimating shadow prices for binding constraints because the zero-heavy training distribution pulls predictions toward zero.
- Constraint VC@K: 34.1%@100, 70.0%@500, 82.7%@1000 — need to compare vs v008 smoke for same period.

**Risk factors for v012**:
1. **Class imbalance in regression**: 95%+ of samples are zeros. XGBoost may predict near-zero for everything, giving only weak signal for truly binding constraints. The log1p transform helps somewhat but doesn't fully correct.
2. **Monotonic constraint issue**: Step 2 features include monotonic constraints (e.g., prob_exceed_100 must have positive slope). These constraints were derived for the binding-only setting. With all samples (many zeros), the feature-target relationship might not be monotone in the expected direction for all features.
3. **Feature selection behavior**: `select_features()` uses Spearman and AUC checks. With a zero-heavy target, these checks may reject features that actually matter for the non-zero cases.
4. **Tweedie loss might fit better**: For a zero-inflated positive outcome, Tweedie regression (`objective='reg:tweedie'`) is more appropriate than squared error. Standard `reg:squarederror` will produce biased predictions in the presence of many zeros.

---

## 7. Open Hypotheses to Test (if v012 disappoints)

### H1: Tweedie Loss for Unified Regressor
Replace `objective='reg:squarederror'` with `objective='reg:tweedie'` and `tweedie_variance_power=1.5` in the unified regressor. This is the statistically correct loss for zero-inflated positive outcomes and should reduce the zero-bias problem.

### H2: Zero-Weight Downsampling
Train unified regressor on all binding samples + random 10% of non-binding samples. Preserves the "all samples" spirit while avoiding the 95% zero domination.

### H3: Fix Coverage Problem Without Unified Regressor
Lower the threshold dramatically (to 0.3 or even 0.2) while keeping the binding-only regressor. The classifier is good (AUC=0.69) — at lower thresholds it recovers more binding constraints. Trade precision for coverage.

### H4: Two-Stage Prediction with First Stage as Filter Only
Keep Stage 1 classifier at beta=0.7 (recall-heavy threshold). Run regressor on ALL Stage-1-positive samples (might be 30-40% of constraints). Then rank by regressor output, not by combined score. This is essentially a soft version of the EV scoring but with a better-calibrated regressor.

### H5: Change Training Labels to Full Shadow Price (Not Binary)
Instead of binary label for classifier, train a single XGBoost with target=shadow_price directly. Use quantile regression (`objective='reg:quantileerror'`, `quantile_alpha=0.5`) to find the median, and add a second model for the mean. This collapses Stage 1 and Stage 2 into one model but loses the monotonic constraints on Stage 1 features.

---

## 8. Architectural Decisions Made This Session

### 8a. Gate Restructuring (2026-02-25)

**Before**: `HARD_GATES` = 11 gates including S1-REC, S1-PREC, S2-SPR (threshold-dependent).
**After**: `PROMOTION_GATES` (8 threshold-independent) + `MONITORING_GATES` (7 threshold-dependent, no floors).

**Why**: Comparing v000 (threshold ~0.5) vs v008 (threshold ~0.9) using recall or Spearman-on-TPs is fundamentally unfair. The metrics change dramatically as the threshold changes. Moving them to "monitoring" (informational) prevents threshold-tuning experiments from masquerading as genuine improvements.

**Files changed**: `registry.py` (replaced HARD_GATES), `run_experiment.py` (_compute_gates now includes S1-AP, R-REC@500, C-NDCG as promotion gates).

### 8b. Value-Weighted Training Flag Split

Added two independent flags:
- `value_weighted: bool = False` — applies to classifiers (weights per-sample by log1p(shadow_price) of binding events)
- `value_weighted_reg: bool = False` — applies to regressors only

v010 showed that value-weighting the classifier causes it to underperform on low-value binding events, dropping S1-REC below the floor. Separating the flags lets us apply value weighting only where it helps (regressors).

### 8c. Expected Value Scoring Mode

Added `ThresholdConfig.expected_value_scoring=True` + `regression_prob_floor=0.05`. When enabled:
- Regressor runs on all samples with P(binding) ≥ 0.05
- Output = P(binding) × regressor_prediction

This was theoretically appealing but empirically neutral on ranking metrics because the regressor was never trained on non-binding samples (so its "predictions" for them are arbitrary).

---

## 9. Current Code State

### Files Changed This Session

| File | Changes |
|------|---------|
| `src/shadow_price_prediction/config.py` | Added `value_weighted`, `value_weighted_reg`, `unified_regressor` to `TrainingConfig`; added `expected_value_scoring`, `regression_prob_floor` to `ThresholdConfig` |
| `src/shadow_price_prediction/models.py` | Branch + default regressor training: conditional `unified_regressor` path (all samples) vs standard (binding-only). Changed regressor training from `value_weighted` → `value_weighted_reg`. |
| `src/shadow_price_prediction/prediction.py` | Stage 2 regression block: conditional on `unified_regressor` (no gate), `expected_value_scoring` (soft gate + EV multiply), or standard (binary gate). |
| `src/shadow_price_prediction/registry.py` | `HARD_GATES` → `PROMOTION_GATES` + `MONITORING_GATES`. `check_gates()` default changed to `PROMOTION_GATES`. |
| `scripts/run_experiment.py` | `_compute_gates()` adds S1-AP, R-REC@500, C-NDCG as promotion metrics. |
| `scripts/_experiment_worker.py` | Added overrides for `expected_value_scoring`, `regression_prob_floor`, `value_weighted`, `value_weighted_reg`, `unified_regressor`. |
| `docs/runbook.md` | Updated Stage 2 description, gate architecture section, run log, added TrainingConfig table. |

---

## 10. Key Concerns for the Next AI Session

1. **v012 bias issue**: The smoke test showed a large negative bias ($468 mean pred vs $1,302 actual on TPs). This may indicate the unified regressor approach is undermined by class imbalance. Check whether the full 32-run results show improvement or regression in ranking metrics vs v008/v000.

2. **If v012 fails**: Try Tweedie loss first (H1 above), then zero-weight downsampling (H2). Both are single-flag changes that can be tested quickly.

3. **Feature selection with unified target**: The `select_features()` function uses Spearman correlation and AUC checks against the target. With a zero-heavy target (log1p of 95%-zeros), Spearman correlation for features like `prob_exceed_100` may still be positive (higher prob_exceed → more likely non-zero shadow price). But AUC checks may behave differently. Monitor feature selection output carefully.

4. **Monotonic constraints and unified target**: Monotone constraints require that higher feature values → higher predictions. With a zero-heavy target, this relationship is preserved in expectation but individual monotone splits may be harder to find. XGBoost may fall back to fewer monotone features.

5. **Gate floors still based on threshold-dependent history**: The MONITORING_GATES have no floors, so they can't be used to pass/fail. But the PROMOTION_GATES floors were set based on v000 performance levels — they should be reviewed periodically as we improve the model.

6. **v009 and v010 confusion in registry**: There are two v010 entries in the versions dir — `versions/v010-threshold-050` (an older experiment) and `versions/v010-value-weighted-ev` (the actual v010). This is a naming artifact from session context, not a real conflict. The benchmark parquets for `v010-value-weighted-ev` are the canonical v010 results.

---

## 11. How to Resume Work

```bash
# Activate venv
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# Check v012 benchmark progress
ls /opt/temp/tmp/pw_data/spice6/experiments/v012-unified-regressor/*.parquet | wc -l

# Score v012 once all 32 parquets are done
PYTHONPATH=/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src:$PYTHONPATH \
  python /home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/scripts/run_experiment.py \
  --mode score --version-id v012-unified-regressor

# Run full benchmark from scratch (if needed)
PYTHONPATH=.../src:$PYTHONPATH python .../scripts/run_experiment.py \
  --mode full --version-id v012-unified-regressor --concurrency 4 \
  --overrides '{"unified_regressor": true}'
```

```python
# Compare results after scoring
from shadow_price_prediction.registry import ModelRegistry
reg = ModelRegistry('versions')
v = reg.get_version('v012-unified-regressor')
print(v.gate_values)
```

---

## 12. Next Steps (Prioritized)

1. **[Immediate]** Wait for v012 full benchmark to complete (was 9/32 at last check). Run `--mode score`.
2. **[If v012 improves ranking]** Check per-segment gate values. Write NOTES.md for v012. Update runbook run log.
3. **[If v012 fails on ranking]** Try H1 (Tweedie loss) as v013 — single-line change in EnsembleConfig + new version.
4. **[After benchmark]** Consider whether the branch regressor should also use unified mode or only the default regressor.
5. **[Architecture]** Consider whether to use `reg:tweedie` + `tweedie_variance_power=1.5` for ALL future regressors (not just unified), as it better handles the zero-inflated distribution even in the binding-only training case.
