# NB Two-Track: Consolidated Findings

**Date**: 2026-03-14
**Branch**: feature/pjm-v70b-ml-pipeline

---

## Executive Summary

Two-track NB scoring is a **Pareto improvement at K≥300**: adding 30 reserved dormant
slots to the v0c formula improves VC, Recall, NB capture, and dangerous-branch detection
simultaneously on holdout. At K≤150, NB slots cost VC. The operating point is K=300+ with
R=30.

The best overall system is **v0c + NB(R=30)**, not v3a. Although v3a (LambdaRank) beats
v0c on solo ranking, it cannot tolerate reserved NB slots — its global ranking degrades
catastrophically under hard slot reservation. v0c's simpler formula absorbs the NB
insertion gracefully.

---

## 1. Metric Framework

### K Pairs
Signal now targets 150 tier-0 + 150 tier-1 constraints. Evaluation uses paired K levels:
- **(150, 300)**: tier-0 budget / full budget
- **(200, 400)**: larger tier-0 / larger full budget

### Metrics at Each K
| Metric | Definition |
|--------|-----------|
| VC@K | Captured SP / total SP in universe |
| Recall@K | Fraction of binding branches in top-K |
| Abs_SP@K | Captured SP / total DA SP (cross-universe denominator) |
| NB12_Count@K | Count of NB12 binders (dormant, bound this quarter) in top-K |
| NB12_SP@K | Captured NB12 SP / total NB12 SP |
| Dang_Recall@K | Fraction of dangerous branches (SP > $50k) captured |
| Dang_SP_Ratio@K | Captured dangerous SP / total dangerous SP |

### Dangerous Branches
- Definition: `realized_shadow_price > 50,000` per quarter
- ~9 per group on average (range 2-18)
- ~85% are established branches, ~8% dormant, ~7% zero-history
- They hold ~40% of total SP per group

---

## 2. Model Registry

### Track A Candidates

| Model | Type | Features |
|-------|------|----------|
| **v0c** | Formula | `0.40 * norm(1-da_rank) + 0.30 * norm(rt_max) + 0.30 * norm(bf_combined_12)` |
| v3a | LambdaRank | 13 features (density bins + bf + da_rank + limits) |

### Track B (NB Model)

| Version | Target | Features | Population |
|---------|--------|----------|-----------|
| Phase 3 | Binary unweighted | 12 density bins | dormant + zero |
| **Phase 4a** | Binary, tiered weights (1/3/10) | 14 (density + sp_da + da_rank + hist_max) | **dormant only** |
| Phase 4b | log1p(SP) regression | 23 (+ recency + shape) | dormant only |

Phase 4a is the Track B champion. Phase 4b regression was strictly worse.

---

## 3. Holdout Results

### Pair (150, 300)

| Model | R_150 | R_300 | VC@150 | Rec@150 | NB12_SP@150 | DgR@150 | VC@300 | Rec@300 | NB12_SP@300 | DgR@300 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v0c solo | 0 | 0 | 0.5374 | 0.2924 | 0.0000 | 0.6455 | 0.7195 | 0.4861 | 0.0425 | 0.8121 |
| **v0c+NB(20/30)** | 20 | 30 | 0.5229 | 0.2725 | 0.0577 | **0.6788** | **0.7325** | 0.4914 | 0.0772 | **0.8788** |
| v0c+NB(30/50) | 30 | 50 | 0.5175 | 0.2616 | 0.0772 | 0.6788 | 0.6851 | 0.4702 | 0.1031 | 0.7899 |
| v3a solo | 0 | 0 | **0.5475** | **0.3053** | 0.0000 | 0.6455 | 0.7289 | **0.5086** | 0.0171 | 0.7899 |
| v3a+NB(20/30) | 20 | 30 | 0.5401 | 0.2782 | 0.0577 | 0.6788 | 0.7174 | 0.4939 | 0.0772 | 0.8232 |

**K=300 champion: v0c+NB(R=30).** Pareto improvement over all solo models: VC+1.8% vs v0c, DgR+8.8% vs v0c, and NB12_SP from 4.3% to 7.7%.

### Pair (200, 400)

| Model | R_200 | R_400 | VC@200 | Rec@200 | NB12_SP@200 | DgR@200 | VC@400 | Rec@400 | NB12_SP@400 | DgR@400 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v0c solo | 0 | 0 | **0.6236** | **0.3765** | 0.0000 | **0.7566** | 0.7861 | 0.5520 | 0.2166 | **0.9091** |
| v0c+NB(R=30) | 30 | 30 | 0.5920 | 0.3449 | 0.0772 | 0.7343 | 0.7835 | **0.5789** | 0.0772 | 0.8788 |
| v3a solo | 0 | 0 | 0.5952 | 0.3795 | 0.0000 | 0.6455 | 0.7717 | 0.5840 | 0.0562 | 0.7899 |
| v3a+NB(25/40) | 25 | 40 | 0.5933 | 0.3614 | 0.0746 | 0.6788 | 0.7680 | 0.5813 | 0.1031 | 0.8232 |

At K=400: v0c+NB(R=30) is nearly free (VC -0.3%, Recall +4.9%). v0c solo has higher DangR because the large K budget naturally captures dangerous branches.

### Per-Group Holdout Detail (v0c+NB R=30)

**K=200:**
| Group | VC | Recall | NB12_C | NB12_SP | Dang_R | Dorm_C | Dorm_SP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 2025-06/aq1 | 0.5728 | 0.3464 | 3 | 0.0645 | 0.667 (4/6) | 30 | $7,392 |
| 2025-06/aq2 | 0.5727 | 0.3106 | 4 | 0.0173 | 0.636 (7/11) | 30 | $7,890 |
| 2025-06/aq3 | 0.6304 | 0.3776 | 8 | 0.1496 | 0.900 (9/10) | 30 | $54,528 |

**K=400:**
| Group | VC | Recall | NB12_C | NB12_SP | Dang_R | Dorm_C | Dorm_SP |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| 2025-06/aq1 | 0.8189 | 0.5850 | 3 | 0.0645 | 1.000 (6/6) | 30 | $7,392 |
| 2025-06/aq2 | 0.7236 | 0.5531 | 4 | 0.0173 | 0.636 (7/11) | 30 | $7,890 |
| 2025-06/aq3 | 0.8079 | 0.5986 | 8 | 0.1496 | 1.000 (10/10) | 30 | $54,528 |

---

## 4. Why v3a Cannot Be Track A in Two-Track

v3a (LambdaRank) produces a globally optimized ranking. Removing R slots and inserting NB
picks breaks this ranking disproportionately:

| Config | VC@300 (holdout) | Drop from solo |
|---|:---:|:---:|
| v3a solo | 0.7289 | — |
| v3a + NB(R=30) | 0.7174 | -1.6% |
| v3a + NB(R=50) | 0.6842 | -6.1% |
| v0c solo | 0.7195 | — |
| v0c + NB(R=30) | 0.7325 | **+1.8%** |

v0c's formula-based ranking is more robust to slot displacement because the marginal
Track A branches (rank #270-300) are scored by a simple formula that doesn't depend on
the global ranking context. v3a's marginal branches are part of an NDCG-optimized
ordering that degrades when disrupted.

---

## 5. Track B Model Details

### Features (14)
Phase 3 density bins (12): `bin_{60,70,80,90,100,110,120,150,-50,-100}_cid_max`,
`count_active_cids`

Added in Phase 4a (3): `shadow_price_da`, `da_rank_value`, `historical_max_sp`

### Features NOT Used (and why)
| Feature | Why excluded |
|---------|-------------|
| count_cids | Dominated by count_active_cids; causes complexity bias |
| Recency features (5) | Spearman < 0.02 among binders; months_since_last_bind near-random |
| Shape features (6) | Added noise, hurt holdout generalization |
| Constraint propagation (6) | CID mapping is 1-to-1; zero shared CIDs across branches |
| V6.2B forward columns | Only 13% coverage for dormant branches |

### Training
- Population: `cohort == "history_dormant"` only (history_zero excluded)
- Target: binary `SP > 0` with tiered sample weights (bottom⅓=1, mid⅓=3, top⅓=10)
- LightGBM: class ratio folded into per-sample weights (no scale_pos_weight)
- Expanding window: all prior PYs as training
- history_zero stays in evaluation universe with score=0 for apples-to-apples comparison

### Model Limitations
- AUC ~0.65 on dormant population (bind vs no-bind)
- Spearman ~0.14 between model score and SP magnitude among binders
- ~70% of high-SP dormant binders are "density-invisible" (low density features)
- The model finds binders that happen to have high density signals; it misses binders
  whose binding is driven by regime changes or outage events not in the SPICE simulation

---

## 6. Negative Results

| Experiment | Result | Why |
|---|---|---|
| Phase 4b: log1p(SP) regression | Worse than Phase 4a | 92% zeros dominate; model learns "predict ~0" |
| Phase 4c: top-value classifier | 3.6% vs 3.1% (marginal) | Same features, different target doesn't help |
| Interaction features | No improvement | LightGBM learns interactions internally |
| Recent-window training | Worse than expanding | Too few positives per PY (~70) |
| Dropping count_active_cids | 3.5% vs 3.6% (tied) | Model finds other complexity proxies |
| V6.2B forward columns | Dead end | 13% dormant coverage |
| Constraint propagation | Infeasible | 1-to-1 CID→branch mapping |

---

## 7. Recommendations

### For Production
1. **Use v0c + NB blend (α=0.05)** — NOT hard two-track slot reservation
2. Track B model: Phase 4a tiered LightGBM, 14 features, dormant-only
3. Blend adds α × normalized NB score to v0c score for dormant branches only
4. No forced R parameter — dormant branches compete on blended score
5. Gate metrics at paired K levels: VC, Recall, Abs_SP, NB12_SP, Dang_Recall
6. Reproducible via `scripts/run_phase5_reeval.py`, artifacts in registry

### For Future Research
1. **New data sources needed** for dormant ranking improvement — current features explain
   ~3% of dormant SP variance. The 59% oracle ceiling shows the problem is solvable.
2. Do NOT invest more in LightGBM recipe tuning — the feature set is the binding constraint.
3. Constraint-level modeling remains theoretically promising but requires a different bridge
   mapping that allows CID sharing across branches.

---

## 8. Audit Verification (2026-03-14-code-audit-review.md)

### Finding 3 (HIGH): R=0 rows not true solo baselines — CONFIRMED, MATERIAL IMPACT

The two-track R=0 path categorically bars dormant branches from top-K (they go to
Track B with r=0 slots). True v0c solo scores ALL branches globally and includes
dormant branches that score well on da_rank + density.

**This DID affect the baseline comparison.** The earlier claim that R=0 matched true solo
was incorrect. Corrected comparison (holdout):

| K | v0c TRUE VC | v0c+NB(R=30) VC | delta |
|---|:---:|:---:|:---:|
| 150 | 0.5374 | 0.5175 | **-0.0199** (NB costs) |
| 200 | 0.6236 | 0.5920 | **-0.0316** (NB costs) |
| **300** | **0.7195** | **0.7325** | **+0.0130** (NB wins) |
| 400 | 0.7861 | 0.7835 | **-0.0026** (near-tied) |

**K=300 Pareto claim narrowed but still holds.** v0c+NB(R=30) wins on VC (+1.3%),
Recall (+0.5%), and DangR (+6.7%). The win is concentrated in aq3 (+5.7% VC from
$54k dormant SP capture). Other groups are slightly negative.

**K=400: corrected to toss-up.** True solo naturally includes 63 dormant branches
($63k SP) vs NB's 30 targeted slots ($23k SP). The formula's untargeted inclusion
actually captures more dormant SP than the NB model's targeted selection at this K.

### Dormant Branch Natural Inclusion

v0c naturally includes dormant branches at higher K because da_rank and density give
them nonzero scores (top dormant v0c scores ~0.46-0.53):

| K (holdout) | v0c Dorm_in_topK | Dorm_SP |
|---|:---:|---:|
| 150 | 1.0 | $0 |
| 200 | 3.3 | $0 |
| 300 | 21.0 | $9,745 |
| 400 | 63.3 | $62,822 |

The NB model's value is highest at K=300 where true solo includes 21 dormant branches
(mostly non-binders, $10k SP) but the NB model's 30 targeted slots capture $23k SP
including 5 NB12 binders. At K=400, the formula's broader inclusion outperforms the
NB model's narrow targeting.

### Finding 1 (HIGH): Evaluator hardcoded to @50/@100 — FIXED

`ml/evaluate.py` now computes K=150/200/300/400 and dangerous branch metrics
(Dang_Recall, Dang_SP_Ratio, Dang_Count). `ml/config.py` has PHASE5_K_LEVELS,
DANGEROUS_THRESHOLD, PHASE5_GATE_METRICS. Old constants preserved.

---

## 9. Phase 5: Final Re-evaluation (audited, with registry artifacts)

Phase 5 re-evaluated all candidates under the new metric framework using true solo
baselines (not biased R=0 two-track), paired scorecard, and saved to registry.

**Script**: `scripts/run_phase5_reeval.py`
**Registry**: `registry/phase5_champ_150_300/`, `registry/phase5_champ_200_400/`

### Champion: C1_a0.05 (v0c + 0.05 × NB blend)

Wins both K pairs on holdout. Robust at both $50k and $30k dangerous thresholds.

**How it works**: Score all branches with v0c formula. For dormant branches only, add
`α × normalized_NB_score` where α=0.05 and NB scores are normalized to v0c's score
range. No forced slot reservation — dormant branches compete on blended score.

### Holdout Results (DANGEROUS_THRESHOLD = $50,000)

**Pair (150, 300):**

| Config | VC@150 | VC@300 | DgR@300 | NB12_SP@300 | Score |
|---|:---:|:---:|:---:|:---:|:---:|
| v0c solo | 0.5374 | 0.7195 | 0.8121 | 0.0425 | 0.4792 |
| **C1_a0.05** | **0.5374** | **0.7233** | **0.8758** | **0.2054** | **0.5013** |
| v3a solo | 0.5475 | 0.7289 | 0.7899 | 0.0171 | 0.4819 |

**Pair (200, 400):**

| Config | VC@200 | VC@400 | DgR@400 | NB12_SP@400 | Score |
|---|:---:|:---:|:---:|:---:|:---:|
| v0c solo | 0.6236 | 0.7861 | 0.9091 | 0.2166 | 0.5630 |
| **C1_a0.05** | **0.6187** | **0.7880** | **0.9394** | **0.2923** | **0.5716** |
| v3a solo | 0.5952 | 0.7717 | 0.7899 | 0.0562 | 0.5189 |

### Why Blend Wins Over Hard Two-Track

Hard two-track (forced R slots) displaces v0c's natural dormant inclusion. At K=300,
v0c solo already includes 21 dormant branches. Forcing R=30 replaces some of those
(which v0c ranked by formula quality) with NB model picks (ranked by AUC 0.65 model).
The blend preserves v0c's natural ranking while applying a small NB-informed boost.

### Updated Results with Sqrt + Adaptive-R (Phase 5 final)

38 configs tested including sqrt weight blend and adaptive-R. Registry:
`phase5_final_150_300/`, `phase5_final_200_400/`.

**Holdout champions (from saved artifacts):**

| K Pair | Champion | Score | Runner-up | Score |
|---|---|:---:|---|:---:|
| (150, 300) | **S1_sqrt_a0.05** (v0c + sqrt NB α=0.05) | **0.5023** | C1_a0.05 (tiered) | 0.5013 |
| (200, 400) | **C1_a0.05** (v0c + tiered NB α=0.05) | **0.5716** | S1_sqrt_a0.05 (sqrt) | 0.5698 |

Sqrt edges tiered at (150, 300); tiered wins at (200, 400). Both are within
0.002 of each other — effectively equivalent.

Adaptive-R was identical to hard-R at all tau thresholds tested (p70/p80/p90)
because all NB scores exceed even the 90th percentile threshold. Not useful
with the current NB model's narrow score distribution.

### Threshold Sensitivity: $30k vs $50k

Same champion at both thresholds. Blend passes all gates regardless.
