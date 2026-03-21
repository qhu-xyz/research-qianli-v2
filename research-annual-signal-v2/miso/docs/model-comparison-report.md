# Annual Signal Model Comparison Report

**Date**: 2026-03-15
**Repo**: research-annual-signal-v2

> **IMPORTANT**: All models below were trained on **combined class_type** data
> (`realized_shadow_price = onpeak_sp + offpeak_sp`). V6.1 production signals are
> class-specific (different rankings for onpeak vs offpeak). The class-specific
> pipeline is the next step — these results serve as the combined-ctype baseline.

---

## 1. Model Inventory

### 1.1 Formula Models (no ML training)

**v0a — Pure da_rank_value**
- Features: 1 (`da_rank_value`)
- Score: `1 - normalize(da_rank_value)` (lower da_rank = more binding historically)
- Rationale: baseline — how much does historical DA congestion alone predict annual binding?

**v0b — 5-feature blend**
- Features: 5 (`da_rank_value`, `density_ori_rank_value`, `density_mix_rank_value`, `shadow_price_da`, `ori_mean`)
- Score: weighted combination of rank columns
- Rationale: V6.1's original formula replicated

**v0c — 6-feature formula (Track A champion)**
- Features: 6 (`da_rank_value`, `bin_80/90/100/110_cid_max`, `bf_combined_12`)
- Score: `0.40 × norm(1 - da_rank) + 0.30 × norm(rt_max) + 0.30 × norm(bf_combined_12)`
- Rationale: simplified formula balancing historical congestion, density right-tail, and binding frequency

### 1.2 ML Models (LightGBM LambdaRank)

**v3a — 13-feature LambdaRank**
- Features: 13 (density bins + bf windows + da_rank + limits)
- Target: tiered labels (0/1/2/3) from realized SP quantiles
- Training: expanding window, one model per eval PY
- Rationale: full ML approach using all available features

**v3e_nb — 18-feature LambdaRank + NB features**
- Features: 18 (v3a + density NB-focused features)
- Same training as v3a
- Rationale: test whether NB-specific features improve global ranking

### 1.3 NB Track B Model (dormant-only binary classifier)

**Phase 4a NB model — tiered/sqrt LightGBM binary**
- Features: 14 (12 density bins + `shadow_price_da` + `da_rank_value` + `historical_max_sp`)
- Target: `realized_shadow_price > 0` (binary bind/no-bind)
- Sample weights: tiered (1/3/10 by SP tertile) or sqrt(SP)
- Population: history_dormant branches only (bf_combined_12 = 0, has_hist_da = True)
- Purpose: NOT a standalone ranker. Used only as a blend boost for dormant branches.

### 1.4 Blend (Production Champion)

**v0c + NB blend (α=0.05)**
- Track A: v0c formula scores ALL branches
- Track B: Phase 4a NB model scores dormant branches only
- Blend: `final = v0c_score + 0.05 × normalized_NB_score` for dormant; `v0c_score` for others
- No forced slot reservation — dormant branches compete on blended score

---

## 2. Holdout Results — Legacy Metrics (@50, @100)

These were the original evaluation metrics. Included for historical context.

| Model | VC@50 | VC@100 | Recall@50 | NDCG | Spearman | NB12_Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| v0a (1f formula) | 0.3453 | 0.4121 | 0.0925 | 0.7899 | 0.2960 | 0 |
| v0b (5f blend) | 0.2748 | 0.4189 | 0.0888 | 0.7847 | 0.3032 | 0 |
| **v0c (6f formula)** | **0.3547** | **0.4658** | **0.1229** | **0.8599** | **0.3645** | **0** |
| **v3a (13f ML)** | **0.3675** | **0.4768** | **0.1207** | **0.8670** | **0.3963** | **0** |
| v3e_nb (18f ML+NB) | 0.3444 | 0.4821 | 0.1200 | 0.8646 | 0.3907 | 0 |

**Key observations**:
- v3a wins VC@50, NDCG, Spearman. v0c wins Recall@50.
- No model captures ANY NB binders at @50 or @100.
- v3e_nb's NB features don't help — NB branches still rank last in global NDCG.

---

## 3. Holdout Results — New Metrics (@150, @300)

Pair (150, 300) with dangerous branch recall (SP > $50,000).

| Model | VC@150 | Rec@150 | DgR@150 | VC@300 | Rec@300 | DgR@300 | NB12_SP@300 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v0c solo | 0.5374 | 0.2924 | 0.6455 | 0.7195 | 0.4861 | 0.8121 | 0.0425 |
| v3a solo | **0.5475** | **0.3053** | 0.6455 | **0.7289** | **0.5086** | 0.7899 | 0.0171 |
| v0c hard R=10/20 | 0.5298 | 0.2882 | 0.6455 | 0.7346 | 0.4872 | **0.8788** | 0.0577 |
| **v0c blend sqrt α=0.05** | 0.5374 | 0.2924 | 0.6455 | 0.7237 | 0.4712 | 0.8758 | **0.2159** |
| v0c blend tiered α=0.05 | 0.5374 | 0.2934 | 0.6455 | 0.7233 | 0.4711 | 0.8758 | 0.2054 |

## 4. Holdout Results — New Metrics (@200, @400)

| Model | VC@200 | Rec@200 | DgR@200 | VC@400 | Rec@400 | DgR@400 | NB12_SP@400 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v0c solo** | **0.6236** | 0.3765 | **0.7566** | 0.7861 | 0.5520 | 0.9091 | 0.2166 |
| v3a solo | 0.5952 | **0.3795** | 0.6455 | 0.7717 | **0.5840** | 0.7899 | 0.0562 |
| v0c hard R=10/20 | 0.6229 | 0.3652 | 0.7899 | **0.7931** | 0.5796 | 0.8788 | 0.0577 |
| **v0c blend tiered α=0.05** | 0.6187 | 0.3713 | 0.7566 | 0.7880 | 0.5372 | **0.9394** | **0.2923** |
| v0c blend sqrt α=0.05 | 0.6171 | 0.3701 | 0.7566 | 0.7813 | 0.5318 | **0.9394** | 0.2974 |

---

## 5. Champion Selection (Paired Scorecard)

Champions selected by paired composite score:
`score = 0.5 × (0.4×VC + 0.2×Recall + 0.2×DangR + 0.2×NB12_SP)` at each K level.

Hard gates: VC ≤ 2% regression, DangR ≤ 5% regression vs best solo at each K.

| K Pair | Champion | Score | Runner-up | Score |
|---|---|:---:|---|:---:|
| **(150, 300)** | **S1_sqrt_a0.05** (v0c + sqrt NB blend) | **0.5023** | C1_a0.05 (tiered) | 0.5013 |
| **(200, 400)** | **C1_a0.05** (v0c + tiered NB blend) | **0.5716** | S1_sqrt_a0.05 (sqrt) | 0.5698 |

Both blend variants within 0.002 of each other — effectively equivalent.

---

## 6. Why Each Model Wins or Loses

| Model | Strength | Weakness |
|---|---|---|
| v0a | Simplest, good VC@50 | No BF signal, poor Recall |
| v0b | Balanced formula | Worse VC@50 than v0a (over-diversified) |
| **v0c** | Best formula VC, good DangR | Zero NB capture; BF dominates dormant ranking |
| **v3a** | Best Recall, highest NDCG | Fails DangR gate at (200,400); breaks under NB slot reservation |
| v3e_nb | Slightly better VC@100 | NB features don't help global ranking |
| **Blend** | Best NB12_SP + DangR with no VC cost | Requires two models; NB model is weak (AUC 0.65) |

---

## 7. Key Finding: v3a Cannot Be Track A

v3a produces a globally optimized LambdaRank ordering. Inserting NB reserved slots
breaks this ordering disproportionately — at K=300/R=30, v3a loses 6% VC vs only 1% for
v0c. v3a also fails the DangR gate at (200, 400): DgR@200 = 0.6455 vs v0c's 0.7566.

The simpler v0c formula absorbs NB insertion gracefully because its marginal branches
(rank #270-300) are scored by a simple formula that doesn't depend on global ranking
context.
