# Phase 5 Design: Re-evaluation Under New Metric Framework

**Date**: 2026-03-14
**Status**: Draft (rev 2 — review findings addressed)
**Depends on**: Phase 3 (two-track infra), Phase 4a (Track B model), code audit review

---

## 1. Problem Statement

All prior champion decisions were made under @50/@100 metrics. The operational policy
has changed to paired K levels: **(150, 300)** and **(200, 400)**, with 150 tier-0 and
150 tier-1 constraints. This invalidates prior conclusions in three ways:

1. **R=0 two-track was not a true solo baseline.** The two-track R=0 path bars dormant
   branches from top-K by construction. True v0c solo naturally includes 21 dormant
   branches at K=300 (holdout) with $10k SP. All prior "solo baseline" comparisons in
   the two-track scripts are biased.

2. **NB economics flip at higher K.** At K=50, every NB slot displaces a strong
   established branch (net SP loss). At K=300, the marginal established branch is weak
   enough that NB picks can add net value. The entire Phase 4 "NB is too weak" narrative
   needs re-examination.

3. **v0c vs v3a tradeoff is different at K=300.** v3a wins Recall; v0c wins VC and
   Dangerous Recall. This tradeoff was never evaluated at the new K levels with true
   solo baselines.

## 2. Scope

Phase 5 is a clean re-evaluation, not new model development:

1. **Update the official evaluator** to support K=150/200/300/400 and dangerous metrics
2. **Compute true solo baselines** for v0c and v3a at all K levels
3. **Test three NB integration approaches** against true baselines:
   - Hard two-track (forced R slots) — what we have
   - Blend (base_score + α × NB for dormant) — soft integration
   - Adaptive R (forced slots but only for high-confidence NB picks)
4. **Joint paired-K parameterization** — configs are (R_lo, R_hi) or (α_lo, α_hi) pairs
5. **Dangerous branch analysis** as a first-class metric
6. **Pick champions** for (150, 300) and (200, 400) pairs via paired scorecard

Phase 5 does NOT:
- Train new Track B models (use Phase 4a tiered/lgbm 14f)
- Change Track A models (use existing v0c and v3a)
- Build new features
- Change the merge module beyond adding blend support

## 3. True Solo Baselines

Score ALL branches globally with a single model. No track splitting.

### v0c TRUE solo
```python
scores = compute_v0c_scores(group_df)  # scores ALL branches
topk = argsort(scores)[::-1][:K]
```
Dormant branches get nonzero scores from da_rank + density (top dormant scores ~0.46-0.53).
At K=300, v0c naturally includes ~21 dormant branches.

### v3a TRUE solo
```python
scored_df, _ = train_and_predict(model_table, train_pys, eval_pys, V3A_FEATURES)
# scored_df has scores for ALL branches including dormant
topk = argsort(scores)[::-1][:K]
```
v3a LambdaRank scores all branches. At K=300, v3a naturally includes ~6 dormant branches
(fewer than v0c because LambdaRank learns to rank dormant lower).

### Track A Score Caching (MANDATORY)

Track A base scores are computed **once per split** and cached:

```python
# v0c: deterministic formula, compute once per group
v0c_cache[py, aq] = compute_v0c_scores(group_df)

# v3a: train once per split, score all eval groups
v3a_scored, _ = train_and_predict(model_table, train_pys, eval_pys, V3A_FEATURES)
v3a_cache[py, aq, branch] = v3a_scored["score"]
```

All blend / hard-two-track / adaptive variants reuse these cached base scores.
No retraining inside the candidate loop.

### Known holdout numbers (already computed)

| K | v0c TRUE VC | v0c Dorm_inK | v3a TRUE VC | v3a Dorm_inK |
|---|:---:|:---:|:---:|:---:|
| 150 | 0.5374 | 1.0 | 0.5475 | 0.0 |
| 200 | 0.6236 | 3.3 | 0.5952 | 0.0 |
| 300 | 0.7195 | 21.0 | 0.7289 | 5.7 |
| 400 | 0.7861 | 63.3 | 0.7717 | 25.3 |

## 4. NB Integration Approaches

### 4.1 Hard Two-Track (existing)

Split universe: Track A = established, Track B = dormant. Merge with R reserved slots.
History_zero appended with score=0 for correct evaluation denominators.

**Problem**: At K=300, v0c naturally includes 21 dormant branches. Forcing R=30 displaces
some of those natural dormant picks and replaces them with NB model picks — which may be
worse (the NB model is only AUC 0.65).

Tested for both v0c and v3a as Track A.

### 4.2 Blend (new)

Score all branches with the Track A model. For dormant branches only, add a scaled NB boost.

**Normalization contract** (applies to BOTH v0c and v3a):

```python
base_scores = track_a_cache[group]  # v0c or v3a scores for ALL branches
nb_raw = nb_model.predict(dormant_features)

# Normalize NB to base score scale WITHIN this group
base_range = base_scores.max() - base_scores.min()
nb_norm = (nb_raw - nb_raw.min()) / (nb_raw.max() - nb_raw.min() + 1e-10) * base_range

# Blend: dormant branches get boosted, others unchanged
final_scores = base_scores.copy()
final_scores[is_dormant] += alpha * nb_norm
```

The normalization maps NB scores to [0, base_range] regardless of the Track A model's
score distribution. This works for both v0c (scores in ~[0, 1]) and v3a (LambdaRank
scores in arbitrary range) because both sides are scaled to the same group-level range.

**Alpha sweep**: {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}

**Advantage**: No forced slots. Dormant branches that the Track A model already ranks
well stay in position. The NB model only promotes dormant branches it's confident about.

**Disadvantage**: If NB model signal is weak (AUC 0.65), the boost may promote wrong
dormant branches and displace marginal established branches.

### 4.3 Adaptive R (new)

Hard two-track but only insert NB picks that exceed a confidence threshold:

```python
R_actual = min(R_max, count(nb_score >= tau))
# Unused slots return to Track A
```

Already implemented in `merge_tracks(tau=...)`.

**Tau calibration**: percentiles {50, 70, 80, 90, 95} of NB scores among dormant
branches in the training split.

Tested for both v0c and v3a as Track A.

## 5. Metrics

### Primary (gated)

| Metric | Definition |
|--------|-----------|
| VC@K | Captured SP / total SP |
| Recall@K | Fraction of binding branches in top-K |
| Abs_SP@K | Captured SP / total DA SP |

### NB-specific (monitored)

| Metric | Definition |
|--------|-----------|
| NB12_Count@K | Count of NB12 binders in top-K |
| NB12_SP@K | Captured NB12 SP / total NB12 SP |
| Dorm_inK | Count of dormant branches in top-K |
| Dorm_SP | Total SP from dormant branches in top-K |

### Dangerous branch (new, gated)

| Metric | Definition |
|--------|-----------|
| Dang_Recall@K | Fraction of branches with SP > $50k captured in top-K |
| Dang_SP_Ratio@K | Captured dangerous SP / total dangerous SP |
| DangNB_Count@K | Count of dangerous dormant binders in top-K |

### Evaluation K levels
- K ∈ {150, 200, 300, 400}
- Champions chosen per pair: (150, 300) and (200, 400)

## 6. Candidate Matrix — Paired Configurations

Each candidate specifies parameters for BOTH K levels in the pair simultaneously.
No independent per-K optimization — the pair is the atomic unit.

### Pair (150, 300)

| ID | Track A | Integration | (R_150 or α_150) | (R_300 or α_300) |
|---|---|---|---|---|
| A1 | v0c | solo | — | — |
| A2 | v3a | solo | — | — |
| B1a | v0c | hard R | (5, 10) | |
| B1b | v0c | hard R | (10, 15) | |
| B1c | v0c | hard R | (10, 20) | |
| B1d | v0c | hard R | (15, 30) | |
| B1e | v0c | hard R | (20, 30) | |
| B2a | v3a | hard R | (5, 10) | |
| B2b | v3a | hard R | (10, 15) | |
| B2c | v3a | hard R | (10, 20) | |
| B2d | v3a | hard R | (15, 30) | |
| C1a | v0c | blend | (0.05, 0.05) | |
| C1b | v0c | blend | (0.1, 0.1) | |
| C1c | v0c | blend | (0.1, 0.2) | |
| C1d | v0c | blend | (0.2, 0.2) | |
| C1e | v0c | blend | (0.3, 0.3) | |
| C2a | v3a | blend | (0.1, 0.1) | |
| C2b | v3a | blend | (0.2, 0.2) | |
| C2c | v3a | blend | (0.3, 0.3) | |
| D1a | v0c | adaptive R | R_max=(10,20), tau sweep | |
| D2a | v3a | adaptive R | R_max=(10,20), tau sweep | |

### Pair (200, 400)

Same structure with scaled R values:
- Hard R pairs: (10, 15), (15, 20), (15, 30), (20, 40), (25, 50)
- Blend α pairs: same as (150, 300)
- Adaptive R: R_max=(15, 30), tau sweep

Total: ~22 configs per pair × 2 pairs = ~44 configs.

## 7. Paired Scorecard and Champion Selection

### Paired Composite Score

Each candidate is scored on a **paired composite** that weights both K levels:

```
paired_score = 0.5 * score_lo + 0.5 * score_hi
```

where `score_K` for a given K is:

```
score_K = w_vc * VC@K + w_rec * Recall@K + w_dang * Dang_Recall@K + w_nb * NB12_SP@K
```

Default weights: `w_vc=0.4, w_rec=0.2, w_dang=0.2, w_nb=0.2`

### Hard gates (must pass BOTH K levels)

1. **VC@K ≥ best TRUE solo VC@K - 0.02** at each K (max 2% regression)
2. **Dang_Recall@K ≥ best TRUE solo Dang_Recall@K - 0.05** at each K

### Champion selection

Among candidates passing hard gates at both K levels, pick highest paired_score.

If no NB integration candidate passes, the champion is the best TRUE solo.

## 8. Implementation

### 8.1 Update evaluate.py

Add K=150/200/300/400 to `all_ks` in `evaluate_group()`.
Add dangerous branch metrics (SP > 50000 threshold).

**Backward compatibility**: Add new constants `PHASE5_K_LEVELS = [150, 200, 300, 400]`
and `PHASE5_GATE_METRICS` in `ml/config.py`. Keep existing `TIER1_GATE_METRICS` and
`TWO_TRACK_GATE_METRICS` unchanged. Phase 5 evaluation uses the new constants; existing
tests and scripts continue to use the old ones.

### 8.2 Single experiment script

One script: `scripts/run_phase5_reeval.py`

1. Build data + enrich (once)
2. Compute and cache Track A base scores per split (v0c + v3a)
3. Train NB model per split (once)
4. Sweep all paired configs on dev
5. Rank by paired_score, select top 5
6. Validate on holdout
7. Save to registry

### 8.3 Files changed

| File | Change |
|------|--------|
| `ml/evaluate.py` | Add K=150/200/300/400 to all_ks, add dangerous branch metrics |
| `ml/config.py` | Add PHASE5_K_LEVELS, PHASE5_GATE_METRICS, DANGEROUS_THRESHOLD. Keep old constants unchanged |
| `scripts/run_phase5_reeval.py` | NEW — full paired candidate matrix evaluation |
| `registry/phase5_*/` | NEW — results |

### 8.4 No changes

| File | Reason |
|------|--------|
| `ml/merge.py` | Already has tau support |
| `ml/features_trackb.py` | No new features |
| `ml/train.py` | Track A training unchanged |

## 9. Expected Runtime

- Data build + enrichment: ~90s (shared)
- Track A caching: v0c ~1s total, v3a ~15s per split × 4 splits = ~60s
- NB model training: ~1s per split × 4 = ~4s
- Per paired config evaluation: ~0.5s per group (reuses cached scores)
- 44 configs × 12 dev groups × 2 K levels: ~528 evals × 0.5s ≈ ~5 min
- Holdout (top 5 × 3 groups × 2 K levels): ~15s
- Total: ~8 min
