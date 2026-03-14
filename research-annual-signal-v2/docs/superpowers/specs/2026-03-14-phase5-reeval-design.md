# Phase 5 Design: Re-evaluation Under New Metric Framework

**Date**: 2026-03-14
**Status**: Draft
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
   - Blend (v0c + α × NB for dormant) — soft integration
   - Adaptive R (forced slots but only for high-confidence NB picks)
4. **R sweep at the right granularity** for K=300: R ∈ {0, 5, 10, 15, 20, 30}
5. **Dangerous branch analysis** as a first-class metric
6. **Pick champions** for (150, 300) and (200, 400) pairs

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

**R sweep for K=300**: {0, 5, 10, 15, 20, 30}. Prior experiments only tested R=30.

### 4.2 Blend (new)

Score all branches with v0c. For dormant branches, add a scaled NB boost:

```python
v0c_scores = compute_v0c_scores(group_df)  # all branches
nb_raw = nb_model.predict(dormant_features)  # dormant only

# Normalize NB to v0c scale
v0c_range = v0c_scores.max() - v0c_scores.min()
nb_norm = (nb_raw - nb_raw.min()) / (nb_raw.max() - nb_raw.min()) * v0c_range

# Blend: dormant branches get boosted, others unchanged
final_scores = v0c_scores.copy()
final_scores[is_dormant] += alpha * nb_norm
```

**Alpha sweep**: {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}

**Advantage**: No forced slots. Dormant branches that v0c already ranks well stay in
position. The NB model only promotes dormant branches it's confident about. At K=400
where v0c naturally includes 63 dormant branches, this approach preserves v0c's good
dormant picks while potentially reordering them.

**Disadvantage**: If NB model signal is weak (AUC 0.65), the boost may promote wrong
dormant branches and displace marginal established branches that would have contributed
more SP.

### 4.3 Adaptive R (new)

Hard two-track but only insert NB picks that exceed a confidence threshold:

```python
R_actual = min(R_max, count(nb_score >= tau))
# Unused slots return to Track A
```

Already implemented in `merge_tracks(tau=...)`. The tau sweep from Phase 4b was
ineffective because all NB scores exceeded the tested thresholds. Need to calibrate
tau on actual NB score percentiles per split.

**Tau calibration**: percentiles {50, 70, 80, 90, 95} of NB scores among dormant
branches in the training split.

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

## 6. Candidate Matrix

| ID | Track A | NB Integration | Parameters |
|---|---|---|---|
| A1 | v0c TRUE solo | None | — |
| A2 | v3a TRUE solo | None | — |
| B1 | v0c + hard two-track | R ∈ {5, 10, 15, 20, 30} at each K | Phase 4a NB model |
| B2 | v3a + hard two-track | R ∈ {5, 10, 15, 20, 30} at each K | Phase 4a NB model |
| C1 | v0c + blend | α ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} | Phase 4a NB model |
| C2 | v3a + blend | α ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} | Phase 4a NB model |
| D1 | v0c + adaptive R | R_max=20, tau sweep | Phase 4a NB model |

Total: 2 solo + 10 hard R + 12 blend α + 1 adaptive = ~25 configs per K level.

## 7. Success Criteria

A Phase 5 candidate is champion for a K pair if on holdout:

1. **VC@K_hi ≥ best TRUE solo VC@K_hi** (no regression on the higher K level)
2. **Dang_Recall@K_hi ≥ best TRUE solo Dang_Recall@K_hi**
3. Among candidates meeting (1) and (2), **maximize NB12_SP@K_hi**
4. At K_lo: tolerate ≤ 2% VC regression vs TRUE solo if K_hi is a clear win

If no NB integration candidate meets (1) and (2), the champion is the best TRUE solo
model for that K pair.

## 8. Implementation

### 8.1 Update evaluate.py

Add K=150/200/300/400 to `all_ks` in `evaluate_group()`.
Add dangerous branch metrics (SP > 50000 threshold).
Keep backward compatibility with existing @50/@100 tests.

### 8.2 Single experiment script

One script: `scripts/run_phase5_reeval.py`

Runs the full candidate matrix on dev, selects top configs, validates on holdout.
Saves results to registry with the new metric schema.

### 8.3 Files changed

| File | Change |
|------|--------|
| `ml/evaluate.py` | Add K=150/200/300/400, dangerous branch metrics |
| `ml/config.py` | Add EVAL_K_LEVELS, DANGEROUS_THRESHOLD, updated gate metrics |
| `scripts/run_phase5_reeval.py` | NEW — full candidate matrix evaluation |
| `registry/phase5_*/` | NEW — results |

### 8.4 No changes

| File | Reason |
|------|--------|
| `ml/merge.py` | Already has tau support |
| `ml/features_trackb.py` | No new features |
| `ml/train.py` | Track A training unchanged |

## 9. Expected Runtime

- Data build: ~90s (shared across all configs)
- Per config per group: ~0.5s (v0c scoring) or ~2s (v3a train+predict)
- 25 configs × 4 K levels × 12 dev groups: ~30 min
- Holdout (top 5 configs × 4 K × 3 groups): ~2 min
- Total: ~35 min
