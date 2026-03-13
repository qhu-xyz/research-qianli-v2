# Phase 3 Design: Two-Track NB Reranking

## Problem Statement

NB12_Recall@50 = 0.0 for every model in the registry, including v0c (champion) and v3a (best ML).
The `cohort_contribution` metric confirms: top-50 is 100% established branches in every eval group.
Dormant and zero-history cohorts contribute zero branches to the shortlist, zero SP captured.

This is not a bug — a single model optimizing global NDCG (or any global ranking metric) will always
prefer established branches with bf_combined_12 >> 0 over NB candidates with bf_combined_12 = 0.
The density features have discriminative ratios of 1.3-1.7x for NB12 binders vs non-binders (dev-only),
which is far too weak to overcome the history signal. Adding density features to the global ranker (v3a)
improved VC@50 but left NB12_Recall at 0.0.

**Population size (dev, 12 groups):**
- NB12 binders: 1,443 / 4,403 total binders = 33% of all binders
- NB12 SP share: 19.4% of total binding SP
- This is material — one-fifth of SP is invisible to the shortlist

## Key Definitions

- **NB candidate**: a branch with `cohort in {"history_dormant", "history_zero"}`, i.e.,
  `bf_combined_12 == 0`. This is the Track B population. Known at prediction time.
- **NB12 binder**: an NB candidate that actually binds in the target quarter
  (`is_nb_12 == True`, which requires `realized_shadow_price > 0`). Only known in evaluation.
- **Established branch**: `cohort == "established"` (`bf_combined_12 > 0`). Track A population.

The cohort assignment (`features.py:101-108`) and `is_nb_12` flag (`nb_detection.py`) both use
`bf_combined_12` / `combined_bound` over the same 12-month window with the same cutoff logic,
so the populations are aligned: all `is_nb_12 == True` branches are in the NB candidate pool.

## Approach: Two-Track Scoring with Reserved NB Capacity

Split the universe into two populations using the existing cohort contract (`features.py:101`),
model each separately, merge with reserved slots.

### Track A: Established Branches

**Population**: `cohort == "established"` (bf_combined_12 > 0)

**Model**: v0c formula baseline (current champion) or v3a LambdaRank.
No changes to Track A scoring — the goal is to not regress on established-binder performance.

**Output**: ranked list of established branches, scored as today.

### Track B: NB Candidate Branches

**Population**: `cohort in {"history_dormant", "history_zero"}`

These branches have bf_combined_12 = 0 by definition. History features are dead here.
The model must rank using forward-looking (density) and structural (limits) features only.

**Features** (dev-validated NB12 discriminators, sorted by ratio):

| Feature | NB12/NonB Ratio (dev) | Signal Type |
|---------|----------------------:|-------------|
| bin_-100_cid_max | 1.70x | Counter-flow density |
| bin_-50_cid_max | 1.46x | Counter-flow density |
| bin_110_cid_min | 1.34x | Right-tail density (min across CIDs) |
| bin_100_cid_max | 1.31x | Right-tail density |
| bin_80_cid_max | 1.28x | Right-tail density |
| bin_110_cid_max | 1.26x | Right-tail density |
| bin_90_cid_max | 1.25x | Right-tail density |
| bin_70_cid_max | 1.22x | Mid-tail density |
| count_cids | 1.21x | Structural metadata |
| limit_min | 0.82x | Structural (NB lower) |
| count_active_cids | 1.17x | Structural metadata |
| bin_60_cid_max | 1.16x | Mid-tail density |

All ratios measured on dev groups only (2022-06 through 2024-06), no holdout contamination.

**NOT included**: da_rank_value (1.08x — useless), all bf_* (zero by definition),
shadow_price_da (correlated with da_rank_value).

**Model candidates** (to be swept in Phase 3.2):
1. LightGBM binary classifier (bind vs not-bind) — outputs probability, rank by descending P(bind)
2. Logistic regression on density features — simpler, more interpretable
3. LightGBM LambdaRank on NB population only (optional) — same tiered labels, smaller population.
   Note: query groups will have only ~25-38 positives per (PY, quarter), which is borderline
   for ranking objectives. This reinforces why binary classification is the primary candidate.

The choice depends on Track B population size per query group. With ~1,200-1,600 NB candidates
per (PY, quarter) and only ~100-150 actual binders, class imbalance is severe (~8% positive rate).
Binary classification with class weights may be more appropriate than LambdaRank here.

### Merge Policy

**Top-K construction**: For top-K (K=50), allocate (K - R) slots from Track A and R reserved
slots from Track B, where R is a policy parameter.

**R sweep**: {0, 5, 10, 15} for K=50. Evaluate each R on dev first, then validate best R on holdout.

**Tie-breaking within tracks**: branches ranked by their track-specific score descending.

**Selection from Track B**: top-R branches by Track B score, regardless of whether they are
predicted binders. The model's job is to rank the most likely NB binders highest; the merge
policy's job is to decide how many to include.

**Evaluation of merged results**: Track A and Track B use different models, so raw scores are
on different scales. Rather than synthesizing fake scores (which would distort metrics beyond
the merged top-K), the merge is handled via a `top_k_indices` override:

- `merge_tracks()` returns a pre-computed ordered list of branch indices for the merged top-K:
  first (K-R) indices from Track A (by Track A score), then R indices from Track B (by Track B score).
- `evaluate_group()` gains an optional `top_k_override: np.ndarray | None` parameter.
  When provided, ALL K=50 metrics — including `cohort_contribution()` — use the override
  indices instead of `argsort(scores)[::-1][:50]`. This ensures cohort mix reporting is
  consistent with the actual merged shortlist.
- This is a **top-50 reranking policy**. The policy only controls which branches enter the
  top-50 shortlist. It does not define a full global ranking. Therefore:
  - **Gate metrics are restricted to top-50 only**: VC@50, Recall@50, Abs_SP@50.
  - VC@100, Recall@100, and NDCG are computed from Track A scores and reported as
    **monitoring metrics only** — they are NOT gated, because they do not reflect the
    two-track policy. Gating on metrics insensitive to the reserved NB slots would
    validate Track A, not the two-track system.
  - See Success Criteria for the precise gate set.

### Evaluation Metrics

**Gate metrics** (evaluated on merged top-50, must not regress vs v0c):
- VC@50, Abs_SP@50, Recall@50 — these reflect the actual two-track policy

**Monitoring metrics** (reported but NOT gated):
- VC@100, Recall@100, NDCG — evaluated on Track A scores; insensitive to reserved NB slots
- Spearman — Track A scores only

**New NB metrics to add** (computed in `evaluate_group()`, on merged top-50):

| Metric | Definition | Type |
|--------|-----------|------|
| NB12_Count@50 | Count of NB12 binders (is_nb_12 & label_tier > 0) in top-50 | Monitor |
| NB12_SP@50 | SP from NB12 binders in top-50 / sum(realized_shadow_price where is_nb_12) | Monitor |
| NB6_Recall@50 | Recall of NB6 binders in top-50 | Monitor |
| NB24_Recall@50 | Recall of NB24 binders in top-50 | Monitor |

**NB12_SP@50 denominator**: sum(realized_shadow_price) for all branches where `is_nb_12 == True`
in the group. This is the total NB12 SP available to capture.

**NB gate architecture**:

1. **Remove `NB12_Recall@50` from `TIER1_GATE_METRICS`** in `config.py:171`. Currently this
   metric is permanently inert: both v0c and all candidates score 0.0, so the strict-inequality
   gate (0.0 > 0.0 = False) always fails. It cannot validate improvement and blocks all models
   equally. After removal, `TIER1_GATE_METRICS` contains only: VC@50, VC@100, Recall@50,
   Recall@100, NDCG, Abs_SP@50.

2. **New NB metrics are NOT added to `TIER1_GATE_METRICS`**. They are monitoring metrics only,
   computed and reported but not checked by `check_gates()`.

3. **NB performance is gated exclusively via `check_nb_threshold()`**, a new function separate
   from `check_gates()`:
   ```python
   def check_nb_threshold(
       per_group: dict,
       holdout_groups: list[str],
       min_total_count: int = 3,
   ) -> dict:
       """Cross-group NB gate: sum NB12_Count@50 across holdout groups >= min_total_count.
       Returns dict with passed, total_count, per_group_counts.
       """
   ```
   This is called separately from `check_gates()` and only applies to two-track candidates.

4. **Persistence**: `registry.py:save_experiment()` gains an `nb_gate_results: dict | None`
   parameter (same pattern as the existing `gate_results` parameter). When provided, writes
   `registry/{version}/nb_gate_results.json`.

### Success Criteria

A Phase 3 candidate is considered successful if:

1. **NB12_Count@50 >= 3** total across 3 holdout quarters (via `check_nb_threshold()`)
2. **NB12_SP@50 > 0** on at least 2/3 holdout quarters
3. **Top-50 per-group gates pass** via `check_gates()` vs v0c on the two-track gate set:
   **VC@50, Recall@50, Abs_SP@50** only (win-count >= 2/3 + mean > baseline).
   VC@100, Recall@100, and NDCG are excluded from the gate because they are evaluated on
   Track A scores alone and are insensitive to the reserved NB slots — gating on them would
   validate Track A, not the two-track policy. They are still computed and reported as
   monitoring metrics.
   Note: `NB12_Recall@50` is also excluded (removed from `TIER1_GATE_METRICS` in Phase 3.0.1).
4. Track B model achieves **NB12 AUC > 0.60** within the NB candidate population (dev-only)

Criteria 1-3 are hard gates. Criterion 4 is a sanity check that Track B has learned signal.

**Config change for two-track gate set**: Phase 3.0.1 defines a separate constant
`TWO_TRACK_GATE_METRICS = ["VC@50", "Recall@50", "Abs_SP@50"]` in `ml/config.py`.
The two-track experiment scripts pass this to `check_gates()` instead of `TIER1_GATE_METRICS`.
The existing `TIER1_GATE_METRICS` (minus NB12_Recall@50) continues to govern non-two-track
candidates. This avoids modifying `check_gates()` itself.

## Implementation Plan

### Phase 3.0: Infrastructure (no modeling)

**3.0.1 — Gate contract cleanup**
- Remove `NB12_Recall@50` from `TIER1_GATE_METRICS` in `ml/config.py:171`.
  This metric is permanently inert (0.0 vs 0.0) and blocks all models equally.
  After removal: `["VC@50", "VC@100", "Recall@50", "Recall@100", "NDCG", "Abs_SP@50"]`
- Add `TWO_TRACK_GATE_METRICS = ["VC@50", "Recall@50", "Abs_SP@50"]` to `ml/config.py`.
  Two-track experiment scripts pass this to `check_gates()` instead of `TIER1_GATE_METRICS`.
  Restricts gating to top-50 metrics that actually reflect the two-track merge policy.

**3.0.2 — Add new NB metrics to evaluate.py**
- `NB12_Count@50`: count of (is_nb_12 & label_tier > 0) branches in top-50
- `NB12_SP@50`: SP from NB12 binders in top-50 / total NB12 SP in group
- `NB6_Recall@50` and `NB24_Recall@50`: same pattern as existing NB12_Recall@50
- These are monitoring metrics only — NOT added to `TIER1_GATE_METRICS`
- Add `check_nb_threshold()` as a separate cross-group gate function
- Add `top_k_override` parameter to `evaluate_group()` for two-track merge support.
  When provided, ALL K=50 metrics including `cohort_contribution()` use the override indices.

**3.0.3 — Extend registry.py persistence**
- Add `nb_gate_results: dict | None` parameter to `save_experiment()` (same pattern as
  existing `gate_results`). Writes `registry/{version}/nb_gate_results.json` when provided.

**3.0.4 — Update baseline_contract.json**
- Add NB metric baselines (all 0.0 / 0 for v0c)
- Document that NB12_Recall@50 was removed from TIER1_GATE_METRICS and why
- Document that NB cross-group gate uses `check_nb_threshold()`, not `check_gates()`

**3.0.5 — Compute NB metrics for existing registry entries**
- Rerun evaluation for v0c and v3a to compute the new NB monitoring metrics
- Save as **supplementary artifacts** (`registry/v0c/nb_metrics_supplement.json`,
  `registry/v3a/nb_metrics_supplement.json`) — do NOT overwrite existing `metrics.json`
- This preserves frozen baseline lineage while adding the new metric columns

### Phase 3.1: NB Population Analysis (dev-only)

**3.1.1 — NB population at multiple windows**
- Compute NB6, NB12, NB24 populations on dev groups
- Report: count, SP share, cohort breakdown (dormant vs zero-history) at each window
- Key question: does NB24 capture a meaningfully different population than NB12?

**3.1.2 — Track B population profiling**
- For each dev group: how many NB candidates (dormant + zero-history)? How many are NB12 binders?
- What is the NB base rate (NB12 binders / total NB candidates) per group?
- Distribution of Track B features across NB12 binders vs NB non-binders
- Compute per-feature AUC for NB12 binary classification (bind vs not-bind)
  within the NB candidate population only

**3.1.3 — Feature correlation within Track B**
- Correlation matrix of density features within the NB candidate population
- Prune highly correlated pairs (|r| > 0.85) — keep the one with higher AUC

### Phase 3.2: Track B Model Development (dev-only)

**3.2.1 — Binary classifier (primary candidate)**
- LightGBM binary classification on NB candidate population (cohort in {dormant, zero})
- Target: label_tier > 0 (any binding = positive)
- Features: density bins + limits + count_cids from Phase 3.1 (pruned set)
- Class weights to handle ~8% positive rate
- Expanding-window training matching existing EVAL_SPLITS
- Metric: AUC, precision@10, recall@10 within Track B population

**3.2.2 — Logistic regression baseline**
- Same features, same population
- L2-regularized logistic regression
- Purpose: establish a simple baseline; if LightGBM doesn't beat logistic by >3% AUC,
  prefer the simpler model

**3.2.3 — LambdaRank on NB population (optional)**
- Only if binary classifier AUC > 0.65 and there's signal to rank within binders
- Use tiered labels (1/2/3) on the NB binder subset
- Lower priority: query groups have ~25-38 positives per (PY, quarter), borderline for
  ranking objectives. Primary goal is detection (find binders), not fine-grained SP ranking.

### Phase 3.3: Merge Policy Sweep (dev-only)

**3.3.1 — Build merge scorer**
- New module `ml/merge.py`:
  - `merge_tracks(track_a_df, track_b_df, k, r) -> (merged_df, top_k_indices)`
  - `track_a_df`: established branches with Track A `score` column
  - `track_b_df`: NB candidates with Track B `score` column
  - Returns:
    - `merged_df`: full universe concatenation with `track` provenance column ("A" / "B")
    - `top_k_indices`: np.ndarray of length K — first (K-R) are top Track A indices,
      last R are top Track B indices (in the merged_df index space)
  - Track A scores are preserved for K>50 evaluation. Track B scores are only used to
    select the top-R NB candidates; beyond that, they do not affect the ranking.

**3.3.2 — Sweep R on dev**
- For each R in {0, 5, 10, 15}:
  - Compute merged top-50 for each dev group (using `top_k_override`)
  - Evaluate: VC@50, Abs_SP@50, Recall@50 (on merged top-50);
    VC@100, Recall@100, NDCG (on Track A scores);
    NB12_Count@50, NB12_SP@50, NB12_Recall@50 (on merged top-50)
- Report the Pareto frontier: NB12_Count@50 vs VC@50 regression
- Select R* that maximizes NB12_Count@50 while keeping standard gates passing vs v0c

**3.3.3 — Track A model selection**
- Test both v0c and v3a as Track A:
  - v0c + Track B at R*
  - v3a + Track B at R*
- Pick the combination with best aggregate performance

### Phase 3.4: Holdout Validation

**3.4.1 — Run best configuration on holdout**
- Best (Track A model, Track B model, R*) from dev
- Report all metrics including new NB metrics
- Standard gate check via `check_gates()` against v0c baseline
- NB cross-group gate via `check_nb_threshold()`

**3.4.2 — Registry and reporting**
- Save to registry with full config (track_a_version, track_b_version, R, features_track_b)
- Save `nb_gate_results.json` via extended `save_experiment()` (Phase 3.0.3)
- Save `gate_results.json` via existing `save_experiment()` path
- Generate comparison table: v0c vs two-track candidate
- Update baseline_contract.json if candidate passes all gates

### Phase 3.5: Target Definition Investigation (if time permits)

**3.5.1 — Tiered label methods**
- Current: sorted-index tertiles (n//3 boundaries)
- Alternative 1: SP-quantile cut (qcut by SP value, not count)
- Alternative 2: log-SP quantile cut
- Compare: does label method affect LambdaRank or Track B classifier performance?

**3.5.2 — Track B target alternatives**
- Binary (bind vs not-bind) — already the primary Track B target
- Continuous SP — regression objective
- Compare AUC and precision@10 on dev

## File Changes Summary

| File | Change | Phase |
|------|--------|-------|
| `ml/config.py` | Remove NB12_Recall@50 from TIER1_GATE_METRICS; add TWO_TRACK_GATE_METRICS | 3.0.1 |
| `ml/evaluate.py` | Add NB12_Count@50, NB12_SP@50, NB6/24_Recall@50; add check_nb_threshold(); add top_k_override param to evaluate_group() + cohort_contribution() | 3.0.2 |
| `ml/registry.py` | Add nb_gate_results parameter to save_experiment() | 3.0.3 |
| `registry/baseline_contract.json` | Add NB metric baselines, document gate changes | 3.0.4 |
| `registry/v0c/nb_metrics_supplement.json` | NEW — supplementary NB metrics (does not overwrite metrics.json) | 3.0.5 |
| `registry/v3a/nb_metrics_supplement.json` | NEW — supplementary NB metrics (does not overwrite metrics.json) | 3.0.5 |
| `ml/merge.py` | NEW — two-track merge logic returning top_k_indices | 3.3.1 |
| `scripts/run_nb_analysis.py` | NEW — Phase 3.1 population analysis script | 3.1 |
| `scripts/run_track_b_experiment.py` | NEW — Phase 3.2 Track B model training | 3.2 |
| `scripts/run_two_track_experiment.py` | NEW — Phase 3.3-3.4 merge sweep + holdout | 3.3 |

## Risks and Mitigations

**Risk 1: Track B signal too weak (AUC < 0.55)**
- Mitigation: If density features can't discriminate NB binders from NB non-binders at all,
  fall back to a density-composite heuristic (rank by max(bin_80, bin_90, bin_100) within NB pool)
- This would be a formula, not ML, but still better than the current 0.0

**Risk 2: VC@50 regression exceeds 5% at useful R**
- Mitigation: R=5 is the minimum meaningful slot count. If even R=5 causes >5% regression,
  consider expanding K (top-55 instead of top-50) rather than displacing established branches

**Risk 3: Track B overfits on small NB population**
- Mitigation: use logistic regression as the default; only upgrade to LightGBM if it beats
  logistic by >3% AUC. Strong L2 regularization in either case.

**Risk 4: Dormant vs zero-history are too heterogeneous for one model**
- Mitigation: monitor per-cohort AUC in Track B. If dormant AUC >> zero AUC (or vice versa),
  split Track B into two sub-tracks.

## Dependencies

- All Phase 1-2 infrastructure (data_loader, features, ground_truth, history_features, bridge)
- v0c and v3a registry entries (for Track A baselines)
- No external data sources beyond what Phase 1-2 already uses
- No Ray cluster needed (all experiments run in <2 minutes locally)

## Non-Goals

- This phase does NOT change the universe definition or threshold
- This phase does NOT modify Track A scoring (v0c formula or v3a LambdaRank)
- This phase does NOT explore new data sources (outages, network topology, etc.)
- This phase does NOT implement the signal writer — that is a separate deliverable
