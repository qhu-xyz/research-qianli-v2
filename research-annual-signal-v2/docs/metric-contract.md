# Metric Contract: Cross-Model Comparison (2026-03-24, revised)

## Purpose

This metric suite answers three different questions without conflating them:

1. **Which model captures the most real realized DA SP if used standalone?** (Native Standalone View)
2. **Which model is best at finding NB-hist-12 branches?** (NB-Specialist View)
3. **How much of the observed difference is ranking quality vs universe coverage?** (Coverage + Overlap Views)

## CRITICAL: Rank Type Naming

Two types of rank exist. They are NOT interchangeable. Every table, report, and code comment MUST specify which one is being used.

| Name | Definition | Use for |
|------|-----------|---------|
| **`rank_native`** | Model's rank of this branch within its **own full universe**. Bucket_6_20: out of ~2,700. V4.4: out of ~1,200. These are the ranks that determine top-K selection in production. | Deployment decisions, top-K hit rates, "does this branch get selected?" |
| **`rank_overlap`** | Model's rank of this branch after **reranking on the shared overlap set only** (branches both models can score). Smaller denominator. | Pure ranking quality comparison, controlling for universe size. Diagnostic only — does NOT reflect deployment behavior. |

**Forbidden**: Presenting `rank_overlap` numbers as if they are `rank_native`. Presenting `rank_native` from different-sized universes as directly comparable without noting the universe sizes.

## CRITICAL: Rank Direction Conventions

Different signals use **opposite** rank directions. Getting this wrong silently inverts the entire comparison.

| Signal | Convention | Top branch has | Sort order for top-K |
|--------|-----------|---------------|---------------------|
| **V4.4** | Lower rank = better | `rank ≈ 0.001` | **ascending** |
| **V7.0B (v0c)** | Higher rank = better | `rank ≈ 0.79` | **descending** |
| **V6.1** | Lower rank_ori = better | `rank_ori ≈ 0.001` | **ascending** |

**Before sorting ANY signal by rank, verify the direction.** Check: does the top-ranked branch have high or low `shadow_price_da`? If high historical DA → that branch should be near the top → that tells you the direction.

**Past mistake (2026-03-24)**: V7.0B was sorted ascending (like V4.4), which selected the WORST 200 branches. Result: V7.0B showed $13K SP vs V4.4's $421K — a 30x underperformance that was actually a sort bug, not a model failure. Correct sort (descending) shows V7.0B at $619-710K, beating V4.4.

**Example of the confusion this prevents**:
- MNTCELO 2025 onpeak: `rank_native` Bucket_6_20 = 874/2705, V4.4 = 377/1483
- MNTCELO 2025 onpeak: `rank_overlap` Bucket_6_20 = 236/811, V4.4 = 145/811
- Both are "absolute rank" but they answer different questions

## Evaluation Unit

All comparisons at **branch_name** level.
- If a model outputs constraints, collapse to unique branches before scoring.
- A branch appears only once in a model's top-K list.
- Ground truth = branch-level realized DA SP for the eval slice.

## Ground Truth Contract

For every selected branch:
- `SP > 0` → binder
- `SP == 0` → labeled non-binder
- **unlabeled must remain unlabeled** — never silently convert to zero

## Model List Types

| Tag | Examples | Notes |
|-----|---------|-------|
| `general_ranker` | v0c, Bucket_6_20, V4.4 standalone | Full branch ranking |
| `nb_specialist` | NB-hist-12-only models | Trained on dormant only |
| `deployment_combo` | R30_nb, R50_nb | v0c + reserved NB slots |

## Mandatory Metric Views

### A. Native Standalone View

**Question**: If I take this model's own top-K from its own universe, how much real value do I capture?

- **Universe**: model's native candidate set (our density universe for Bucket_6_20, V4.4's own universe for V4.4)
- **Selection**: model's native top-K
- **Rank type**: `rank_native`
- **Metrics**:
  - `Branch_SP@K_native`: total realized SP of top-K branches
  - `Binders@K_native`: count of top-K branches with SP > 0
  - `Precision@K_native`: Binders / K
  - `NB_SP@K_native`: realized SP from NB-hist-12 branches in top-K
  - `NB_Binders@K_native`: count of NB binders in top-K
  - `Label_Coverage@K_native`: fraction of top-K branches with GT labels

### B. Overlap-Only View

**Question**: Where both models can score the same branches, whose ranking is better?

- **Universe**: intersection of branches scorable by all compared models
- **Selection**: each model's scores restricted to that overlap, **reranked within the overlap set**
- **Rank type**: `rank_overlap`
- **Metrics**:
  - `Branch_SP@K_overlap`
  - `Binders@K_overlap`
  - `NB_SP@K_overlap`
  - `NB_Binders@K_overlap`
  - `avg_rank_overlap` of top-N binders (reranked on shared set)

**Important**: `rank_overlap` numbers are smaller than `rank_native` because the denominator is smaller. Do NOT compare `rank_overlap` against `rank_native` from a different table.

### C. Deployment View

**Question**: How does the model behave inside our actual production shortlist setup?

- **Universe**: our current branch universe
- **Selection**: projected shortlist used by deployment logic
- **Rank type**: `rank_native` (our universe)
- **Metrics**:
  - `VC@K`: SP captured / total SP in our universe
  - `Abs_SP@K`: SP captured / total DA SP (cross-universe denominator)
  - `Recall@K`: binders found / total binders in our universe
  - `NB_SP@K`
  - `NB_Binders@K`
  - `Fill_Rate@K`: if reserved-slot/backfill logic applies

## Case Study Tables

When showing individual branch ranks across models:
- **Always state the rank type** in the table header
- **Always include the denominator** (e.g., `874/2705` not just `874`)
- If comparing `rank_native` across models with different universe sizes, note this explicitly
- Never present `rank_overlap` results as contradicting `rank_native` results — they measure different things

## Coverage Metrics

Because V4.4 and our models have different universes, coverage must be reported separately from ranking quality.

**For every model**:
- `Candidate_Branches`: number of branches the model can score
- `Universe_SP_Coverage`: total realized DA SP on candidate branches / total DA SP in slice

**For V4.4 specifically**:
- `Outside_Our_Universe_Count`
- `Outside_Our_Universe_SP`

## NB-Specialist Contract

NB-only models must also be evaluated on the NB task directly:

- **Universe**: NB-hist-12 branches only (per-ctype dormant)
- **Metrics**: `NB_only_SP@K`, `NB_only_VC@K`, `NB_only_Recall@K`, `NB_only_Precision@K`
- **K levels**: 30, 50, 100

## K Levels

| View | K values |
|------|----------|
| NB-only | 50, 100 |
| Standalone + Deployment | 200, 400 |

## Forbidden Practices

- Treat unlabeled as zero
- Compare branch models to constraint models without branch collapse
- Report only projected-to-our-universe results as if they were native standalone results
- Use NDCG as primary cross-model metric when shortlist logic is not pure score order
- **Present `rank_overlap` as `rank_native` or vice versa**
- **Compare absolute rank numbers across different-sized universes without noting the sizes**

## Interpretation Rules

| View | Answers |
|------|---------|
| Native Standalone | Real-world selector quality — "what does this model actually capture?" |
| Overlap-Only | Pure ranking quality diagnostic — "on the same branches, who ranks better?" |
| Coverage Metrics | Universe mismatch explanation |
| Deployment | Shipping usefulness |

## Decision Rule

- **General models**: primary = `Branch_SP@200_native`, then `Branch_SP@400_native`
- **NB specialists**: primary = `NB_only_VC@50/100` and `NB_SP@200_native`
- **Shipping decisions**: primary = deployment metrics, not standalone metrics
