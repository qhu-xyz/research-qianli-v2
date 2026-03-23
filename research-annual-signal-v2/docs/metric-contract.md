# Metric Contract: Cross-Model Comparison (2026-03-23)

## Purpose

This metric suite answers three different questions without conflating them:

1. **Which model captures the most real realized DA SP if used standalone?** (Native Standalone View)
2. **Which model is best at finding NB-hist-12 branches?** (NB-Specialist View)
3. **How much of the observed difference is ranking quality vs universe coverage?** (Coverage + Overlap Views)

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
| `general_ranker` | v0c, V4.4 standalone | Full branch ranking |
| `nb_specialist` | NB-hist-12 ML models | Trained on dormant only |
| `deployment_combo` | R30_nb, R50_nb | v0c + reserved NB slots |

## Mandatory Metric Views

### A. Native Standalone View

**Question**: If I take this model's own top-K from its own universe, how much real value do I capture?

- **Universe**: model's native candidate set (our density universe for v0c/ML, V4.4's own universe for V4.4)
- **Selection**: model's native top-K
- **Metrics**:
  - `Branch_SP@K_native`: total realized SP of top-K branches
  - `Binders@K_native`: count of top-K branches with SP > 0
  - `Precision@K_native`: Binders / K
  - `NB_SP@K_native`: realized SP from NB-hist-12 branches in top-K
  - `NB_Binders@K_native`: count of NB binders in top-K
  - `Label_Coverage@K_native`: fraction of top-K branches with GT labels (SP resolved vs unlabeled)

### B. Overlap-Only View

**Question**: Where both models can score the same branches, whose ranking is better?

- **Universe**: intersection of branches scorable by all compared models
- **Selection**: each model's scores restricted to that overlap
- **Metrics**:
  - `Branch_SP@K_overlap`
  - `Binders@K_overlap`
  - `NB_SP@K_overlap`
  - `NB_Binders@K_overlap`

### C. Deployment View

**Question**: How does the model behave inside our actual production shortlist setup?

- **Universe**: our current branch universe
- **Selection**: projected shortlist used by deployment logic (e.g., R30 reserved slots)
- **Metrics**:
  - `VC@K`: SP captured / total SP in our universe
  - `Abs_SP@K`: SP captured / total DA SP (cross-universe denominator)
  - `Recall@K`: binders found / total binders in our universe
  - `NB_SP@K`
  - `NB_Binders@K`
  - `Fill_Rate@K`: if reserved-slot/backfill logic applies

## Coverage Metrics

Because V4.4 and our models have different universes, coverage must be reported separately from ranking quality.

**For every model**:
- `Candidate_Branches`: number of branches the model can score
- `Candidate_Label_Coverage`: fraction of candidate branches with GT labels
- `Universe_SP_Coverage`: total realized DA SP on labeled candidate branches / total DA SP in slice
- `Universe_NB_SP_Coverage`: same restricted to NB-hist-12 branches

**For V4.4 specifically**:
- `Outside_Our_Universe_Count`: V4.4 branches not in our density universe
- `Outside_Our_Universe_SP`: realized SP from those branches
- `Outside_Our_Universe_NB_SP`: NB SP from those branches

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

## Interpretation Rules

| View | Answers |
|------|---------|
| Native Standalone | Real-world selector quality |
| Overlap-Only | Pure ranking quality (controls for universe) |
| Coverage Metrics | Universe mismatch explanation |
| Deployment | Shipping usefulness |
| NB-specialist can lose on overall SP and still win the NB task |

## Minimum Report Table Set

Every comparison report must contain:

### Table 1: Standalone Native Top-K
`model, Branch_SP@200_native, Binders@200, NB_SP@200, NB_Binders@200, Label_Coverage@200`

### Table 2: Coverage
`model, Candidate_Branches, Universe_SP_Coverage, Universe_NB_SP_Coverage, Outside_Our_Universe_SP (V4.4)`

### Table 3: NB-Only
`model, NB_only_VC@50, NB_only_Recall@50, NB_only_VC@100, NB_only_Recall@100`

### Table 4: Deployment
`config, VC@200, Abs_SP@200, NB_SP@200, VC@400, Abs_SP@400, NB_SP@400`

## Decision Rule

- **General models**: primary = `Branch_SP@200_native`, then `Branch_SP@400_native`
- **NB specialists**: primary = `NB_only_VC@50/100` and `NB_SP@200_native`
- **Shipping decisions**: primary = deployment metrics, not standalone metrics
