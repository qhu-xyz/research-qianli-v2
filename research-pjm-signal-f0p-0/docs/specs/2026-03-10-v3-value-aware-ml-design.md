# V3: Value-Aware ML with Enriched Features — Design Spec

**Date**: 2026-03-10
**Status**: Approved
**Problem**: V2 ML (9 features + tiered labels) loses to v0 formula on VC@20 (-6.7% holdout) despite winning on Recall (+9.5%) and Spearman (+17.8%). PJM's formula baseline is inherently strong (holdout VC@20=0.55) because da_rank_value is 3-5x more predictive than MISO's. ML finds more moderate binders but misses high-value whales.

## Root Cause Analysis

1. **Feature gap**: V2's binding_freq features predict *binary binding* (did it bind?), not *value magnitude* (how much?). The formula implicitly encodes value via shadow_price_da → da_rank_value.
2. **Label gap**: Tiered labels (0/1/2/3) collapse value information. A $50K whale and $50 binder both get tier 3. LambdaRank optimizes NDCG over these tiers, so has no incentive to rank whales above moderate binders.

## Solution: Two Changes

### Change 1: Enriched Feature Set (14 features)

| # | Feature | Source | Monotone | Notes |
|---|---------|--------|:--------:|-------|
| 1-5 | binding_freq_{1,3,6,12,15} | Realized DA | +1 | Recurrence signal (kept from v2) |
| 6 | v7_formula_score | V6.2B blend | -1 | Formula baseline (kept) |
| 7 | da_rank_value | V6.2B | -1 | Historical DA rank (kept) |
| 8 | **shadow_price_da** | V6.2B | +1 | Historical DA magnitude — range 0-15K, gives model value info lost by ranking |
| 9 | **binding_probability** | spice6 ml_pred | +1 | Independent binding estimate (0-1) |
| 10 | **predicted_shadow_price** | spice6 ml_pred | +1 | Spice6 regression value prediction |
| 11 | prob_exceed_110 | spice6 density | +1 | Density overflow (kept) |
| 12 | **prob_exceed_100** | spice6 ml_pred | +1 | Near-limit density threshold |
| 13 | constraint_limit | spice6 density | 0 | Physical limit (kept) |
| 14 | **hist_da** | spice6 ml_pred | +1 | Historical DA indicator from spice6 |

New features (8-10, 12, 14) address the value-prediction gap. shadow_price_da is already loaded but never used as a feature. spice6 ml_pred features join on branch_name and are confirmed available (92 auction months, 2018-06 to 2026-01).

### Change 2: Value-Aware Labels

Replace 4-tier discrete labels with log-transformed continuous relevance:

```python
# Non-binding: label = 0
# Binding: label = log1p(realized_sp) / scale, winsorized at p99
```

LambdaRank supports continuous relevance. log1p compresses the heavy tail while preserving ordering:
- $50K → label ~10.8
- $5K → label ~8.5
- $500 → label ~6.2
- $50 → label ~3.9
- $0 → label 0

This teaches the model that whales matter proportionally more.

## Experiment Variants

| Variant | Features | Labels | Purpose |
|---------|----------|--------|---------|
| v3a | 9 (v2 set) | value-aware | Isolate label change |
| v3b | 14 (enriched) | tiered (v2) | Isolate feature change |
| v3c | 14 (enriched) | value-aware | Combined |
| v3d | v3c + two-stage | value-aware | Fallback if v3c loses VC@20 |

v3d two-stage: top-K by formula ranking preserved, remainder re-ranked by ML. Guarantees VC@20 >= v0.

## Data Flow

```
V6.2B parquet ──→ shadow_price_da (already loaded, add to features)
       │
spice6 ml_pred ──→ binding_probability, predicted_shadow_price, hist_da, prob_exceed_100
  (join on constraint_id + flow_direction, keyed by auction_month + market_month + class_type)
       │
Realized DA ──→ log1p(realized_sp) as continuous label
       │
LightGBM LambdaRank (100 trees, lr=0.05, 31 leaves, num_threads=4)
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| shadow_price_da ≈ da_rank_value redundancy | LightGBM handles correlated features; rank loses magnitude |
| spice6 ml_pred missing for early months | Fill with 0 (same as density loader) |
| Log labels sensitive to outliers | Winsorize at p99 before log |
| 14 features overfit on 8-month train | Monotone constraints + regularization; explicit 14f vs 9f comparison |

## Success Criteria

v3 must beat v0 on ALL metrics simultaneously (mean across 6 slices, holdout):
- VC@20 > 0.553 (v0 holdout mean)
- Recall@20 > 0.255
- Spearman > 0.246
- NDCG > 0.525
- No individual slice regression > 10% on any metric
