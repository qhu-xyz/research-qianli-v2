# V6.2B Universe Coverage Gap Analysis

**Date**: 2026-03-11
**Slice**: f0/onpeak (85 months, 2018-01 to 2025-01)

## Key Finding

V6.2B aggressively filters the raw density universe, missing the majority of binding constraints and value.

| Metric | V6.2B | Raw Density | Gap |
|--------|-------|-------------|-----|
| Avg constraints/month | ~450 branches | ~3,120 branches | 7x smaller |
| Binding constraints captured | 33% (47/142) | 89% (+54%) | **65% missed** |
| Binding value captured | 47% | 89% (+42%) | **53% missed** |
| Top-20 binders captured | ~9/20 | ~18/20 | 11 missed |

## Data Pipeline

```
Raw spice6 density: ~11,600 constraint_ids × 11 outage_dates
    ↓ constraint_info mapping
~3,120 unique branches
    ↓ V6.2B filtering (unknown criteria)
~400-580 rows → ~300-450 unique branches
```

## What V6.2B Misses

- Missed binders have **similar mean shadow prices** to captured ones (not low-value noise)
- Examples from 2024-06: JUN-TMI ($21,165), CNS-NOR1 ($23,262), HAN-JUN1 ($20,680) — all absent from V6.2B
- The missed constraints are real, high-value binding events

## Implications

The biggest improvement opportunity is NOT better ML within V6.2B's 450-constraint universe.
It's expanding the constraint universe by building from raw density data (~3,120 branches vs ~450).

This would require:
1. Loading raw spice6 density scores (score.parquet) for all ~11,600 constraints
2. Mapping to branches via constraint_info
3. Aggregating across outage_dates (mean score per branch)
4. Joining DA history + ml_pred features
5. Training ML on the full universe

## Reproduction

Analysis was run interactively. Key data paths:
- Raw density: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/density/auction_month={M}/market_month={M}/market_round=1/outage_date={D}/score.parquet`
- Constraint info: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/constraint_info/auction_month={M}/market_round=1/{ptype}/{ctype}/`
- V6.2B: `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/{M}/{ptype}/{ctype}`
- Realized DA cache: `data/realized_da/`
