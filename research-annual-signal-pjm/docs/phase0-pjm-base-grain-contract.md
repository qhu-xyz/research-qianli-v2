# PJM Annual Signal — Base Grain Contract

**Date**: 2026-03-25
**Status**: Phase 0 output — defines the PJM evaluation and publish grain

---

## 1. Planning Year Grid

PJM uses the same June–May planning year as MISO.

- **PY 2019** = Jun 2019 – May 2020
- **PY 2025** = Jun 2025 – May 2026
- `auction_month` format: `"2025-06"` (June of start year)

Available PYs in V4.6 benchmark: **2019-06 through 2024-06** (R1), **2019-06 through 2025-06** (R2-R4).

## 2. Auction Rounds

PJM annual has **4 rounds** (vs MISO's 3).

| Round | Close Date (template) | History Cutoff (inclusive) |
|-------|----------------------|--------------------------|
| R1 | April 4 | April 3 |
| R2 | April 11 | April 10 |
| R3 | April 18 | April 17 |
| R4 | April 25 | April 24 |

All rounds fall in April. Month-level cutoff for all rounds: **March** (last full safe month).

Already implemented in `ml/core/calendars.py` in the MISO repo:
```python
PJM_ANNUAL_ROUND_CLOSE = {1: Apr 4, 2: Apr 11, 3: Apr 18, 4: Apr 25}
VALID_PJM_ROUNDS = {1, 2, 3, 4}
```

## 3. Period Type

The V4.6 **publish surface** uses period type **`a`** (annual) — one output per planning year.

However, the underlying **bridge data** (PJM_SPICE_CONSTRAINT_INFO) IS partitioned by `aq1`–`aq4`, exactly like MISO. Different constraints may be active in different quarters. GT mapping must be done per-quarter, then aggregated to annual for the publish surface.

**Implication**: The evaluation grain is `(planning_year, round, class_type)`. The quarterly bridge structure is internal to the GT mapping pipeline, not exposed in the publish grain.

## 4. Class Types

PJM has **3 class types** (vs MISO's 2):

| Class Type | Description |
|------------|-------------|
| `onpeak` | Weekday peak hours |
| `dailyoffpeak` | Daily off-peak hours |
| `wkndonpeak` | Weekend on-peak hours |

In the V4.6 notebook, `wkndonpeak` is auto-generated alongside `onpeak` (same signal, different SF). All three are published.

**No `offpeak` class** — PJM uses `dailyoffpeak` instead.

## 5. Evaluation Grain

The base grain cell is:

```
(planning_year, market_round, class_type)
```

Total cells per PY: **4 rounds × 3 class types = 12 cells**

For 6 historical PYs (2019-06..2024-06) × 1 round × 3 classes = **18 cells per round**. Total across all 4 rounds: **72 cells** (R2-R4 can add 2025-06 for up to 21 cells each).

## 6. V4.6 Benchmark — Publish Surface

### Path layout
```
/opt/data/xyz-dataset/signal_data/pjm/
  constraints/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}/{PY}/a/{class_type}/
  sf/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}/{PY}/a/{class_type}/
```

### Constraint schema (34 columns)
| Column Group | Columns |
|-------------|---------|
| Deviation features (20) | `{0,60,65,70,75,80,85,90,95,100}_{max,sum}` |
| Rank columns (4) | `deviation_max_rank`, `deviation_sum_rank`, `shadow_rank`, `rank` |
| Metadata (10) | `constraint`, `equipment`, `flow_direction`, `convention`, `shadow_sign`, `shadow_price`, `shadow_price_da`, `tier`, `deviation_max`, `deviation_sum` |

### Rank direction (CRITICAL)

**Low rank = best.** Verified empirically:

| rank | shadow_price_da | tier | interpretation |
|------|----------------|------|----------------|
| 0.001 | 602,672 | 0 | Most important |
| 1.000 | 0.0 | 4 | Least important |

- `rank` ∈ [0, 1], where 0 = most important constraint
- `tier` ∈ {0, 1, 2, 3, 4}, where 0 = top 12% (best)
- Correlation(rank, shadow_price_da) = **-0.28**
- Same direction as MISO V4.4; opposite to V7.1B

### SF schema
- Rows: `pnode_id` (PJM pricing nodes, ~11k)
- Columns: `{constraint}|{shadow_sign}|spice` (one per published constraint, ~1k)
- Values: shift factors (float64)

### Constraint counts (PY 2024-06)
| Round | Constraints |
|-------|------------|
| R1 | 1,043 |
| R2 | 1,049 |
| R3 | 1,066 |
| R4 | 1,091 |

## 7. MISO → PJM Structural Differences

| Dimension | MISO | PJM |
|-----------|------|-----|
| Rounds | 3 (R1, R2, R3) | 4 (R1, R2, R3, R4) |
| Period type (publish) | aq1-aq4 (quarterly) | a (annual) |
| Period type (bridge/GT internal) | aq1-aq4 | aq1-aq4 (same as MISO) |
| Class types | onpeak, offpeak | onpeak, dailyoffpeak, wkndonpeak |
| Close dates | Apr 8, Apr 22, May 5 | Apr 4, 11, 18, 25 |
| Base grain cell | (PY, quarter, round, ctype) | (PY, round, ctype) |
| Cells per PY | 4q × 3r × 2c = 24 | 1p × 4r × 3c = 12 |
| Benchmark rank dir | low = best (V4.4) | low = best (V4.6) |

## 8. What Must Be PJM-Native (Not Copied from MISO)

1. **Data paths**: density, bridge, limits, SF — all PJM-specific NFS paths
2. **Bridge mapping**: PJM constraint_id = `monitored_facility:contingency_facility` (ACTUAL → BASE normalization)
3. **DA loading**: candidate loader is `PjmApTools.get_da_shadow_by_peaktype()` (delegates to `PjmDaShadowPrice`). Returns constraint-level hourly data. **Pending Phase 1 confirmation** of daily vs monthly behavior, replacement logic, and effective date range
4. **Class type handling**: 3 classes, `wkndonpeak` auto-derived from `onpeak`
5. **Quarterly bridge internally, annual publish externally**: bridge data uses aq1-aq4 per-quarter partitions, but V4.6 publishes as annual `a`. GT mapping must be per-quarter, aggregated to annual
6. **Density root**: all PJM annual data now under single root `network_model=miso/spice_version=v6/auction_type=annual/`. PYs 2018-06 through 2025-06 available. 2022-06 has R1-R2 only. See `phase1-data-source-contract.md` for current coverage table.
7. **Universe threshold**: must be calibrated for PJM independently
8. **Supplement key fallback**: may not exist or may differ for PJM

## 9. What to Reuse from MISO (Architecture Only)

1. **Calendar module** (`ml/core/calendars.py`) — already has PJM round dates
2. **Evaluation metrics** (`ml/evaluate.py`) — `abs_sp_at_k` is market-agnostic
3. **Training pipeline** (`ml/train.py`) — LightGBM wrapper is market-agnostic
4. **Registry schema** (`ml/core/registry.py`) — spec.json / metrics.json structure
5. **Feature assembly pattern** (`ml/features.py`) — LEFT JOIN universe + GT + history
6. **GT vs model-universe separation** — same architectural discipline
7. **Output schema pattern** (`ml/products/annual/output_schema.py`) — adapt columns for PJM

## 10. Known Risks for Phase 1

1. **R1 has no PY 2025-06** in V4.6 — benchmark gap for holdout year
2. **DA loading behavior unknown** — must verify PjmApTools daily/monthly DA availability and date coverage
3. **Bridge coverage unknown** — PJM may have different CID→branch recovery rates
4. **Density split across two roots** — loader must resolve correct root per (PY, round). PY 2022-06 entirely missing, PY 2024-06 split by round
5. **wkndonpeak auto-derivation** — need to understand whether this is a separate model or just onpeak with different SF

---

## 11. Reconstruction Constraint

For this repo, V4.6 is a **benchmark**, not a reproducible source dependency.

Current project constraint:

- do not depend on hidden upstream flow simulator artifacts such as `flow_memo.parquet`
- use reproducible shared inputs such as density + DA history to build our own annual signal
- evaluate success by branch-level benchmark comparison against V4.6, not by exact internal formula replication
