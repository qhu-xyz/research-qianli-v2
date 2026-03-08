# Production Migration Assessment

## How V6.2B Production Signal is Consumed

**Consumer**: `pmodel/base/ftr23/v6_ori/` via `pbase/analysis/tools/all_positions.py:get_signal_expo2()`

**Columns actively consumed by pmodel** (from `all_positions.py:1437-1503`):
1. `shadow_price` — used for exposure calculation (`sf_expo * shadow_price`)
2. `shadow_sign` — used for unit exposure direction (`sf_expo * shadow_sign`)
3. `tier` — mapped to trades for filtering/sampling (`expo["tier"]`, `ppp["tier"]`)
4. `rank_ori` — mapped to trades if present (`expo["rank_ori"]`)
5. `constraint_id` — join key with shift factor matrix

**Config consumed** (from `miso_models_f0.py:105-119`):
- `shadow_price_upper_clip: 30000` — clips signal shadow price
- `white_list_tiers: None` — no tier filtering for V6.2B (all tiers used)
- `unit_expo_th: None` — no exposure threshold

**Other columns in parquet but NOT directly consumed by pmodel**:
- `rank` (dense rank normalized), `density_mix_rank`, flow features, `branch_name`, etc.
- These are informational / used by other analysis tools

## Production V6.2B Output Schema (21 columns)

| Column | Type | Source | Consumed? |
|--------|------|--------|-----------|
| constraint_id | String | V6.2B signal | YES (join key) |
| flow_direction | Int64 | V6.2B signal | no |
| mean_branch_max | Float64 | Density model | no |
| mean_branch_max_fillna | Float64 | Density model | no |
| ori_mean | Float64 | Density model | no |
| branch_name | String | MISO topology | no |
| bus_key | String | MISO topology | no |
| bus_key_group | String | MISO topology | no |
| mix_mean | Float64 | Density model | no |
| shadow_price_da | Float64 | Historical DA (60mo) | no |
| density_mix_rank_value | Float64 | Computed rank | no |
| density_ori_rank_value | Float64 | Computed rank | no |
| da_rank_value | Float64 | Computed rank | no |
| **rank_ori** | Float64 | **Formula output** | **YES** |
| density_mix_rank | Float64 | Computed rank | no |
| **rank** | Float64 | **dense_rank(rank_ori)** | no (but readable) |
| **tier** | Int64 | **From rank (5 tiers)** | **YES** |
| **shadow_sign** | Int64 | **Direction multiplier** | **YES** |
| **shadow_price** | Float64 | **shadow_price_da * shadow_sign** | **YES** |
| equipment | String | MISO topology | no |
| __index_level_0__ | String | Pandas index artifact | no |

## Tier Assignment (Production)

Production uses 5 tiers (0-4), roughly equal-sized buckets from rank:
- tier 0: ~20% (best/most binding, lowest rank_ori)
- tier 1: ~20%
- tier 2: ~20%
- tier 3: ~20%
- tier 4: ~20% (least binding)

Sample: tier 0=110, tier 1=112, tier 2=111, tier 3=111, tier 4=113 (n=557)

## Q1: Can v0 Reproduce 100% of Production Signal for f0?

**YES for `rank_ori`.** Verified exact match: `max|formula - rank_ori| = 0.0` across all months.

**YES for `shadow_price`.** It's `shadow_price_da * shadow_sign` — both present in the parquet.

**YES for `shadow_sign`.** Present in the parquet as-is.

**ALMOST for `tier`.** Production assigns 5 tiers (0-4) via roughly-equal quantile bins on rank.
Our pipeline doesn't currently produce tier assignment. Gap: we need to replicate the exact tier
assignment logic (quantile-based, ~20% per tier).

**ALMOST for `rank`.** Production computes `dense_rank(rank_ori) / max_dense_rank`.
We can reproduce this trivially but haven't implemented it as it's not consumed by pmodel.

**Summary: v0 can reproduce the 4 actively consumed columns with minimal work** (tier assignment
is the only gap, and it's a simple quantile binning).

## Q2: Can ML Versions Produce the Same Output?

**YES, with a different `rank_ori` (ML score instead of formula).**

The ML versions output a continuous score per constraint (higher = more binding). To produce the
same output format:

1. `rank_ori` → Replace with ML prediction score (or rank-normalized version)
2. `rank` → `dense_rank(ml_score) / max_rank`
3. `tier` → Quantile bins on ml_score (same 5-tier logic as production)
4. `shadow_price` → Unchanged (same source: shadow_price_da * shadow_sign)
5. `shadow_sign` → Unchanged (same source)
6. `constraint_id` → Unchanged

**Gaps to close:**

| Gap | Effort | Description |
|-----|--------|-------------|
| Tier assignment | Small | Quantile-bin ML scores into 5 tiers. Replicate production logic. |
| Score normalization | Small | ML outputs raw regression values; need to normalize to [0,1] like rank_ori |
| Output writer | Medium | Write parquet in production schema via `ConstraintsSignal.save_data()` |
| Signal name | Config | Register new signal name (e.g., `TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1`) |
| f0 only | None | We only train on f0 currently. Production has f0-f3 + quarterly. |
| Offpeak | Medium | We only eval onpeak. Production runs both. Need offpeak ground truth. |

## Codex Audit Findings (Must Fix Before Production)

From `codex-review/audit.md`:

1. **Tier0-AP / Tier01-AP are degenerate** — always 1.0 because binding rate < 20%.
   These metrics and any gates built on them are invalid. **Must remove from gates.**

2. **Recall@100 is tie-contaminated** — fewer than 100 binding constraints per month,
   so the "true top 100" includes non-binding rows. **Use Recall@50 or Recall@20 instead.**

3. **FEATURES_V3 / ml_pred not wired** — `mlpred_loader.py` exists but is never called.
   Missing features silently zero-fill. **Remove or complete.**

4. **Governance artifacts stale** — `champion.json` still points to v0, comparisons stop at v1b.
   **Update to reflect v6b as dev champion.**

## Recommended Migration Steps

1. **Fix codex audit findings** (clean up gates, remove dead code)
2. **Implement tier assignment** (quantile binning matching production)
3. **Implement output writer** (write parquet matching production schema)
4. **Verify v0 reproduction** (write production-format output, diff against actual V6.2B)
5. **Generate ML signal** (v6b output in production format)
6. **Shadow run** (run ML signal alongside V6.2B in pmodel, compare exposure/tier assignments)
7. **Register new signal name** (e.g., V7.0) and configure in pmodel params
