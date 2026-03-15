# Annual Signal v2: Implementer Guide

**Date**: 2026-03-11
**Status**: Ready for review — NO code exists yet. All implementation code was deleted. This doc is the sole spec.
**Context**: MISO Annual FTR constraint ranking ML pipeline, redesign from v1 (research-annual-signal)

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Business Context](#2-business-context)
3. [Critical Decision: V4.4 Abandoned](#3-critical-decision-v44-abandoned)
4. [Data Architecture](#4-data-architecture)
5. [Universe Definition](#5-universe-definition)
6. [Collapse Strategy](#6-collapse-strategy)
7. [Feature Design](#7-feature-design)
8. [Ground Truth Pipeline](#8-ground-truth-pipeline)
9. [Training Approach](#9-training-approach)
10. [Evaluation Metrics & Gates](#10-evaluation-metrics--gates)
11. [New Binding (NB) Detection & Branch Cohorts](#11-new-binding-nb-detection--branch-cohorts)
12. [Implementation Plan (Phases)](#12-implementation-plan-phases)
13. [What v1 Proved (Reusable Knowledge)](#13-what-v1-proved-reusable-knowledge)
14. [What v1 Got Wrong](#14-what-v1-got-wrong)
15. [Traps & Pitfalls (MUST READ)](#15-traps--pitfalls-must-read)
16. [External Dependencies & Environment](#16-external-dependencies--environment)
17. [File Layout](#17-file-layout)
18. [Data Paths Quick Reference](#18-data-paths-quick-reference)

---

## 1. What This Project Does

Rank MISO constraints by likelihood of binding in the Day-Ahead (DA) electricity market for the annual FTR (Financial Transmission Rights) auction. The ranking is used for trade selection — constraints ranked higher are more likely to produce profitable FTR positions.

Universe size varies by quarter (typically 1,100-1,850 branch_names per quarter after right-tail density filtering and constraint-to-branch collapse; see §5 for exact numbers).

**Input**: Raw SPICE density distributions (forward-looking power flow simulations) + realized DA shadow prices (historical binding data)
**Output**: Per-branch_name ranking score for each (planning_year, aq_quarter) group

The signal covers:
- **4 quarterly periods**: aq1 (Jun-Aug), aq2 (Sep-Nov), aq3 (Dec-Feb), aq4 (Mar-May)
- **Per planning year**: PY runs from June to May (e.g., PY 2025-06 covers Jun 2025 - May 2026)
- **Class type**: Class-type agnostic — density features have no class_type. Ground truth uses combined onpeak + offpeak DA. BF features include combined (either ctype) alongside separate onpeak/offpeak.

---

## 2. Business Context

### What Is an FTR?

An FTR (Financial Transmission Right) pays out based on congestion between two nodes in the power grid. When a transmission constraint binds (flow hits the limit), it creates shadow prices that determine FTR payoffs. Our job: predict WHICH constraints will bind and how much.

### What Is a Constraint?

A physical transmission line or transformer with a thermal limit. When power flow hits the limit, the constraint "binds" and creates a shadow price. MISO has ~14,000 modeled constraints, of which ~300-600 actually bind in any given quarter.

### What Is the Annual Auction?

MISO holds an annual FTR auction each April for the upcoming planning year (June-May). Bids must be submitted by ~April 10. The auction clears in 4 quarterly strips (aq1-aq4). Our signal ranks constraints to identify the best trading opportunities.

### Why Ranking, Not Regression?

We don't need to predict the exact shadow price. We need to rank constraints by binding likelihood so traders can focus on the most promising ones. LambdaRank (learning-to-rank) is the natural approach — it directly optimizes the ordering.

### V6.1 → V4.4 → Raw Density (History of This Project)

- **v1 (research-annual-signal)**: Used V6.1 signal (286-395 constraints). Ran 17 experiments (v0-v17). V6.1 had the smallest universe, covering only ~38-42% of realized binding SP.
- **v2 initial plan**: Migrate to V4.4 signal (1,227 constraints, ~68-84% coverage). V4.4 has more constraints and richer features.
- **v2 current plan**: Investigation proved V4.4 is not reproducible from raw data (opaque transformation). Abandoned V4.4. Now using raw SPICE density distribution directly, collapsed to ~1,100-1,850 branch_names per quarter (right-tail density filter, then cid→branch collapse) covering 80-84% of mappable binding SP.

---

## 3. Critical Decision: V4.4 Abandoned

### What V4.4 Is

V4.4 (`TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1`) is a pre-assembled signal with 1,227 constraints and 35 columns including:
- 20 "percentile" features: `{0,60,65,70,75,80,85,90,95,100}_{max,sum}`
- 3 deviation features: `deviation_max_rank`, `deviation_sum_rank`, `deviation`
- Historical DA features: `shadow_price_da`, `shadow_rank`
- Metadata: `flow_direction`, `equipment`, `convention`, `shadow_sign`, `rank`, `tier`, `__index_level_0__`

### Why It Was Abandoned

Investigation on 2026-03-11 proved V4.4's percentile features are **not reproducible** from the raw SPICE density distribution data they claim to be derived from. Five independent tests:

**Test 1: Quantization**. V4.4's `_max` values (e.g., `70_max = 0.96875`, `80_max = 0.9375`) are exact multiples of 1/32 (0.03125). Raw density bin values are NOT multiples of 1/32. This means V4.4 applies some quantization step that is not documented.

**Test 2: Non-monotonicity**. For some constraints, `70_max > 80_max` in V4.4. If these were cumulative exceedance probabilities (P(flow > threshold)), this would be impossible. The raw density data is also non-monotonic — but for a different reason: it's a probability density distribution (peaked, not cumulative). V4.4 claims to derive from the raw density but the non-monotonicity patterns don't match.

**Test 3: Nonzero where raw is zero**. For constraints where the raw density at threshold 100 is exactly 0.0, V4.4's `100_max` can be nonzero (e.g., 0.03125). This means V4.4 is adding probability mass that doesn't exist in the raw data.

**Test 4: No ratio pattern**. If V4.4 were simply rescaling the raw density, V4.4/raw ratios would be constant. They aren't — ratios vary across thresholds for the same constraint.

**Test 5: No tail-sum match**. V4.4's values don't correspond to tail sums (probability above threshold), weighted averages, or any obvious linear combination of the raw density bins.

**Conclusion**: V4.4 is a black box. The transformation involves quantization (1/32 steps), possible smoothing, possibly different aggregation logic (different outage_date weighting?), and possibly even a different density source. We cannot understand, debug, or extend it.

### What Replaces It

Raw SPICE density distribution is now the primary data source. Benefits:
- Full transparency in every transformation
- Larger universe (~1,100-1,850 branches per quarter vs 1,227 cids) with better SP coverage
- Ability to engineer features from all 77 bins (V4.4 only exposed 10 thresholds)
- No dependency on an opaque pipeline we don't control

### V4.4 Features We Lose

V4.4's `deviation_max_rank`, `deviation_sum_rank` are flow deviation features independent from DA history (Spearman ρ < 0.1 with da_rank_value). These came from a different part of the SPICE pipeline and we don't have a direct replacement from raw density. The density bins themselves provide forward-looking signal, but deviation_rank specifically measures how close flow gets to the limit — a different facet.

If deviation features turn out to be important, we may need to investigate the SPICE deviation data source separately.

---

## 4. Data Architecture

### 4.1 Three Naming Systems in MISO

MISO uses three distinct constraint naming conventions. They do NOT match each other:

| System | Format | Example |
|--------|--------|---------|
| **DA (realized market)** | Long human-readable with zone suffix | `ADAMS_I ADAMSHAYWA16_1 1 (LN/ALTW/ALTW)` |
| **SPICE / V4.4 / V6.1 (model)** | Short abbreviated | `08CLO08STI13_1 1` |
| **Bridge table** | Same as SPICE | `LEONIPLYMO13_1 1` |

**Direct match between DA `branch_name` and SPICE `equipment`: 0 out of 423 (0.0%).** The branch_name strings use completely different formats — bridge table is required.

**However, constraint_id numeric IDs DO partially overlap.** ~60% of binding DA constraint_ids are also in the density universe (329/546 for 2025-06/aq1). But direct cid matching alone is insufficient for training targets — see §8.2 for why branch-level target aggregation is required regardless.

### 4.2 Primary Data Sources

#### Density Distribution (PRIMARY — features)
- **Path**: `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet`
- **Partitions**: `spice_version/auction_type/auction_month/market_month/market_round/outage_date`
- **For annual**: `spice_version=v6/auction_type=annual/auction_month={PY}/market_month={month}/market_round=1/outage_date={date}`
- **Granularity**: 1 row per (constraint_id, market_month, outage_date)
- **Columns**: `constraint_id` + 77 shadow price threshold bins
- **NO `flow_direction` column** — density distribution is direction-agnostic
- **Row count**: ~12,841 unique constraint_ids × ~11 outage_dates per month × 3 months per quarter ≈ ~424k rows per quarter
- **pbase loader**: `MisoSpiceDensityDistributionOutput` in `pbase.data.dataset.spice.pipeline` — **DOES NOT WORK for annual data** (see note below)

The 77 bin columns are named by shadow price thresholds from -300 to +300. Exact column names (verified from data):

```
-300, -280, -260, -240, -220, -200, -180, -160, -150, -145, -140, -135, -130, -125, -120,
-115, -110, -105, -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35,
-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 160, 180, 200,
220, 240, 260, 280, 300
```

**SEMANTICS — NOT WHAT YOU MIGHT EXPECT**: These values are NOT cumulative exceedance probabilities and are NOT individually bounded to [0, 1]. Observed properties from actual data:

1. **Every row sums to exactly 20.0** across all 77 bins. This is a normalized distribution, not independent probabilities.
2. **Values are NOT monotonically decreasing** across bins. For a given constraint, the value at bin "10" can be larger than at bin "5" (e.g., 5.32 > 3.90). Bins are NOT "probability of exceeding threshold X."
3. **Individual values can exceed 1.0** (e.g., bin "15" = 5.83 for constraint 1000). They are density weights, not probabilities.
4. **Many bins are zero or near-zero** — typically 15-66 bins are nonzero per row, concentrated around the constraint's typical shadow price range.
5. **The distribution appears to represent shadow price probability density** — higher values near the expected shadow price, tapering to zero in the tails. The constant sum of 20 suggests normalization to a fixed total weight.

Example row (constraint_id=1000, outage_date=2025-07-01):
```
bin "0"  = 0.0000001   bin "5"  = 3.90   bin "10" = 5.32   bin "15" = 5.83
bin "20" = 3.82        bin "30" = 0.15   bin "50" = ~0      bin "70" = 0
Sum = 20.0 (only ~15 bins nonzero, all in the 0-30 range)
```

Example row (constraint_id=100023, same date — wider distribution):
```
bin "0"  = 1.21   bin "10" = 1.04   bin "30" = 1.25   bin "70" = 0.52
bin "90" = 0.08   bin "100" = 0.01  bin "120" = 0.0007
Sum = 20.0 (66 bins nonzero, spread across 0-150 range)
```

**IMPLICATION FOR FEATURE ENGINEERING**: Since the semantics are not fully understood, the implementer should:
- NOT assume monotonicity across bins — do NOT set `monotone_constraints` for density bin features without empirical validation
- Treat each bin as an independent feature and let the model learn relationships
- Consider derived aggregations: sum of positive-side bins (0 to 300), peak bin index, spread/width of the distribution, right-tail weight (sum of bins >= 100)
- Run correlation analysis between adjacent bins to identify redundant groups before pruning

#### Density Signal Score — NOT USED

`MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet` exists in the spice6 data but is **not used** in this pipeline. It is a pre-computed binding probability derived from the 77-bin density distribution by an opaque, undocumented transformation (same problem as V4.4). We use the raw density bins directly — fully transparent and reproducible.

**Universe filter** uses raw density bins instead: `right_tail_max >= threshold` where `right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` across outage_dates per cid. See §5.4 for details.

**`count_active_cids`** uses the same raw filter: cids with `right_tail_max >= threshold`.

#### Constraint Limits (FEATURE)
- **Path**: `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet`
- **Columns**: `constraint_id`, `limit` (thermal limit in MW)
- **~14,007 constraints**
- **pbase loader**: `MisoSpiceConstraintLimitOutput` — **DOES NOT WORK for annual data** (see note below)

#### CRITICAL: pbase Loaders Do NOT Work for Annual Data

The pbase outage-date loaders (`MisoSpiceDensityDistributionOutput`, `MisoSpiceDensitySignalScoreOutput`, `MisoSpiceConstraintLimitOutput`) construct partition paths as:

```
spice_version=v6/auction_month=YYYY-MM/market_month=YYYY-MM/market_round=N/outage_date=YYYY-MM-DD
```

But the actual parquet layout for annual data has an **additional `auction_type` level**:

```
spice_version=v6/auction_type=annual/auction_month=YYYY-MM/market_month=YYYY-MM/market_round=N/outage_date=YYYY-MM-DD
```

The loader path `spice_version=v6/auction_month=2025-06` does not exist — `auction_type=annual` sits between them. The loaders return empty DataFrames for annual partitions.

**Workaround**: Use `pl.scan_parquet()` with `hive_partitioning=True` directly. This discovers all partition levels automatically:

```python
import polars as pl

density = pl.scan_parquet(
    '/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet',
    hive_partitioning=True
).filter(
    (pl.col('auction_type') == 'annual')
    & (pl.col('auction_month') == '2025-06')
    & (pl.col('market_month') == '2025-07')
).collect()
```

This is the **only reliable path for loading annual spice6 data**. Do NOT attempt to use pbase loaders for annual partitions. (Monthly data may work with the loaders since both `auction_type=monthly` exists and the loader's path construction may resolve, but verify before relying on it.)

#### Bridge Table (GROUND TRUTH MAPPING)
- **Path**: `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet`
- **Partitions**: `spice_version/auction_type/auction_month/market_round/period_type/class_type`
- **Key columns**: `constraint_id` (join key to DA), `branch_name` (join key to signal universe)
- **~14,000 constraint_ids per partition**
- **Schema issue**: Different partitions have different schemas (column `device_type` is Null in some, String in others). ALWAYS read partition-specific paths or use `pl.scan_parquet()` with `hive_partitioning=True` and filter to a single partition.

#### Realized DA Shadow Prices (GROUND TRUTH + BF FEATURES)
- **Access**: `MisoApTools().tools.get_da_shadow_by_peaktype(st='YYYY-MM-01', et_ex='YYYY-MM-01', peak_type='onpeak')` (requires Ray)
- **Columns**: `constraint_id`, `constraint_name`, `branch_name`, `contingency_description`, `shadow_price`, `monitored_line`, `year`, `month`, `day`
- **Cache**: Build your own in `data/realized_da/` via `scripts/fetch_realized_da.py` (requires Ray). Fetch BOTH onpeak and offpeak per month. See §8.3 Step 1 for pattern.
- **DA `branch_name`** uses long format (e.g., `ADAMS_I ADAMSHAYWA16_1 1 (LN/ALTW/ALTW)`) — does NOT match SPICE `equipment`/`branch_name`. Bridge table is required.

### 4.3 Data Lineage

```
Raw density distribution (77 bins, per constraint per outage_date)
  → [V4.4 was here — ABANDONED, opaque transformation from density to percentiles]
  → [density_signal_score was here — DROPPED, opaque derivation from same bins]

v2 data flow:
  1. Load density distribution for target (PY, quarter) market_months
  2. Compute right_tail_max = max(bin_80, bin_90, bin_100, bin_110) per cid across outage_dates
  3. Filter universe: right_tail_max >= threshold → ~2,400-3,900 cids per quarter
  4. Level 1 collapse: density to (constraint_id, quarter) — mean across outage_dates/months
  5. Level 2 collapse: (constraint_id, quarter) → (branch_name, quarter) — max/min across cids
     Result: ~1,100-1,850 branch_names per quarter (see §6.2)
  6. Join constraint_limit (collapsed: min/mean/max/std across cids per branch)
  7. Add branch metadata: count_cids, count_active_cids
  8. Compute BF features from realized DA cache (onpeak + offpeak + combined) — naturally branch-level
  9. Compute da_rank_value from historical realized DA — naturally branch-level
  10. Get ground truth: realized DA (combined ctype) → annual bridge + monthly fallback → labels
      Target = sum(abs(SP)) per branch. (Monthly fallback critical for 2025-06: recovers +13.4% SP. See §8.2-8.3.)
```

### 4.4 How V6.2B Monthly Signal Works (for comparison)

The monthly signal (V6.2B, `TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1`) uses the same underlying SPICE data but for monthly (f0/f1/f2/f3) period types. It has:
- 489 constraints per (month, period_type)
- Columns: `da_rank_value`, `density_mix_rank_value`, `density_ori_rank_value`, `ori_mean`, `mix_mean`, etc.
- Formula: `rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value`
- `da_rank_value` is rank of HISTORICAL DA shadow price (NOT realized/leaked)
- `shadow_price_da` is HISTORICAL — Spearman ~0.81 with V6.4B hist_shadow, only ~0.36 with actual realized shadow

Our annual v2 will NOT use V6.2B directly but follows a similar pattern: combine historical DA features with forward-looking density features. v0b formula baseline (`0.60 * da_rank + 0.40 * right_tail_rank`) is analogous to 6.2B's formula. ML (v1+) must beat v0b/v0c formula baselines to justify added complexity.

---

## 5. Universe Definition

### 5.1 Why Not V4.4's 1,227?

V4.4 applies aggressive filtering: from ~14,000 bridge table constraints down to 1,227. This filtering is opaque (shadow_rank cutoff? flow direction filter? minimum density threshold?). The result captures only 68-84% of mappable binding SP.

### 5.2 Why Not All 13,000?

The full 13,000 spice6 universe has:
- ~4.2% binding rate (extreme class imbalance)
- 95%+ of constraints are noise (never bind)
- Training on 13,000 rows with 4% positive rate wastes compute and may confuse the model

### 5.3 Elbow Analysis (from 2026-03-11 investigation)

We ranked all ~13,000 constraints by right-tail density weight (`right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` across outage_dates) and computed cumulative binding SP captured across 2 planning years (2023-06 and 2024-06). Results showed an elbow at a low threshold but the exact universe size **varies by quarter and year**.

### 5.4 Actual Universe Sizes (Verified 2026-03-12)

**The row unit is `branch_name`.** Multiple constraint_ids (~2.5 per branch on average) are collapsed to one row per branch via aggregation (§6.2). Each row has a unique target — no duplication, no weighting needed.

**Universe filter**: `right_tail_max >= threshold` applied at cid level BEFORE Level 1 collapse.

**Computation order** (important — universe filter and Level 1 collapse are separate operations):
1. For each raw row (cid, outage_date): compute `right_tail = max(bin_80, bin_90, bin_100, bin_110)`
2. For each cid: compute `right_tail_max = max(right_tail)` across all outage_dates in the quarter
3. Filter: keep cids where `right_tail_max >= threshold`
4. THEN do Level 1 collapse (mean across outage_dates) on the filtered cids only

The universe filter uses **max across dates** (a cid passes if ANY outage_date has right_tail >= threshold). This is more inclusive than mean — it keeps constraints with even intermittent signal. **The implementer must use `max` (not `mean`) for the universe filter.**

**Threshold calibration procedure** (Phase 1, Day 1 — must complete before anything else):

Calibration is performed at **BRANCH level** (the model's row unit):
1. For PY 2024-06/aq1, compute `right_tail_max` per cid, then aggregate to max per branch via the annual bridge.
2. Map realized DA to branches (via annual bridge only, no monthly fallback). Sum SP per branch.
3. Sort branches by `right_tail_max` descending. Compute cumulative binding SP captured vs branch count.
4. Find the elbow at 95% SP capture. The SP denominator is annual-bridge-mapped branch SP (not total quarter DA SP).
5. Cross-check: apply the same threshold to 2023-06/aq1 at branch level. It should produce a similar-sized universe (within ±20%).
6. **Freeze the threshold**: record it in `ml/config.py` as `UNIVERSE_THRESHOLD`.

**Application is CID-LEVEL**: `is_active = (right_tail_max >= UNIVERSE_THRESHOLD)` per cid in `load_collapsed()`. Branches with ≥1 active cid are kept. This is consistent because `branch_rtm = max(cid_rtm) >= threshold` implies at least one active cid.

**CALIBRATION COMPLETE (2026-03-12)**: Threshold = 0.0003467728739657263. 95% of annual-bridge-mapped branch SP captured at 2,339 branches. Cross-check 2023-06/aq1: ratio = 1.07. See `registry/threshold_calibration/threshold.json`.

The tables below were generated using the OLD `density_signal_score >= 0.001` filter and are retained for reference only. Actual universe sizes with the calibrated threshold are ~2,339 branches for 2024-06/aq1.

**PY 2025-06:**

| Quarter | Raw cids | cids (filtered) | **branch_names** | cid:branch ratio |
|:---:|:---:|:---:|:---:|:---:|
| aq1 | 12,876 | 3,840 | **1,527** | 2.51x |
| aq2 | 12,833 | 4,264 | **1,604** | 2.66x |
| aq3 | 12,931 | 2,790 | **1,118** | 2.50x |
| aq4 | 12,936 | 3,716 | **1,320** | 2.82x |

**PY 2024-06:**

| Quarter | cids (filtered) | **branch_names** | ratio |
|:---:|:---:|:---:|:---:|
| aq1 | 4,343 | **1,712** | 2.54x |
| aq2 | 4,481 | **1,749** | 2.56x |
| aq3 | 3,239 | **1,280** | 2.53x |
| aq4 | 3,812 | **1,428** | 2.67x |

**PY 2023-06:**

| Quarter | cids (filtered) | **branch_names** | ratio |
|:---:|:---:|:---:|:---:|
| aq1 | 4,577 | **1,852** | 2.47x |
| aq2 | 4,306 | **1,647** | 2.61x |
| aq3 | 4,482 | **1,658** | 2.70x |
| aq4 | 4,898 | **1,821** | 2.69x |

**Key observations**:
- Per-quarter universe is **1,100-1,850 branch_names** (training rows)
- Each branch_name has ~2.5 constraint_ids on average (collapsed via aggregation — see §6.2)
- aq3 (Dec-Feb) consistently has the smallest universe — winter has different congestion
- ~4,400-7,400 rows per PY. With ~4k rows per year, **overfitting is a real risk** — feature count must be managed aggressively

**Design decision**: Universe is defined **per quarter**, not annually. Each (PY, quarter) group has its own universe. The row unit is `(branch_name, planning_year, aq_quarter)`.

### 5.4b Why Branch-Level Training?

Multiple constraint_ids map to the same physical equipment (branch_name) under different **contingencies**. For example, branch "EQIN-HAM-4 A" has 51 constraint_ids — each a different contingency scenario with genuinely different density distributions (verified: `ori_mean` differs for 100% of multi-cid branches in V6.2B).

**Why collapse to branch_name** (not keep cid-level rows):
1. **Clean objective match**: 1 row = 1 branch = 1 target = 1 prediction unit. No duplicate targets, no sample weighting needed.
2. **Training overweight avoided**: With cid-level rows, a branch with 51 cids contributes 51× more gradient than a branch with 1 cid. Weighting (1/n_cids) fixes this but adds complexity. Branch-level training eliminates the problem entirely.
3. **Cross-contingency signal preserved**: Aggregation stats (max, min across cids) capture "what's the worst case?" and "do ALL contingencies agree?" — the same information as per-cid rows, in cleaner form. Empirical analysis (§7.2) showed mean/top2_mean are near-duplicates of max (ρ > 0.994).
4. **Overfitting risk lower**: ~1,200-1,800 rows per quarter vs ~3,000-4,500. With ~4k rows per year, fewer rows with cleaner signal is better.
5. **Metrics naturally branch-level**: No dedup step needed in evaluation.

**Target rule**: `sum(abs(shadow_price))` across all DA constraint_ids mapping to the branch, NOT mean. If one constraint binding on a branch is large, we want recall to be high — averaging would dilute the signal.

**V6.2B comparison**: V6.2B monthly signal uses cid-level rows because it's a formula (no ML training dynamics to worry about). Our branch-level design is a deliberate improvement for ML training.

**Our v2 pipeline stages**:

| Stage | Unit | Responsibility |
|-------|------|----------------|
| **1. Ranking (our ML model)** | `branch_name` | Rank branches by binding likelihood. Output: scored branches. |
| **2. Trade construction (downstream, NOT our job)** | `constraint_id` | For each top-ranked branch, expand to cids, select 2-3 with distinctive SF profiles. |

**Downstream SF-based cid expansion (observed in notebook code)**: V6.2B monthly signal generation (`5.get_signal.ipynb` in `psignal/notebook/hz/2025-planning-year/jan/miso/submission/spice6/`) applies SF-based deduplication AFTER ranking. The notebook code groups constraints by `bus_key_group`, then greedily selects cids with distinctive shift factor profiles (Chebyshev distance and correlation thresholds). This is **observed notebook code, not verified as final production behavior** — exact thresholds and rules may differ in actual production.

### 5.5 Bridge Coverage for Expanded Universe

For the 2024-06 annual partition:
- Total density constraints: 12,997
- With bridge table entries: 12,898 (99.2%)
- Missing bridge entries: 99 (0.8%)

The expanded universe does NOT have a bridge coverage problem.

### 5.6 SP Coverage Comparison

| Universe | Size (per quarter) | Mappable SP captured (2021-2024 avg) | Binding rate |
|----------|:---:|:---:|:---:|
| V6.1 (v1) | 286-395 branches | ~40% | ~36% |
| V4.4 (abandoned) | 700-1,483 cids | ~77% | ~20-28% |
| **v2 (right-tail filter)** | **1,100-1,850 branches** | **~80-84%** | **~14-16%** |
| Full density | ~5,000 branches | ~95%+ | ~4-8% |

"Mappable SP captured" = fraction of DA binding SP where the DA constraint_id has a bridge table entry AND the resulting branch_name is in our universe. Stage 1 loss (no bridge entry at all) is 1-3% for 2021-2024 and applies equally to all universes.

**2025-06 is a severe outlier**: 26.3% Stage 1 loss (DA constraint_ids with no bridge entry). See `bridge-table-gap-analysis.md` for the full 5-year quantified analysis.

---

## 6. Collapse Strategy

### 6.1 Problem

Raw density has multiple rows per constraint per quarter:
- ~11 outage_dates per market_month
- 3 market_months per quarter
- = ~33 rows per constraint per quarter
- ~2.5 constraint_ids per branch_name (different contingencies)

We need 1 row per **(branch_name, planning_year, aq_quarter)** for training. This requires a **two-level collapse**.

### 6.2 Two-Level Collapse

**Level 1: outage_dates → constraint_id** (temporal aggregation within a quarter)

Load density data for all 3 market_months in the quarter, concatenate, then compute **mean** per bin per constraint_id across all outage_dates and months. This gives one value per bin per constraint_id.

```python
# Load density for all 3 market_months via partition-specific paths
density_path = '/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet'
frames = []
for mm in market_months:  # e.g., ['2024-06', '2024-07', '2024-08'] for aq1
    mm_path = f'{density_path}/spice_version=v6/auction_type=annual/auction_month={PY}/market_month={mm}/market_round=1/'
    df = pl.read_parquet(mm_path)
    frames.append(df.select(['constraint_id'] + selected_bins))
raw = pl.concat(frames, how='diagonal')

# Level 1 collapse: mean across all outage_dates and months per cid
cid_level = raw.group_by('constraint_id').agg([
    pl.col(b).mean().alias(f'bin_{b}') for b in selected_bins
])
```

(We use only mean at Level 1 because Level 2 adds richer statistics across contingencies. Stacking multiple stats at both levels would explode feature count.)

For constraint_limit: single value per constraint per outage_date (take mean across dates if it varies).

**Level 2: constraint_id → branch_name** (contingency aggregation)

**Prerequisite**: Before Level 2 collapse, join cid→branch_name via bridge table with `convention < 10` filter. Without this filter, the bridge fans out ~2.3× (see §8.4). The convention filter must be applied BEFORE aggregation, not after.

Multiple constraint_ids share the same branch_name (different contingencies on the same equipment). Collapse using richer summaries:

**Per density bin** (2 stats — see §7.2 for empirical justification):
- `max_across_cids` — worst-case contingency (if ANY scenario predicts severe binding). Best single stat (highest Spearman ρ with target).
- `min_across_cids` — best-case contingency. Most independent from max (ρ ≈ 0.70-0.74). Captures whether ALL contingencies agree on binding signal.

**Why only 2 stats?** Empirical analysis (§7.2) showed mean, max, and top2_mean are near-duplicates (ρ > 0.994 with each other). With ~4k rows/year, including all 4 stats wastes degrees of freedom.

**Which 2 stats? — DECISION REQUIRED**

Two independent analyses reached different conclusions:
- **Univariate Spearman** (12 dev groups, all quarters): `max` > `top2` > `mean` > `std` > `min`
- **Leave-one-year-out model score** (aq1 only, onpeak labels): `std` > `top2` > `max` > `mean` > `min`

The discrepancy likely comes from different evaluation methods (univariate correlation vs model composite).

> **STOP AND THINK before implementing.** The Phase 2 build-up (§12) is designed to resolve this empirically. The three plausible choices are:
>
> | Choice | Features | Rationale | Risk |
> |--------|:---:|-----------|------|
> | **max + min** | 20 density | min is most independent from max (ρ ≈ 0.70-0.74). Captures "do ALL contingencies agree?" | min may not add signal in model context |
> | **max + std** | 20 density | std was best in leave-one-year-out model probe. Captures contingency disagreement. | std is moderately correlated with max (ρ ≈ 0.92-0.94) |
> | **max only** | 10 density | Simplest. Only add a second stat if Step 2d proves it helps. | Might miss real signal from cross-contingency variation |
>
> **How to decide**: Run Step 2b (max-only) first. Then run Step 2d with BOTH min and std variants. Compare holdout metrics. If neither second stat improves NB12_Recall@50 or VC@50 by more than noise (±2%), keep max-only. Document the comparison numbers and your reasoning.

**Constraint limit** (4 features):
- `limit_min` — tightest contingency (most binding-relevant)
- `limit_mean` — average across contingencies
- `limit_max` — loosest contingency
- `limit_std` — variability (high std = limit depends heavily on contingency)

**Branch metadata** (2 features):
- `count_cids` — total constraint_ids for this branch (captures equipment complexity)
- `count_active_cids` — cids with `right_tail_max >= threshold` (captures how many contingencies have active signal)

**BF / da_rank_value**: already branch-level, no collapse needed.

```python
# Level 2 collapse example:
bin_cols = [c for c in cid_features.columns if c.startswith('bin_')]
branch_features = cid_features.group_by(['branch_name', 'planning_year', 'aq_quarter']).agg([
    # Density bins: 2 stats per bin (max + min)
    *[pl.col(c).max().alias(f'{c}_cid_max') for c in bin_cols],
    *[pl.col(c).min().alias(f'{c}_cid_min') for c in bin_cols],
    # Constraint limit: 4 stats
    pl.col('constraint_limit').min().alias('limit_min'),
    pl.col('constraint_limit').mean().alias('limit_mean'),
    pl.col('constraint_limit').max().alias('limit_max'),
    pl.col('constraint_limit').std().alias('limit_std'),
    # Branch metadata
    pl.col('constraint_id').count().alias('count_cids'),
    pl.col('constraint_id').filter(pl.col('right_tail_max') >= threshold).count().alias('count_active_cids'),
])
```

### 6.3 Expected Sizes (After Both Collapse Levels)

| Scope | Rows (branch_names) |
|-------|:---:|
| Per (PY, quarter) | ~1,100-1,850 (varies by quarter; see §5.4) |
| Per PY (4 quarters) | ~4,400-7,400 |
| All 7 PYs (2019-2025) | ~31,000-52,000 |
| Training set for 2025 holdout (2019-2024) | ~26,000-42,000 |

**Overfitting note**: With ~4k rows per year and ~34 features, the row:feature ratio (~120:1) is adequate but should be monitored. Prune features with < 2% importance after first training run. A strong non-ML formula baseline is essential for comparison — see §12.

### 6.4 Quarter-to-Market-Month Mapping

| Quarter | Market months (PY 2025-06 example) |
|---------|:---:|
| aq1 | Jun 2025, Jul 2025, Aug 2025 |
| aq2 | Sep 2025, Oct 2025, Nov 2025 |
| aq3 | Dec 2025, Jan 2026, Feb 2026 |
| aq4 | Mar 2026, Apr 2026, May 2026 |

The density distribution is partitioned by `market_month`. For aq1 of PY 2025-06, load market_months 2025-06, 2025-07, 2025-08.

### 6.5 Training Unit

Each training/evaluation instance is `(branch_name, planning_year, aq_quarter)`. The model ranks branches within each (PY, quarter) group. LambdaRank query groups = (PY, quarter).

**Target**: `sum(abs(shadow_price))` per branch — across all DA constraint_ids mapping to the branch, both onpeak + offpeak, across all months in the quarter. See §8.3 for the mapping pipeline.

Since the row unit is branch_name, each row has a unique target. No sample weighting needed. No metric dedup needed. This is the cleanest training design.

---

## 7. Feature Design

### 7.1 Feature Categories

| Type | Features | Source | Signal for NB? | Count |
|------|----------|--------|:-:|:-:|
| **Density bins (cid_max)** | Max across cids per bin | MISO_SPICE_DENSITY_DISTRIBUTION | **Yes** | 10 |
| **Density bins (cid_2nd)** | Second stat — see DECISION REQUIRED below | MISO_SPICE_DENSITY_DISTRIBUTION | **Yes** | 10 |
| **Constraint limit** | `limit_min`, `limit_mean`, `limit_max`, `limit_std` | MISO_SPICE_CONSTRAINT_LIMIT | **Yes** | 4 |
| **Branch metadata** | `count_cids`, `count_active_cids` | Derived from bridge + right-tail filter | **Yes** | 2 |
| **Historical DA** | `da_rank_value` (rank of historical DA SP) | Computed from realized DA | Partial | 1 |
| **Onpeak BF** | `bf_6`, `bf_12`, `bf_15` | Computed from realized DA (onpeak) | No | 3 |
| **Offpeak BF** | `bfo_6`, `bfo_12` | Computed from realized DA (offpeak) | No | 2 |
| **Combined BF** | `bf_combined_6`, `bf_combined_12` | Computed from realized DA (either ctype) | No | 2 |

Total: ~34 features initially (10 bins × 2 cid-stats + 4 limit + 2 metadata + 1 DA + 7 BF).

**Design rationale**: The feature count was reduced from 82 (17 bins × 4 stats) to ~34 based on empirical research (§7.2). Key cuts: (a) bins -10/0/20/40 dropped (negligible or inverted signal), (b) redundant adjacent bins dropped (85/95/105), (c) mean and top2_mean dropped (ρ > 0.994 with max — near-duplicates). See §7.2 for full analysis.

**Pruning**: After first training run, drop features with < 2% importance. With ~4k rows/year and ~34 features, the ratio is manageable but still monitor for overfitting. A strong non-ML formula baseline (§12) is essential — ML must beat it to justify complexity.

**Combined BF**: `bf_combined_N = (months with binding in EITHER onpeak OR offpeak) / N`. Captures structural congestion regardless of class type. See §7.5b.

### 7.2 Density Bin Selection Strategy

V4.4 used only 10 thresholds: {0, 60, 65, 70, 75, 80, 85, 90, 95, 100}. This misses important bins at 110, 120, 150 (high shadow price thresholds), and negative thresholds (counter-flow binding).

#### Empirical Research (2026-03-12, dev data only: 2022-2024, 12 groups)

Spearman correlation (ρ) between each density bin feature and branch-level target (`sum(abs(SP))`), averaged across all 12 dev groups. No holdout data was used.

**Signal by bin (best Level 2 stat per bin):**

| Bin | Best stat | Mean ρ | Signal level |
|:---:|---|:---:|---|
| -100 | max | 0.180 | Moderate (counter-flow) |
| -50 | max | 0.174 | Moderate (counter-flow) |
| -10 | max | 0.057 | Negligible |
| 0 | std | 0.039 | Negligible |
| 20 | min | -0.103 | **Inverted** — drop |
| 40 | top2_mean | 0.143 | Weak |
| 60 | top2_mean | 0.225 | Good |
| **70** | **max** | **0.239** | **Strong** |
| **80** | **max** | **0.243** | **Strongest** |
| **85** | **max** | **0.242** | **Strong** |
| **90** | **max** | **0.241** | **Strong** |
| **95** | **max** | **0.238** | **Strong** |
| **100** | **max** | **0.236** | **Strong** |
| **105** | **max** | **0.233** | **Strong** |
| **110** | **max** | **0.229** | **Strong** |
| 120 | max | 0.223 | Good |
| 150 | top2_mean | 0.217 | Good |
| 200 | top2_mean | 0.196 | Moderate |
| 300 | max | 0.092 | Weak |

**Key findings:**
1. **Bins 70-120 are the sweet spot** (ρ ≈ 0.22-0.24). Bin 80 is marginally the best single bin.
2. **Adjacent bins are highly correlated** — 80 vs 85 vs 90 carry nearly identical signal. Including all of them adds redundancy, not information.
3. **Counter-flow bins (-100, -50) carry moderate signal** (ρ ≈ 0.17-0.18) — worth including as they capture a different facet.
4. **Bins -10, 0, 20 are near-zero or inverted** — drop them.
5. **Extreme tail (300) is weak** (ρ = 0.09) — too sparse to be useful.

**Level 2 stat comparison (which aggregation works best?):**

| Stat | Typical ρ gap vs max | Redundancy with max |
|---|---|---|
| **max** | baseline (best) | — |
| **top2_mean** | -0.002 to -0.005 | ρ = 0.997-0.999 with max (near-duplicate) |
| **mean** | -0.01 to -0.02 | ρ = 0.994-0.998 with max (near-duplicate) |
| **std** | -0.02 to -0.04 | ρ = 0.92-0.94 with max (moderate redundancy) |
| **median** | -0.05 to -0.06 | ρ = 0.90-0.93 with max |
| **min** | -0.14 to -0.16 | ρ = 0.70-0.74 with max (most independent) |

**Critical redundancy finding**: mean, max, and top2_mean are near-duplicates (ρ > 0.994). Including all 3 wastes degrees of freedom with ~4k rows/year. **Use max (best signal) + one complementary stat (min or std)**, not all 4.

**Calibration (does high max → actually binds?):**

| Feature | Threshold | N above | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| bin_110_max | > 0.05 | 3,342 | 35.9% | 16.1% |
| bin_100_max | > 0.2 | 3,061 | 37.5% | 15.5% |
| bin_90_max | > 0.5 | 3,196 | 36.4% | 15.6% |
| bin_80_max | > 0.5 | 6,064 | 33.4% | 27.3% |

Max captures the strongest-signal contingency, but ~63% of high-max branches still don't bind. This is expected — density is one signal among several (BF, da_rank_value carry most of the weight). Density's value is for NB detection where BF=0.

#### v2 bin selection (evidence-based)

Based on the empirical analysis, use **10 bins** (not 17):
- Counter-flow: **-100, -50** (moderate signal, independent from positive bins)
- Mid-range: **60** (good signal, pre-plateau)
- **Core plateau: 70, 80, 90, 100, 110** (strongest signal band — but adjacent bins are redundant; if pruning is needed, keep 80 + 100 + 110 and drop 70/90)
- Extended tail: **120, 150** (good signal, captures extreme events)

**Dropped**: -10, 0, 20, 40 (negligible or inverted signal), 85, 95, 105 (redundant with adjacent bins), 200, 300 (too sparse).

#### v2 Level 2 stats (evidence-based)

Use **max** as the primary stat. The second stat (min, std, or none) is resolved empirically in Phase 2 (see §6.2 DECISION REQUIRED):
- **`max`** — best single stat, captures worst-case contingency. Always included.
- **`min`** — most independent from max (ρ ≈ 0.70-0.74), captures whether ALL contingencies agree. Candidate for second stat.
- **`std`** — strongest in leave-one-year-out model probes (ρ ≈ 0.92-0.94 with max — more correlated than min). Candidate for second stat.

If 2 stats: 10 bins × 2 = **20 density features**. Combined with limit (4), metadata (2), DA (1), BF (7) = **34 total**.
If 1 stat: 10 bins × 1 = **10 density features**. Total = **24 features**.

**Two-level aggregation per selected bin**:
- **Level 1** (outage_dates → cid): **mean** across all outage_dates and months in the quarter → 1 value per bin per cid
- **Level 2** (cid → branch): **max, min** across cids → 2 values per bin per branch

### 7.3 Historical DA Features

In v1, `shadow_price_da` and `da_rank_value` came pre-computed in V4.4/V6.1. In v2, compute them from realized DA:

```python
# For each branch_name, sum |shadow_price| across all DA cids mapped to that branch
# across ALL available months from 2017-04 through March of the submission year
# (both onpeak + offpeak, same backfill as BF features)
# Then rank: branch with highest total SP gets rank 1 (= most binding)
# da_rank_value = rank (lower = more binding)
# Branches with zero historical SP get rank = max_rank + 1 (worst rank)
```

**Lookback window**: All available realized DA from `2017-04` through March of submission year (same backfill floor as BF features — see §7.4). For PY 2025-06 (submitted April 2025): months 2017-04 through 2025-03 = up to 96 months. This gives the most stable ranking — cumulative SP over many years is less noisy than a short window.

**IMPORTANT**: The lookback cutoff must match what was available at signal submission time (see Trap 1). For annual R1 submitted ~April 10, use realized DA through March only. April data is partial and must NOT be used.

**Scale note**: V4.4's `shadow_price_da` was cumulative (~8459 mean) vs V6.1's monthly average (~791 mean). Since we compute from scratch, the raw value doesn't matter — use the **rank** (scale-invariant) as the feature.

### 7.4 Binding Frequency (BF) Features

The #1 feature family from v1. Measures the fraction of recent months where a branch bound:

```python
# bf_N = (# months in last N where branch bound) / N
# "bound" = any DA cid mapped to this branch has abs(shadow_price) > 0
# Separate onpeak (bf_N) and offpeak (bfo_N) versions
```

**Windows**: bf_6, bf_12, bf_15 (onpeak), bfo_6, bfo_12 (offpeak)

**Backfill**: Use `floor_month="2017-04"` for maximum history. This means BF features can look back up to 107 months (2017-04 through 2026-02). Backfill was the key v16 champion insight — more history hurts dev but dominates holdout (+25% VC@20).

**BF computation requires bridge table**: Realized DA uses MISO constraint_ids. Our universe uses SPICE branch_names. Must map DA constraint_id → branch_name via bridge table, then compute BF per branch_name.

### 7.5 Offpeak BF — Why It Matters

Offpeak BF (bfo_6, bfo_12) became the #2 feature family in v16 champion (29% importance). Why:
- Constraints that bind offpeak often bind onpeak too
- Offpeak BF captures structural congestion patterns invisible to onpeak-only history
- Provides signal for constraints "new" to onpeak but established in offpeak

### 7.5b Combined BF — Class-Type Agnostic Binding

The annual density signal is class-type agnostic (no `class_type` column in density_distribution). Since we produce ONE ranking used for both onpeak and offpeak trading, binding frequency should also capture class-type-agnostic signal:

```python
# bf_combined_N = (months with binding in EITHER onpeak OR offpeak) / N
# A month "counts" if any DA cid mapped to this branch had |shadow_price| > 0 in onpeak OR offpeak
bf_combined_6 = count_months_any_binding(window=6) / 6
bf_combined_12 = count_months_any_binding(window=12) / 12
```

This is strictly >= the onpeak-only bf_N (more months count as "binding"). A branch that binds offpeak 10/12 months but never onpeak has bf_12=0, bfo_12=0.83, bf_combined_12=0.83. The combined feature correctly identifies it as an active binder.

Keep separate bf_ (onpeak) and bfo_ (offpeak) features alongside bf_combined_ — the model can learn whether the distinction matters.

### 7.6 Monotone Constraints (CRITICAL)

LightGBM supports monotone constraints — forcing the model to respect that "higher feature value → higher/lower ranking." Wrong signs silently degrade the model without any error.

| Feature | Monotone direction | Reasoning |
|---------|:-:|---|
| bf_6, bf_12, bf_15 | +1 | Higher freq = more binding |
| bfo_6, bfo_12 | +1 | Higher freq = more binding |
| bf_combined_6, bf_combined_12 | +1 | Higher freq = more binding |
| da_rank_value | -1 | Lower rank value = more binding |
| density bins (cid_max/cid_min) | **0 (none)** | NOT monotone — bins are density weights, not exceedance probabilities. Must validate empirically before constraining. See §4.2 for semantics. |
| limit_min, limit_mean, limit_max | 0 (none) | Relationship unclear; may not be monotone |
| limit_std | 0 (none) | Higher variability across contingencies — direction unclear |
| count_cids, count_active_cids | 0 (none) | More contingencies ≠ more/less binding |

### 7.7 Feature from v1 Champion (v16)

v16 used 7 features with this importance split:
- da_rank_value: 35%
- bfo_12: 29%
- bf_15: 12%
- bf_6: 8%
- bfo_6: 7%
- shadow_price_da: 6%
- bf_12: 3%

v2 adds density bins (max + min/std) + constraint_limit + bf_combined on top of these. The hope: density features provide signal for NB constraints where all BF and DA features are zero.

---

## 8. Ground Truth Pipeline

### 8.1 Overview

"Ground truth" = which constraints actually bound in DA and how much. This is what we train against and evaluate on.

### 8.2 Mapping Architecture (CRITICAL — Investigated 2026-03-12)

**The DA and density universes use overlapping but NOT identical constraint_id systems.** Direct ID intersection is only ~60% for binding constraints. The bridge table is required for correct mapping.

**Why branch-level targets even though ~60% of cids match directly**: A branch with 5 SPICE cids (A-E) may have only 1 direct DA match (cid A, SP=300). With cid-level targets: A gets 300, B-E get 0 — but B-E are the same physical line and SHOULD be positive. This creates ~300-400 false negatives per quarter, roughly equal to the number of true positives. Branch-level aggregation assigns the correct positive target to ALL cids on a binding branch, eliminating systematic label noise. For 2025-06/aq1, the bridge adds only ~9 extra cids (+0.4% SP) beyond direct matching. For training years (2021-2024), the bridge adds more (+1.8-4.1% SP). But coverage gain is not the primary reason for using the bridge — **label correctness is** (eliminating false negatives from sibling cids).

#### ID Systems

| Source | constraint_id format | Example | Count |
|---|---|---|---|
| **Density distribution** | String, 93% numeric + 7% SPICE-style | `"1000"`, `"1006FG"` | ~12,876 per quarter |
| **Realized DA** | String, all numeric | `"1101"`, `"121696"` | ~546 binding per quarter |
| **Bridge table** | String, matches density IDs | `"1000"`, `"100023"` | ~14,083 per partition |

#### Verified Coverage (2025-06/aq1, onpeak DA only — pre-combined-GT investigation)

| Mapping approach | DA cids mapped | SP captured |
|---|---|---|
| Direct cid match (DA ∩ density) | 329/546 (60%) | 75.6% |
| Annual bridge only | 338/546 (62%) | 75.7% |
| **Annual bridge + monthly fallback** | **461/546 (84%)** | **89.2%** |
| Still unmapped | 85/546 (16%) | 10.8% |

For comparison, training years have much better coverage:
- 2024-06/aq1: annual bridge alone maps **98.7%** of SP
- 2023-06/aq1: annual bridge alone maps **95.8%** of SP

The 2025-06 shortfall is because 208 DA constraint_ids are **completely absent from the annual bridge** — these are newer constraints added to the MISO grid after the annual bridge was built (April 2025). The monthly bridge tables (updated more frequently) contain 123 of these 208.

### 8.3 Four-Step Pipeline with Monthly Fallback

**Ground truth uses COMBINED onpeak + offpeak DA data.** Since the density signal is class-type agnostic (no `class_type` column), training labels should also be class-type agnostic. A constraint that binds offpeak but not onpeak is still a binder.

```
Step 1: Fetch realized DA shadow prices for target quarter (BOTH class types)
  - Load monthly realized DA for the 3 market months in the quarter
  - Load BOTH onpeak AND offpeak: get_da_shadow_by_peaktype(peak_type='onpeak') + peak_type='offpeak'
  - Source: cached parquet in data/realized_da/ (this project's local cache)
  - Build cache: run scripts/fetch_realized_da.py (requires Ray) — fetches all months for all PYs
  - Pattern per month: MisoApTools().tools.get_da_shadow_by_peaktype(st, et_ex, peak_type)
    → group_by(constraint_id).agg(shadow_price.sum().abs()) → write_parquet
  - Fetch BOTH peak_type='onpeak' AND peak_type='offpeak' per month
  - Cache naming: {YYYY-MM}.parquet (onpeak), {YYYY-MM}_offpeak.parquet
  - Do NOT reuse caches from other projects — build your own to ensure reproducibility
  - See stage5-tier/ml/realized_da.py for the proven fetch_and_cache_month() pattern
  - Aggregate: sum(abs(shadow_price)) per constraint_id across the 3 months AND both class types
  - Result: DataFrame with (constraint_id, total_abs_shadow_price)

Step 2: Map DA constraint_id → branch_name via ANNUAL bridge table (PRIMARY)
  - Load bridge: MISO_SPICE_CONSTRAINT_INFO, partition-specific path
  - Load BOTH class_type='onpeak' AND class_type='offpeak' bridge partitions
  - Apply convention < 10 filter to each (following pbase pattern — keeps real constraints only)
  - Filter out null branch_names from each
  - UNION the two partitions: combined = concat(onpeak, offpeak).unique()
  - Log any constraint_ids where onpeak and offpeak map to DIFFERENT branch_names
    (verified 2026-03-12: this occurs for 0-2 cids per slice, ≤0.87% of SP in worst case)
  - LEFT JOIN DA constraint_ids to combined bridge on constraint_id
  - Track which DA cids are UNMAPPED (no bridge entry)

Step 3: MONTHLY BRIDGE FALLBACK for unmapped DA cids
  - For each market_month in the quarter, load the monthly bridge:
    Load BOTH class_type='onpeak' AND class_type='offpeak', UNION them
    auction_type='monthly', auction_month={market_month}, period_type='f0'
  - Filter: convention < 10, non-null branch_name
  - Match remaining unmapped DA cids against all monthly bridges
  - This recovers ~60% of unmapped cids (critical for 2025-06; marginal for 2022-2024)
  - Log: how many cids recovered, SP recovered, still unmapped

Step 4: Aggregate to branch_name and attach to universe
  - Combine annual-mapped + monthly-recovered mappings
  - Multiple DA constraint_ids can map to same branch_name → SUM their shadow prices (NOT mean)
  - Target per branch = sum(abs(shadow_price)) across ALL mapped DA cids, ALL months, BOTH ctypes
  - LEFT JOIN per-branch SP onto universe DataFrame
  - Universe branches without a DA match get realized_shadow_price = 0.0
  - Rationale: if any constraint on a branch binds heavily, the branch matters for recall
```

```python
def load_bridge_partition(bridge_path: str, auction_type: str, auction_month: str,
                          market_round: str, period_type: str) -> pl.DataFrame:
    """Load bridge for BOTH class types and UNION them.

    RAISES FileNotFoundError if NEITHER class type partition exists.
    Logs a warning if only one class type is found (partial coverage).
    """
    frames = []
    missing = []
    for ctype in ['onpeak', 'offpeak']:
        part_path = (
            f'{bridge_path}/spice_version=v6/auction_type={auction_type}'
            f'/auction_month={auction_month}/market_round={market_round}'
            f'/period_type={period_type}/class_type={ctype}/'
        )
        if not Path(part_path).exists():
            missing.append(ctype)
            continue
        df = pl.read_parquet(part_path).filter(
            (pl.col('convention') < 10) & pl.col('branch_name').is_not_null()
        ).select(['constraint_id', 'branch_name']).unique()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No bridge partition found for {auction_type}/{auction_month}/"
            f"{period_type} in either onpeak or offpeak"
        )
    if missing:
        import logging
        logging.warning(
            f"Bridge partition missing for class_type={missing} "
            f"({auction_type}/{auction_month}/{period_type}). "
            f"Using {['onpeak', 'offpeak'][0] if 'offpeak' in missing else 'offpeak'} only."
        )
    return pl.concat(frames).unique()

def build_gt_mapping(auction_month: str, period_type: str, market_months: list[str]) -> pl.DataFrame:
    """Build constraint_id → branch_name mapping with monthly fallback."""
    bridge_path = '/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet'

    # Primary: annual bridge (UNION of onpeak + offpeak)
    ann = load_bridge_partition(bridge_path, 'annual', auction_month, '1', period_type)

    # Monthly fallback: union of monthly bridges for the quarter's market months
    monthly_frames = []
    for mm in market_months:
        mb = load_bridge_partition(bridge_path, 'monthly', mm, '1', 'f0')
        if len(mb) > 0:
            monthly_frames.append(mb)

    if monthly_frames:
        monthly = pl.concat(monthly_frames).unique()
        # Only use monthly for cids NOT in annual bridge
        monthly_only = monthly.filter(~pl.col('constraint_id').is_in(ann['constraint_id']))
        combined = pl.concat([ann, monthly_only]).unique()
    else:
        combined = ann

    return combined
```

**Why combined GT**: The density features predict binding regardless of class type. Training against onpeak-only GT would teach the model to ignore offpeak-binding constraints that the density correctly identifies. Combined GT aligns labels with feature semantics.

**Evaluation split**: Report metrics against combined GT (primary) AND separately against onpeak-only and offpeak-only GT (monitoring). This shows whether the model favors one class type.

### 8.4 Convention Filter (from pbase)

The bridge table has a `convention` column with values: -1, 1, 999. pbase's `get_equipment_mapping()` filters `convention < 10`, which keeps convention -1 and 1 (real binding constraints) and removes 999 (derived/non-binding).

In our data:
- Convention -1: ~6,373 rows (one branch_name per cid)
- Convention 1: ~7,793 rows (one branch_name per cid)
- Convention 999: ~18,479 rows (secondary mappings, multiple per cid)

Convention 999 rows have valid branch_names but are alternative mappings. The convention filter does NOT change the set of mapped constraint_ids (same 14,083 cids with or without filter). It only affects WHICH branch_name is chosen for cids with multiple mappings.

**Rule**: Always filter `convention < 10` to match pbase production behavior.

### 8.5 Bridge Table Details

**Many-to-one mapping**: 2,720 out of 4,473 branch_names have >1 constraint_id (max: 51 cids per branch). This is expected — multiple constraint_ids can correspond to the same physical equipment monitored under different contingencies. When mapping DA → branch_name, multiple DA cids map to the same branch → sum their shadow prices.

**Partition filtering (CRITICAL)**: Filter on **four columns** to avoid cross-partition fan-out. Use partition-specific paths (not hive scan) to avoid the `SchemaError` from mismatched `device_type` columns across partitions:

```python
# Use partition-specific path (not pl.scan_parquet with hive_partitioning)
# Load BOTH class types and UNION — see load_bridge_partition() in §8.3
bridge = load_bridge_partition(bridge_path, 'annual', auction_month, '1', period_type)
```

**Why all four partition levels**: A constraint_id that maps to branch_name="FOO" in aq1 may map to "BAR" in aq3. Using fewer filters produces incorrect many-to-many mappings.

**Bridge class_type UNION rule (verified 2026-03-12)**: Always load BOTH onpeak and offpeak bridge partitions and UNION them. Investigation across 19 (PY, quarter) slices showed:
- 18/19 slices: onpeak and offpeak have **identical** branch mappings for all shared cids
- 1/19 slices (2021-06/aq4): 2 cids with divergent mappings (0.87% of SP)
- 0-21 class-type-exclusive cids per slice (avg 0.02% of bridge)
- The UNION captures all mappings from both class types at negligible cost (one extra parquet read of ~14K rows)

### 8.6 Tiered Labels

Convert continuous realized_shadow_price to tiered relevance labels for LambdaRank:
- **0** = non-binding (realized_shadow_price = 0)
- **1** = binding, bottom tertile of positive SP
- **2** = binding, middle tertile
- **3** = binding, top tertile (highest shadow price)

Tertile boundaries are computed per (PY, quarter) group. The 4-tier scheme outperformed binary (bind/no-bind) by +36% VC@20 in v1 monthly experiments.

### 8.7 Ground Truth Quality (Verified 2026-03-12)

SP coverage with annual bridge + monthly fallback:

| Planning Year | Annual bridge SP | + Monthly fallback | Still unmapped SP |
|:---:|:---:|:---:|:---:|
| 2022-06/aq1 | 95.5% | 99.2% | 0.8% |
| 2023-06/aq1 | 95.8% | 99.5% | 0.5% |
| 2024-06/aq1 | 98.7% | 99.2% | 0.8% |
| **2025-06/aq1** | **75.7%** | **89.2%** | **10.8%** |

2025-06 is a severe outlier — 208 DA binding constraint_ids are completely absent from the annual bridge (newer constraints added after bridge build in April 2025). Monthly fallback recovers 123 of these 208 cids (+13.4% SP).

**For training (2022-2024)**: Annual bridge alone gives 95.5-98.7% SP. Monthly fallback is marginal but still worth implementing for consistency.

**For holdout (2025-06)**: Monthly fallback is **critical** — without it, 24.3% of binding SP is lost as false negatives.

**False negatives**: The remaining unmapped SP creates false negatives in training labels (binding constraints labeled as non-binding). The model is still usable because this noise is consistent across train and eval periods. But for 2025-06 holdout, higher false negative rate means holdout metrics are slightly pessimistic (real model performance is better than measured).

See `bridge-table-gap-analysis.md` for the full 5-year quantified analysis.

### 8.8 Production pbase Comparison

The production codebase uses the **same mapping pattern** but without monthly fallback:

```python
# pbase/analysis/tools/miso.py
MisoTools.get_equipment_mapping(auction_month_str, auction_round, period_type, ...)
# Reads spice3 constraint_info, filters convention < 10, returns constraint_id → branch_name
# Also combines equipment + convention into "equ:con" key, groups by constraint_id

MisoTools.get_da_sp_by_equipment(da, mapping)
# LEFT JOIN DA to mapping on constraint_id, then groupby equipment
# Converts constraint_id to string before join (DA uses numeric, bridge uses string)
```

Note: spice3 path only has `period_type=f0` for annual — NOT aq1/aq2/aq3/aq4. Only the spice6 bridge table (MISO_SPICE_CONSTRAINT_INFO) has annual period types.

---

## 9. Training Approach

### 9.1 Backend: LightGBM LambdaRank

NOT regression, NOT XGBoost. LightGBM LambdaRank directly optimizes the ranking ordering using tiered relevance labels (0/1/2/3).

### 9.2 Parameters

From v1 champion (v16). Simple params beat tuned params in all experiments:

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "num_threads": 4,           # CRITICAL: 64-CPU container causes contention
    "verbose": -1,
}
```

**`num_threads=4`**: This container has 64 CPUs. LightGBM auto-detects all 64 and creates massive thread contention on small datasets (57s → 0.1s training time). ALWAYS set this.

### 9.3 Expanding Window Split

Train on all prior years, eval on target year. Train once per eval year, reuse model for 4 quarters.

**Data availability (verified)**: Annual density distribution exists for all 7 PYs: 2019-06, 2020-06, 2021-06, 2022-06, 2023-06, 2024-06, 2025-06. Bridge table and constraint limits also exist for all 7 PYs. Realized DA cache covers 2017-04 through 2026-02.

| Eval Year | Training PYs | Train groups | Eval groups |
|:-:|---|:-:|:-:|
| 2022-06 | 2019, 2020, 2021 | 12 | 4 (dev) |
| 2023-06 | 2019-2022 | 16 | 4 (dev) |
| 2024-06 | 2019-2023 | 20 | 4 (dev) |
| 2025-06 | 2019-2024 | 24 | 3 (holdout: aq1-aq3 only) |

**Dev**: 12 groups (2022-2024 × 4 quarters)
**Holdout**: 3 groups (2025-06/aq1-aq3). aq4 (Mar-May 2026) is incomplete as of 2026-03 — March has only partial data, April/May have not occurred. aq4 is **monitored** when data becomes available but excluded from gate decisions.

**Gate rule adjustment**: With 3 holdout groups, candidate must beat baseline on at least **2 of 3** holdout groups (not 3 of 4) for each blocking metric, AND mean must be >= baseline mean.

### 9.4 Query Groups

LambdaRank requires specifying which rows belong to the same query (= should be ranked against each other). Each (planning_year, aq_quarter) is one query group.

```python
# For LightGBM Dataset
group_sizes = df.group_by(["planning_year", "aq_quarter"]).len()["len"].to_list()
train_set = lgb.Dataset(X_train, label=y_train, group=group_sizes)
```

### 9.5 Why Not Regression?

Regression predicts absolute shadow price. But we care about **ranking**, not absolute values. LambdaRank:
- Directly optimizes NDCG (ranking metric)
- Is robust to label noise (consistent false negatives across groups)
- Handles tiered relevance naturally (label 3 > label 2 > label 1 > label 0)

---

## 10. Evaluation Metrics & Gates

### 10.1 Metric Definitions

| Metric | Definition |
|--------|-----------|
| **VC@K** | Value Captured at K: fraction of total binding SP **within universe** in the top-K ranked branches |
| **Recall@K** | Fraction of binding branches (count) found in top-K |
| **Abs_SP@K** | Absolute SP captured: total binding SP of top-K branches, measured against **all DA binding SP** (universe-independent) |
| **Abs_Binders@K** | Count of top-K branches that actually bound in DA |
| **NDCG** | Normalized Discounted Cumulative Gain (standard ranking metric) |
| **Spearman** | Spearman rank correlation between predicted and actual rankings |
| **Tier0-AP** | Average Precision for top-tier (label=3) branches |
| **NB12_Recall@K** | Of all New Binding branches (see §11), what fraction did the model rank in top-K? |
| **NB_SP_Capture@K** | Of all NB binding SP, what fraction is in the model's top-K? |
| **NB_Median_Rank** | Median rank the model assigns to NB binder branches (lower = better detection) |

### 10.2 Tier 1: Blocking Gates

Must pass to promote a new model version. **7 metrics, gate on 2/3 holdout groups + mean >= baseline.**

| # | Metric | Why blocking |
|---|--------|-------------|
| 1 | VC@50 | Core portfolio metric — top 50 is realistic bid size |
| 2 | VC@100 | Broader portfolio coverage |
| 3 | Recall@50 | Missing a top binder is costly |
| 4 | Recall@100 | Broader completeness |
| 5 | NDCG | Full ranking quality |
| 6 | **Abs_SP@50** | **Cross-universe gate**: total real binding SP in top-50. Universe-independent — directly measures trading value. Enables comparison across different universe sizes (v1 V6.1 vs v2 density). |
| 7 | **NB12_Recall@50** | **New binder detection gate**: fraction of NB12 branches (no binding on any cid in either ctype for 12 months, see §11) that the model ranks in top-50. Measures whether density features add value for branches invisible to historical features. |

**Gate rule**: Candidate must beat baseline on at least **2 of 3** holdout groups for each blocking metric, AND mean must be >= baseline mean. (3 holdout groups because aq4 is incomplete — see §9.3.)

### 10.3 Tier 2: Monitoring (reported, not gated)

| Metric | Notes |
|--------|-------|
| VC@20, VC@200 | Head/tail value capture |
| Recall@20 | Too volatile for gating with ~3k universe |
| Spearman | Global correlation, doesn't weight top-K |
| Tier0-AP, Tier01-AP | Precision for top-tier binders |
| Abs_Binders@50 | Count of top-50 that actually bound |
| NB6_Recall@50 | NB with 6-month window |
| NB24_Recall@50 | NB with 24-month window |
| NB_Median_Rank | Median rank of NB binders |
| NB_SP_Capture@50 | SP capture for NB branches |
| onpeak-only VC@50 | Metrics against onpeak-only GT (class-type split monitoring) |
| offpeak-only VC@50 | Metrics against offpeak-only GT |

### 10.4 Tier 3: Cohort Contribution (new in v2)

See §11 for cohort definitions. Measure each cohort's contribution to the **global** top-K shortlist:

| Metric | What it measures |
|--------|-----------------|
| Cohort-in-Top50 | Count of cohort members in global top-50 |
| Cohort-SP-in-Top50 | Binding SP from cohort members in global top-50 |
| Cohort-Recall@50 | Fraction of cohort's total binding SP captured in global top-50 |
| Cohort-Miss-Rate | Fraction of cohort's binders ranked below position 100 |

### 10.5 Why VC@20 Is NOT Blocking

- With ~1,100-1,850 branches per quarter and ~14-16% binding rate, top-20 is a very small head
- A single constraint moving in/out of top-20 swings VC@20 by 5-10%
- With only 3 holdout groups, VC@20 gate decisions are essentially coin flips
- VC@50 captures head performance more stably

### 10.6 Cross-Universe Comparison

**Abs_SP@50** is the primary metric for comparing models across different universes (e.g., v1's V6.1 vs v2's density universe). It measures the total binding SP in the model's top-50 picks against ALL realized DA binding SP for the quarter. The denominator is the same regardless of universe size.

```python
abs_sp_50 = total_binding_sp_in_top_50 / total_da_binding_sp_all_constraints
```

This answers the question: "which model+universe combination picks the 50 branches with the most real binding value?"

### 10.7 Reporting Requirements

Every experiment MUST report:
1. Per-group breakdown for all 3 holdout quarters (not just mean)
2. Per-quarter min/max/spread
3. NB breakdown: NB6/NB12/NB24 recall and median rank
4. Cohort breakdown: metrics split by cohort (see §11)
5. Dev AND holdout numbers (dev alone is misleading — see Trap 2)
6. Class-type split: combined GT metrics (primary) + onpeak-only and offpeak-only (monitoring)
7. Walltime for the experiment run

---

## 11. New Binding (NB) Detection & Branch Cohorts

### 11.1 New Binding (NB) Definition — BRANCH LEVEL

A **New Binding branch** is one where NO constraint_id on the branch has bound in **either** onpeak or offpeak DA for the last N months, but the branch DOES bind in the target quarter's realized DA.

The unit of NB detection is **branch_name** (matching the training row unit), NOT constraint_id. A branch is "new" only if NONE of its mapped DA cids have binding history.

Since the annual signal is class-type agnostic (no `class_type` in density data), NB detection must also be class-type agnostic. A branch that regularly binds offpeak is NOT "new" even if it has never bound onpeak.

**NB windows**: NB6, NB12, NB24 (no binding in either ctype for 6, 12, or 24 months).

```python
def is_new_binder_branch(branch_name, target_quarter_months, N, realized_da_cache, bridge):
    """Check if branch is NB_N: no binding on ANY cid mapped to this branch
    in last N months (either ctype), but branch binds in target quarter.

    All operations are branch-level — this is NOT a per-constraint check.
    """
    lookback_months = get_last_n_months(cutoff="March", N=N)

    # Get ALL constraint_ids that map to this branch via bridge
    branch_cids = bridge.filter(pl.col('branch_name') == branch_name)['constraint_id'].to_list()

    # Check binding in BOTH onpeak AND offpeak for the lookback window
    any_binding = False
    for month in lookback_months:
        for peak_type in ['onpeak', 'offpeak']:
            da = load_realized_da(month, peak_type)
            # Check if ANY cid on this branch had binding
            branch_da = da.filter(pl.col('constraint_id').is_in(branch_cids))
            if branch_da['realized_sp'].sum() > 0:
                any_binding = True
                break
        if any_binding:
            break

    # Check if branch actually bound in target quarter (combined GT)
    target_sp = get_branch_gt(branch_name, target_quarter_months)  # sum(abs(SP)) per branch
    target_binding = target_sp > 0

    return (not any_binding) and target_binding
```

**For branches without N months of history**: If a branch first appears in the bridge table less than N months before the target, treat it as NB_N-eligible (no observed binding = no binding). This is conservative and correct — if it had bound, DA records would show it.

**NB12 is the primary gate window** (§10.2, gate #7). NB6 and NB24 are monitoring.

**Per-ctype NB (monitoring only)**: Also compute per-ctype variants for diagnostic reporting:
- `nb_onpeak_12`: no onpeak binding for 12 months (but may bind offpeak) — measures "new to onpeak"
- `nb_offpeak_12`: no offpeak binding for 12 months (but may bind onpeak) — measures "new to offpeak"

These are NOT used for gating. They help answer: "can the model find branches that are new to one class type but established in the other?" This tests cross-ctype signal transfer.

### 11.2 Branch Cohorts

Cohorts describe the model's signal availability for each **branch**. They complement NB detection with a feature-based view:

| Cohort | Definition | Model signal available |
|--------|-----------|----------------------|
| **History-zero** | `has_hist_da = False AND bf_combined_12 = 0` | Only density features — all historical features are zero/sentinel |
| **History-dormant** | `has_hist_da = True AND bf_combined_12 = 0` | DA history exists but no recent binding in either ctype |
| **Established** | `bf_combined_12 > 0` | All features available, recent binding confirmed |

**Assignment priority** (non-overlapping): established > history-dormant > history-zero.

Note: These are defined on branch-level features (da_rank_value and bf_combined_12 are already branch-level after §7.3 and §7.5b). No constraint_id-level logic needed.

Note: The old "cross-peak" cohort (`bf_12=0 AND bfo_12>0`) is subsumed by `bf_combined_12` — a branch with bfo_12 > 0 has bf_combined_12 > 0 and falls into "established."

### 11.3 Why NB Detection Matters

v1 champion gives 80%+ feature importance to historical features (BF + da_rank_value). For NB branches, the model has **zero signal from these features** and must rely entirely on density features. The **NB12_Recall@50 gate** (§10.2) directly measures: "can the model find branches that historical features would miss?"

This is the core value proposition of adding density features. If density can't detect NB branches, a dual-model architecture may be needed:
- If NB binders are rare (<5/quarter) and low-value → single model suffices
- If NB binders are valuable (>$50k SP/quarter) and badly ranked → dual model justified

### 11.4 Measurement Approach

**The right question**: Of the model's global top-K picks, how many came from each cohort? And separately: of all NB binder branches, how many did the model find?

**NOT the right question**: How well does the model rank within each cohort? (This is subgroup ranking, irrelevant for trading)

```python
def nb_detection_metrics(df, scores, actual, N=12, K=50):
    """Measure model's capability to detect New Binding branches.

    All inputs are branch-level arrays (one entry per branch_name).
    """
    nb_mask = df[f"is_nb_{N}"].to_numpy()  # pre-computed NB labels (branch-level)
    top_k_idx = np.argsort(-scores)[:K]

    # NB binders: branches that are NB AND actually bound
    nb_binders = nb_mask & (actual > 0)
    nb_binder_count = nb_binders.sum()
    nb_binder_sp = actual[nb_binders].sum()

    # How many NB binder branches did the model rank in top-K?
    top_k_mask = np.zeros(len(scores), dtype=bool)
    top_k_mask[top_k_idx] = True
    nb_in_topk = (top_k_mask & nb_binders).sum()
    nb_sp_in_topk = actual[top_k_mask & nb_binders].sum()

    nb_recall = nb_in_topk / nb_binder_count if nb_binder_count > 0 else 0
    nb_sp_capture = nb_sp_in_topk / nb_binder_sp if nb_binder_sp > 0 else 0

    # Median rank of NB binders (even if not in top-K)
    ranks = np.argsort(np.argsort(-scores)) + 1  # 1-indexed ranks
    nb_binder_ranks = ranks[nb_binders]
    median_rank = np.median(nb_binder_ranks) if len(nb_binder_ranks) > 0 else float('inf')

    return {
        f"NB{N}_count": nb_binder_count,
        f"NB{N}_Recall@{K}": nb_recall,
        f"NB{N}_SP_Capture@{K}": nb_sp_capture,
        f"NB{N}_Median_Rank": median_rank,
    }
```

### 11.5 Cohort Contribution (monitoring)

```python
def cohort_contribution(df, scores, actual, K=50):
    """How much does each cohort contribute to the global top-K?
    All inputs are branch-level arrays (one entry per branch_name).
    """
    top_k_idx = np.argsort(-scores)[:K]
    top_k_actual = actual[top_k_idx]
    top_k_cohort = df["cohort"].to_numpy()[top_k_idx]

    for cohort in ['history_zero', 'history_dormant', 'established']:
        mask = top_k_cohort == cohort
        n_in_topk = mask.sum()
        sp_in_topk = top_k_actual[mask].sum()
        total_sp = actual[df["cohort"] == cohort].sum()
        recall = sp_in_topk / total_sp if total_sp > 0 else 0
        print(f"{cohort}: {n_in_topk}/{K} slots, ${sp_in_topk:.0f} SP, {recall:.1%} recall")
```

---

## 12. Implementation Plan (Phases)

### Phase 1: Foundation (formula baselines + diagnostics)

**Goal**: Get the pipeline running on raw density data with strong formula baselines and proper metrics. No ML yet.

1. **Data loader**: Load density distribution + constraint_limits → two-level collapse to (branch_name, quarter)
   - Compute `right_tail_max = max(bin_80, bin_90, bin_100, bin_110)` per cid across outage_dates
   - Filter universe by `right_tail_max >= threshold`
   - Level 1: mean across outage_dates/months per cid (see §6.2)
   - Level 2: max/min across cids per branch_name (see §6.2, §7.2 for why 2 stats)
   - Bridge join with `convention < 10` for cid→branch mapping (see §8.4)
   - Cache collapsed data to local parquet (avoid NFS re-scans)

2. **Ground truth pipeline**: Realized DA → bridge table → labels
   - Load BOTH onpeak AND offpeak DA (combined GT — see §8.3)
   - Annual bridge (convention < 10) + monthly bridge fallback for unmapped cids (see §8.2-8.3)
   - Log unmapped cid count and SP for every (PY, quarter)
   - Tiered labels (0/1/2/3)

3. **BF computation**: Port from v1
   - Compute bf_6/12/15 (onpeak), bfo_6/12 (offpeak), AND bf_combined_6/12 (either ctype)
   - Backfill from 2017-04
   - BF cutoff: only months through March of submission year (leakage prevention)

4. **da_rank_value computation**: From realized DA history — **BRANCH LEVEL**
   - For each branch_name, sum |shadow_price| across ALL DA cids mapped to the branch via bridge (combined onpeak + offpeak)
   - Sum across all months from 2017-04 through March of submission year
   - Rank branches (lower rank = more binding)
   - Branches with zero historical SP get rank = max_rank + 1

5. **NB labeling**: For each (PY, quarter), label **branches** as NB6/NB12/NB24
   - A branch is NB_N if NO cid on the branch bound in either ctype for N months (see §11.1)
   - Also compute per-ctype NB: `nb_onpeak_12`, `nb_offpeak_12` (monitoring only)
   - For branches with < N months of history, treat as NB_N-eligible

6. **Formula baselines** (fixed weights, no ML, no overfitting):
   - **v0a**: Pure da_rank_value — rank by historical DA shadow price rank. Floor baseline.
   - **v0b**: da_rank + density blend:
     ```python
     # Normalization convention: higher = more binding for ALL terms
     # da_rank_value: lower = more binding → invert: da_rank_norm = 1 - (rank - 1) / (n - 1)
     # right_tail: higher = more binding → right_tail_rank_norm = (rank - 1) / (n - 1)
     #   where rank 1 = highest right_tail_mean = most binding
     # bf_combined_12: higher = more binding → bf_combined_12_rank_norm = (rank - 1) / (n - 1)
     #   where rank 1 = highest bf_combined_12
     # All terms: 1.0 = most binding, 0.0 = least binding
     da_rank_norm = 1.0 - (da_rank_value - 1) / (n_branches - 1)
     right_tail_mean = mean(bin_80 + bin_90 + bin_100 + bin_110)  # per branch (Level 2 mean of cid_max)
     right_tail_rank_norm = rank_descending_normalized(right_tail_mean)  # 1.0 = highest
     formula_score = 0.60 * da_rank_norm + 0.40 * right_tail_rank_norm
     ```
   - **v0c**: da_rank + density + bf blend:
     ```python
     bf_combined_12_rank_norm = rank_descending_normalized(bf_combined_12)  # 1.0 = highest bf_combined_12
     formula_score = 0.40 * da_rank_norm + 0.30 * right_tail_rank_norm + 0.30 * bf_combined_12_rank_norm
     ```
   - Run on dev (12 groups) and holdout (3 groups: aq1-aq3)
   - Report ALL Tier 1/2/3 metrics + per-group breakdown + NB metrics

7. **Cohort diagnostics**: Assign every **branch** to a cohort (§11.2), measure global top-K contribution
   - NB analysis: count NB6/NB12/NB24 binder branches per quarter, total SP
   - Per-ctype NB: report `nb_onpeak_12` and `nb_offpeak_12` counts and SP (monitoring)
   - Compute `months_since_last_binding` distribution (branch-level, combined ctype)

### Phase 2: ML — Build Up Feature Groups Incrementally

**Goal**: Determine if ML + density features add value beyond formula baselines. **Do NOT dump all features at once.** Add one group at a time, measure the marginal lift of each group, and stop when adding more features stops helping.

**Rationale**: With ~4k rows/year, throwing 34 features at the model risks overfitting and makes it impossible to tell which features actually contribute. Building up lets you:
- Confirm each feature group earns its place
- Detect when additional features start hurting (overfitting signal)
- Produce a final model with only features that demonstrably help

#### Step 2a: ML baseline — historical features only (8 features)

Features: `da_rank_value` + BF family (`bf_6, bf_12, bf_15, bfo_6, bfo_12, bf_combined_6, bf_combined_12`)

This is the v1 champion feature set adapted for branch-level training. Must beat v0c formula to justify ML. If ML doesn't beat the formula on these features alone, the ML framework has a problem independent of density.

**Report**: All Tier 1/2/3 metrics, per-group breakdown, NB metrics, feature importance.

#### Step 2b: Add core density bins — max only (5 features → 13 total)

Add: `bin_80_cid_max, bin_100_cid_max, bin_110_cid_max, bin_120_cid_max, bin_150_cid_max`

These are the 5 strongest bins from the core plateau + tail (§7.2, ρ = 0.22-0.24). Use only `max` (best single stat). This tests: **does density signal add anything on top of historical features?**

**Key metric**: NB12_Recall@50. Density's value proposition is detecting new binders invisible to BF/da_rank. If NB recall doesn't improve, density features are not earning their place.

**Report**: Compare vs Step 2a on all metrics. Focus on NB12_Recall@50 and VC@50 delta.

#### Step 2c: Add counter-flow + mid-range bins (3 features → 16 total)

Add: `bin_-100_cid_max, bin_-50_cid_max, bin_60_cid_max`

These capture counter-flow binding (ρ ≈ 0.17-0.18) and the pre-plateau range. Tests: **do non-core bins add marginal signal?**

**Report**: Compare vs Step 2b. If delta < noise (±1% VC@50 across dev groups), these bins are not contributing — drop them.

#### Step 2d: Add second Level-2 stat (10 features → 26 total)

Test **two variants** (run both, compare):
- **2d-min**: Add `bin_{X}_cid_min` for each of the 10 selected bins. Min is the most independent from max (ρ ≈ 0.70-0.74). Captures "do ALL contingencies agree?"
- **2d-std**: Add `bin_{X}_cid_std` for each of the 10 selected bins. Std was the strongest single stat in leave-one-year-out model probes (§7.2). Captures "how much do contingencies disagree?"

Tests: **does a second Level-2 stat add information beyond max?** And **which second stat is better: min or std?**

**Report**: Compare 2d-min and 2d-std vs Step 2c. Check feature importance — if all second-stat features get < 2% importance, drop the entire group and keep max-only.

#### Step 2e: Add structural features (6 features → up to 32 total)

Add: `limit_min, limit_mean, limit_max, limit_std, count_cids, count_active_cids`

Tests: **do constraint limit and branch metadata help?**

**Report**: Compare vs Step 2d. Prune features with < 2% importance.

#### Step 2f: Final pruning

From the best step above:
1. Drop all features with < 2% importance
2. Run correlation check — if two features have ρ > 0.95 with each other, keep the one with higher importance
3. Re-train with pruned set, confirm metrics don't degrade
4. This is the **candidate champion**

#### Step 2g: Density-only model (diagnostic)

Train on density features only (no BF, no da_rank_value). This is NOT a candidate for promotion — it's a diagnostic to measure: "can density alone rank?" Key for understanding NB detection capability.

**If density-only model has NB12_Recall@50 >> full model's NB12_Recall@50**: Consider a dual-model blend (density-only for NB cohort, full model for established cohort).

### Phase 3: Universe & Further Exploration (evidence-driven)

Only if Phase 2 build-up shows clear signal from density:
- Try remaining bins (70, 90 — adjacent to 80/100, likely redundant)
- Try `std` instead of `min` at Level 2
- Universe expansion → test lower right_tail_max thresholds
- Dual model blend if NB detection warrants it

---

## 13. What v1 Proved (Reusable Knowledge)

### 13.1 Key Findings

1. **Binding frequency is the #1 feature family**. Multi-window (bf_6/12/15) beats single window. BF alone outperforms any signal formula.
2. **Offpeak BF is a strong complementary signal** — became #2 feature at 29% importance in v16 champion.
3. **Backfill (2017-04+) helps generalization** — hurts dev but dominates holdout (+25% VC@20). More history = more robust BF features.
4. **Dev eval can be misleading** — v10e was dev champion (VC@20=0.3389) but v16 won holdout (0.3920 vs 0.3124). ALWAYS run holdout before declaring winner.
5. **LightGBM LambdaRank with tiered labels works** — NOT regression, NOT XGBoost. Tiered labels (4 levels) fix rank-transform noise: +36% VC@20 vs binary labels.
6. **Formula score as feature helps** — da_rank_value (the dominant formula component) consistently adds +10-18% VC@20.
7. **Simple params beat tuned** — lr=0.03, 31 leaves, 200 trees. No hyperparameter tuning needed.
8. **Feature pruning works** — removing features with <2% importance barely hurts, sometimes helps.
9. **Expanding window, train once per eval year** — efficient and proven. Expanding >> fixed window.
10. **8mo train / 0 val >> 6mo train / 2 val** — biggest non-feature improvement in monthly pipeline.
11. **Always evaluate ALL Group A metrics** — composite rank across 7 metrics is more robust than any single metric.

### 13.2 v1 Experiment Summary (17 versions)

| Version | Features | Holdout VC@20 | Key finding |
|---------|----------|:-:|---|
| v0 | V6.1 formula (0.60/0.30/0.10) | 0.2329 | Formula baseline |
| v0b | Pure da_rank_value only | 0.2997 | da_rank_value alone beats full formula |
| v7d | 7f ML (no BF backfill) | 0.3033 | ML > formula |
| v8-v9 | Feature variations | 0.30-0.31 | Incremental exploration |
| v10e | 8f (dev-best) | 0.3124 | Dev champion but lost on holdout |
| **v16** | **7f (BF + offpeak BF + backfill from 2017-04)** | **0.3920** | **Holdout champion (+68% vs v0)** |
| v17 | v16 + partial April BF | ~v16 | Dead end: 0% importance for partial features |

### 13.3 v1 Monthly Pipeline Findings (research-stage5-tier)

The monthly pipeline (V6.2B, f0/f1 period types) validated the same approach on monthly data:

| Slice | v0 (Formula) | v2 (ML) | Improvement |
|-------|:-:|:-:|:-:|
| f0/onpeak | 0.1835 | **0.3529** | +92% |
| f0/offpeak | 0.2075 | **0.3780** | +82% |
| f1/onpeak | 0.2209 | **0.3677** | +66% |
| f1/offpeak | 0.2492 | **0.3561** | +43% |

Same 9 features: bf_1, bf_3, bf_6, bf_12, bf_15, v7_formula_score, prob_exceed_110, constraint_limit, da_rank_value.

---

## 14. What v1 Got Wrong

1. **VC@20 tunnel vision**: All 17 versions optimized for VC@20 only. Other metrics (VC@50/100, Recall, NDCG) were reported but not weighted in decisions.

2. **Universe too small**: V6.1 had 286-395 constraints covering only ~38-42% of binding SP. Major binders like BENTON and MNTCELO were missing or under-covered.

3. **No new-binder measurement**: Never defined or measured performance on branches with zero binding history. Never measured how many binding branches are invisible to the universe.

4. **No per-quarter analysis**: Aggregate mean across holdout groups hides quarter-specific effects. aq1 (summer) has different congestion patterns than aq3 (winter).

5. **Experiment sprawl**: 17 versioned scripts (run_v0.py through run_v17.py), mostly incremental. No systematic framework for comparing across universe changes.

---

## 15. Traps & Pitfalls (MUST READ)

### Trap 1: Temporal Leakage — Auction Timing

**THE SINGLE MOST DANGEROUS BUG CLASS IN THIS CODEBASE.**

Annual R1 is submitted ~April 10 of each year. At submission time:
- Realized DA through **March** is complete
- April data is only ~10 days in (partial, unreliable)
- BF features MUST use only months **<= March** of the submission year

Using April data (even partial) is leakage. This was discovered in the monthly pipeline (stage5-tier) where it inflated binding_freq results by **6-20%**.

The general lag rule for monthly period type fN: `lag = N + 1` (months). For annual: realized DA through March only.

**For any realized-data-derived feature, always ask: "at the moment we submit this signal, is this data actually available?"**

### Trap 2: Dev vs Holdout Divergence

v10e was dev champion (VC@20=0.3389) but v16 won holdout (0.3920 vs 0.3124). Dev has 12 groups across 3 years, holdout has 4 groups in 1 year — high variance.

**Backfill was the key example**: adding 2017-04+ history hurt dev metrics but dominated holdout (+25% VC@20). If we'd stopped at dev, we'd have missed the champion.

**Rule**: Never declare a winner on dev alone. Always run holdout. Report per-group numbers.

### Trap 3: LightGBM Thread Contention (57s → 0.1s)

Container has 64 CPUs. LightGBM auto-detects all 64 → massive thread contention on small datasets (~2,400-3,900 rows per quarter). Training goes from 0.1s to 57s.

**Always set `num_threads=4` in ALL LightGBM usage** (training, prediction, feature importance).

### Trap 4: Bridge Table Partition Sensitivity

Must filter on ALL FOUR partition levels: `auction_type`, `auction_month`, `period_type`, `class_type`. Use partition-specific file paths (not hive scan). See §8.5 for code. Using fewer filters produces incorrect many-to-many mappings.

### Trap 5: Bridge Table Schema Mismatch

Reading the full bridge table parquet fails with `SchemaError` due to `device_type` column mismatch across partitions. **Use partition-specific paths** (e.g., `pl.read_parquet(f'{path}/spice_version=v6/auction_type=annual/...')`) instead of `pl.scan_parquet(..., hive_partitioning=True)` for the bridge table. See §8.5.

### Trap 6: `aggregate_months()` Return Format

`evaluate.py::aggregate_months()` returns `{"mean": {...}, "std": {...}, "bottom_2_mean": {...}}` — NOT a flat dict. Doing `agg["VC@20"]` gives KeyError. Must do `agg["mean"]["VC@20"]`. This caused silent bugs in early experiment scripts.

### Trap 7: Monotone Constraints Directionality

BF features (bf_, bfo_, bf_combined_): +1 (higher = more binding). da_rank_value: -1 (lower = more binding). Density bins: do NOT constrain (semantics unclear — see §4.2 and §7.6). Getting the sign wrong silently degrades the model — LightGBM won't error, it'll just learn a worse model. See §7.6 for the full table.

### Trap 8: v17 Partial BF — Dead End

v17 tested adding first-12-days-of-April binding data (available at submission but discarded):
- `bf_partial` / `bfo_partial` — single-year current April partial
- `bf_april` / `bfo_april` — multi-year April BF

Result: bf_april got 0% feature importance; bf_partial got <1%. The signal is too noisy from 12 days. **Don't re-explore.**

### Trap 9: Glob Picks Up Partial Files

When loading realized DA monthly data for BF computation, glob can pick up both full-month and partial-month files. Partial files have `_partial_` in the filename. If you don't filter these out, BF values get corrupted by double-counting.

### Trap 10: Bidding Window API Quirks

`pbase.data.dataset.ftr.market.base.get_market_info("MISO")`:
- Returns **tz-aware timestamps** — handle timezone when comparing to naive dates
- **PY 2021-06 is missing** — causes KeyError if not handled
- Some dates from previous calendar year for certain auction types

### Trap 11: NFS Stale File Handles

Worktrees get stale NFS handles during long benchmarks (35+ min). Process continues but file reads silently return old data or fail.

**Mitigation**: Commit changes BEFORE running long benchmarks. Cache data locally (not on NFS).

### Trap 12: Memory Budget (128 GiB Pod)

| Component | Budget |
|-----------|--------|
| Cursor + extensions | ~3 GiB |
| Pyright | ≤4 GiB |
| Claude Code | ~1 GiB |
| Research scripts | ≤40 GiB |
| Safety margin | ~80 GiB |

Rules:
- Use **polars** (not pandas) — 2-4x less memory
- Use `pl.scan_parquet().filter().collect()` (lazy scan) for large files
- `del df; gc.collect()` between pipeline stages
- Print `mem_mb()` at each stage
- Cache enriched data to local parquet

```python
import resource
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
```

### Trap 13: DA Ground Truth False Negatives

Unmapped DA constraint_ids create false negatives in training labels (binding constraint labeled as non-binding). The model is still usable because this noise is consistent across train and eval. See §8.6 and `bridge-table-gap-analysis.md`.

### Trap 14: Polars Date Comparison

`pl.col('outage_date') == '2025-07-01'` fails with `InvalidOperationError: cannot compare date to string`. Must use `date(2025, 7, 1)` from the `datetime` module.

### Trap 15: Polars numpy.int64 Indexing

`series[numpy_int64_value]` fails with `TypeError`. Cast to Python int: `series[int(idx)]`.

### Trap 16: (removed — density_signal_score no longer used)

### Trap 17: V4.4 Is NOT Reproducible

Do not attempt to use V4.4 features or reproduce them from raw density. The transformation is opaque. See §3 for the full investigation. If someone suggests using V4.4, point them to this section.

### Trap 18: `shadow_price_da` Scale Varies by Signal

V6.1's `shadow_price_da` is monthly average (~791 mean). V4.4's is cumulative sum (~8459 mean). V6.2B's is historical (not realized). Since v2 computes from scratch, use **rank** (scale-invariant) not raw value.

### Trap 19: Ray Required for pbase Data Loaders

Any pbase data loader call requires Ray to be initialized. Always init at script start, BEFORE any data access:

```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])
```

### Trap 20: Never Use Multiprocessing/Dask/Joblib

Always use Ray for parallelism. Other frameworks cause deadlocks in this environment.

### Trap 21: pbase Spice Loaders Broken for Annual Data

`MisoSpiceDensityDistributionOutput` and `MisoSpiceConstraintLimitOutput` construct paths without `auction_type` in the partition hierarchy. The actual data has `auction_type=annual` between `spice_version` and `auction_month`. Loader calls return empty DataFrames silently. Use `pl.scan_parquet(..., hive_partitioning=True)` with explicit `auction_type` filter instead. See §4.2 for the workaround pattern.

### Trap 22: Density Distribution Bins Are NOT Probabilities

The 77 bin values are density weights that sum to exactly 20.0 per row. They are NOT independent probabilities bounded to [0, 1]. Individual values can exceed 1.0 (e.g., 5.83). They are NOT monotonically decreasing across thresholds. Do not assume monotone_constraints on density bin features. See §4.2 for full investigation.

### Trap 23: Ground Truth Must Be Combined Onpeak + Offpeak

The density signal is class-type agnostic (no `class_type` column). Training against onpeak-only GT would teach the model to ignore offpeak-binding constraints that the density correctly identifies. Always load BOTH onpeak and offpeak DA for ground truth labels. See §8.2.

Similarly, NB detection (§11.1) must check binding in BOTH class types at the **branch** level. A branch with bfo_12 > 0 is NOT a new binder.

### Trap 24: DA and Density constraint_id Systems Only Partially Overlap (~60%)

Direct constraint_id intersection between DA binding constraints and density universe is only ~60% (329/546 for 2025-06/aq1). The remaining ~40% of binding DA cids are NOT in the density universe. But even for the 60% that match, **cid-level targets create false negatives** — sibling cids on the same branch get target=0 when the branch actually bound. The bridge table + branch-level target aggregation fixes both problems: it maps the extra 40% AND eliminates false negatives. See §8.2 for the full mapping architecture.

Additionally, the annual bridge table for 2025-06 is missing 208 DA constraint_ids (newer constraints). The monthly bridge fallback (§8.3) recovers 123 of these. **Always implement the monthly fallback** — it's critical for holdout.

### Trap 25: (removed — density_signal_score no longer used)

### Trap 26: Two-Level Collapse — Don't Skip Level 2

The row unit is `branch_name`, NOT `constraint_id`. Density features must go through TWO collapse levels:
1. **Level 1** (outage_dates → cid): mean across outage_dates per bin per cid
2. **Level 2** (cid → branch): max/min across cids per bin per branch (see §7.2 for why only 2 stats)

If you skip Level 2 and train at cid level:
- A branch with 51 cids contributes 51× more gradient than a branch with 1 cid (training bias)
- Metrics inflate from multiple cids of the same branch in top-K
- Model wastes capacity learning to distinguish contingencies — zero trading value
- Level 2 aggregation (max/min) captures cross-contingency range without these problems

---

## 16. External Dependencies & Environment

### Ray (MANDATORY)

```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])  # add lightgbm if needed
```

Ray cluster: `ray://10.8.0.36:10001` (standard dev cluster)
Shutdown when done: `ray.shutdown()`

### Virtual Environment

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

### Key Python Packages

- `polars 1.31.0` — data frames (NOT pandas for data loading)
- `lightgbm` — LambdaRank model
- `numpy` — array operations
- `ray` — parallel/distributed computing + pbase data loading

### Key pbase Modules

| Module | Purpose |
|--------|---------|
| `pbase.data.dataset.spice.pipeline.MisoSpiceDensityDistributionOutput` | Load density distribution — **broken for annual data** (see §4.2); use `pl.scan_parquet()` with `hive_partitioning=True` instead |
| ~~`MisoSpiceDensitySignalScoreOutput`~~ | ~~Not used — density_signal_score dropped from pipeline~~ |
| `pbase.data.dataset.spice.pipeline.MisoSpiceConstraintLimitOutput` | Load constraint limits — **broken for annual data**; use direct polars scan |
| `pbase.analysis.tools.all_positions.MisoApTools` | Load realized DA shadow prices |
| `pbase.data.dataset.ftr.market.base.get_market_info` | Auction calendar/bidding windows |
| `pbase.utils.ray.ray_map_bounded` | Bounded Ray concurrency |

### v1 Code (reference, not to be ported blindly)

Located at `/home/xyz/workspace/research-qianli-v2/research-annual-signal/`. Key files:
- `ml/config.py` — V6.1 paths, feature sets, splits (rebuild for v2)
- `ml/data_loader.py` — V6.1 loader (rebuild for density)
- `ml/ground_truth.py` — DA → bridge → labels (adapt join key logic)
- `ml/binding_freq.py` — BF computation (reusable if join key alignment handled)
- `ml/evaluate.py` — metric computation (extend for cohorts)
- `ml/train.py` — LightGBM LambdaRank training (reusable as-is)
- `ml/features.py` — feature prep + query groups (update feature lists)
- `registry/v16_champion/config.json` — champion config reference

---

## 17. File Layout

```
research-annual-signal-v2/
  docs/
    implementer-guide.md          ← THIS FILE (the spec)
    bridge-table-gap-analysis.md  ← 5-year GT mapping gap analysis (preserved)
  ml/
    __init__.py
    config.py          ← paths, feature sets, splits, gates, leaky features
    data_loader.py     ← load density + score + limits → collapsed (cid, quarter) rows
    ground_truth.py    ← realized DA (combined ctype) → bridge table → tiered labels
    nb_detection.py    ← NB6/NB12/NB24 labeling with combined onpeak+offpeak
    features.py        ← feature preparation, query groups, monotone constraints
    train.py           ← LightGBM LambdaRank training + prediction
    evaluate.py        ← Tier 1/2/3 metrics, NB gates, Abs_SP, cohort contribution
  scripts/
    fetch_realized_da.py          ← Build realized DA cache (requires Ray) — run FIRST
    run_v0a_da_rank.py            ← formula: pure da_rank_value
    run_v0b_blend.py              ← formula: da_rank + right_tail density blend
    run_v0c_full_blend.py         ← formula: da_rank + right_tail density + bf blend
    run_v1_ml_historical.py       ← ML: da_rank + BF features only
    run_v2_ml_density.py          ← ML: + density bin features
  registry/
    (versioned experiment results: config.json, metrics.json per version)
```

---

## 18. Data Paths Quick Reference

### Spice6 Pipeline Data (raw)

| Data | Path |
|------|------|
| Density distribution | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet` |
| ~~Density signal score~~ | ~~Not used — dropped from pipeline~~ |
| Constraint limits | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet` |
| Bridge table | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` |

### Signal Data (pre-assembled, for reference only)

| Data | Path | Status |
|------|------|--------|
| V4.4 annual | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1/` | **ABANDONED** |
| V6.1 annual (v1) | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/` | v1 only |
| V6.2B monthly | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1` | Monthly reference |
| V4.5 annual | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R1/` | Subset of V4.4 |

### Other

| Data | Path |
|------|------|
| Realized DA cache (BUILD YOUR OWN) | `data/realized_da/` (this project — fetch via `MisoApTools`, see §8.3 Step 1) |
| Realized DA cache (stage5 reference) | `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/` (NOT for direct use — build your own) |
| Spice6 density (legacy path) | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` |
| v1 code | `/home/xyz/workspace/research-qianli-v2/research-annual-signal/` |
| Monthly pipeline (stage5) | `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/` |
