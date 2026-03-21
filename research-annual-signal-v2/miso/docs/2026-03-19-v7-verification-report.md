# V7.0 Publication Verification Report

**Date**: 2026-03-19
**Signal**: `TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1`
**Published**: 54 constraint files + 54 SF files (2019-06 through 2025-06, 2025-06/aq4 excluded)
**Verification slices**: 2021-06/aq1/onpeak (early/clean) and 2025-06/aq2/offpeak (late/degraded)

---

## 1. File-Level Sanity (all 54 files)

All 54 published files pass structural checks.

### Constraints

| Check | Result |
|-------|--------|
| All 1000 rows | Yes (54/54) |
| All zero nulls | Yes (54/54) |
| All 21 columns (20 data + index) | Yes (54/54) |
| Index format `{cid}\|{sign}\|spice` | Yes (54/54) |
| Unique branches per file | 733-777 (mean 752) |
| Duplicate rows per file | 223-267 (mean 248) |

### SF

| Check | Result |
|-------|--------|
| All 1001 columns (pnode_id + 1000) | Yes (54/54) |
| CID columns match constraint file | EXACT (54/54) |
| NaN values | None (54/54) |
| SF value range | [-1.0, +1.0] |
| Pnode count | 2032 (2023-06) to 2266 (2025-06) |
| Zero-SF columns | Present — see Section 4 |

### Missing slices

- `2025-06/aq4/onpeak` — no data (aq4 incomplete)
- `2025-06/aq4/offpeak` — no data (aq4 incomplete)

### Pnode drift

Pnode count varies by PY (2032 to 2266). Published SF uses raw SPICE pnode_ids without nodal replacement. `MisoNodalReplacement` in pbase maps `to_node -> from_node` when pnodes are renamed. Downstream consumers joining SF across years must apply nodal replacement themselves. This is a downstream concern, not a publisher issue.

---

## 2. Loss Waterfall

For each selected slice, we traced every realized DA CID through four stages to measure where value gets lost.

### 2025-06/aq2/offpeak (worst slice)

| Stage | CIDs | Branches | SP | % of DA | Lost SP |
|-------|:---:|:---:|---:|:---:|---:|
| 1. Total DA | 728 | - | $1,402,323 | 100% | |
| 2. Mapped to branches | 441 | 337 | $932,923 | 66.5% | $469,400 |
| 3. In model universe | 372 | 281 | $867,977 | 61.9% | $64,946 |
| 4. Published | 218 | 154 | $597,735 | 42.6% | $270,242 |

### 2021-06/aq1/onpeak (clean slice)

| Stage | CIDs | Branches | SP | % of DA | Lost SP |
|-------|:---:|:---:|---:|:---:|---:|
| 1. Total DA | 525 | - | $1,260,419 | 100% | |
| 2. Mapped to branches | 510 | 397 | $1,235,423 | 98.0% | $24,996 |
| 3. In model universe | 413 | 318 | $1,190,597 | 94.5% | $44,826 |
| 4. Published | 186 | 138 | $766,925 | 60.8% | $423,672 |

### Three distinct loss sources

**Loss A — Unmapped CIDs** (stage 1 -> 2): DA constraint_ids with no bridge entry in either annual or monthly fallback. Growing rapidly over time.

| PY | Unmapped CID % of DA SP |
|----|:---:|
| 2021-06 | 2.0% |
| 2022-06 | 3.7% |
| 2023-06 | 3.1% |
| 2024-06 | 8.1% |
| 2025-06 | 21.4% |

Concrete example: CID `511847` in 2025-06/aq2/offpeak, $81,878 SP. Not in annual bridge, not in any monthly fallback. Completely invisible to the model.

**Loss B — Out-of-universe branches** (stage 2 -> 3): DA CIDs map to branches via the bridge, but those branches are not in the density model universe. Stable across PYs.

| PY | Out-of-universe % of DA SP | Avg branches lost |
|----|:---:|:---:|
| 2021-06 | 3.6% | ~85 |
| 2022-06 | 6.9% | 104 |
| 2023-06 | 8.4% | 107 |
| 2024-06 | 4.5% | 89 |
| 2025-06 | 6.0% | 71 |

Root cause: these branches have CIDs in the bridge and density data exists, but all CIDs have `right_tail_max` effectively zero (below `UNIVERSE_THRESHOLD`). The density model says "this won't bind" but DA says it did.

Concrete example: branch `AUST_TAYS_1545 A` in 2025-06/aq2/offpeak, $30,386 SP. Has 9 CIDs in CONSTRAINT_INFO, 7 with density data — but all have right_tail_max = 0.0. Filtered out by `is_active` check in `load_collapsed()`.

**Loss C — Publication capacity** (stage 3 -> 4): Branches in the model universe that rank below the 1000-constraint slot limit (~750 unique branches after branch_cap expansion). This is the largest loss in both slices.

| Slice | Binding branches in universe | Published | Not published | Lost SP |
|-------|:---:|:---:|:---:|---:|
| 2021-06/aq1/onpeak | 318 | 138 | 180 | $423,672 |
| 2025-06/aq2/offpeak | 281 | 154 | 127 | $270,242 |

This is a capacity limit, not a filter bug. With 1000 constraint slots and branch_cap=3, the publisher can cover ~750 unique branches. Binding branches ranked below ~750 don't get published.

---

## 3. Tier Binding Monotonicity

v0c ranking works — higher tiers have higher binding rates in both slices.

| Tier | 2021-06/aq1/onpeak | 2025-06/aq2/offpeak |
|------|:---:|:---:|
| 0 (top 200 constraints) | 57.0%, mean SP=$4,217 | 60.0%, mean SP=$3,552 |
| 1 | 17.5%, mean SP=$389 | 28.0%, mean SP=$672 |
| 2 | 12.0%, mean SP=$560 | 6.0%, mean SP=$62 |
| 3 | 9.0%, mean SP=$74 | 6.5%, mean SP=$106 |
| 4 | 7.0%, mean SP=$62 | 6.5%, mean SP=$23 |

Binding rate drops monotonically tier 0 -> 4. Mean SP concentration in tier 0 is strong.

---

## 4. Zero-SF Constraints

### Finding

Some published constraints have SF = 0.0 on every pnode in the SF matrix. These constraints have no price impact on any node.

### Root cause

The raw SPICE SF source (`MISO_SPICE_SF.parquet`) produces these constraints with all-zero shift factors. Verified for CID `276150` (`WOADHODELL11_1 1`): zero on all 2032 pnodes across all months and all outage dates. The publisher does not cause this — it passes through the raw SPICE data.

### Why they survive publication

Each zero-SF CID is in a separate `bus_key_group`. The SF dedup (`_chebyshev_distance` and `_correlation` in `signal_publisher.py`) operates within groups, so it would reject a second zero vector within the same group. But these are separate groups, so they pass. There is no explicit all-zero-SF rejection rule in the publisher.

### Blast radius

| PY | Zero-SF per file | In tier 0 | Slots wasted |
|----|:---:|:---:|:---:|
| 2019-06 | 1 | 0-1 | 0.1% |
| 2020-06 | 1-2 | 0 | 0.1-0.2% |
| 2021-06 | 1-3 | 0-1 | 0.1-0.3% |
| 2022-06 | 1-3 | 0 | 0.1-0.3% |
| **2023-06** | **13-14** | **3** | **1.3-1.4%** |
| 2024-06 | 1-2 | 0 | 0.1-0.2% |
| 2025-06 | 1-4 | 0 | 0.1-0.4% |

**2023-06 is a clear outlier** with 13-14 zero-SF constraints per file, 3 in tier 0. All other PYs have 1-4.

Total across all 54 files: 183 zero-SF constraints, 31 in tier 0 (17%).

Average: 3.4 per file = 0.34% of slots.

### Recommendation

Add a one-liner filter in `signal_publisher.py:170`: skip candidates where `sf_pd[cid].abs().sum() == 0`. Low blast radius, clean fix, no downstream impact (zero-SF constraints have no price effect anyway).

---

## 5. Verification Artifacts

All saved to `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/data/v7_verification/`:

```
data/v7_verification/
├── step1_sanity.json                    # Full sanity results for all 54 files
├── 2021-06_aq1_onpeak/
│   ├── real_da_merged.parquet           # 525 DA CIDs with mapping status
│   ├── training_universe_map.parquet    # 2813 branches with GT + published flag
│   ├── published_signal_map.parquet     # 1000 constraints with branch-level DA
│   ├── loss_waterfall.parquet           # 525 CIDs with loss_reason labels
│   └── summary.json
├── 2025-06_aq2_offpeak/
│   ├── real_da_merged.parquet           # 728 DA CIDs with mapping status
│   ├── training_universe_map.parquet    # 2210 branches with GT + published flag
│   ├── published_signal_map.parquet     # 1000 constraints with branch-level DA
│   ├── loss_waterfall.parquet           # 728 CIDs with loss_reason labels
│   └── summary.json
```

These files are inspection/debug outputs for teammate review. They are NOT model registry artifacts.

---

## 6. Concrete Examples for Spot-Checking

### Working example

- Branch `FORMAFORMN11_1 1`, DA CID `314718`
- Maps cleanly through annual bridge
- Published in tier 0
- Binding with substantial SP in both slices

### Failure Mode 1: Unmapped CID

- DA CID `511847`, 2025-06/aq2/offpeak
- Offpeak SP = $81,878
- Not in annual bridge, not in any monthly fallback
- Completely invisible to branch modeling

### Failure Mode 2: Out-of-universe branch

- Branch `AUST_TAYS_1545 A`, DA CID `316316`, 2025-06/aq2/offpeak
- Offpeak SP = $30,386
- 9 CIDs in CONSTRAINT_INFO (bridge exists)
- 7 CIDs in density data (density exists)
- All CIDs have right_tail_max = 0.0 (6 exactly zero, 1 at 5.2e-105)
- Filtered out by `is_active` threshold in `load_collapsed()`
- Branch never enters model universe

### Zero-SF example

- CID `276150`, branch `WOADHODELL11_1 1`, 2023-06/aq1/offpeak
- Published as tier 0, rank 0.518
- SF = 0.0 on all 2032 pnodes across all months and outage dates
- Scores well on v0c (da_rank + bf + density) — none depend on SF
- Wastes a tier-0 slot with no price impact

---

## 7. Open Items

| Priority | Item | Type |
|----------|------|------|
| 1 | Zero-SF filter in publisher | One-liner fix |
| 2 | Doc corrections (retract "ALL binding" claim, update Abs_SP, fix PLESNLEEDS) | Documentation |
| 3 | 2023-06 zero-SF outlier investigation | Investigative |
| 4 | Publication capacity (1000 slots -> ~750 branches) | Design decision |
| 5 | Bridge coverage for 2024-aq4 through 2025-06 | Upstream infrastructure |
