# V7.0 Coverage Investigation Handoff

**Date**: 2026-03-20
**Author**: Claude Code (research-annual-signal-v2)
**For**: Teammate investigating 2024-2025 coverage drop
**Branch**: `feature/pjm-v70b-ml-pipeline`

---

## 1. Background

### What is the annual signal?

The MISO annual signal ranks transmission constraints by predicted binding value. It is used to prioritize which constraints to trade in the MISO annual FTR auction.

The signal is published as two parquet files per (planning_year, quarter, class_type):
- **Constraints parquet**: 1,000 rows, each a ranked constraint with metadata (branch, tier, rank, flow direction, density features)
- **SF parquet**: Shift factor matrix (pnode × constraint), used downstream for price impact estimation

Published at:
```
/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/{py}/{aq}/{class_type}/signal.parquet
/opt/data/xyz-dataset/signal_data/miso/sf/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/{py}/{aq}/{class_type}/signal.parquet
```

54 files published (7 PYs × 4 quarters × 2 classes, minus 2025-06/aq4).

### What is the coverage problem?

When we evaluate the signal against realized DA (day-ahead) shadow prices, a growing fraction of DA binding value is invisible to the signal. In 2021-2023, ~3-5% of DA SP was invisible. In 2025, it's **24-38%**.

---

## 2. ML Experiment Setup: Branch-to-Branch Modeling

### Data flow

```
SPICE Density Distribution (~13,000 CIDs per PY)
    │
    ├── CID → branch mapping via annual bridge (CONSTRAINT_INFO)
    │   (many CIDs map to same branch; ambiguous CIDs dropped)
    │
    ├── Density threshold filter: is_active = right_tail_max >= 0.000347
    │   (branch kept if ≥1 CID is active)
    │
    └── ~2,200 branches per slice = MODEL UNIVERSE
            │
            ├── Features: density bins, limits, da_rank_value, bf_12/bfo_12, history
            ├── Ground truth: realized DA shadow price per branch (class-specific)
            ├── Model: v0c formula = 0.40*da_rank + 0.30*rt_max + 0.30*bf
            │
            └── Evaluation at branch level (unique branch per row)
                    VC@K, Recall@K, Abs_SP@K, NB metrics, Dang metrics
```

### Key design choices

- **Unit of analysis**: branch (not constraint). Multiple CIDs collapse to one branch.
- **Champion model**: v0c — a 3-feature formula, confirmed through 10 phases of attempted improvement (ML, blend, cross-class, two-population all NEGATIVE).
- **Evaluation**: branch-level VC@K (no double counting). K=50/200/250/400.
- **Dev/holdout split**: PY 2022-06 through 2024-06 = dev (12 groups), PY 2025-06 = holdout (3 groups, aq4 excluded).

### Filtering logic (4 stages)

```
Stage 1: Bridge mapping (data_loader.py:191, bridge.py:18)
  Rule: CID must exist in annual bridge partition with unique branch mapping
  Source: MISO_SPICE_CONSTRAINT_INFO.parquet, partitioned by (spice_version, auction_type, auction_month, market_round, period_type, class_type)
  Filter: convention < 10
  Ambiguity: CIDs mapping to >1 branch are dropped (bridge.py:86-108)

Stage 2: Density threshold (data_loader.py:217)
  Rule: branch must have ≥1 CID with right_tail_max >= UNIVERSE_THRESHOLD (0.000347)
  right_tail_max = max(bin_80, bin_90, bin_100, bin_110) across all outage_dates per CID
  Effect: drops branches where SPICE density model predicts near-zero binding probability

Stage 3: v0c ranking + capacity (signal_publisher.py:142-194)
  Rule: branches sorted by v0c_score descending
  Each branch can contribute up to branch_cap=3 CIDs to the published signal
  SF dedup within bus_key_group rejects similar CIDs (Chebyshev < 0.05 or correlation > -0.21)
  Stop at 1,000 total constraints

Stage 4: SF dedup (signal_publisher.py:170-183)
  Rule: within same bus_key_group, reject sibling CIDs too similar to already-selected ones
  Does NOT reject zero-SF CIDs (known gap — see Section 7)
```

---

## 3. Production Port

### How the published signal is built

`signal_publisher.py:publish_signal()` implements a 7-step pipeline:

1. **Score branches**: build class model table → score with v0c
2. **Expand branch → constraints**: join scored branches to CID mapping (1 branch → N CIDs)
3. **Join metadata**: flow_direction, bus_key, bus_key_group, density features
4. **Compute ranks**: rank_ori, density_mix_rank, da_rank_value (normalized within published set)
5. **Build SF matrix**: load and aggregate shift factors from MISO_SPICE_SF.parquet
6. **Grouped dedup**: walk top-down by v0c score, select up to 1,000 constraints with branch_cap=3 and SF similarity filtering
7. **Validate + format**: schema check, null check, save

### Round-trip: CID → branch → CID

- Forward (modeling): SPICE CIDs → bridge → branches → v0c scoring
- Backward (publication): top branches → expand back to CIDs → dedup → publish 1,000
- The round-trip works within the SPICE universe. No CIDs are lost in this direction.

### What the published artifact contains

Per constraint: constraint_id, branch_name, flow_direction, shadow_sign, tier (0-4), rank (v0c score), da_rank_value, density features, bus_key, equipment.

Per SF: pnode_id × 1,000 constraint columns, values in [-1, +1].

---

## 4. Coverage Statistics

### 4.1 Overall coverage by PY

Coverage = fraction of class-specific total DA SP that reaches the model universe (branch-level).

| PY | Unmapped CIDs (% of DA SP) | Out-of-Universe (% of DA SP) | Total gap |
|----|:---:|:---:|:---:|
| 2019-06 | 0.3-3.1% | — | — |
| 2020-06 | 0.7-1.5% | — | — |
| 2021-06 | 1.3-5.4% | 3.6% avg | 5-9% |
| 2022-06 | 3-4% | 4-10% | 8-15% |
| 2023-06 | 2-5% | 4-14% | 6-18% |
| 2024-06 aq1-2 | 2% | 4-6% | 5-9% |
| **2024-06 aq3-4** | **8-20%** | **2-7%** | **8-26%** |
| **2025-06** | **23-30%** | **2-5%** | **24-38%** |

### 4.2 Loss waterfall (two anchor slices)

**2021-06/aq1/onpeak** (clean baseline):

| Stage | CIDs | Branches | SP | % of DA |
|-------|:---:|:---:|---:|:---:|
| 1. Total DA | 525 | - | $1,260,419 | 100% |
| 2. Mapped to branches | 510 | 397 | $1,235,423 | 98.0% |
| 3. In model universe | 413 | 318 | $1,190,597 | 94.5% |
| 4. Published | 186 | 138 | $766,925 | 60.8% |

**2025-06/aq2/offpeak** (worst slice):

| Stage | CIDs | Branches | SP | % of DA |
|-------|:---:|:---:|---:|:---:|
| 1. Total DA | 728 | - | $1,402,323 | 100% |
| 2. Mapped to branches | 441 | 337 | $932,923 | 66.5% |
| 3. In model universe | 372 | 281 | $867,977 | 61.9% |
| 4. Published | 218 | 154 | $597,735 | 42.6% |

### 4.3 Per-group detail (all 30 eval slices)

| Group | Class | N_branches | N_binding | BranchSP | TotalDA_SP | Abs/VC | Unmapped% | OoU% |
|-------|-------|:---:|:---:|---:|---:|:---:|:---:|:---:|
| 2022-06/aq1 | onpeak | 2427 | 299 | 1,912,106 | 2,089,770 | 0.915 | 3.2% | 5.8% |
| 2022-06/aq1 | offpeak | 2427 | 261 | 1,335,440 | 1,469,828 | 0.909 | 3.2% | 5.2% |
| 2022-06/aq2 | onpeak | 2225 | 366 | 1,418,417 | 1,649,973 | 0.860 | 3.7% | 9.1% |
| 2022-06/aq2 | offpeak | 2225 | 314 | 1,422,034 | 1,586,181 | 0.897 | 3.7% | 7.8% |
| 2022-06/aq3 | onpeak | 1953 | 264 | 908,695 | 1,066,875 | 0.852 | 4.4% | 10.0% |
| 2022-06/aq3 | offpeak | 1953 | 257 | 1,084,630 | 1,228,854 | 0.883 | 4.4% | 7.6% |
| 2022-06/aq4 | onpeak | 2395 | 347 | 1,148,901 | 1,264,009 | 0.909 | 3.5% | 6.1% |
| 2022-06/aq4 | offpeak | 2395 | 313 | 1,327,026 | 1,440,088 | 0.921 | 3.5% | 3.8% |
| 2023-06/aq1 | onpeak | 2512 | 290 | 1,125,850 | 1,284,360 | 0.877 | 4.6% | 7.7% |
| 2023-06/aq1 | offpeak | 2512 | 260 | 598,445 | 682,347 | 0.877 | 4.6% | 7.7% |
| 2023-06/aq2 | onpeak | 2278 | 382 | 1,621,641 | 1,753,223 | 0.925 | 2.0% | 5.1% |
| 2023-06/aq2 | offpeak | 2278 | 330 | 1,797,798 | 1,903,863 | 0.944 | 2.0% | 4.0% |
| 2023-06/aq3 | onpeak | 2274 | 300 | 1,200,271 | 1,466,719 | 0.818 | 3.2% | 14.1% |
| 2023-06/aq3 | offpeak | 2274 | 277 | 1,216,018 | 1,461,073 | 0.832 | 3.2% | 14.4% |
| 2023-06/aq4 | onpeak | 2382 | 381 | 1,270,737 | 1,413,911 | 0.899 | 2.4% | 7.4% |
| 2023-06/aq4 | offpeak | 2382 | 327 | 1,194,718 | 1,313,957 | 0.909 | 2.4% | 7.0% |
| 2024-06/aq1 | onpeak | 2339 | 345 | 795,653 | 859,251 | 0.926 | 2.2% | 5.6% |
| 2024-06/aq1 | offpeak | 2339 | 295 | 516,381 | 555,520 | 0.930 | 2.2% | 4.2% |
| 2024-06/aq2 | onpeak | 2281 | 394 | 1,131,289 | 1,240,442 | 0.912 | 2.2% | 6.2% |
| 2024-06/aq2 | offpeak | 2281 | 366 | 1,037,973 | 1,098,009 | 0.945 | 2.2% | 3.9% |
| 2024-06/aq3 | onpeak | 1918 | 244 | 797,432 | 970,915 | 0.821 | 7.8% | 6.8% |
| 2024-06/aq3 | offpeak | 1918 | 223 | 826,930 | 902,618 | 0.916 | 7.8% | 4.1% |
| 2024-06/aq4 | onpeak | 2004 | 278 | 701,878 | 948,591 | 0.740 | 20.3% | 2.8% |
| 2024-06/aq4 | offpeak | 2004 | 241 | 756,512 | 938,413 | 0.806 | 20.3% | 2.1% |
| 2025-06/aq1 | onpeak | 2218 | 241 | 747,092 | 1,016,356 | 0.735 | 23.3% | 2.2% |
| 2025-06/aq1 | offpeak | 2218 | 224 | 476,780 | 627,197 | 0.760 | 23.3% | 2.4% |
| 2025-06/aq2 | onpeak | 2210 | 281 | 938,095 | 1,398,348 | 0.671 | 30.5% | 5.4% |
| 2025-06/aq2 | offpeak | 2210 | 281 | 867,977 | 1,402,323 | 0.619 | 30.5% | 4.6% |
| 2025-06/aq3 | onpeak | 1706 | 230 | 1,173,560 | 1,534,835 | 0.765 | 10.5% | 12.0% |
| 2025-06/aq3 | offpeak | 1706 | 222 | 1,279,668 | 1,581,468 | 0.809 | 10.5% | 9.6% |

### 4.4 DA CID profile by year (aq2/offpeak, sampled month)

| PY | Sample month | DA CIDs | DA-only | DA-only % | DA-only SP % |
|----|:---:|:---:|:---:|:---:|:---:|
| 2021-06 | 2021-10 | 306 | 8 | 3% | 2.1% |
| 2022-06 | 2022-10 | 326 | 17 | 5% | 2.7% |
| 2023-06 | 2023-10 | 358 | 28 | 8% | 1.7% |
| 2024-06 | 2024-10 | 355 | 19 | 5% | 1.9% |
| **2025-06** | **2025-10** | **368** | **136** | **37%** | **30.1%** |

The total number of DA CIDs per month is similar across years (306-368). What changed is the fraction that doesn't exist in the SPICE bridge.

---

## 5. Two Failure Modes with Concrete Examples

### Failure Mode 1: DA CID never maps to any branch

**What it means**: A constraint_id appears in realized DA data but does not exist in the SPICE annual bridge (CONSTRAINT_INFO), does not exist in any monthly fallback bridge, and does not exist in the SPICE density distribution. It was never part of any SPICE planning model.

**Example**: CID `511847` in 2025-06/aq2/offpeak

```
constraint_name:  MNTCELO TR6 XF FLO MNTCELO-QUARRYN
branch_name:      MNTCELO TR6 TR6__2 (XF/NSP/*)
contingency:      MONTICELLO-QUARRY 345
monitored_line:   MNTCELO TR6 XF
offpeak SP:       $81,878 ($9K in Oct + $73K in Nov 2025)
```

Checked against every data source:
- Annual bridge (2025-06, all quarters): **NOT FOUND**
- Annual bridge (2023-06, 2024-06): **NOT FOUND**
- Raw CONSTRAINT_INFO (all partitions): **NOT FOUND**
- Density distribution: **0 rows**
- Monthly bridges (2025-06 through 2025-11): **NO BRIDGE FILES EXIST**

This constraint exists only in MISO's real-time DA operations. The SPICE planning model has never included it.

**Key finding**: In 2021-2024, DA-only CIDs were mostly constraints that appeared in later PY bridges (timing issue — the bridge evolves year to year). **In 2025, the DA-only CIDs don't exist in ANY PY's bridge, ever.** This is not a bridge refresh problem — it's a SPICE model coverage gap. MISO's DA model uses constraints that the SPICE planning model has never included.

**Top 10 unmapped CIDs (2025-06/aq2/offpeak)**:

| DA CID | Offpeak SP | Constraint Name |
|:---:|---:|---|
| 511847 | $81,878 | MNTCELO TR6 XF FLO MNTCELO-QUARRYN |
| 512454 | $36,515 | LAKEGEO-TOWER_RD FLO MICH CITY-BABCOCK |
| 519135 | $18,355 | (not looked up) |
| 509235 | $16,835 | (not looked up) |
| 513621 | $15,045 | CLDONIAW-FARGO FLO JAMESTOWN-PICKERT 230 |
| 235714 | $11,228 | RIVR 115-BRAINRD FLO MUDLAKE-RIVERTON |
| 514142 | $11,109 | OSEOL2-WILSON FLO DELL 500/161 AT2 |
| 488321 | $10,249 | (not looked up) |
| 513025 | $9,248 | MAPLE R-WINGER FLO JAMSTN-PICK-GRFRK |
| 513520 | $8,330 | (not looked up) |

8 CIDs with SP > $10K account for $201K (43% of all unmapped SP).
77 CIDs with SP > $1K account for $431K (92% of all unmapped SP).

### Failure Mode 2: Branch exists in GT but not in model universe

**What it means**: A DA CID successfully maps to a branch through the bridge. The branch has realized DA SP in the ground truth. But the branch is NOT in the model universe because all of its SPICE CIDs have near-zero right-tail density — the density model predicts zero binding probability.

**Example**: Branch `AUST_TAYS_1545 A` in 2025-06/aq2/offpeak

```
DA CID:           316316
Branch:           AUST_TAYS_1545 A
Offpeak SP:       $30,386
SPICE CIDs:       7 (from bridge)
```

All 7 SPICE CIDs have density data, but:

| SPICE CID | right_tail_max | is_active |
|:---:|:---:|:---:|
| 270954 | 0.0 | False |
| 275490 | 0.0 | False |
| 275558 | 0.0 | False |
| 275590 | 0.0 | False |
| 275678 | 0.0 | False |
| 279030 | 0.0 | False |
| 316316 | 5.24e-105 | False |

The density model says all CIDs have essentially zero probability of binding at $80+/MWh. The `is_active` threshold (0.000347) filters out the branch. But DA says it bound for $30K.

**Near-miss example**: Branch `78L_TNATIO11_1 1` in 2021-06/aq1/onpeak. CID with right_tail_max = 1.5e-4, just below the threshold of 3.47e-4. $15,583 SP lost because of a 2× miss on the threshold.

**Top out-of-universe branches (2025-06/aq2/offpeak)**:

| Branch | DA SP | SPICE CIDs | Active CIDs |
|--------|---:|:---:|:---:|
| AUST_TAYS_1545 A | $30,386 | 7 | 0 |
| BENTOMNTCE23_1 1 | $9,927 | 2 | 0 |
| WILISLTLMU11_1 1 | $4,389 | 6 | 0 |
| EAU_CLA TR9 | $2,394 | 12 | 0 |
| WISHELINTO11_1 1 | $2,351 | 0 (DA-only) | — |
| EAU_CLA TR9_2 | $1,446 | 3 | 0 |
| HORNLK_FREP125 A | $1,381 | 3 | 0 |
| PILOTBLK_D11_1 1 | $1,278 | 2 | 0 |

---

## 6. What Specifically Changed in 2025?

### CID-level view (initial analysis, overstates the problem)

The bridge size is roughly constant (~14,100-14,700 CIDs per PY). DA produces a similar number of binding CIDs per month (306-368). 129 DA CIDs in 2025-06 don't match any CID in the SPICE bridge. This was initially reported as 29% of DA SP being "unmapped."

### Branch-level view (corrected analysis)

**CID-level matching overstates the problem.** Many DA CIDs with new constraint_ids monitor branches that already exist in the SPICE universe under different CIDs.

DA branch_name format: `STATION SPICE_BRANCH (TYPE/AREA/AREA)`. Extracting the SPICE branch name and matching against the bridge recovers a large fraction.

| PY | DA CIDs | CID-unmapped | CID SP% | Branch recovered | Recovered SP% | Truly unmapped | True SP% |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019-06 | 231 | 7 | 0.1% | 2 | 0.0% | 5 | **0.1%** |
| 2020-06 | 338 | 9 | 2.3% | 3 | 2.1% | 6 | **0.1%** |
| 2021-06 | 306 | 8 | 2.1% | 0 | 0.0% | 8 | **2.1%** |
| 2022-06 | 326 | 17 | 2.7% | 2 | 1.6% | 15 | **1.0%** |
| 2023-06 | 358 | 28 | 1.7% | 5 | 0.0% | 23 | **1.6%** |
| 2024-06 | 355 | 19 | 1.8% | 10 | 0.5% | 9 | **1.3%** |
| **2025-06** | **368** | **129** | **29.2%** | **46** | **18.0%** | **83** | **11.2%** |

**Key finding**: The true 2025-06 coverage gap is ~11% (truly new branches), not 29%. The other 18% is new CIDs on branches we already model.

Even the 11% is partly overstated: transformer-style DA branch names (e.g., `MNTCELO TR6 TR6__2 (XF/NSP/*)`) extract poorly — they yield `TR6 TR6__2` instead of `MNTCELO TR6`. Fixing transformer parsing would recover more.

### Recovered example

DA CID 513025 ($9,248 SP): branch_name = `MAPLE_R MAPLEWINGE23_1 1 (LN/OTP/OTP)`.
Extract → `MAPLEWINGE23_1 1`. This branch exists in SPICE (CID 426288 maps to it).
The CID is new, but the physical branch (Maple River–Winger 230kV line) is already modeled.

### Truly unmapped example

DA CID 513621 ($15,045 SP): branch_name = `CLDONIAW CLDONFARGO11_1 1 (LN/WAUE/WAUE)`.
Extract → `CLDONFARGO11_1 1`. This branch does NOT exist in any SPICE bridge.
The Caledonia–Fargo 115kV line was never in any SPICE planning model.

### Top 10 truly unrecoverable DA CIDs (2025-06/aq2/offpeak)

| Rank | CID | SP | Extracted Branch | Constraint Name | Category |
|:---:|:---:|---:|---|---|---|
| 1 | 511847 | $81,878 | `TR6 TR6__2` | MNTCELO TR6 XF FLO MNTCELO-QUARRYN | XF extraction failure |
| 2 | 519135 | $18,355 | `WESTWMEI_I11_1 1` | WESTWD2-MEI INT FLO MONTECELLO TR6 | Genuinely new branch |
| 3 | 513621 | $15,045 | `CLDONFARGO11_1 1` | CLDONIAW-FARGO FLO JAMESTOWN-PICKERT 230 | Genuinely new branch |
| 4 | 513520 | $8,330 | `AT3 AT3` | HORNLK AT3 FLO GENTRX T1+GENTX-FREPT | XF extraction failure |
| 5 | 516279 | $7,387 | `RACELA_NVILLE3 A` | NVILLE-RACELA FLO RICHARDSON-ADDIS | Genuinely new branch |
| 6 | 518353 | $7,289 | `BOXC_EMST_1339 A` | BOXC-EMAINST FLO PANA-AUSTIN | Genuinely new branch |
| 7 | 475465 | $5,315 | `BUGL_MASN_6564 B` | HNTT-MASN 6564 FLO WARSON-MASON-4 138 | Genuinely new branch |
| 8 | 508074 | $5,231 | `ROOT_MTGY_5218 A` | ROOT-MTGY FLO CALLAWAY-BLAN | Genuinely new branch |
| 9 | 516133 | $4,606 | *(empty)* | WISHEKNW WISHNWLINT11_1 1 MDU23010 | DA branch_name is None |
| 10 | 511647 | $3,916 | `BISMAJAMES23_1 1` | BISMARK2-JAMESTN FLO CENTER2-JMSTNOTP | Genuinely new branch |

Top 10 = $157K = 71% of all truly unmapped SP.

### Corrected decomposition (V3: generalized algorithm)

The matching algorithm uses a cascading match — no LN/XF-specific logic — so it generalizes to PY 2026+ without modification:

```
1. Strip parenthetical (.*) from DA branch_name
2. Handle semicolons (take first segment)
3. Try full cleaned string against SPICE branches
4. Try drop first token (remainder) — catches LN-type lines
5. Try first two tokens (station + device) — catches XF-type transformers
6. First match wins
```

**Per-year results (generalized algorithm, aq2/offpeak):**

| PY | CID-unmapped | CID SP% | Recovered | Rec SP% | Truly unmapped | True SP% | No branch |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019-06 | 7 | 0.1% | 2 | 0.0% | 4 | **0.1%** | 1 |
| 2020-06 | 9 | 2.3% | 3 | 2.1% | 5 | **0.1%** | 1 |
| 2021-06 | 8 | 2.1% | 0 | 0.0% | 6 | **2.0%** | 2 |
| 2022-06 | 17 | 2.7% | 2 | 1.6% | 9 | **1.0%** | 6 |
| 2023-06 | 28 | 1.7% | 5 | 0.0% | 13 | **1.4%** | 10 |
| 2024-06 | 19 | 1.8% | 11 | 0.5% | 4 | **1.0%** | 4 |
| **2025-06** | **129** | **29.2%** | **53** | **21.8%** | **30** | **6.2%** | **46** |

| Layer | 2019-2024 | 2025-06 | Fix |
|-------|:---:|:---:|---|
| CID-level "unmapped" (old metric) | 0.1-2.7% | 29.2% | *(overstates problem)* |
| Recovered via `drop_first` (LN-like) | 0-2.1% | 17.9% | Generalized cascade step 4 |
| Recovered via `first_two` (XF-like) | 0% | 3.8% | Generalized cascade step 5 |
| No `branch_name` in DA | 0-1.4% | 1.2% | DA data quality — cannot fix by matching |
| **Truly new branches** | **0.1-2.0%** | **6.2%** | Cannot fix — genuinely new transmission elements |

---

## 7. Other Findings

### Zero-SF constraints

13-14 published constraints in 2023-06 (1.3% of slots) have SF = 0.0 on every pnode. Other PYs have 1-4 (0.1-0.3%). These survive publication because each is in a separate bus_key_group (no within-group dedup trigger). The publisher has no all-zero-SF rejection rule. Low blast radius overall but wastes tier-0 slots.

Example: CID `276150` (`WOADHODELL11_1 1`), published as tier 0, rank 0.518. SF = 0.0 on all 2032 pnodes across all months. Raw SPICE SF source confirms — the SPICE model produces zero shift factors for this constraint.

### Abs_SP@K bug (fixed)

Abs_SP@K was identical to VC@K because `total_da_sp_quarter` was set to the branch-level SP sum (same denominator). Now fixed: uses per-class total DA SP from GT diagnostics, including unmapped CIDs. Registry results regenerated. VC@K unchanged; Abs_SP@K is now 81-91% of VC on dev, 62-84% on holdout.

### Publication capacity loss

The largest single loss in both anchor slices: 127-180 binding branches are in the model universe but rank below ~750 in v0c score, so they don't fit in 1,000 constraint slots.

Example: Branch `13831 A` in 2025-06/aq2/offpeak, $64,588 offpeak SP, 8 SPICE CIDs, active — but v0c score too low to make the published set.

---

## 8. Verification Artifacts

### Mapping parquets (for spot-checking)

Saved at `/opt/temp/tmp/qianli/miso_trash/v7_verification/`:

```
2021-06_aq1_onpeak/
├── spice_cid_to_branch.parquet   # 13,257 SPICE CIDs → branch + is_active + is_published + tier
├── da_cid_to_branch.parquet      # 525 DA CIDs → branch + realized_sp + mapping_status/source
└── branch_summary.parquet        # 4,381 branches: n_spice_cids, n_da_cids, da_sp, flags

2021-06_aq1_offpeak/              # same 3 files
2025-06_aq2_onpeak/               # same 3 files
2025-06_aq2_offpeak/              # same 3 files + loss_waterfall + published_signal_map + summary.json
```

### How to use these for review

```python
import polars as pl

# Load mapping tables
base = '/opt/temp/tmp/qianli/miso_trash/v7_verification/2025-06_aq2_offpeak'
spice = pl.read_parquet(f'{base}/spice_cid_to_branch.parquet')
da = pl.read_parquet(f'{base}/da_cid_to_branch.parquet')
bs = pl.read_parquet(f'{base}/branch_summary.parquet')

# Q: What SPICE CIDs map to branch X?
spice.filter(pl.col('branch_name') == 'AUST_TAYS_1545 A')

# Q: What DA CIDs are unmapped?
da.filter(pl.col('da_mapping_status') == 'unmapped').sort('realized_sp', descending=True)

# Q: Which branches have DA SP but are not in the model universe?
bs.filter((pl.col('da_sp_total') > 1000) & (~pl.col('in_model_universe')))

# Q: Which published branches have the most DA SP?
bs.filter(pl.col('is_published')).sort('da_sp_total', descending=True)
```

### Scripts

All in `research-annual-signal-v2/scripts/`:
- `v7_step1_sanity.py` — file-level sanity on all 54+54 published files
- `v7_step2_da_merge.py` — builds loss waterfall + verification tables for one slice
- `v7_mapping_tables.py` — builds SPICE↔branch↔DA mapping parquets

---

## 9. Questions for Investigation

1. **Implement generalized branch matching in the GT pipeline**: The cascading algorithm (full → drop_first → first_two) recovers 53/129 CID-unmapped constraints (74.7% of SP) for 2025-06. It works identically across all PYs without type-specific logic, and should generalize to PY 2026+. This is the highest-leverage fix.

2. **Handle DA CIDs with `branch_name = None`**: 46 CIDs in 2025-06 ($7,652 SP) have no branch_name in DA data at all. These need `monitored_line` or `constraint_name` parsing as a secondary fallback.

3. **Are the ~30 truly unmapped branches real transmission constraints, or modeling artifacts** (e.g., RDT/interface constraints, temporary emergency constraints)?

4. **Is the density threshold (0.000347) too aggressive?** Branch `78L_TNATIO11_1 1` missed by 2× ($15K SP lost). Lowering the threshold would include more branches but increase the universe size and dilute the model.

5. **Should the publisher filter out zero-SF constraints?** Currently 3.4 per file (0.34% of slots), concentrated in 2023-06 (13-14 per file). Simple one-liner fix.
