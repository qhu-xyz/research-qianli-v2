# Coverage Analysis Runbook

**Purpose**: Measure how much realized DA binding value is visible to the V7.0 signal pipeline, broken down by failure mode.

---

## Prerequisites

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
export RAY_ADDRESS=ray://10.8.0.36:10001
export PYTHONPATH=/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2
```

Ray is needed for `MisoApTools.get_da_shadow_by_peaktype()`.

---

## Step 1: DA CID vs Bridge CID Profile by PY

For each PY, sample one month of raw DA (aq2, offpeak, middle month) and compare:
- How many DA CIDs exist?
- How many are in the annual bridge?
- What are the DA-only CIDs? Do they exist in any other PY's bridge?
- What are their constraint names?

```bash
uv run python -c "
import os
os.environ['RAY_ADDRESS'] = 'ray://10.8.0.36:10001'
from pbase.config.ray import init_ray
init_ray()

from pbase.analysis.tools.all_positions import MisoApTools
tools = MisoApTools().tools

import polars as pl
from ml.bridge import load_bridge_partition

PYS = [
    ('2021-06', '2021-10'),
    ('2022-06', '2022-10'),
    ('2023-06', '2023-10'),
    ('2024-06', '2024-10'),
    ('2025-06', '2025-10'),
]

print(f'{\"PY\":<10} {\"Sample\":<10} {\"DA_CIDs\":>8} {\"DA-only\":>8} {\"DA-only%\":>9} {\"DA-onlySP%\":>11} {\"DA-only_med\":>12} {\"Mapped_max\":>11}')
print('-' * 90)

for py, sample_month in PYS:
    st = f'{sample_month}-01'
    next_m = int(sample_month[-2:]) + 1
    next_y = int(sample_month[:4])
    if next_m > 12:
        next_m = 1
        next_y += 1
    et = f'{next_y}-{next_m:02d}-01'

    da = tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type='offpeak')
    if da is None or len(da) == 0:
        print(f'{py:<10} {sample_month:<10} NO DATA')
        continue

    da_cids = da.groupby('constraint_id').agg({'shadow_price': 'sum'}).reset_index()
    da_cids['abs_sp'] = da_cids['shadow_price'].abs()
    da_cid_set = set(da_cids['constraint_id'].astype(str).tolist())

    bridge = load_bridge_partition('annual', py, 'aq2')
    bridge_cids = set(bridge['constraint_id'].to_list())

    in_both = da_cid_set & bridge_cids
    da_only = da_cid_set - bridge_cids

    da_only_sp = da_cids[da_cids['constraint_id'].astype(str).isin(da_only)]['abs_sp'].sum()
    total_sp = da_cids['abs_sp'].sum()
    da_only_pct = len(da_only) / len(da_cid_set) * 100
    da_only_sp_pct = da_only_sp / total_sp * 100 if total_sp > 0 else 0

    da_only_ids = sorted([int(c) for c in da_only if c.isdigit()])
    in_both_ids = sorted([int(c) for c in in_both if c.isdigit()])
    da_only_med = da_only_ids[len(da_only_ids)//2] if da_only_ids else 0
    mapped_max = max(in_both_ids) if in_both_ids else 0

    print(f'{py:<10} {sample_month:<10} {len(da_cid_set):>8} {len(da_only):>8} {da_only_pct:>8.0f}% {da_only_sp_pct:>10.1f}% {da_only_med:>12} {mapped_max:>11}')

print()

# Cross-PY bridge check for DA-only CIDs in 2025-06
print('=== 2025-06 DA-only CIDs: exist in any other PY bridge? ===')
da_2025 = tools.get_da_shadow_by_peaktype(st='2025-10-01', et_ex='2025-11-01', peak_type='offpeak')
da_cid_set_2025 = set(da_2025['constraint_id'].astype(str).unique())
bridge_2025 = load_bridge_partition('annual', '2025-06', 'aq2')
da_only_2025 = da_cid_set_2025 - set(bridge_2025['constraint_id'].to_list())
print(f'  DA-only in 2025-06: {len(da_only_2025)} CIDs')
for other_py in ['2019-06','2020-06','2021-06','2022-06','2023-06','2024-06']:
    try:
        ob = load_bridge_partition('annual', other_py, 'aq2')
        overlap = da_only_2025 & set(ob['constraint_id'].to_list())
        if overlap:
            print(f'  Found in {other_py}: {len(overlap)}')
    except:
        pass

print()

# Top 10 DA-only CIDs with names for 2025-06
print('=== Top 10 DA-only CIDs (2025-06, 2025-10/offpeak) by SP ===')
da_only_rows = da_2025[da_2025['constraint_id'].astype(str).isin(da_only_2025)]
da_only_agg = da_only_rows.groupby('constraint_id').agg({'shadow_price': lambda x: x.abs().sum()}).reset_index()
da_only_agg.columns = ['constraint_id', 'abs_sp']
da_only_top = da_only_agg.nlargest(10, 'abs_sp')
for _, r in da_only_top.iterrows():
    cid = str(r['constraint_id'])
    sp = r['abs_sp']
    name_row = da_2025[da_2025['constraint_id'].astype(str) == cid].iloc[0]
    name = name_row.get('constraint_name', '?')
    print(f'  CID {cid:>10}: \${sp:>10,.0f}  {name}')
"
```

---

## Step 2: Interpret Results

Key columns:
- **DA_CIDs**: total unique constraint_ids in DA for that month
- **DA-only**: CIDs not in the annual bridge (cannot map to any branch)
- **DA-only%**: fraction of DA CIDs that are unmapped
- **DA-onlySP%**: fraction of DA SP from unmapped CIDs (more important than count)
- **DA-only_med**: median numeric CID of unmapped constraints
- **Mapped_max**: highest numeric CID that exists in BOTH DA and bridge

If DA-only_med >> Mapped_max, the unmapped CIDs are numerically newer than anything in the bridge.
If DA-only CIDs don't exist in any other PY's bridge, they were never part of any SPICE planning model.

---

## Results

### Run 1: 2026-03-20 (pre-refresh)

| PY | DA CIDs | DA-only | DA-only % | DA-only SP% | DA-only median CID | Mapped max CID |
|----|:---:|:---:|:---:|:---:|:---:|:---:|
| 2021-06 | 306 | 8 | 3% | 2.1% | 350,272 | 351,769 |
| 2022-06 | 326 | 17 | 5% | 2.7% | 384,627 | 390,097 |
| 2023-06 | 358 | 28 | 8% | 1.7% | 414,702 | 426,143 |
| 2024-06 | 355 | 19 | 5% | 1.9% | 462,093 | 465,848 |
| 2025-06 | 368 | 136 | 37% | 30.1% | 511,233 | 476,330 |

2025-06 DA-only CIDs: 136 checked against all other PY bridges — only 1 found.

### Run 2: 2026-03-20 (post-refresh)

| PY | DA CIDs | DA-only | DA-only % | DA-only SP% | DA-only median CID | Mapped max CID |
|----|:---:|:---:|:---:|:---:|:---:|:---:|
| 2021-06 | 306 | 8 | 3% | 2.1% | 350,272 | 351,769 |
| 2022-06 | 326 | 17 | 5% | 2.7% | 384,627 | 390,097 |
| 2023-06 | 358 | 28 | 8% | 1.7% | 414,702 | 426,143 |
| 2024-06 | 355 | 19 | 5% | 1.9% | 462,093 | 465,848 |
| 2025-06 | 368 | **129** | **35%** | **29.3%** | **511,558** | **482,684** |

**Changes post-refresh:**
- 2021-2024: identical (no change)
- 2025-06: DA-only dropped from 136 → 129 (7 CIDs recovered), SP% from 30.1% → 29.3%
- Mapped max CID increased from 476,330 → 482,684 (bridge gained ~6K new CIDs)
- DA-only median CID shifted from 511,233 → 511,558 (slight)

**Interpretation**: The bridge refresh added ~7 new CID mappings for 2025-06, recovering about 1% of DA SP.

2025-06 DA-only CIDs: 129 checked against all other PY bridges — only 1 found in each of 2020/2022/2023/2024.

**Top 10 DA-only CIDs (2025-06, 2025-10/offpeak, post-refresh):**

| DA CID | SP | Constraint Name |
|:---:|---:|---|
| 512454 | $36,515 | LAKEGEO-TOWER_RD FLO MICH CITY-BABCOCK |
| 513621 | $14,213 | CLDONIAW-FARGO FLO JAMESTOWN-PICKERT 230 |
| 514142 | $11,180 | OSEOL2-WILSON FLO DELL 500/161 AT2 |
| 513025 | $9,248 | MAPLE R-WINGER FLO JAMSTN-PICK-GRFRK |
| 511847 | $9,073 | MNTCELO TR6 XF FLO MNTCELO-QUARRYN |
| 513520 | $8,330 | HORNLK AT3 FLO GENTRX T1+GENTX-FREPT |
| 514750 | $7,914 | LAKEGEO-TOWER_RD FLO BABCOCK3-DUNEACRE |
| 515317 | $7,228 | WINGRIV-VERNDLE FLO BRAINRD-MUDLAKE |
| 511239 | $6,315 | MAPLE_R-WINGER FLO AUDUBON-SHEYNNE |
| 512028 | $6,159 | WILISTN2-LTLMUDDY FLO LELANDO2 KU2A XF |

---

## CORRECTION: CID-level mapping overstates the problem (2026-03-20)

### The issue

Runs 1 and 2 above match DA to SPICE by **constraint_id**. A DA CID not found in the SPICE bridge was classified as "unmapped." But this overstates the problem because:

**A DA constraint with a new CID may monitor a branch that already exists in the SPICE universe under different CIDs.**

The raw DA data has a `branch_name` column (e.g., `MNTCELO TR6 TR6__2 (XF/NSP/*)`). The SPICE bridge also maps CIDs to branch_names. If we match on **branch_name** instead of constraint_id, we may find that many "unmapped" DA CIDs are actually on branches we already know about — just with new constraint formulations.

Example: DA CID 511847 (`MNTCELO TR6 XF FLO MNTCELO-QUARRYN`) has DA branch `MNTCELO TR6 TR6__2 (XF/NSP/*)`. The SPICE bridge might have other CIDs that also map to a MNTCELO TR6 branch. The CID is new, but the physical branch is not.

### What to do

Re-run the coverage analysis matching by branch_name:
1. Extract monitored_line or branch_name from raw DA
2. Match against SPICE branch_names in the bridge
3. Measure: how many "CID-unmapped" DA constraints are actually on known branches?

This will split the "unmapped" bucket into:
- **Truly new branches**: DA constraint on a branch not in SPICE at all
- **Known branches, new CIDs**: DA constraint on a branch we already model, just a new CID

### Run 3: Branch-level matching (post-refresh, 2026-03-20)

DA branch_name format: `STATION SPICE_BRANCH (TYPE/AREA/AREA)`.
Extraction: strip station prefix + parenthetical → compare against SPICE bridge branch_names.
Verified on mapped CIDs: 62% exact match (transformers extract poorly due to `STATION DEVICE DEVICE__2` format).

**Per-year coverage with branch recovery (aq2/offpeak, sampled month):**

| PY | DA CIDs | CID-unmapped | CID SP% | Branch recovered | Recovered SP% | Truly unmapped | True SP% |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019-06 | 231 | 7 | 0.1% | 2 | 0.0% | 5 | **0.1%** |
| 2020-06 | 338 | 9 | 2.3% | 3 | 2.1% | 6 | **0.1%** |
| 2021-06 | 306 | 8 | 2.1% | 0 | 0.0% | 8 | **2.1%** |
| 2022-06 | 326 | 17 | 2.7% | 2 | 1.6% | 15 | **1.0%** |
| 2023-06 | 358 | 28 | 1.7% | 5 | 0.0% | 23 | **1.6%** |
| 2024-06 | 355 | 19 | 1.8% | 10 | 0.5% | 9 | **1.3%** |
| **2025-06** | **368** | **129** | **29.2%** | **46** | **18.0%** | **83** | **11.2%** |

**Key finding**: CID-level matching overstated the 2025-06 gap by ~2.6×. The true gap (branches not in any SPICE model) is ~11%, not 29%. The other 18% is recoverable — new CIDs on branches we already model.

Even the 11% is overstated: some "truly unmapped" are transformer extraction failures (e.g., CID 511847 `MNTCELO TR6` extracts as `TR6 TR6__2` instead of `MNTCELO TR6`). Fixing transformer parsing would recover more.

**Top 10 truly unrecoverable DA CIDs (2025-06/aq2/offpeak, all 3 months):**

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

### Corrected decomposition

| Layer | 2019-2024 | 2025-06 | Fix |
|-------|:---:|:---:|---|
| CID-level "unmapped" (old metric) | 0.1-2.7% | 29.2% | *(overstates problem)* |
| Branch-recoverable (new CIDs on known branches) | 0-2.1% | 18.0% | Match DA branch_name → SPICE branch_name |
| XF extraction failures (known branch, bad parse) | ~0% | ~5% (est.) | Fix transformer parsing |
| Truly new branches | 0.1-1.6% | ~6% (est.) | Cannot fix — genuinely new transmission elements |
