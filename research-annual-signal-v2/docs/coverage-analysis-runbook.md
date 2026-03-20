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

**Interpretation**: The bridge refresh added ~7 new CID mappings for 2025-06, recovering about 1% of DA SP. The core problem persists: 129 DA CIDs (29.3% of SP) are still not in any SPICE bridge.

2025-06 DA-only CIDs: 129 checked against all other PY bridges — only 1 found in each of 2020/2022/2023/2024. These constraints were never part of any SPICE planning model.

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
