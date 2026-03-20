# Coverage Analysis Runbook

**Purpose**: Measure how much realized DA binding value is visible to the V7.0 signal pipeline, broken down by failure mode.

**Canonical method**: Use `MisoDaShadowPriceSupplement` structured keys to construct branch names from DA CIDs, then match against SPICE branches. This replaces the earlier string-parsing approach.

---

## Prerequisites

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
export RAY_ADDRESS=ray://10.8.0.36:10001
export PYTHONPATH=/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2
```

Ray is needed for `MisoApTools.get_da_shadow_by_peaktype()`.

---

## Data Sources

| Source | Path | What it provides |
|--------|------|-----------------|
| DA Shadow Price | via `MisoApTools.get_da_shadow_by_peaktype()` | Realized DA CIDs + shadow prices |
| DA Supplement | `/opt/data/xyz-dataset/modeling_data/miso/MISO_DA_SHADOW_PRICE_SUPPLEMENT.parquet` | Structured keys: `key1`, `key2`, `key3`, `device_type` per CID |
| SPICE Bridge | `MISO_SPICE_CONSTRAINT_INFO.parquet` (via `load_bridge_partition()`) | CID → branch_name mapping |

---

## Branch Matching Algorithm

### Why CID-level matching is insufficient

A DA constraint may have a brand-new CID not in the SPICE bridge, but the physical branch it monitors may already exist in the SPICE universe under different CIDs.

Example: DA CID 513025 for `MAPLE R-WINGER` has a new CID, but branch `MAPLEWINGE23_1 1` already exists in SPICE with other CIDs.

### Supplement key rules

The DA supplement parquet has structured keys per CID. The `device_type` field determines the rule:

| device_type | Rule | Example CID | key1 | key2 | key3 | Constructed branch |
|:-----------:|------|:-----------:|------|------|------|-------------------|
| **XF** (transformer) | `key1 + key3` | 511847 | MNTCELO | TR6 | TR6__2 | `MNTCELO TR6__2` |
| **LN** (line) | `key2 + key3` | 513025 | MAPLE_R | MAPLEWINGE23_1 | 1 | `MAPLEWINGE23_1 1` |

Then normalize whitespace and match against SPICE bridge branch_names.

### Why supplement is preferred over string parsing

| Approach | Recovered (2025-06) | Method |
|----------|:---:|---|
| String parsing (drop first token) | 46 CIDs, $114K | Regex on free-text DA `branch_name` |
| String parsing (generalized cascade) | 53 CIDs, $138K | Try full → drop first → first two tokens |
| **Supplement keys** | **86 CIDs, $142K** | Structured `key1/key2/key3` from MISO data |

Supplement is more accurate (+33 CIDs), uses structured data (no regex), and covers 128/129 unmapped CIDs. It generalizes to PY 2026+ because the keys come from MISO's own data model.

---

## Results (Canonical — Supplement Method)

**Per-year coverage (aq2/offpeak, sampled month, post-refresh):**

| PY | DA CIDs | CID-unmapped | CID SP% | Recovered | Rec SP% | Truly unmapped | True SP% | No supplement |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2019-06 | 231 | 7 | 0.1% | 2 | 0.0% | 4 | **0.1%** | 1 |
| 2020-06 | 338 | 9 | 2.3% | 1 | 0.2% | 5 | **0.1%** | 3 |
| 2021-06 | 306 | 8 | 2.1% | 1 | 0.0% | 6 | **2.0%** | 1 |
| 2022-06 | 326 | 17 | 2.7% | 4 | 1.6% | 12 | **1.0%** | 1 |
| 2023-06 | 358 | 28 | 1.7% | 7 | 0.2% | 18 | **1.4%** | 3 |
| 2024-06 | 355 | 19 | 1.8% | 10 | 0.5% | 7 | **1.2%** | 2 |
| **2025-06** | **368** | **129** | **29.2%** | **86** | **22.4%** | **42** | **6.7%** | **1** |

### Key findings

1. **CID-level matching overstated the problem by ~4×** for 2025-06. CID SP% = 29.2%, True SP% = 6.7%.
2. **86/129 CID-unmapped constraints in 2025-06 are on branches we already model** — new constraint formulations on known physical lines/transformers.
3. **For 2019-2024, the true gap is 0.1-2.0%** — essentially negligible.
4. **The supplement covers 128/129 CID-unmapped constraints** — only 1 has no supplement entry.

### Decomposition

| Layer | 2019-2024 | 2025-06 | Fix |
|-------|:---:|:---:|---|
| CID-level "unmapped" (old metric) | 0.1-2.7% | 29.2% | *(overstates problem)* |
| **Recovered via supplement keys** | 0-1.6% | **22.4%** | Match `key1+key3` (XF) or `key2+key3` (LN) → SPICE branch |
| **Truly unmapped (branch not in SPICE)** | **0.1-2.0%** | **6.7%** | Cannot fix — genuinely new transmission elements |
| No supplement entry | 0-0.4% | 0.4% | DA data gap |

---

## Concrete Examples

### Recovered: CID 511847 (XF transformer)

```
DA CID:           511847
SP:               $9,073
device_type:      XF
key1:             MNTCELO
key2:             TR6
key3:             TR6__2
Constructed:      key1 + key3 = "MNTCELO TR6__2"
SPICE branch:     "MNTCELO  TR6__2" ✓ (exists in bridge)
```

The old string-parsing approach FAILED on this CID (extracted `TR6 TR6__2`). The supplement gets it right.

### Recovered: CID 513025 (LN line)

```
DA CID:           513025
SP:               $9,248
device_type:      LN
key1:             MAPLE_R
key2:             MAPLEWINGE23_1
key3:             1
Constructed:      key2 + key3 = "MAPLEWINGE23_1 1"
SPICE branch:     "MAPLEWINGE23_1 1" ✓ (exists in bridge)
```

### Truly unmapped: CID 513621 (LN line)

```
DA CID:           513621
SP:               $15,045
device_type:      LN
key1:             CLDONIAW
key2:             CLDONFARGO11_1
key3:             1
Constructed:      key2 + key3 = "CLDONFARGO11_1 1"
SPICE branch:     NOT FOUND — genuinely new branch
```

The Caledonia–Fargo 115kV line was never in any SPICE planning model.

---

## How to Re-Run

See the full script in the Prerequisites section. The core logic:

```python
import polars as pl, re
from ml.bridge import load_bridge_partition

def norm(x):
    return re.sub(r'\s+', ' ', str(x).strip()) if x else ''

# Load supplement keys
cols = ['constraint_id','key1','key2','key3','device_type','year','month']
supp = pl.scan_parquet(
    '/opt/data/xyz-dataset/modeling_data/miso/MISO_DA_SHADOW_PRICE_SUPPLEMENT.parquet'
).select(cols).filter(
    (pl.col('year') == YEAR) & (pl.col('month') == MONTH)
).collect().unique('constraint_id')

# Load SPICE branches
bridge = load_bridge_partition('annual', PY, AQ)
spice_norm = {norm(b): b for b in bridge['branch_name'].unique().to_list()}

# For each CID-unmapped DA constraint:
row = supp.filter(pl.col('constraint_id') == cid)
dt = row['device_type'][0]
k1, k2, k3 = row['key1'][0], row['key2'][0], row['key3'][0]
branch = norm(f'{k1} {k3}') if dt == 'XF' else norm(f'{k2} {k3}')
matched = branch in spice_norm
```
