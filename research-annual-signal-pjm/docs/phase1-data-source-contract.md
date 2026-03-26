# PJM Annual Signal — Data Source Contract

**Date**: 2026-03-25
**Status**: Phase 1.1 output — canonical paths, schemas, root-resolution rules

---

## Root Resolution Rule

**Updated 2026-03-25**: all PJM SPICE annual data now lives under a single root. The earlier "newer root" (`spice_version=v6/` without `network_model`) has been removed from NFS. No dual-root resolution needed.

**Canonical root**: `…/network_model=miso/spice_version=v6/auction_type=annual/`

This applies to density, bridge, limit, and SF.

### Coverage per PY

| PY | Density rounds | Bridge rounds | Notes |
|----|---------------|---------------|-------|
| 2018-06 | R1-R4 | R1-R4 | Recently appeared |
| 2019-06 | R1-R4 | R1-R4 | |
| 2020-06 | R1-R4 | R1-R4 | |
| 2021-06 | R1-R4 | R1-R4 | |
| 2022-06 | R1-R4 | R1-R4 | Fully backfilled as of 2026-03-25 |
| 2023-06 | R1-R4 | R1-R4 | |
| 2024-06 | R1-R4 | R1-R4 | Previously split across roots, now consolidated |
| 2025-06 | R1-R4 | R1-R4 | Previously newer root only, now in legacy root |

```python
# Root resolution is now trivial
SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"

def get_annual_root(dataset: str) -> str:
    return f"{SPICE_BASE}/{dataset}/network_model=miso/spice_version=v6/auction_type=annual"
```

**Warning**: NFS data is not static. The dual-root split observed earlier in this session was resolved during the session. Always verify paths before assuming any root structure from docs.

---

## 1. Density

**Dataset**: `PJM_SPICE_DENSITY_DISTRIBUTION.parquet`

**Base path**: `/opt/data/xyz-dataset/spice_data/pjm/PJM_SPICE_DENSITY_DISTRIBUTION.parquet/`

**Partition layout**: `{root}/auction_month={PY}/market_month={MM}/`

**Schema** (80 columns):

| Column | Type | Description |
|--------|------|-------------|
| `constraint_id` | String | CID, format: `FACILITY_NAME:CONTINGENCY` |
| `-300` .. `300` | Float64 | 78 density bin columns (percentile deviation distributions) |
| `market_round` | Int64 | Round number (1-4), embedded as column |
| `outage_date` | Date | Outage scenario date |

**Key metrics (PY 2024-06, one month)**:
- ~11,667 unique CIDs per market_month
- ~341,874 rows per market_month (CIDs × outage_dates × rounds)

**Coverage**: see root resolution table above. Each complete PY has 12 market_months.

---

## 2. Bridge (Constraint Info)

**Dataset**: `PJM_SPICE_CONSTRAINT_INFO.parquet`

**Base path**: `/opt/data/xyz-dataset/spice_data/pjm/PJM_SPICE_CONSTRAINT_INFO.parquet/`

**Partition layout**: `{root}/auction_month={PY}/market_round={R}/period_type=aq{Q}/`

Note: no class_type hive partition — `class_type` is a column inside the parquet.

**Schema** (39 columns):

| Column | Type | Description |
|--------|------|-------------|
| `constraint_id` | String | CID key |
| `branch_name` | String | SPICE branch name (the mapping target) |
| `convention` | Int64 | Convention flag (filter < 10 for valid) |
| `class_type` | String | `onpeak` or `dailyoffpeak` |
| `contingency` | String | Contingency name |
| `device_name` | String | Device name |
| `from_name`, `to_name` | String | Bus names |
| `from_bus_kv`, `to_bus_kv` | Float64 | Voltage levels |
| `rate_a`, `rate_b`, `rate_c` | Float64 | Thermal ratings |
| `limit`, `base_case_limit`, `contingency_case_limit` | Float64 | Constraint limits |
| `factor` | Float64 | Shift factor |
| ... | | (39 columns total, see bridge parquet for full list) |

**Key metrics (PY 2025-06, R1, aq1)**:
- ~18,369 unique CIDs
- ~4,854 unique branch_names
- Class types present: `onpeak`, `dailyoffpeak` (no `wkndonpeak` in bridge)

**Coverage**: follows root resolution table. Each (PY, round) has 4 period_type partitions (aq1-aq4).

---

## 3. Limit

**Dataset**: `PJM_SPICE_CONSTRAINT_LIMIT.parquet`

**Base path**: `/opt/data/xyz-dataset/spice_data/pjm/PJM_SPICE_CONSTRAINT_LIMIT.parquet/`

**Partition layout**: `{root}/auction_month={PY}/market_month={MM}/market_round={R}/`

**Schema** (3 columns):

| Column | Type | Description |
|--------|------|-------------|
| `constraint_id` | String | CID |
| `limit` | Float32 | Constraint limit (MW) |
| `outage_date` | Date | Outage scenario date |

**Coverage**: follows root resolution table. Same PY/round split as density.

---

## 4. Shift Factors (SF)

**Dataset**: `PJM_SPICE_SF.parquet`

**Base path**: `/opt/data/xyz-dataset/spice_data/pjm/PJM_SPICE_SF.parquet/`

**Partition layout**: `{root}/auction_month={PY}/market_month={MM}/`

Note: no round partition — SF data does not vary by round within a given root.

**Schema** (~11,833 columns):

| Column | Type | Description |
|--------|------|-------------|
| `pnode_id` | first column | PJM pricing node identifier |
| `{constraint_name}` | Float64 | One column per constraint, format: `FACILITY:CONTINGENCY` |

**Key metrics**: ~11,833 columns (constraints) × ~11,337 rows (pnodes) per partition.

**Coverage**:
- Legacy: 2019-06 through 2024-06
- Newer: 2024-06, 2025-06

---

## 5. DA Shadow Prices

**Dataset**: `PJM_DA_SHADOW_PRICE.parquet`

**Base path**: `/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet/`

**Partition layout**: `year={YYYY}/` (single root, no migration split)

**Schema** (7 columns):

| Column | Type | Description |
|--------|------|-------------|
| `datetime_beginning_utc` | Datetime[ns, US/Eastern] | Hourly timestamp |
| `monitored_facility` | String | Monitored element |
| `contingency_facility` | String | Contingency element |
| `shadow_price` | Float64 | Hourly DA shadow price ($/MWh) |
| `__index_level_0__` | Int64 | Legacy index |
| `month` | Int64 | Calendar month |
| `day` | Int64 | Calendar day |

**Date coverage**: 2010-01-01 through 2026-03-31

**Per-PY availability**:

| PY | Settlement months | DA complete? |
|----|-------------------|-------------|
| 2019-06 | Jun 2019 – May 2020 | Yes |
| 2020-06 | Jun 2020 – May 2021 | Yes |
| 2021-06 | Jun 2021 – May 2022 | Yes |
| 2022-06 | Jun 2022 – May 2023 | Yes (DA exists, no density) |
| 2023-06 | Jun 2023 – May 2024 | Yes |
| 2024-06 | Jun 2024 – May 2025 | Yes |
| 2025-06 | Jun 2025 – May 2026 | **Partial** (through Mar 2026) |

**Candidate loader**: `PjmApTools.get_da_shadow_by_peaktype()` → `PjmDaShadowPrice.load_data()`. Pending Phase 1.3 confirmation.

**CID construction** (from notebook):
```python
constraint_id = monitored_facility + ":" + contingency_facility.replace("ACTUAL", "BASE")
```

---

## 6. V4.6 Benchmark (Reference Only)

**Base path**: `/opt/data/xyz-dataset/signal_data/pjm/`

**Partition layout**: `{constraints|sf}/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}/{PY}/a/{class_type}/`

**Coverage**: see Phase 0 contract note. R1: PYs 2019-06..2024-06. R2-R4: PYs 2019-06..2025-06.

Not a pipeline input — used for benchmark comparison only.
