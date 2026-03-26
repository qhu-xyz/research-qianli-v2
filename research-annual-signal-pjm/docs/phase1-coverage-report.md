# PJM Annual Signal — GT and Universe Coverage Report

**Date**: 2026-03-25
**Slice**: R2, all available PYs
**Mapping method**: monitored-line → branch (primary), + f0 monthly bridge fallback where available

**CAVEAT**: All coverage numbers in this report use **all-hours DA** (not filtered by peak type). The tables are labeled by PY and round but are NOT class-type-specific. Class-type-specific coverage will differ because the bridge maps different constraints per class type. These numbers represent the total branch-level mapping coverage across all hours.

---

## 1. GT Mapping Coverage

### Method

The model is branch-level. Many constraints (different contingencies on the same monitored line) collapse onto one branch. The mapping is:

```
DA.monitored_facility → (normalize whitespace) → bridge monitored part → branch_name
```

99.9% of PJM monitored lines map to exactly 1 branch. The 0.1% exceptions are generic interface names (APSouth, WEST, AEP-DOM) — dropped as ambiguous.

For each quarter (aq1-aq4), the bridge partition may have different active constraints. GT is mapped per-quarter then summed to annual.

### Recovery by PY × Quarter (full ladder: annual bridge + f0 fallback)

f0 monthly bridge fallback applies where monthly bridge data exists (2026-01..2026-04). No effect on historical PYs (no monthly bridge for those settlement months).

| PY | aq1 | aq2 | aq3 | aq4 | Annual |
|----|-----|-----|-----|-----|--------|
| 2019-06 | 93.8% | 90.1% | 92.7% | 90.4% | 91.6% |
| 2020-06 | 96.1% | 92.3% | 93.5% | 80.2% | 90.1% |
| 2021-06 | 90.4% | 81.4% | 81.4% | 94.2% | 86.4% |
| 2023-06 | 99.0% | 93.9% | 91.7% | 94.3% | 94.5% |
| 2024-06 | 96.7% | 98.7% | 96.5% | 79.7% | 93.0% |
| 2025-06 | 89.1% | 65.0% | 71.8% | 92.0% | 76.7% |

### Analysis

**Historical PYs (training set: 2019, 2020, 2023, 2024)**

Recovery is 90-95% annually. The 5-10% loss comes from DA monitored lines that genuinely don't exist in the bridge — the DA observed binding on equipment that the SPICE model didn't enumerate. This is a hard floor; no fallback can recover it.

Worst individual quarters: 2020-06 aq4 (80.2%), 2024-06 aq4 (79.7%). The aq4 dips in these years are driven by a handful of high-SP natural-language facility names (e.g., "Jordan - WFrankfort E 138kV", "Pres - Tibbs 138 kV") that use a different naming convention than the bridge.

**PY 2021-06 (training)**

86.4% annual recovery. aq2/aq3 show 81.4% recovery — likely naming mismatches. DA year=2022 has since been backfilled (73k rows), so the earlier claim of missing aq3/aq4 GT is no longer valid. Coverage numbers in this report predate the backfill and may undercount.

**PY 2022-06 (training, R1-R2 only)**

Recently backfilled. Density, bridge, limit, SF, and DA all now available. R1-R2 only (R3-R4 bridge absent). Coverage not yet measured — numbers in this report predate the backfill.

**PY 2025-06 (holdout)**

55.0% with annual bridge only, 76.7% with f0 fallback. The breakdown:

| Quarter | Annual bridge | + f0 | Cause of gap |
|---------|--------------|------|-------------|
| aq1 (Jun-Aug 2025) | 89.1% | 89.1% | Small gap, no f0 available |
| aq2 (Sep-Nov 2025) | 65.0% | 65.0% | Format change in DA monitored_facility, no f0 |
| aq3 (Dec 2025-Feb 2026) | 29.0% | 71.8% | Format change; f0 recovers Jan+Feb 2026 (no Dec bridge) |
| aq4 (Mar-May 2026) | 44.7% | 92.0% | Format change; f0 recovers Mar+Apr 2026 (no May bridge) |

Root cause: PJM DA `monitored_facility` encoding changed in recent months. Same physical equipment, different string format:
- Bridge: `BEDINGTO500 KV  BEDINGTO.T1`
- New DA: `BEDINGTO_T1_500TRAN2_XF`

### Checkpoint slice detail (PY=2024-06, R2, onpeak)

| Quarter | DA lines | Mapped | Unmapped | Branches | Total |SP| | Mapped |SP| | Recovery |
|---------|----------|--------|----------|----------|-----------|------------|----------|
| aq1 | 388 | 365 | 23 | 352 | 1,731,681 | 1,674,092 | 96.7% |
| aq2 | 340 | 326 | 14 | 309 | 1,289,166 | 1,271,855 | 98.7% |
| aq3 | 352 | 309 | 43 | 295 | 2,178,164 | 2,102,168 | 96.5% |
| aq4 | 333 | 285 | 48 | 269 | 1,602,485 | 1,276,969 | 79.7% |
| **Annual** | | | | | **6,801,496** | **6,325,084** | **93.0%** |

Top unmapped monitored lines (aq4, where most loss occurs):
1. `Pres - Tibbs 138 kV l/o Commodore - Jord 345 kV` — 68,911 |SP|
2. `Jordan - WFrankfort E 138kV l/o Jordan - Massac 345kV` — 65,133 |SP|
3. `Goosecre TX1 XFORMER H 500 KV` — 56,007 |SP|
4. `Bunsonville - Eugene Tie 345kV l/o Dumont - Wilton Center 765kV` — 49,949 |SP|
5. `ASPENSLR115 KV  ASP-ROXB` — 19,062 |SP|

These are natural-language names that don't match any bridge monitored line.

---

## 2. Model-Universe Coverage

### Method

Separate from GT mapping. Measures how much of the *already-mapped* GT SP survives the density universe threshold filter. Uses MISO's default threshold (0.000347) as starting point — needs PJM-specific calibration.

Steps:
1. Load density for all 12 months, filter to R2
2. Compute right-tail max per CID (bins 80, 90, 100, 110)
3. Active CIDs = right_tail_max >= threshold
4. Map active CIDs to branches via bridge (union across all quarters)
5. Compare: which GT branches are in the universe?

### Results by PY × Quarter (R2, onpeak)

| PY | aq1 | aq2 | aq3 | aq4 | Annual |
|----|-----|-----|-----|-----|--------|
| 2019-06 | 82.4% | 87.6% | 91.0% | 93.7% | 88.0% |
| 2020-06 | 86.8% | 83.7% | 86.0% | 89.1% | 86.6% |
| 2021-06 | 82.0% | 89.0% | 72.0% | 78.8% | 79.7% |
| 2023-06 | 90.5% | 90.8% | 93.1% | 95.3% | 92.6% |
| 2024-06 | 82.4% | 85.3% | 91.3% | 92.6% | 88.0% |
| 2025-06 | 90.3% | 89.6% | 90.7% | 93.0% | 90.3% |

### Threshold sensitivity

| Threshold | Active CIDs | % of total |
|-----------|-------------|------------|
| 0.0000 | 11,822 | 100.0% |
| 0.0001 | 7,629 | 64.5% |
| 0.0003 | 7,566 | 64.0% |
| **0.000347** (MISO default) | **7,563** | **64.0%** |
| 0.001 | 7,512 | 63.5% |
| 0.005 | 7,325 | 62.0% |
| 0.01 | 7,130 | 60.3% |

The threshold is not very sensitive in the 0.0001–0.001 range — the CID count plateau suggests a natural gap between active and inactive CIDs. MISO's threshold is reasonable as a starting point.

---

## 3. Combined Coverage Ceiling

*Table 1 × Table 2 = fraction of total DA SP that the model can see.*

| PY | GT mapping | Universe filter | **Combined** |
|----|-----------|-----------------|-------------|
| 2019-06 | 91.6% | 88.0% | **~81%** |
| 2020-06 | 90.1% | 86.6% | **~78%** |
| 2021-06 | 86.4% | 79.7% | **~69%** |
| 2023-06 | 94.5% | 92.6% | **~88%** |
| 2024-06 | 93.0% | 88.0% | **~82%** |
| 2025-06 | 76.7% | 90.3% | **~69%** |

For training PYs (excluding 2021, 2025), the model sees **78-88%** of total DA SP. The rest splits between GT mapping loss (monitored lines not in bridge) and universe filter loss (branches below density threshold).

---

## 4. Usable PY Grid

| PY | Density | DA | GT recovery | Universe | Status |
|----|---------|-----|-------------|----------|--------|
| 2019-06 | R1-R4 | Complete | 91.6%* | Not measured | Training |
| 2020-06 | R1-R4 | Complete | 90.1%* | Not measured | Training |
| 2021-06 | R1-R4 | Complete (backfilled) | 86.4%* | Not measured | Training |
| 2022-06 | R1-R4 | Complete (backfilled) | Not measured | Not measured | Training |
| 2023-06 | R1-R4 | Complete | 94.5%* | Not measured | Training |
| 2024-06 | R1-R4 | Complete | 93.0%* | 93.5%* | Training / eval anchor |
| 2025-06 | R1-R4 | Through Mar 2026 | 76.7%* (+f0) | Not measured | Holdout (incomplete) |

*All-hours coverage — not class-type-specific. See caveat at top of report.

Note: root resolution has been simplified — all data is now under the legacy root (`network_model=miso`). The earlier dual-root split was resolved during data backfill.

---

## 5. Open Issues

1. **All-hours caveat**: coverage numbers in this report are NOT class-type-specific. Class-type-specific coverage will differ because the bridge maps different constraints per class type. Need to re-run with peak-type filtering for authoritative per-ctype coverage.
2. **2025-06 format change**: DA `monitored_facility` encoding changed. Annual bridge cannot match ~35-71% of DA SP in aq2/aq3 without f0 help.
3. **PY 2021-06 DA gap**: year=2022 DA previously had zero rows but has since been backfilled (73k rows). The 2021-06 @400 loss in benchmark results may improve if the lookback window now has full 2022 data. Needs revalidation.
4. **Universe threshold**: MISO default used. PJM-specific calibration pending.
5. **NERC holidays**: not yet incorporated into peak-type filtering. Affects onpeak/wkndonpeak boundary.
