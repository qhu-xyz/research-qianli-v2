# PJM June Monthly Auction — MCP Percentage Distribution

## Objective
Given a path's annual MCP, determine the monthly percentage distribution for the June auction (f0-f11) to improve MTM over the current flat `annual_mcp / 12`.

## Methodology

### Data
- **Source**: `PjmMcp` (obligation node MCPs, 24h class) + `PjmClearedFtrs` (annual cleared paths)
- **Years**: Planning years 2017-2025 (9 years)
- **Monthly auction**: June auction, round 1, f0 (Jun) through f11 (May)
- **Annual auction**: All 4 rounds combined for path universe; round 4 node MCPs for computing annual path MCP
- **Path universe**: All unique source-sink pairs from annual Obligation 24H cleared FTRs
- **Split**: Paths with annual MCP > $100 (positive MTM) vs < -$100 (negative MTM)

### Computation
For each path: `pct_fx = (monthly_node_fx[sink] - monthly_node_fx[source]) / sum_f0_to_f11 × 100`

### Validation
- Obligation MCPs verified: `cleared.mcp == node[sink] - node[source]` (93/93 R4 obligations match)
- Hub-to-hub paths cross-verified between scripts (diffs < 0.05%)
- All percentage sums = 100.0000%
- Options excluded (different pricing from obligations)

## Results (9-year, PY 2017-2025)

### Positive MTM paths (annual > 0) — Value-weighted aggregate

| f# | Month | '17 | '18 | '19 | '20 | '21 | '22 | '23 | '24 | '25 | **9yr avg** | **Std** |
|----|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------------|---------|
| f0 | Jun | 6.7 | 7.4 | 7.3 | 6.1 | 10.3 | 8.6 | 8.0 | 8.3 | 8.1 | **7.9%** | 1.1% |
| f1 | Jul | 8.6 | 6.9 | 7.7 | 6.5 | 9.6 | 8.7 | 8.7 | 8.8 | 9.7 | **8.4%** | 1.1% |
| f2 | Aug | 8.6 | 5.9 | 7.5 | 6.1 | 9.8 | 8.6 | 8.6 | 8.2 | 8.9 | **8.0%** | 1.2% |
| f3 | Sep | 7.8 | 9.0 | 9.2 | 9.2 | 12.8 | 9.7 | 8.7 | 8.7 | 7.8 | **9.2%** | 1.4% |
| f4 | Oct | 8.0 | 9.3 | 9.5 | 9.3 | 12.5 | 11.1 | 8.7 | 8.5 | 7.8 | **9.4%** | 1.4% |
| f5 | Nov | 7.8 | 9.0 | 9.2 | 8.7 | 10.5 | 7.7 | 8.0 | 7.7 | 7.1 | **8.4%** | 1.0% |
| f6 | Dec | 10.2 | 11.5 | 8.2 | 7.5 | 5.4 | 8.0 | 7.5 | 7.8 | 9.0 | **8.4%** | 1.7% |
| f7 | Jan | 10.2 | 11.5 | 8.2 | 10.6 | 1.5 | 9.0 | 10.0 | 10.1 | 10.9 | **9.1%** | 2.9% |
| f8 | Feb | 9.2 | 10.4 | 7.7 | 9.5 | 1.5 | 8.5 | 8.4 | 8.5 | 8.6 | **8.0%** | 2.4% |
| f9 | Mar | 7.7 | 6.4 | 8.6 | 9.4 | 8.6 | 7.3 | 7.5 | 7.5 | 7.3 | **7.8%** | 0.9% |
| f10 | Apr | 7.5 | 6.2 | 8.3 | 8.7 | 8.6 | 6.1 | 8.0 | 7.8 | 7.2 | **7.6%** | 0.9% |
| f11 | May | 7.7 | 6.4 | 8.6 | 8.6 | 9.0 | 6.6 | 7.9 | 8.1 | 7.6 | **7.8%** | 0.8% |

Path counts: 697, 2504, 1162, 923, 1586, 1005, 1321, 2614, 2297

### Negative MTM paths (annual < 0) — Value-weighted aggregate

| f# | Month | '17 | '18 | '19 | '20 | '21 | '22 | '23 | '24 | '25 | **9yr avg** | **Std** |
|----|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------------|---------|
| f0 | Jun | 6.0 | 6.7 | 5.6 | 7.1 | 6.5 | 7.0 | 8.2 | 7.7 | 7.8 | **7.0%** | 0.8% |
| f1 | Jul | 8.7 | 6.7 | 6.8 | 7.4 | 6.4 | 7.3 | 8.5 | 9.5 | 10.4 | **8.0%** | 1.3% |
| f2 | Aug | 9.6 | 5.9 | 7.0 | 7.0 | 6.4 | 7.5 | 8.1 | 9.2 | 9.6 | **7.8%** | 1.3% |
| f3 | Sep | 7.6 | 14.8 | 7.8 | 10.4 | 8.5 | 8.8 | 9.5 | 9.0 | 8.7 | **9.4%** | 2.1% |
| f4 | Oct | 7.8 | 15.3 | 8.1 | 10.7 | 9.2 | 11.6 | 9.4 | 8.8 | 8.0 | **9.9%** | 2.3% |
| f5 | Nov | 7.6 | 14.8 | 7.8 | 10.3 | 10.2 | 8.3 | 8.4 | 7.7 | 7.2 | **9.1%** | 2.3% |
| f6 | Dec | 10.5 | 5.7 | 12.2 | 7.5 | 8.8 | 9.0 | 7.2 | 8.1 | 8.4 | **8.6%** | 1.8% |
| f7 | Jan | 10.5 | 5.7 | 12.2 | 5.6 | 11.5 | 9.2 | 8.3 | 9.0 | 9.6 | **9.1%** | 2.2% |
| f8 | Feb | 9.5 | 5.1 | 11.4 | 6.0 | 9.9 | 9.3 | 7.5 | 7.8 | 7.7 | **8.3%** | 1.9% |
| f9 | Mar | 7.4 | 6.4 | 7.1 | 9.4 | 9.4 | 7.5 | 8.0 | 7.5 | 7.4 | **7.8%** | 1.0% |
| f10 | Apr | 7.2 | 6.2 | 6.8 | 9.0 | 6.8 | 7.2 | 8.6 | 7.5 | 7.2 | **7.4%** | 0.8% |
| f11 | May | 7.4 | 6.4 | 7.1 | 9.6 | 6.4 | 7.3 | 8.1 | 8.0 | 8.0 | **7.6%** | 0.9% |

Path counts: 471, 750, 451, 319, 546, 627, 575, 1288, 1486

## Key Findings

### 1. Moderate Sep-Oct peak confirmed across 9 years
Both positive and negative MTM paths show Sep (f3) and Oct (f4) at ~9-10% vs flat 8.33%. This is the most consistent pattern.

### 2. Winter is NOT depressed — it's slightly elevated for positive MTM
Positive paths show Jan (f7) at 9.1% avg — ABOVE flat 8.33%. This contradicts the initial node-level analysis that showed negative winter. The median confirms this (10.3% for Jan).

### 3. Spring (Mar-May) and Jun are modestly below flat
f0 (Jun) = 7.9%, f9 (Mar) = 7.8%, f10 (Apr) = 7.6%, f11 (May) = 7.8%. These are consistently ~0.5% below flat.

### 4. PY2021 is the outlier year
PY2021 shows extreme deviation: Oct=12.5%, Jan=1.5%. Excluding PY2021, the 8-year avg for Jan would be ~10%, not 9.1%.

### 5. Year-to-year std is 1-3%
Most months have std of 1-2% across 9 years. Jan has the highest at 2.9% (positive paths). This is moderate — the signal (deviation from flat) is comparable to the noise (year-to-year variation).

### 6. Positive vs negative paths are similar in shape
Both groups peak in Sep-Oct. The negative group has slightly higher Oct (9.9% vs 9.4%) and lower Jun (7.0% vs 7.9%). The differences are within noise.

## Recommended Distribution

### For positive MTM paths (annual > 0):
```
Jun=7.9%  Jul=8.4%  Aug=8.0%  Sep=9.2%  Oct=9.4%
Nov=8.4%  Dec=8.4%  Jan=9.1%  Feb=8.0%  Mar=7.8%
Apr=7.6%  May=7.8%
```

### For negative MTM paths (annual < 0):
```
Jun=7.0%  Jul=8.0%  Aug=7.8%  Sep=9.4%  Oct=9.9%
Nov=9.1%  Dec=8.6%  Jan=9.1%  Feb=8.3%  Mar=7.8%
Apr=7.4%  May=7.6%
```

### Or a single universal distribution (average of both):
```
Jun=7.4%  Jul=8.2%  Aug=7.9%  Sep=9.3%  Oct=9.7%
Nov=8.8%  Dec=8.5%  Jan=9.1%  Feb=8.2%  Mar=7.8%
Apr=7.5%  May=7.7%
```

## Comparison to flat (1/12 = 8.33%)

| Month | Recommended | Flat | Diff |
|-------|-------------|------|------|
| Jun | 7.4% | 8.33% | -0.9% |
| Jul | 8.2% | 8.33% | -0.2% |
| Aug | 7.9% | 8.33% | -0.4% |
| Sep | **9.3%** | 8.33% | **+1.0%** |
| Oct | **9.7%** | 8.33% | **+1.3%** |
| Nov | 8.8% | 8.33% | +0.4% |
| Dec | 8.5% | 8.33% | +0.2% |
| Jan | **9.1%** | 8.33% | **+0.8%** |
| Feb | 8.2% | 8.33% | -0.2% |
| Mar | 7.8% | 8.33% | -0.5% |
| Apr | 7.5% | 8.33% | -0.8% |
| May | 7.7% | 8.33% | -0.6% |

The maximum deviation from flat is **+1.3%** (Oct) and **-0.9%** (Jun). The distribution is moderately non-uniform with three peaks: Sep-Oct (fall congestion), Jan (winter congestion), and a trough in Apr-Jun (spring shoulder).

## Errors Caught and Corrected

| Issue | Resolution |
|-------|-----------|
| Initial node-level system sum showed Jan=-8.4%, Oct=17.4% | Misleading — dominated by extreme nodes. Switched to path-level analysis. |
| class_type mismatch: '24H' in cleared vs '24h' in MCP | Fixed filters per source. |
| Options included | Filtered to `hedge_type == 'Obligation'` only. |
| Only R4 paths initially | Expanded to all 4 annual rounds. |
| Only 3 years (2020-2022) | Expanded to 9 years (2017-2025). |

## Scripts
| Script | Purpose |
|--------|---------|
| `15_audit.py` | Correctness audit |
| `16_final.py` | Final 3-year analysis |
| `18_all_years.py` | **Full 9-year analysis (definitive)** |
