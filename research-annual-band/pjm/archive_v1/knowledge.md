# PJM Annual Band Research — Knowledge Base

**Last updated:** 2026-03-16
**Status:** Baseline research complete. Ready for banding phase.

---

## Auction Structure

| Aspect | PJM Annual | MISO Annual |
|--------|-----------|-------------|
| Period type | `a` (12-month, Jun-May) | `aq1`-`aq4` (quarterly) |
| Rounds | **4** (R1-R4) | 3 (R1-R3) |
| Settlement | 12 months | 3 months per quarter |
| All rounds clear | **~April** (same auction event) | ~April (R1), ~May (R2), ~Jun (R3) |
| Auction month convention | June (month 6) | June (month 6) |
| Class types (production) | `onpeak`, `dailyoffpeak`, `wkndonpeak` | `onpeak`, `offpeak` |
| Hedge types | `obligation` (97%) + `option` (3%) | `obligation` only |

## Baseline Decision: `mtm_1st_mean` for ALL Rounds (FINAL)

### Why mtm_1st_mean

- All 4 rounds have `mtm_1st_mean` with 99.9-100% coverage
- No need for H baseline, nodal_f0 stitch, or any fallback
- Simpler than MISO (no Ray, no nodal lookup, no fallback chain)

### What mtm_1st_mean Is

| Round | Source | When It Clears |
|-------|--------|:---:|
| R1 | Long-term yr1 Round 5 (same PY) | ~March |
| R2 | Annual R1 (same PY) | ~April |
| R3 | Annual R2 (same PY) | ~April |
| R4 | Annual R3 (same PY) | ~April |

Verified from raw M2M columns: `lr5Jun24 == mtm_1st` (exact match, diff=0.0000).

### Why We Can't Improve It (Exhaustive Search)

| Approach | Best MAE (monthly) | vs mtm_1st | Verdict |
|----------|---:|---:|---|
| **mtm_1st_mean (lr5)** | **65.7** | — | **BEST** |
| Nodal f0 stitch (PjmCalculator) | 110.3 | +68% | Much worse |
| ML regression (8 features) | 82.8 | +26% | Worse, overfits |
| LT round blend (lr5+lr4+lr3) | 65.6 | -0.2% | Negligible |
| Prior year R1 MCP blend | 54.8 | -2.7% | Low coverage |
| Best nodal blend (90/10) | 64.7 | -1.4% | Negligible |
| Monthly f0 path MCPs | 78.6 | +20% | Worse, 18% coverage |

For R2-R4: same conclusion. `1.2*m1 - 0.2*m2` gives at most -3.6% on R3, -2.7% on R4 — not worth the complexity. For R2, nothing beats mtm_1st at all.

### R1 MAE Is Improving Over Time

| Period | R1 MAE (monthly) | R1/R2 Ratio |
|--------|---:|:---:|
| PY 2017-2020 | ~105 | 3.4-4.8x |
| PY 2021 | 51.8 | 2.5x |
| PY 2022 | 95.8 | 2.3x |
| PY 2023-2025 | **~52** | **1.6-2.8x** |

For PY 2026 production: expect R1 MAE ~40-50 monthly (480-600 annual).
R2-R4 MAE is stable at 15-25 monthly across all years (except PY 2022: 38-43).

### R2-R4 MAE Is Stable (No Improvement Needed)

| PY | R1 | R2 | R3 | R4 |
|----|---:|---:|---:|---:|
| 2017 | 101.4 | 30.2 | 22.0 | 15.4 |
| 2018 | 99.5 | 22.1 | 18.1 | 13.5 |
| 2019 | 105.7 | 26.3 | 22.9 | 17.8 |
| 2020 | 114.6 | 23.9 | 17.8 | 15.3 |
| 2021 | 51.8 | 20.6 | 19.5 | 19.0 |
| **2022** | **95.8** | **42.5** | **38.9** | **43.4** |
| 2023 | 62.0 | 22.0 | 16.5 | 15.8 |
| **2024** | **41.2** | **16.3** | **12.7** | **11.7** |
| 2025 | 49.9 | 30.3 | 21.9 | 20.4 |

PY 2022 is the universal worst year. PY 2024 is the best.

## Scale Convention (CONFIRMED)

| Column | Scale | Formula |
|--------|-------|---------|
| `mcp` | **Annual total** (12-month) | The actual clearing price |
| `mcp_mean` | **Monthly average** | `mcp / 12` (confirmed: ratio = 12.0000) |
| `mtm_1st_mean` | **Monthly average** | Prior round's `mcp_mean` |

**For banding:** Use `mcp` as target, `baseline = mtm_1st_mean * 12`. All band widths in annual scale.

## Data Profile

| Dimension | Value |
|-----------|-------|
| Raw data | `/opt/temp/qianli/annual_research/pjm_annual_cleared_all.parquet` (238 MB) |
| With MTM | `/opt/temp/qianli/annual_research/pjm_annual_with_mcp.parquet` (403 MB) |
| Total rows (obligations, prod classes) | 19.4M |
| PYs | 2017-2025 (9 years) |
| Unique paths | 194,778 |
| Rows per round | R1: 3.9M, R2: 4.7M, R3: 5.4M, R4: 5.4M |

### Important: PY 2017-2022 R1 was onpeak-only

`dailyoffpeak` and `wkndonpeak` only appear in R1 from PY 2023. This means:
- Per-class stratification falls back to pooled for early PYs in R1
- Recent years (2023+) give more accurate per-class bands for R1
- R2-R4 have all 3 classes starting PY 2023 as well

## Preliminary Band Results (V1, Dev)

| Round | P95 Cov | P95 HW (annual) | PY2022 (worst) | onpeak P95 | dailyoff P95 | wkndon P95 |
|-------|:---:|---:|:---:|:---:|:---:|:---:|
| R1 | 97.1% | 2,863 | 94.0% | 96.6% | 97.7% | 98.1% |
| R2 | 95.6% | 803 | 88.3% | 94.5% | 97.7% | 97.2% |
| R3 | 95.8% | 670 | 86.0% | 94.5% | 97.8% | 97.9% |
| R4 | 94.6% | 569 | 82.4% | 92.7% | 97.9% | 97.5% |

## Banding Phase Plan

1. **Write PJM band script** — adapted from MISO `run_v9_bands.py`:
   - 4 rounds (not 3)
   - 3 class types (not 2)
   - Period type `a` (not `aq1`-`aq4`)
   - Annual scale × 12 (not quarterly × 3)
   - `mtm_1st_mean * 12` baseline for ALL rounds (no nodal stitch needed)
   - `hedge_type == 'obligation'` filter

2. **Dev run** — temporal expanding CV, min_train_pys=2, dev PYs 2017-2024

3. **Holdout** — PY 2025, min_train_pys=3

4. **Comprehensive report** — same format as MISO v9/v10 reports

5. **Production port planning** — much simpler than MISO (no nodal lookup needed)

## Completed Steps

1. ~~Data loading + profiling~~ ✅
2. ~~Scale verification (mcp/12)~~ ✅
3. ~~MTM source tracing (lr5 for R1)~~ ✅
4. ~~Baseline improvement exhaustive search~~ ✅ mtm_1st is best
5. ~~MAE trend analysis (all rounds × all PYs)~~ ✅
6. ~~R2-R4 improvement search~~ ✅ Can't beat mtm_1st
7. ~~Preliminary band calibration~~ ✅ V1 results saved
