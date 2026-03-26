# MISO R2/R3: Consolidated Report

**Date:** 2026-03-20
**Status:** Champion spec frozen
**Script:** `miso/scripts/run_r2r3.py`

## Pxx Definition

**Pxx = single bid price with xx% clearing probability on training data.**
- P20 = baseline + quantile(residual, 0.20) → 20% clearing chance
- P95 = baseline + quantile(residual, 0.95) → 95% clearing chance
- Each level is one bid price, NOT a pair. No "band width" concept.

## Frozen Spec

### Baseline
```
baseline = mtm_1st_period   (= mtm_1st_mean × 3, quarterly)
```
- Pure prior-round MCP. No blending.
- 1(rev) oracle w = 1.00 for every cell — zero improvement from revenue blending.
- R2 baseline MAE = 232, R3 baseline MAE = 186 (vs R1 = 789).

### Calibration Cells (Hybrid)
```
q1-q3: calibrate by (bin, flow_type)     — 3 × 2 = 6 cells
q4-q5: calibrate by (bin) only           — 2 cells (flow pooled)
class_type: dropped everywhere           — <0.5pp effect
```
Total: 8 calibration cells per (quarter, fold).

**Why hybrid:**
- q1-q3 are well-calibrated regardless of cell definition (bias < ±50)
- q4-q5 with flow split: prevail over-clears and counter under-clears at P10-P50
  because the expanding window mixes volatile and calm years differently for each flow
- Pooling flow for q4-q5 reduces worst-case P10/P20 miss by 2-4pp
- Tradeoff: P90/P95 for q5 counter over-covers by ~3pp (98% vs 95% target)

**Why NOT pool flow for q1-q3:**
- No improvement from pooling (within ±0.5pp)
- Flow split doesn't hurt (bias is small, cell sizes are large)

### Other Parameters
- 5 quantile bins on training |baseline|
- Expanding window, min 2 training PYs, min 300 paths per calibration cell
- Governance cap: ±15,000 quarterly $ on bid prices

## Holdout Results (PY2023-2025)

### R2 Overall (406,916 paths)

| Level | Target | Actual | Miss |
|-------|--------|--------|------|
| P05 | 5% | 5.2% | +0.2pp |
| P10 | 10% | 10.8% | +0.8pp |
| P20 | 20% | 22.0% | +2.0pp |
| P50 | 50% | 53.0% | +3.0pp |
| P80 | 80% | 82.6% | +2.6pp |
| P95 | 95% | 95.8% | +0.8pp |

### R3 Overall (415,645 paths)

| Level | Target | Actual | Miss |
|-------|--------|--------|------|
| P05 | 5% | 4.8% | -0.2pp |
| P10 | 10% | 10.5% | +0.5pp |
| P20 | 20% | 22.3% | +2.3pp |
| P50 | 50% | 53.9% | +3.9pp |
| P80 | 80% | 83.3% | +3.3pp |
| P95 | 95% | 96.6% | +1.6pp |

Slight over-calibration at P50 (+3-4pp) — the expanding window includes PY2022's high-volatility
residuals, making quantile estimates too wide for the calmer PY2023-2025.

### Holdout by (PY, flow) — R2

| PY | Flow | N | bias | MAE | P10 | P20 | P50 | P90 | P95 |
|----|------|---:|-----:|----:|----:|----:|----:|----:|----:|
| 2023 | prevail | 79,890 | +32 | 265 | 13.1% | 26.1% | 55.7% | 90.7% | 95.0% |
| 2023 | counter | 61,065 | +13 | 231 | 13.2% | 24.4% | 52.3% | 89.8% | 94.6% |
| 2024 | prevail | 68,773 | +113 | 243 | 7.4% | 14.6% | 42.4% | 88.9% | 94.5% |
| 2024 | counter | 45,493 | +16 | 206 | 10.6% | 21.8% | 53.2% | 90.9% | 95.0% |
| 2025 | prevail | 88,066 | -21 | 208 | 10.8% | 23.9% | 59.6% | 94.5% | 97.6% |
| 2025 | counter | 63,629 | +23 | 197 | 9.7% | 20.1% | 52.5% | 93.3% | 97.2% |

### Holdout by (PY, flow) — R3

| PY | Flow | N | bias | MAE | P10 | P20 | P50 | P90 | P95 |
|----|------|---:|-----:|----:|----:|----:|----:|----:|----:|
| 2023 | prevail | 83,733 | +7 | 198 | 11.1% | 23.7% | 56.6% | 92.5% | 96.5% |
| 2023 | counter | 60,586 | +18 | 173 | 11.1% | 21.9% | 50.5% | 90.6% | 95.5% |
| 2024 | prevail | 74,510 | +32 | 177 | 7.4% | 16.5% | 47.6% | 91.6% | 96.1% |
| 2024 | counter | 51,223 | -5 | 174 | 10.0% | 20.5% | 53.2% | 92.6% | 96.6% |
| 2025 | prevail | 84,907 | -73 | 182 | 14.3% | 31.0% | 65.7% | 95.1% | 97.9% |
| 2025 | counter | 60,686 | +44 | 159 | 8.3% | 17.5% | 45.5% | 91.3% | 96.7% |

### Worst Holdout Cells at q5 (PY, flow)

**R2 q5:**

| PY | Flow | N | P10 | P20 | P50 | P90 | P95 | bias |
|----|------|---:|----:|----:|----:|----:|----:|-----:|
| 2023 | prevail | 27,784 | 7.8% | 22.5% | 55.3% | 95.3% | 97.6% | +83 |
| 2023 | counter | 13,882 | 12.2% | 26.8% | 60.9% | 97.1% | 98.7% | -7 |
| 2024 | prevail | 17,760 | 4.9% | 11.4% | 37.9% | 90.9% | 95.9% | +278 |
| 2024 | counter | 7,475 | 14.6% | 29.2% | 61.2% | 95.3% | 97.6% | -16 |
| 2025 | prevail | 25,606 | 9.5% | 24.8% | 67.2% | 98.2% | 99.4% | -67 |
| 2025 | counter | 14,740 | 9.6% | 22.7% | 58.1% | 97.0% | 98.9% | +37 |

**R3 q5:**

| PY | Flow | N | P10 | P20 | P50 | P90 | P95 | bias |
|----|------|---:|----:|----:|----:|----:|----:|-----:|
| 2023 | prevail | 28,541 | 6.6% | 18.3% | 55.1% | 95.3% | 98.4% | +38 |
| 2023 | counter | 14,114 | 7.3% | 18.3% | 51.8% | 97.2% | 98.8% | +48 |
| 2024 | prevail | 22,058 | 7.2% | 17.2% | 50.9% | 94.7% | 97.6% | +53 |
| 2024 | counter | 12,101 | 16.9% | 29.0% | 59.3% | 95.7% | 98.7% | -57 |
| 2025 | prevail | 24,442 | 14.9% | 35.0% | 71.5% | 98.6% | 99.6% | -160 |
| 2025 | counter | 12,093 | 5.9% | 14.4% | 41.0% | 93.4% | 98.4% | +132 |

Note: PY2025 R3 prevail q5 over-clears (+15pp at P20, +22pp at P50). PY2025 R3 counter q5
under-clears at P20 (-6pp) but over-clears at P95 (+3pp). This is the same non-stationarity
as R1 but lower magnitude (bias ±160 vs R1's ±1,800).

## What We Tried

### 1(rev) blending — REJECTED
- Computed 1(rev) for all rounds via nodal DA + MisoNodalReplacement (90.1% coverage)
- Oracle blend weight = 1.00 for EVERY cell — zero improvement
- Prior-round MCP already incorporates whatever DA revenue information exists

### Calibration cell experiments — A/B/C/D comparison

| Config | Cell definition | Best for |
|--------|----------------|----------|
| A: (bin, flow, class) | Current R1 | P90/P95 tails |
| B: (bin, class) | Drop flow | P10-P50 for q5 |
| C: (bin, flow) | Drop class | ≈ A (class is noise) |
| D: (bin) | Drop both | ≈ B |
| **H2: hybrid** | q1-q3 by (bin,flow), q4-q5 by (bin) | **Best overall** |

H2 is champion because:
- q1-q3: flow split is fine (small bias, large cells)
- q4-q5: pooling flow reduces P10/P20 worst-case miss by 2-4pp
- P90/P95 tradeoff is acceptable (+1-3pp over-coverage for q5 counter)

### Cap/floor — NO EFFECT
Tested baseline ±15k + bid ±20k, bid ±15k, baseline ±12k + bid ±15k.
All produced identical coverage under correct Pxx definition because P20 bids
are well within cap range. Cap is a no-op governance safety net.

## PY2018-2019 Note

PY2018-2019 are present in the data but skipped in evaluation because they have < 2
training PYs (the expanding window needs ≥ 2). PY2020 is the first evaluable year
(trained on PY2018-2019). This is by design, not a data gap.

## Baseline Diagnostics

### R2 bias by (PY, flow, bin)

| PY | Flow | q1 | q2 | q3 | q4 | q5 |
|----|------|---:|---:|---:|---:|---:|
| 2023 | prevail | +15 | +3 | +8 | -9 | +101 |
| 2023 | counter | +15 | +19 | +21 | +17 | -9 |
| 2024 | prevail | +24 | +45 | +56 | +91 | +292 |
| 2024 | counter | +17 | +21 | +27 | +27 | -22 |
| 2025 | prevail | +6 | +3 | -1 | -14 | -68 |
| 2025 | counter | +2 | +10 | +24 | +41 | +36 |

### R3 bias by (PY, flow, bin)

| PY | Flow | q1 | q2 | q3 | q4 | q5 |
|----|------|---:|---:|---:|---:|---:|
| 2023 | prevail | +2 | -9 | -13 | -16 | +51 |
| 2023 | counter | +2 | +9 | +10 | +22 | +53 |
| 2024 | prevail | +12 | +15 | +25 | +39 | +54 |
| 2024 | counter | +6 | +17 | +11 | +16 | -64 |
| 2025 | prevail | -7 | -17 | -41 | -70 | -163 |
| 2025 | counter | +1 | +11 | +31 | +49 | +135 |

Key observations:
- Bias is 3-10x smaller than R1 (±300 max vs R1's ±2,000)
- Prevail bias is NOT always negative (unlike R1). Direction flips across years.
- q5 concentrates all large biases. q1-q3 bias is < ±60 for all cells.
- PY2025 R3 shows the largest holdout asymmetry: prevail -163 vs counter +135
