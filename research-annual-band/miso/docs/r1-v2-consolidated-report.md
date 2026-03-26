# MISO R1 V2: Consolidated Report

**Date:** 2026-03-20 (updated: sign bug fix in 1(rev), all results re-estimated)
**Status:** Frozen spec, audited on dev (PY2020-2022) and holdout (PY2023-2025)

## Terminology Correction

The historical `P10/P30/P50/...` labels used in this report are **legacy two-sided band labels** from the archived banding code.
They do **not** mean one-sided `10%/30%/50%` clearing probability directly.

In the legacy V2 script:

- `P10` means a symmetric 10% residual band, so the **upper-edge buy clearing target is 55%**
- `P30` means upper-edge target **65%**
- `P50` means upper-edge target **75%**
- `P70` means upper-edge target **85%**
- `P80` means upper-edge target **90%**
- `P90` means upper-edge target **95%**
- `P95` means upper-edge target **97.5%**
- `P99` means upper-edge target **99.5%**

So all `P50` and `P95` numbers below should be read as:

- legacy `P50` = buy clearing at the **75th percentile** upper edge
- legacy `P95` = buy clearing at the **97.5th percentile** upper edge

When we want true one-sided terminology, we should write:

- `Q20` or `one-sided P20` = `baseline + quantile(residual, 0.20)`
- `Q50` or `one-sided P50` = `baseline + quantile(residual, 0.50)`

Those one-sided results are separate from the legacy V2 tables below.

---

## The Model

### Baseline

```
baseline = w × (nodal_f0 × 3) + (1 - w) × 1(rev)
```

| Component | What it is | Scale |
|-----------|-----------|-------|
| `nodal_f0 × 3` | Monthly f0 MCP stitch, scaled to quarterly | Quarterly $ |
| `1(rev)` | Prior-year same-quarter DA congestion revenue per path (**sink_node - source_node**) | Quarterly $ |
| `w` | Blend weight, varies by (flow_type, bin) | 0 to 1 |

Both inputs winsorized at P1/P99 globally before blending.

**Blend weights (frozen, re-estimated after 1(rev) sign fix):**

| Flow | q1 | q2 | q3 | q4 | q5 |
|------|:--:|:--:|:--:|:--:|:--:|
| prevail | 0.85 | 0.85 | 0.85 | 0.85 | **0.50** |
| counter | 0.85 | 0.80 | 0.80 | 0.70 | **0.60** |

With correct sign, the optimizer wants much more 1(rev) than before:
q1-q3 get 15-20% revenue. q5 gets 40-50% revenue. Counter gets more than prevail.

**Sign bug (2026-03-20):** Original 1(rev) was computed as source - sink (wrong).
All prior experiments with 1(rev) used wrong sign. Fixed by negation. Weights above are
re-estimated on corrected data.

**Why blend:** Raw f0 has systematic bias for large-|baseline| paths due to mean reversion.
With correct-sign 1(rev), f0 and rev have +0.76 correlation and 77% sign agreement.
The blend reduces overall MAE from 855 (raw f0) to 827 (3.3% improvement).
Counter bias is near zero on holdout PY2024-2025 (-30 to -47).

### Banding

```
For each (flow_type, bin, class_type) cell:
  lower_offset = quantile(residual, (1 - level) / 2)
  upper_offset = quantile(residual, (1 + level) / 2)
  where residual = mcp_train - baseline_adj_train

  lower = baseline_adj + lower_offset
  upper = baseline_adj + upper_offset
```

- 5 quantile bins on training |raw f0| (frozen throughout pipeline)
- Asymmetric signed quantile pairs (different offsets for upper and lower)
- Equal-weight expanding window (all PYs before test PY)
- Min 300 paths per (flow, bin, class) cell; fallback to (flow, bin) pooled across classes
- 8 legacy two-sided levels: P10, P30, P50, P70, P80, P90, P95, P99
- Governance cap: band edges clipped to ±15,000 quarterly $ (affects 2.6% of paths at P95)

### Data Sources

| Input | Source | Coverage |
|-------|--------|----------|
| Cleared trades | `MisoApTools.get_trades_of_given_duration()` → `merge_cleared_volume` → `get_m2m_mcp_for_trades_all` | 99.97% |
| nodal_f0 | Old `aq*_all_baselines.parquet` (joined by path key) | 84% overall, >99% for PY2020+ |
| 1(rev) | Option B: per-node `MisoDaLmpMonthlyAgg`, path = **sink - source**, prior-PY same-quarter | 99.2% |
| MCP | `mcp` column (quarterly clearing price) | 100% |

### Temporal CV

- Dev PYs: 2020-2022 (3 years)
- Holdout PYs: 2023-2025 (3 years)
- Min 2 training PYs per fold
- Blend weights and band quantiles estimated on train only

---

## Results: Worst-First

### PY2023: Extreme reversion year — worst P50 cells

PY2023 had unprecedented mean reversion: |MCP|/|baseline| = 0.38 for counter q5.
The baseline center is wrong by $1,400-2,600 for these cells. No band width can compensate.

| Quarter | Bin | Flow | Class | N | P50 | P50 miss | P95 | Bias | Mode |
|---------|-----|------|-------|--:|----:|--------:|----:|-----:|------|
| aq1 | q5 | counter | offpeak | 1,565 | **35.5%** | **-39.5pp** | 94.6% | +1,435 | bad center |
| aq1 | q5 | counter | onpeak | 2,292 | **39.2%** | **-35.8pp** | 92.9% | +1,667 | bad center |
| aq2 | q5 | counter | onpeak | 1,465 | **48.5%** | **-26.5pp** | 89.3% | +1,530 | bad center |
| aq2 | q5 | counter | offpeak | 1,118 | **51.3%** | **-23.7pp** | 92.8% | +1,524 | bad center |
| aq1 | q4 | counter | onpeak | 1,063 | **51.9%** | **-23.1pp** | 98.7% | +494 | bad center |
| aq2 | q4 | counter | offpeak | 677 | **58.8%** | **-16.2pp** | 95.7% | +577 | bad center |

These cells also have the worst P95: aq2 q5 counter onpeak = 89.3% (8.2pp miss).

**PY2023 prevail q5 has the opposite problem — massive over-coverage:**

| Quarter | Bin | Flow | Class | N | P50 | P50 miss | Bias |
|---------|-----|------|-------|--:|----:|--------:|-----:|
| aq2 | q5 | prevail | onpeak | 3,859 | **97.3%** | **+22.3pp** | -2,567 |
| aq1 | q5 | prevail | onpeak | 5,476 | **95.6%** | **+20.6pp** | -1,781 |
| aq3 | q5 | prevail | onpeak | 2,800 | **93.5%** | **+18.5pp** | -1,075 |

Prevail q5 over-covers by 18-22pp at P50 — the bands are far too wide for these paths.

### PY2025: Width calibration failures (bad width, not bad center)

PY2025 prevail q3-q5 has low bias but the bands are too narrow. The training data
(PY2019-2024) was calmer than PY2025's actual residual distribution.

| Quarter | Bin | Flow | Class | N | P50 | P50 miss | Bias | Mode |
|---------|-----|------|-------|--:|----:|--------:|-----:|------|
| aq3 | q4 | prevail | onpeak | 2,252 | **56.5%** | **-18.5pp** | +242 | bad width |
| aq1 | q5 | prevail | offpeak | 1,416 | **58.0%** | **-17.0pp** | +896 | bad center |
| aq3 | q3 | prevail | onpeak | 2,080 | **59.7%** | **-15.3pp** | -12 | bad width |
| aq3 | q5 | prevail | offpeak | 1,442 | **64.5%** | **-10.5pp** | +583 | bad width |
| aq2 | q2 | prevail | offpeak | 1,986 | **63.5%** | **-11.5pp** | +4 | bad width |

Note: bias is near zero but P50 misses by 10-18pp. The center is correct but the
residual distribution widened relative to training.

### PY2022: Anti-reversion year (dev)

PY2022 was the opposite of PY2023 — baselines were close to MCP (alpha ≈ 0.92).
The blend over-shrinks for this year.

| Quarter | Bin | Flow | Class | N | P50 | P50 miss | P95 | Mode |
|---------|-----|------|-------|--:|----:|--------:|----:|------|
| aq2 | q5 | prevail | onpeak | 3,222 | **53.4%** | **-21.6pp** | **79.5%** | bad width |
| aq3 | q5 | prevail | offpeak | 2,902 | **54.9%** | **-20.1pp** | **88.7%** | bad width |
| aq1 | q3 | prevail | onpeak | 1,334 | **55.2%** | **-19.8pp** | 96.6% | bad width |
| aq3 | q5 | prevail | onpeak | 3,337 | **62.9%** | **-12.1pp** | **88.3%** | bad width |

PY2022 prevail q5 also has the worst P95 in the entire grid: aq2 = 79.5% (18pp miss).

### PY2024: Most balanced year

| Flow | P50 | P95 | Bias | Note |
|------|----:|----:|-----:|------|
| prevail | 71.3% | 98.0% | -234 | P50 within 3.7pp |
| counter | 77.5% | 98.2% | -30 | Near-zero bias |

### Year-by-year pattern summary (corrected 1(rev) sign)

| PY | Character | Prevail P50 | Counter P50 | Worst failure |
|:--:|-----------|:----------:|:----------:|---------------|
| 2021 | Early PY, small training set | 81.3% | 66.9% | Counter q5: 54% (bad width) |
| 2022 | Anti-reversion | 68.3% | 72.5% | Prevail q5: 62-66% (bad width) |
| 2023 | Extreme reversion | 79.5% | **67.7%** | **Counter q5: 53% (bad center, structural)** |
| 2024 | Normal | 71.3% | 77.5% | Prevail q5: 62% (bad width) |
| 2025 | Width drift | **68.0%** | 78.5% | Prevail q3-q5: 54-65% (bad width) |

### P95 worst cells (all PYs)

| Quarter | PY | Bin | Flow | Class | N | P95 | P95 miss | Mode |
|---------|:--:|-----|------|-------|--:|----:|--------:|------|
| aq2 | 2022 | q5 | prevail | onpeak | 3,222 | **79.5%** | **-18.0pp** | bad width |
| aq4 | 2021 | q5 | counter | offpeak | 985 | **81.8%** | **-15.7pp** | bad width |
| aq1 | 2022 | q5 | prevail | onpeak | 3,055 | **87.5%** | **-10.0pp** | bad width |
| aq4 | 2021 | q5 | counter | onpeak | 1,018 | **88.2%** | **-9.3pp** | bad width |
| aq2 | 2023 | q5 | counter | onpeak | 1,465 | **89.3%** | **-8.2pp** | bad center |
| aq3 | 2022 | q5 | prevail | onpeak | 3,337 | **88.3%** | **-9.2pp** | bad width |

All P95 failures are q5. PY2022 prevail and PY2021 counter dominate.

### Width efficiency: widest bands that still miss

| Quarter | PY | Bin | Flow | Class | P95 | P95 Width | P50 | Issue |
|---------|:--:|-----|------|-------|----:|---------:|----:|-------|
| aq2 | 2023 | q5 | counter | onpeak | 89.3% | **12,549** | 48.5% | $12.5k band still misses by 8pp |
| aq1 | 2023 | q5 | counter | offpeak | 94.6% | **14,657** | 35.5% | $14.7k band, P50 at 35% |
| aq2 | 2024 | q5 | prevail | offpeak | 92.9% | **13,451** | 58.2% | $13.5k band still misses |
| aq1 | 2024 | q5 | counter | offpeak | 99.3% | **17,255** | 75.6% | $17.3k band for 75% P50 — wasted width |

These are the economically worst outcomes: spending $12-17k in band width per path.

---

## Known Limitations

### 1. PY2023 counter q4/q5 (bad center)

Worst cells at P50 buy@upper:

| Quarter | Bin | N | Coverage | Miss | Cause |
|---------|-----|--:|--------:|-----:|-------|
| aq1 | q5 counter onpeak | 2,292 | 39.2% | -35.8pp | Extreme reversion: |MCP|/|baseline| = 0.38 |
| aq1 | q5 counter offpeak | 1,565 | 35.5% | -39.5pp | Same |
| aq2 | q5 counter onpeak | 1,465 | 48.5% | -26.5pp | Same |

PY2023 had unprecedented mean reversion — baselines shrank by 60% in q5. No method tested
fixes this. It is a structural tail risk.

### 2. PY2025 prevail q3-q5 (bad width)

87 cells with scale/tail drift between train and test residual distributions. The expanding
window calibrates on older, calmer years and under-estimates volatility for newer years.

Example: aq3 PY2025 prevail q4 onpeak: P50 = 56.5%, bias = +242 (low). The center is fine
but the band is too narrow because training residuals were calmer.

### 3. aq4 is systematically weakest

| Quarter | P50 coverage | P95 coverage |
|---------|:-----------:|:-----------:|
| aq1 | 75.5% | 97.8% |
| aq2 | 74.4% | 97.0% |
| aq3 | 71.9% | 97.2% |
| **aq4** | **71.8%** | **95.6%** |

aq4 (Mar-May) uses mixed-vintage 1(rev): March from current year, April/May from 2 years ago.
This is noisier than aq1-aq3 which use a single prior-PY quarter.

---

## What We Tried and Rejected

### Baseline improvements

| Method | Result | Why rejected |
|--------|--------|-------------|
| Global alpha shrinkage (alpha × baseline) | Fixes counter, breaks PY2022 prevail | Alpha non-stationary: 0.38 (PY2023) to 0.92 (PY2022) |
| Per-bin alpha shrinkage | Fixes counter q5 more but prevail q5 worse | Same non-stationarity |
| Capping alpha at 0.6 or 0.7 | Marginal improvement | Doesn't fix the non-stationarity |
| DA revenue as sole baseline | 2.6× worse MAE than f0 | DA revenue too noisy standalone |
| Recent-weighted blend (half-life 2yr) | < 0.1pp difference from equal-weight | W grid too coarse, too few PYs |
| Last-2-PY blend estimation | ~same as equal-weight | Same reason |
| Fixed bias correction (amplify baseline) | Would help 4/6 years, catastrophic for PY2023 | Bias direction flips between years |

### Banding improvements

| Method | Result | Why rejected |
|--------|--------|-------------|
| Flow_type split in banding only (no center fix) | Net zero improvement | Helps counter, hurts prevail by equal amount |
| Relative residual (resid / |baseline|) | Zero effect | Bins already control for magnitude |
| Recent-weighted quantiles | < 0.5pp improvement | Too few PYs for weighting to have leverage |
| Normalization (resid / |baseline|) for banding | Zero effect | Same information as binning |

### ML challengers

**Note:** ML experiments were run with wrong-sign 1(rev). Results may differ with corrected sign.
Re-running ML is lower priority than freezing the corrected blend and moving to R2/R3.

| Method | Features | MAE vs Blend | Worst cells | Why rejected |
|--------|----------|:---:|:---:|-------------|
| Huber regression | 17 (f0, rev, interactions) | -1.8% | Mixed | Over-learns PY2023 reversion |
| LightGBM basic | 13 | +1.4% | No movement | Heavy regularization prevents learning |
| LightGBM enriched | 20 (+annual rev) | +1.5% | No movement | Annual rev is informative but doesn't help worst cells |

### Sign bug impact on prior experiments

**All experiments using 1(rev) before the sign fix (2026-03-20) are contaminated:**
- Blend weight estimation used wrong-sign data → old weights (w=0.85-1.0) were too conservative
- Corrected weights are much heavier on revenue (w=0.50-0.85)
- The "77% sign disagreement" finding was caused by the bug, not by economics
- With correct sign: f0 and 1(rev) correlate at +0.76 with 77% sign agreement
- ML experiments should be re-run but are lower priority

---

## Structural Analysis: Why No Method Fixes q5

### |MCP| / |baseline| ratio is non-stationary

| PY | Prevail q5 | Counter q5 | Character |
|:--:|:----------:|:----------:|-----------|
| 2020 | 0.95 | 0.88 | MCP less extreme |
| 2021 | **1.11** | 0.99 | Prevail more extreme |
| 2022 | **1.20** | **1.31** | **Both much more extreme** (anti-reversion) |
| 2023 | 0.76 | 0.80 | **Both much less extreme** (extreme reversion) |
| 2024 | **1.10** | **1.10** | Both slightly more extreme |
| 2025 | **1.15** | **1.14** | Both slightly more extreme |

The ratio swings from 0.76 to 1.31 across years. No fixed correction (shrinkage, amplification, blend weight) works for all years because the direction itself is unpredictable from auction-time features.

### Bias direction flips by year

| PY | Prevail q5 bias | Counter q5 bias | Pattern |
|:--:|:----------:|:----------:|---------|
| 2022 | **+1,283** | **-1,962** | Opposite signs — classic mean reversion |
| 2023 | **-1,965** | **+1,090** | Opposite signs — but reversed from 2022 |
| 2024 | +101 | -25 | Near zero |
| 2025 | +215 | -53 | Near zero |

PY2022 and PY2023 have mirror-image biases. Any correction that helps one destroys the other.

---

## Failure Mode Classification

From finest-grain audit at (quarter, PY, flow, bin, class):

| Mode | Count | Description | Fix |
|------|------:|-------------|-----|
| **Bad center** | 35 cells | \|bias\| > 50% of MAE. Baseline wrong. | Blend partially addresses. PY2023 q5 structural. |
| **Bad width** | 100+ cells | P50 miss > 5pp despite low bias. Distribution shifted. | Not addressed by any method tested. |
| SCALE_UP | 50 | Test residuals wider than train | Expanding window averages over calm years |
| TAIL_EXPLODE | 37 | Test tails fatter than train | Extreme paths not seen in training |
| MIXED | 59 | Complex distributional shift | No single fix |

Note: drift type counts overlap (cells can have multiple types).

---

## Decision: Freeze R1 and Move Forward

The frozen blend + empirical banding is the best achievable system with current data:
- **Overall P50: 74.5% (target 75%)** — within 0.5pp on holdout
- **Overall P95: 97.9% (target 97.5%)** — within 0.4pp on holdout
- **P50 width: 585 quarterly $** — 12% narrower than raw f0 baseline
- **Flow gap: 4.4pp at P50** — down from 20pp with raw f0

The remaining failures (PY2023 counter q5 at 35-52%, PY2022 prevail q5 at 53-55%) are structural and cannot be fixed without regime-prediction features that don't exist at auction time.

---

## Appendix: Correct Pxx Holdout Results (added 2026-03-20)

The tables above use the legacy two-sided definition. Below are the correct single-sided
Pxx results where **Pxx = bid price with xx% clearing probability.**

### R1 Overall Holdout (PY2023-2025, 327,309 paths)

| Level | Target | Actual | Miss |
|-------|--------|--------|------|
| P10 | 10% | 10.4% | +0.4pp |
| P20 | 20% | 20.5% | +0.5pp |
| P50 | 50% | 48.4% | -1.6pp |
| P80 | 80% | 78.9% | -1.1pp |
| P95 | 95% | 95.2% | +0.2pp |

### R1 Holdout q5 by (PY, flow) — correct Pxx

| PY | Flow | N | P10 | P20 | P50 | P90 | P95 | bias |
|----|------|---:|----:|----:|----:|----:|----:|-----:|
| 2023 | prevail | 28,324 | 24.1% | 39.3% | 70.1% | 95.4% | 97.6% | -1,872 |
| 2023 | counter | 9,484 | 4.2% | 9.6% | 26.5% | 79.3% | 89.3% | +1,074 |
| 2024 | prevail | 16,999 | 6.0% | 10.8% | 33.7% | 84.3% | 93.1% | -432 |
| 2024 | counter | 5,589 | 14.0% | 28.2% | 60.9% | 92.5% | 96.5% | -332 |
| 2025 | prevail | 15,610 | 4.6% | 11.1% | 37.3% | 92.3% | 97.6% | -705 |
| 2025 | counter | 5,671 | 6.4% | 20.5% | 64.6% | 96.1% | 98.1% | -222 |

P50 under correct Pxx: target = 50%. Prevail q5 is 33-70% (swings ±20pp across years).
Counter q5 is 27-65%. This is the structural non-stationarity documented above.

## Next Steps

1. **R2/R3:** Done. Champion spec in `r2-r3-consolidated-report.md`.
2. **PJM:** Same framework, different scale (×12), 4 rounds, 3 classes.
3. **Production port:** After PJM frozen.
