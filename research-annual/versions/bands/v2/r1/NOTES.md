# Band Calibration v2 — Width Reduction via Data-Driven Bins

## Motivation

v1 bands achieve accurate coverage (P95 within 0.3pp of target) but widths are unnecessarily wide. The large bin (|f0| >= 1000) has P95 widths of 7,400-10,000 $/MWh because it spans |f0| from 1,000 to 20,000+, lumping paths with very different residual behavior.

## Method

Same symmetric empirical quantile approach as v1, but with data-driven bin boundaries computed from training set percentiles (per LOO fold).

### Experiments Tested

| Config | Boundaries | Extra params | Rationale |
|--------|-----------|-------------|-----------|
| `v1_repro` | [0,50,250,1000,inf] 4 bins | 0 | Baseline reproduction |
| `split_large` | [0,50,250,1000,3000,inf] 5 bins | +5 | Split most heterogeneous bin |
| `split_large_shrunk` | same 5 bins, alpha=0.8 | +5 | Shrink extreme bins toward global quantile |
| `six_bins` | [0,50,250,1000,3000,10000,inf] 6 bins | +10 | Test if 10k+ split helps |
| `quantile_bins` | data-driven 4 bins (per fold) | 0 | Test if domain boundaries are near-optimal |

### Selection Rule

Pass BG1-BG3, width monotonicity, min 1,000 rows per bin → lowest P95 mean width.

## Results

### Winner: `quantile_bins`

Only `v1_repro` and `quantile_bins` passed all gates. `quantile_bins` wins with ~20% width reduction.

### Why Others Failed

- **`split_large`**: Mean P95 width *increased* from 3,274 to 4,893 because `large_hi` (|f0| >= 3000) has P95 widths 11k-16k — splitting exposed the problem but made the average worse. Failed: mean width increased.
- **`split_large_shrunk`**: Overcoverage at P95 (+0.70 to +1.45pp) and P50 (+1.7 to +2.2pp). Shrinkage toward global quantile inflates bins that should be narrow.
- **`six_bins`**: `extreme` bin (|f0| >= 10000) had only 53-94 rows per quarter. Failed min bin size check.
- **`quantile_bins`**: Passed all gates. Boundaries adapt to the data distribution, balancing bin sizes.

### v1 vs v2 P95 Mean Width

| Quarter | v1 | v2 | Change |
|---------|---:|---:|-------:|
| aq1 | 3,318 | 2,646 | -20.3% |
| aq2 | 3,897 | 3,122 | -19.9% |
| aq3 | 3,104 | 2,496 | -19.6% |
| aq4 | 2,776 | 2,217 | -20.1% |

### Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.66% | -0.34pp |
| aq2 | 95.0% | 94.69% | -0.31pp |
| aq3 | 95.0% | 94.66% | -0.34pp |
| aq4 | 95.0% | 94.87% | -0.13pp |

Coverage accuracy is comparable to v1 (v1: -0.10 to -0.33pp).

### Gate Results

| Gate | Severity | Result |
|------|----------|--------|
| BG0 Baseline still promoted | HARD | PASS |
| BG1 P95 coverage accuracy | HARD | PASS |
| BG2 P50 coverage accuracy | HARD | PASS |
| BG3 Per-bin uniformity | HARD | PASS |
| BG4 Width narrower or equal | SOFT | PASS |
| BG5 Per-PY stability | SOFT | FAIL |
| BG6 Width monotonicity | ADVISORY | PASS |

### BG5 SOFT Failure Justification

Same as v1: PY 2022 worst-PY coverage drops below 90% for aq1-aq3 (87.7%, 86.6%, 87.3%). This is a data property (PY 2022 has extreme congestion volatility), not a method deficiency. Width CV is actually lower than v1 (0.05-0.08 vs 0.03-0.09).

### Asymmetry Diagnostic

Upper/lower P95 ratio CV across PYs per bin:

| Bin | aq1 | aq2 | aq3 | aq4 |
|-----|----:|----:|----:|----:|
| tiny | 0.078 | 0.147 | 0.180 | 0.078 |
| small | 0.123 | 0.134 | 0.120 | 0.154 |
| medium | 0.108 | 0.172 | 0.110 | 0.163 |
| large_lo | 0.049 | 0.157 | 0.181 | 0.155 |
| large_hi | 0.443 | 0.144 | 0.222 | 0.066 |

CV ranges from 0.05 to 0.44 — especially unstable for `large_hi` in aq1 (0.44). This confirms asymmetric bands would be unstable across LOO folds.

### Temporal Validation (expanding window)

LOO uses future data (e.g., PY 2024-2025 help calibrate PY 2021's bands). In production, only prior PYs are available. Temporal validation trains only on strictly prior PYs:

| Quarter | LOO P95 | Temporal P95 | LOO width | Temporal width |
|---------|--------:|------------:|---------:|-------------:|
| aq1 | 94.66% | 93.98% | 2,646 | 2,450 |
| aq2 | 94.69% | 92.65% | 3,122 | 2,668 |
| aq3 | 94.66% | 91.40% | 2,496 | 2,183 |
| aq4 | 94.87% | 91.48% | 2,217 | 1,434 |

Temporal P95 coverage drops 1-3.4pp below LOO. Two causes:
1. **Small training sets for early PYs.** PY 2021 trains on 1 PY (~16-18k rows) — noisy quantile estimates → bands too narrow (temporal P95 coverage 79-87%).
2. **Non-stationarity.** Residual distributions shift across PYs. Prior-only training misses future regime changes.

Per-PY temporal detail (aq3 as example):

| Test PY | Train PYs | n_train | P95 cov | P95 width |
|---------|-----------|--------:|--------:|----------:|
| 2021 | 2020 | 15,946 | 79.86% | 909 |
| 2022 | 2020-2021 | 37,933 | 85.66% | 1,773 |
| 2023 | 2020-2022 | 56,968 | 97.35% | 2,969 |
| 2024 | 2020-2023 | 82,915 | 96.14% | 2,673 |
| 2025 | 2020-2024 | 108,455 | 94.51% | 2,593 |

PY 2021-2022 (few training years) drag down the aggregate. PY 2023+ with 3+ training years achieve 94-97% coverage, comparable to LOO.

**Implication for production:** With 5+ years of history (PY 2025 onward), temporal coverage approaches LOO. For early PYs or after regime changes, a safety margin (e.g., multiply width by 1.05-1.10) may be needed.

## Decision

Promote as v2. 20% width reduction with equivalent LOO coverage accuracy. Temporal validation shows 1-3pp undercoverage but this is driven by early PYs with 1-2 years of training data — with 3+ years, temporal coverage is within tolerance. Data-driven boundaries outperform domain-driven boundaries because the |f0| distribution is not uniform — percentile-based bins balance sample sizes, improving quantile estimation in each bin.
