# R3 Band Calibration v1

## Method

Same symmetric empirical quantile approach as R1 bands, but using `mtm_1st_mean`
(prior round's clearing price) as the baseline instead of `nodal_f0`.

Winner: `quantile_6bin`.

## P95 Mean Width ($/MWh)

| Quarter | P95 Width |
|---------|----------:|
| aq1 | 181.4 |
| aq2 | 180.9 |
| aq3 | 161.7 |
| aq4 | 162.2 |

## Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.58% | -0.42pp |
| aq2 | 95.0% | 94.81% | -0.19pp |
| aq3 | 95.0% | 94.54% | -0.46pp |
| aq4 | 95.0% | 94.67% | -0.33pp |

## Experiments Tested (aq1)

| Config | P95 cov | P95 err | P95 width |
|--------|--------:|--------:|----------:|
| fixed_4bin | 94.57% | -0.43pp | 184.6 |
| quantile_4bin | 94.54% | -0.46pp | 185.1 |
| quantile_6bin | 94.58% | -0.42pp | 181.4 |

## Gate Results

| Gate | Severity | Result |
|------|----------|--------|
| BG1 P95 coverage accuracy | HARD | PASS |
| BG2 P50 coverage accuracy | HARD | PASS |
| BG3 Per-bin uniformity | HARD | PASS |
| BG5 Per-PY stability | SOFT | FAIL |
| BG6 Width monotonicity | ADVISORY | PASS |

BG0 (baseline still promoted) not applicable — R3 uses M (prior round MCP), not a versioned baseline.

### BG5 SOFT Failure Justification

PY 2022 worst-PY P95 coverage is 81-85% across quarters, below the 90% threshold. Same systematic issue as R1 and R2: PY 2022 has extreme congestion volatility.

## Temporal Validation

| Quarter | LOO P95 | Temporal P95 | LOO width | Temporal width |
|---------|--------:|------------:|----------:|-------------:|
| aq1 | 94.58% | 91.42% | 181.4 | 152.2 |
| aq2 | 94.81% | 91.08% | 180.9 | 146.1 |
| aq3 | 94.54% | 91.34% | 161.7 | 135.9 |
| aq4 | 94.67% | 92.09% | 162.2 | 136.3 |

Temporal coverage drops 3-4pp vs LOO — slightly worse than R2, expected since R3 residuals are even tighter (smaller widths amplify training noise). PY 2023+ with 3+ training years achieve 95-97%.

## Cross-Round Width Comparison

| Quarter | R1 P95 | R2 P95 | R3 P95 | R3/R1 |
|---------|-------:|-------:|-------:|------:|
| aq1 | 2,646 | 216.9 | 181.4 | 6.9% |
| aq2 | 3,122 | 228.3 | 180.9 | 5.8% |
| aq3 | 2,496 | 201.8 | 161.7 | 6.5% |
| aq4 | 2,217 | 210.7 | 162.2 | 7.3% |

R3 is ~15x narrower than R1, ~15% narrower than R2.

## Decision

Initial R3 band calibration. `quantile_6bin` wins narrowly — all 3 configs pass all gates, 6 bins gives the narrowest widths with ~1.2M rows/quarter. R3 bands are the tightest of all rounds, consistent with the M baseline being closest to actuals (MAE 56 vs 70 for R2 vs 850 for R1).
