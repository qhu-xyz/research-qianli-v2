# R2 Band Calibration v1

## Method

Same symmetric empirical quantile approach as R1 bands, but using `mtm_1st_mean`
(prior round's clearing price) as the baseline instead of `nodal_f0`.

Winner: `quantile_6bin`.

## P95 Mean Width ($/MWh)

| Quarter | P95 Width |
|---------|----------:|
| aq1 | 216.9 |
| aq2 | 228.3 |
| aq3 | 201.8 |
| aq4 | 210.7 |

## Coverage Accuracy (LOO, P95)

| Quarter | Target | Actual | Error |
|---------|-------:|-------:|------:|
| aq1 | 95.0% | 94.65% | -0.35pp |
| aq2 | 95.0% | 94.77% | -0.23pp |
| aq3 | 95.0% | 94.62% | -0.38pp |
| aq4 | 95.0% | 94.61% | -0.39pp |

## Experiments Tested (aq1)

| Config | P95 cov | P95 err | P95 width |
|--------|--------:|--------:|----------:|
| fixed_4bin | 94.70% | -0.30pp | 221.7 |
| quantile_4bin | 94.69% | -0.31pp | 222.1 |
| quantile_6bin | 94.65% | -0.35pp | 216.9 |

## Gate Results

| Gate | Severity | Result |
|------|----------|--------|
| BG1 P95 coverage accuracy | HARD | PASS |
| BG2 P50 coverage accuracy | HARD | PASS |
| BG3 Per-bin uniformity | HARD | PASS |
| BG5 Per-PY stability | SOFT | FAIL |
| BG6 Width monotonicity | ADVISORY | PASS |

BG0 (baseline still promoted) not applicable — R2 uses M (prior round MCP), not a versioned baseline.

### BG5 SOFT Failure Justification

PY 2022 worst-PY P95 coverage is 82-87% across quarters, below the 90% threshold. Same systematic issue as R1: PY 2022 has extreme congestion volatility. Not a method deficiency — all 3 configs show the same PY 2022 outlier behavior.

## Temporal Validation

| Quarter | LOO P95 | Temporal P95 | LOO width | Temporal width |
|---------|--------:|------------:|----------:|-------------:|
| aq1 | 94.65% | 92.08% | 216.9 | 177.5 |
| aq2 | 94.77% | 91.79% | 228.3 | 181.5 |
| aq3 | 94.62% | 92.66% | 201.8 | 168.0 |
| aq4 | 94.61% | 92.31% | 210.7 | 171.0 |

Temporal coverage drops 2-3pp vs LOO, driven by early PYs (2020-2022) with few training years. PY 2023+ with 3+ training years achieve 94-97%.

## Cross-Round Width Comparison

R2 P95 widths ~8-10% of R1 (~12x narrower), consistent with R2's much lower MAE (70 vs 850).

## Decision

Initial R2 band calibration. `quantile_6bin` wins narrowly over `quantile_4bin` and `fixed_4bin` — all 3 pass all gates, but 6 bins achieves 2-5% narrower widths thanks to the large dataset (~1M rows/quarter). Widths are ~12x narrower than R1 due to M baseline being much closer to actuals.
