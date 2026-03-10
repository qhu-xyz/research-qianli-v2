# Knowledge Test: Constraint-Shadow Monte Carlo Scenarios

## Purpose
This note explains a baseline Monte Carlo construction for uncertainty in constraint shadow prices. The goal is to generate many plausible realizations of shadow prices, then map those realizations into path values and portfolio risk metrics.

## Setup and Notation
- `k`: index of transmission constraints. If there are `K` constraints, then `k \in {1, ..., K}`.
- `s`: index of Monte Carlo scenarios. If there are `M` scenarios, then `s \in {1, ..., M}`.
- `t`: index of historical timestamps used for calibration and residual estimation.
- `M`: total number of simulated scenarios (here `M = 2000`).
- `K`: number of modeled constraints.
- `\mu_k^{(s)}`: simulated shadow price for constraint `k` in scenario `s`.
- `\hat{\mu}_{k,\text{calibrated}}`: calibrated point estimate (deterministic baseline) for constraint `k`.
- `\varepsilon_k^{(s)}`: random forecast error (shock) for constraint `k` in scenario `s`.
- `\varepsilon^{(s)}`: vector of shocks across all constraints in scenario `s`, i.e. `\varepsilon^{(s)} = (\varepsilon_1^{(s)}, ..., \varepsilon_K^{(s)})^\top`.
- `\Sigma_{\varepsilon}`: covariance matrix of residual shocks across constraints (`K x K`).
- `\mathcal{N}(0, \Sigma_{\varepsilon})`: multivariate normal distribution with zero mean vector and covariance `\Sigma_{\varepsilon}`.
- `r_{k,t}`: historical residual for constraint `k` at time `t`.
- `r_t`: residual vector across constraints at time `t`, i.e. `(r_{1,t}, ..., r_{K,t})^\top`.
- `\operatorname{Cov}(\cdot)`: covariance operator.
- `I`: identity matrix.
- `L`: Cholesky factor such that `\Sigma_{\varepsilon} = L L^\top`.
- `z^{(s)}`: standard normal vector draw for scenario `s`, with distribution `\mathcal{N}(0, I)`.
- `^\top`: transpose operator.

## Core Scenario Equation
For each scenario `s` and each constraint `k`:

$$
\mu_k^{(s)} = \hat{\mu}_{k,\text{calibrated}} + \varepsilon_k^{(s)}
$$

Interpretation of terms:
- Left-hand side `\mu_k^{(s)}` is the scenario outcome we want to simulate.
- `\hat{\mu}_{k,\text{calibrated}}` is the best point forecast before randomness.
- `\varepsilon_k^{(s)}` adds uncertainty around that forecast.

This is an additive error model: `realization = baseline + error`.

## Residual Construction (How Errors Are Learned)
Historical residuals are computed as:

$$
r_{k,t} = \mu_{k,t}^{\text{real}} - \hat{\mu}_{k,t}^{\text{calibrated}}
$$

Notation details:
- `\mu_{k,t}^{\text{real}}`: realized historical shadow price.
- `\hat{\mu}_{k,t}^{\text{calibrated}}`: model estimate that would have been available for that historical time.
- `r_{k,t}`: miss of the calibrated model at that time.

If residuals are unbiased, their average is near zero. Their spread and co-movement determine simulation uncertainty.

## Cross-Constraint Dependence
Residual vectors are stacked by time:

$$
r_t = (r_{1,t}, r_{2,t}, ..., r_{K,t})^\top
$$

Then covariance is estimated:

$$
\Sigma_{\varepsilon} = \operatorname{Cov}(r_t)
$$

How to read `\Sigma_{\varepsilon}`:
- Diagonal entry `(k,k)`: variance of residuals for constraint `k`.
- Off-diagonal `(i,j)`: covariance between residuals of constraints `i` and `j`.
- Positive off-diagonal: constraints tend to shock in same direction.
- Negative off-diagonal: constraints tend to shock in opposite directions.

This is crucial because independent shocks would understate or distort portfolio risk when constraints move together.

## Scenario Sampling Mechanics
Assume baseline residual law:

$$
\varepsilon^{(s)} \sim \mathcal{N}(0, \Sigma_{\varepsilon})
$$

A practical way to sample:
1. Draw standard normal vector

$$
z^{(s)} \sim \mathcal{N}(0, I)
$$

2. Factor covariance matrix

$$
\Sigma_{\varepsilon} = L L^\top
$$

3. Transform draw

$$
\varepsilon^{(s)} = L z^{(s)}
$$

Why this works:
- `\mathbb{E}[\varepsilon^{(s)}] = 0` because `\mathbb{E}[z^{(s)}] = 0`.
- `\operatorname{Cov}(\varepsilon^{(s)}) = L I L^\top = L L^\top = \Sigma_{\varepsilon}`.

So the sampled shocks have exactly the target covariance by construction.

## End-to-End Monte Carlo Procedure
1. Compute calibrated baseline `\hat{\mu}_{k,\text{calibrated}}` for all constraints.
2. Build historical residual matrix from realized minus calibrated values.
3. Estimate `\Sigma_{\varepsilon}` from residuals.
4. For `s = 1, ..., M`:
- Draw `\varepsilon^{(s)} \sim \mathcal{N}(0, \Sigma_{\varepsilon})`.
- Form `\mu_k^{(s)} = \hat{\mu}_{k,\text{calibrated}} + \varepsilon_k^{(s)}` for each `k`.
5. Use each scenario vector `\mu^{(s)}` downstream to compute path values, portfolio PnL samples, and risk statistics (for example VaR/CVaR).

## Statistical Meaning of `M = 2000`
- `M` controls Monte Carlo sampling error.
- Higher `M` improves stability of estimated tails and quantiles, but increases compute cost.
- `M = 2000` is often a practical starting point for daily/auction analytics; tail-sensitive risk reports may need larger `M`.

## Baseline Assumptions and Limits
What this baseline assumes:
- Residual shocks are approximately Gaussian.
- Covariance structure estimated from history is representative for future conditions.
- Additive error form is adequate.

Potential issues:
- Heavy tails and regime shifts can make Gaussian tails too light.
- Covariance can be unstable if history is short or structurally changing.
- Extreme event dependence may be nonlinear and not fully captured by covariance alone.

## Practical Validation Checks
- Covariance PSD check: ensure `\Sigma_{\varepsilon}` is positive semidefinite.
- Backtest residuals: compare simulated error quantiles with empirical residual quantiles.
- Stability check: re-estimate `\Sigma_{\varepsilon}` over rolling windows and monitor drift.
- Sensitivity check: compare risk outputs under MVN vs heavier-tail alternatives.

## Summary
The equation

$$
\mu_k^{(s)} = \hat{\mu}_{k,\text{calibrated}} + \varepsilon_k^{(s)}, \quad \varepsilon^{(s)} \sim \mathcal{N}(0, \Sigma_{\varepsilon})
$$

is a standard, coherent baseline Monte Carlo design. It combines a calibrated deterministic view with correlated stochastic errors learned from historical residual behavior.
