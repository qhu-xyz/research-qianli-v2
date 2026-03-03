# Annual FTR Pricing & Bidding Strategy — Full Pipeline Design

## 1. Problem Statement & Objective Function

### What We Solve

Given MISO's annual FTR auction, determine which source-sink paths to bid, at what prices, and at what volumes to maximize risk-adjusted expected profit under capital, concentration, and tail-risk constraints.

### Formal Setup

Let the candidate path universe be P = {p₁, ..., pₙ}. For each path p:

- **Vₚ** = estimated annual congestion revenue ($/MW) — the fair value we model
- **cₚ** = auction clearing price ($/MW) — what we pay
- **qₚ** = our bid volume (MW) on path p
- **K** = total available collateral capital

Per-path expected profit:

```
πₚ = qₚ · (Vₚ − cₚ)
```

Portfolio profit:

```
Π = Σₚ qₚ · (Vₚ − cₚ)
```

### Optimization Objective

```
max   E[Π] − λ · CVaR_α(Π)

subject to:
  (C1)  Σₚ collateral(qₚ, cₚ) ≤ K              [capital/collateral limit]
  (C2)  |qₚ| ≤ Q_max_p                           [per-path position limit]
  (C3)  Σ_{p∈Gₖ} |qₚ · ΔSFₚ,ₖ| ≤ Lₖ   ∀k      [constraint-group concentration]
  (C4)  CVaR₉₉(Π) ≥ −StressLimit                 [hard tail limit]
  (C5)  Σₚ 1[qₚ≠0] ≤ N_max                       [max active positions]
```

Where:

- **λ** = risk-aversion parameter, tuned to target a Sharpe or return-on-capital threshold
- **α** = CVaR confidence level (95% primary, 99% stress)
- **Gₖ** = constraint group — paths sharing dominant binding constraint k; concentration-limited together

### Four Estimation Problems

The pipeline must solve:

1. **Vₚ** — forward congestion value (the DC-OPF valuation engine)
2. **Distribution of Vₚ** — variance and tail scenarios, not just the mean (the risk model)
3. **cₚ** — where the auction clears (auction intelligence / competitive dynamics)
4. **Cross-path covariance** — how path outcomes co-move via shared constraint exposure

---

## 2. Valuation Engine — Constraint-Level Congestion Decomposition

### Core Insight

A path's congestion value is not a property of the path itself — it is the sum of contributions from every binding transmission constraint, weighted by how much each constraint affects that path.

### Fundamental Identity

For path p = (source s, sink d), the congestion rent in hour h:

```
Vₚ(h) = Σₖ μₖ(h) · ΔSFₚ,ₖ
```

Where:

- **μₖ(h)** = shadow price of constraint k in hour h ($/MW)
- **ΔSFₚ,ₖ** = PTDF(d, k) − PTDF(s, k) — the path's shift factor differential on constraint k

Annual value:

```
Vₚ = Σₕ Vₚ(h)    [summed across all 8,760 hours]
```

### Why Constraint-Level Decomposition

We do not model paths directly — we model constraints. Every path value falls out as a linear combination. This gives us:

- **Explainability**: we know exactly which constraints drive each path's value
- **Efficiency**: N constraints × M scenarios is far smaller than P paths × M scenarios
- **Robustness**: calibrating at the constraint level prevents path-level overfitting

### Two Modeling Problems Per Constraint k

**Problem 1 — Binding probability**: P(bind)ₖ(h) — will constraint k bind in hour h?

Depends on:

- Hour type (peak / off-peak / shoulder)
- Season (summer / winter / spring / fall)
- Load forecast percentile for the zone
- Scheduled generation outages near the constraint
- Planned topology changes (new lines, upgrades, retirements)

**Problem 2 — Shadow price severity**: E[μₖ | bind, h] — when it binds, how expensive?

Depends on:

- Generation cost differential across the constraint
- Magnitude of flow violation (how far over the limit demand wants to push)
- Available redispatch capacity (cheap generation on the constrained side)

The DC-OPF solver (Section 3) solves both simultaneously.

---

## 3. DC-OPF Solver Specification

### Formulation

The solver dispatches generation to meet load at minimum cost, subject to transmission limits. Shadow prices on binding constraints are the primary outputs.

```
min   Σᵢ MCᵢ · gᵢ                             [minimize total dispatch cost]

subject to:
  Σᵢ gᵢ = Σₙ Dₙ                               [system power balance]
  Fₖ = Σₙ PTDF(n,k) · Iₙ          ∀k           [flow on constraint k]
  |Fₖ| ≤ Fₖ_max                    ∀k           [transmission limits → dual = μₖ]
  gᵢ_min ≤ gᵢ ≤ gᵢ_max            ∀i           [generator capacity bounds]
```

Where:

- **MCᵢ = heat_rateᵢ × fuel_priceᵢ + VOMᵢ** — marginal cost of generator i
- **Iₙ = Σᵢ∈n gᵢ − Dₙ** — net injection at node n
- **Fₖ** — flow on monitored constraint k
- **μₖ** — the dual variable on the transmission limit constraint (the shadow price we need)

### Scenario Design

We do not solve 8,760 individual hours. We define representative scenario-hours:

```
Scenarios = Hour_Types × Load_Percentiles × Outage_States

  Hour_Types:        {summer_peak, summer_offpeak, winter_peak, winter_offpeak,
                      shoulder_peak, shoulder_offpeak}           → 6 types
  Load_Percentiles:  {P10, P30, P50, P70, P90}                  → 5 levels
  Outage_States:     {base, high_outage, critical}               → 3 states
```

This yields **90 scenario-hours**, each solved as an independent DC-OPF. Each scenario s has a probability weight wₛ derived from historical hour-count frequencies and outage probability distributions.

### Aggregation to Annual Values

```
E[μₖ] = Σₛ wₛ · μₖ(s)

E[Vₚ] = Σₛ wₛ · Σₖ μₖ(s) · ΔSFₚ,ₖ
```

### Required Data Inputs

| Input | Source | Format |
|-------|--------|--------|
| PTDF matrix | MISO auction network model (published pre-auction) | Sparse matrix, nodes × constraints |
| Constraint list + thermal limits | MISO OASIS, filtered to historically relevant + new monitored elements | Table with constraint ID, limit MW, voltage class |
| Generation fleet | EIA-860/923, MISO generator registrations | Table with unit ID, node, capacity, fuel type, heat rate, VOM |
| Fuel price forwards | Henry Hub + regional basis (gas), coal by basin, from broker curves | Time series by month |
| Zonal load forecasts | MISO seasonal load forecast, shaped to hourly profiles | Hourly by zone |
| Outage schedules | MISO planned outage data + probabilistic forced outage rates (EFORd) | Table with unit ID, outage windows, forced outage probability |

---

## 4. Calibration Framework — Model-to-Realized Alignment

### Purpose

Raw DC-OPF output will not match realized congestion. The calibration layer corrects systematic biases at the constraint level.

### Why Constraint-Level Calibration

Paths are linear combinations of constraints. Calibrating constraints automatically calibrates all paths. This is more stable and avoids path-level overfitting.

### Step 1 — Historical Backcast

Run the DC-OPF for the prior 2–3 planning years using realized load, fuel prices, and outage data. For each constraint k and historical month m, compare:

```
Raw model output:    μ̂ₖ(m) = modeled monthly congestion contribution
Realized value:      μₖ(m) = actual DAM shadow price from MISO settlements
```

### Step 2 — Bias Estimation

For each constraint k, fit a calibration function:

```
μₖ_calibrated = αₖ + βₖ · μ̂ₖ + εₖ

where:
  αₖ  = additive bias (captures systematic under/over-prediction)
  βₖ  = scaling factor (captures severity misestimation)
  εₖ  = residual (should be zero-mean; its variance feeds the risk model)
```

Constraints with fewer than 12 binding months use **pooled/hierarchical calibration** — borrow strength from similar constraints grouped by voltage class and region to avoid overfitting.

### Step 3 — Residual Diagnostics

For each constraint, check:

- **Stationarity**: non-stationarity signals a structural topology change → flag for manual review
- **Autocorrelation**: persistence means the model is missing a systematic driver
- **Tail behavior**: fat-tailed εₖ distributions inform the risk model's tail assumptions

### Step 4 — Forward Application

Apply the calibration function to forward-looking DC-OPF outputs:

```
Vₚ_calibrated = Σₖ (αₖ + βₖ · μ̂ₖ_forward) · ΔSFₚ,ₖ
```

### Calibration Discipline

- Parameters estimated on a **rolling 24-month window**, validated on a **12-month holdout**
- If out-of-sample R² per constraint drops below **0.3**, that constraint is flagged as "low-confidence"
- Low-confidence constraint contributions are **haircut by 50%** in the path valuation passed to the bid optimizer

---

## 5. Path Screening & Multi-Factor Signal Scoring

### Universe Definition

All MISO pricing nodes as sources and sinks, filtered to:

- Nodes with non-trivial historical congestion (|avg annual congestion component| > $0.50/MWh)
- Commercially relevant hub-to-hub, hub-to-load-zone, and generator-to-hub paths
- Typical yield: **2,000–5,000 candidate paths**

### Signal Factors

Each path p receives a score on each factor, normalized to [0, 1]:

| Signal | Definition | Weight |
|--------|-----------|--------|
| **S1: Forward model value** | Vₚ_calibrated from the DC-OPF engine | 0.30 |
| **S2: Historical consistency** | Fraction of last 36 months where path congestion rent was positive | 0.15 |
| **S3: Model confidence** | Weighted avg R² of the calibration on path p's dominant constraints | 0.15 |
| **S4: Value-to-clearing spread** | (Vₚ_calibrated − last auction clear) / Vₚ_calibrated — the "edge" | 0.20 |
| **S5: Fundamental catalyst** | Binary/graded flags for topology changes, gen retirements, new load near path nodes | 0.10 |
| **S6: Constraint crowding** | Inverse of how many other high-value paths share the same dominant constraint — penalizes crowded trades | 0.10 |

### Composite Score

```
Scoreₚ = Σⱼ wⱼ · Sⱼ(p)
```

### Selection Rule

Rank all paths by composite score. Select the top N paths where N is determined by:

- Portfolio optimizer capacity (typically **200–500 paths**)
- Minimum score threshold: **Scoreₚ ≥ 0.40**
- **Diversification floor**: at least 3 distinct dominant constraints represented in the top N

### Directional Filter

For each path, the scorer determines whether to bid **prevailing direction** (buy the historically positive flow) or **counterflow** (bet against it).

Counterflow bids are flagged when:

- S4 is negative (market is pricing the path above our fair value in the prevailing direction)
- S5 indicates a structural reversal catalyst (e.g., new generation on the historically import-constrained side)

Paths passing the screen move into the portfolio optimizer with their full valuation distributions.

---

## 6. Risk Model — Scenario-Based Distribution & CVaR

### Purpose

Transform point estimates into a full P&L distribution. Answer: "How bad can this portfolio get in the bottom 5% of outcomes?"

### Scenario Generation

Generate **M = 2,000** Monte Carlo scenarios by perturbing constraint-level shadow prices. For each scenario s and constraint k:

```
μₖ(s) = μ̂ₖ_calibrated + εₖ(s)
```

Where εₖ(s) is drawn from the calibration residual distribution of constraint k.

### Cross-Constraint Correlation

Residuals are correlated across constraints. We estimate the cross-constraint correlation matrix Σ_ε from historical residuals, then draw correlated perturbations:

```
ε(s) ~ MVN(0, Σ_ε)    [multivariate normal as baseline]
```

For fat-tailed constraints (identified in calibration diagnostics): replace the marginal with a **Student-t distribution** using a Gaussian copula — normal correlation structure, heavy-tailed marginals.

### Structural Stress Overlays

Deterministic scenarios added on top of Monte Carlo draws:

| Stress Scenario | Specification |
|----------------|---------------|
| Extreme load high | +15% annual load deviation from forecast |
| Extreme load low | −10% annual load deviation from forecast |
| Fuel shock up | Gas price +40% from forward curve |
| Fuel shock down | Gas price −40% from forward curve |
| Major forced outage | Largest 3 generators on each top-10 constrained interface tripped simultaneously |
| Topology surprise | Unplanned transmission derate on each top-10 binding constraint |

These add **~50 stress scenarios**, weighted at a combined **5% probability mass**.

### Portfolio P&L Distribution

For each scenario s:

```
Πₛ = Σₚ qₚ · (Vₚ(s) − cₚ)

where Vₚ(s) = Σₖ μₖ(s) · ΔSFₚ,ₖ
```

### Risk Metrics

| Metric | Definition | Role |
|--------|-----------|------|
| **VaR₉₅** | 5th percentile of Π distribution | Reporting |
| **CVaR₉₅** | Mean of Π below VaR₉₅ | Primary risk constraint in optimizer |
| **CVaR₉₉** | Mean of Π below 1st percentile | Stress limit — hard ceiling |
| **Max drawdown** | Worst single scenario | Must not exceed 2× CVaR₉₅ |

### Marginal Risk Contribution

The key input to the portfolio optimizer for sizing decisions:

```
MCVaRₚ = ∂CVaR₉₅ / ∂qₚ ≈ E[Vₚ(s) − cₚ | Πₛ ≤ VaR₉₅]
```

Interpretation:

- "If I add 1 MW to path p, how much does portfolio tail risk increase?"
- Paths with high expected value but high MCVaR get **sized down**
- Paths with positive expected value and **negative MCVaR** (natural hedges) get **sized up**

---

## 7. Portfolio Optimizer

### Full Formulation

With all terms now defined:

```
max   Σₚ qₚ · (Vₚ_cal − ĉₚ) − λ · CVaR₉₅(Π)

subject to:
  (C1)  Σₚ collateral(qₚ, ĉₚ) ≤ K                    [capital limit]
  (C2)  |qₚ| ≤ Qₘₐₓ,ₚ                                 [per-path MW cap]
  (C3)  Σ_{p∈Gₖ} |qₚ · ΔSFₚ,ₖ| ≤ Lₖ   ∀k            [constraint-group concentration]
  (C4)  CVaR₉₉(Π) ≥ −StressLimit                       [hard tail limit]
  (C5)  Σₚ 1[qₚ≠0] ≤ N_max                             [max active positions]
```

### Auction Clearing Price Estimate (ĉₚ)

Derived from last year's clearing adjusted for changes in modeled fair value:

```
ĉₚ = clear_prior_year_p · (Vₚ_cal_current / Vₚ_cal_prior)
```

Bounded by historical clearing-to-fair-value ratios. This is imperfect but essential — bidding at fair value without adjusting for expected clearing leaves money on the table.

### Collateral Model (C1 Detail)

For obligation FTRs in MISO:

```
collateral(qₚ, ĉₚ) = qₚ · max(ĉₚ, 0) + qₚ · credit_adder_p
```

Where **credit_adder_p** reflects potential negative settlement exposure, typically set to the historical P10 of monthly path settlement value.

### CVaR Linearization (Rockafellar-Uryasev)

CVaR is linearized into a standard LP:

```
CVaR₉₅ = min_ζ { ζ + (1 / (0.05 · M)) · Σₛ max(−Πₛ − ζ, 0) }
```

This converts the full problem into a **linear program** with M auxiliary variables (one per scenario). Solvable with standard LP solvers (CPLEX, Gurobi, or open-source HiGHS/CBC).

### Output

Optimal positions **{qₚ\*}** — the target MW volume for each path, passed to bid construction.

---

## 8. Bid Curve Construction — From Optimal Positions to Auction Orders

### Why Bid Curves, Not Single Points

MISO's auction accepts price-quantity bid curves, not single-point orders. Laddered bids provide robustness to clearing price uncertainty and improve expected fill quality.

### Bid Ladder Structure

For each path p with target qₚ\*, submit a stepped bid curve with 4 price-quantity tranches:

```
Tranche 1 (aggressive):    q₁ MW  @  price = Vₚ_cal · (1 − margin₁)
Tranche 2 (core):          q₂ MW  @  price = Vₚ_cal · (1 − margin₂)
Tranche 3 (value):         q₃ MW  @  price = Vₚ_cal · (1 − margin₃)
Tranche 4 (opportunistic): q₄ MW  @  price = Vₚ_cal · (1 − margin₄)

where:
  Σᵢ qᵢ = qₚ*
  margin₁ < margin₂ < margin₃ < margin₄
```

### Margin Calibration

Margins control aggressiveness, set per-path based on model confidence:

| Confidence Tier | R² Threshold | Margins [m₁, m₂, m₃, m₄] |
|----------------|-------------|---------------------------|
| High | R² > 0.6 | [0.05, 0.15, 0.25, 0.40] |
| Medium | R² 0.3–0.6 | [0.10, 0.20, 0.35, 0.50] |
| Low | R² < 0.3 | [0.15, 0.30, 0.45, 0.60] |

Logic: when confident in our valuation, bid closer to fair value. When uncertain, demand more edge.

### Volume Allocation Across Tranches

```
q₁ = 0.15 · qₚ*    [small aggressive slice — filled if market is cheap]
q₂ = 0.40 · qₚ*    [core position — expected to clear]
q₃ = 0.30 · qₚ*    [value tranche — clears only if market is favorable]
q₄ = 0.15 · qₚ*    [opportunistic — deep value only]
```

### Counterflow Handling

Counterflow paths (negative ΔSF on the dominant constraint) receive special treatment. MISO awards counterflow FTRs preferentially in the simultaneous feasibility test.

For counterflow bids:

- Margins reduced by 50% (bid more aggressively)
- Counterflow premium effectively lowers cost basis
- Volume allocation shifts toward tranches 1–2

### Pre-Submission Sanity Checks

- No single path exceeds **8%** of total portfolio capital
- Total bid capital (if all tranches clear) does not exceed **1.3×** available capital (allows for partial fills)
- Counterflow exposure does not exceed **25%** of total portfolio notional
- Every bid price is positive (no paying to take obligation FTRs unless explicitly flagged and approved)

---

## 9. Validation & Backtesting Framework

### Walk-Forward Backtest Design

Simulate what the model would have bid in past auctions using only information available at that time.

```
For each historical auction year Y in [Y-5 ... Y-1]:

  1. Training window:   data up to auction_date(Y) − 30 days
  2. Calibration:        fit on [Y-3 ... Y-1] realized data
  3. Forward model:      DC-OPF with fuel forwards and load forecasts AS OF auction_date(Y)
  4. Screen & score:     signal scorer generates candidate paths
  5. Optimize:           portfolio optimizer with capital/risk limits
  6. Construct bids:     generate bid curves
  7. Simulate clearing:  use actual auction clearing prices to determine fills
  8. Settle:             compute realized P&L using actual DAM congestion over delivery year Y
```

### Leakage Controls

| Leakage Type | Prevention |
|---|---|
| Future topology in PTDF matrix | Use only the network model published before auction_date(Y) |
| Future fuel prices | Snapshot forward curves as of auction_date(Y) − 30 days |
| Future outage info | Use only planned outage schedules filed before auction date |
| Calibration on test period | Strict temporal cutoff; calibration window ends before delivery year |
| Survivor bias in path universe | Include all paths that existed at auction time, including later-retired nodes |

### Benchmark Definitions

| Benchmark | Description |
|-----------|-------------|
| **B1: Naive historical** | Bid at trailing 3-year average congestion value, equal-weight portfolio |
| **B2: Auction clearing (market portfolio)** | P&L if you bought every path at its clearing price |
| **B3: Perfect foresight** | Upper bound using realized congestion (unachievable; calibrates expectations) |

### Performance Metrics & Acceptance Thresholds

| Metric | Definition | Threshold to Deploy |
|--------|-----------|---------------------|
| Annualized return on capital | Portfolio P&L / peak capital deployed | > B1 by ≥ 200 bps |
| Information ratio | (Return − B2 return) / tracking error vs B2 | > 0.5 |
| Hit rate | % of paths with positive P&L | > 55% |
| CVaR accuracy | Realized worst-5% outcomes within 1.5× predicted CVaR₉₅ | Must hold in ≥ 4 of 5 backtest years |
| Calibration R² (out-of-sample) | Avg constraint-level R² on holdout year | > 0.25 aggregate |
| Max drawdown ratio | Worst year P&L / avg year P&L | > −1.5 |

### Go/No-Go Decision Rule

- **All 6 thresholds met** across the 5-year backtest → **deploy at full capital**
- **4–5 met** → **deploy at 50% capital allocation**
- **Fewer than 4 met** → **do not deploy**; diagnose which stage is underperforming

---

## 10. Production Contracts — Inputs, Outputs, Versioning & Monitoring

### Module Contracts

| Module | Inputs | Outputs |
|--------|--------|---------|
| 1. Data Ingestion | MISO OASIS, EIA, fuel broker feeds, load forecasts | `ptdf_matrix.parquet`, `gen_stack.parquet`, `constraints.parquet`, `dam_lmps.parquet` |
| 2. DC-OPF Engine | All Stage 1 outputs, `scenario_defs.yaml` | `shadow_prices.parquet` (μₖ per scenario) |
| 3. Calibration | `shadow_prices`, historical DAM LMPs | `calibrated_values.parquet`, `residual_stats.json` |
| 4. Path Screening | `calibrated_values`, prior auction clears, catalyst flags | `ranked_paths.parquet` (top N with scores) |
| 5. Risk Model | `calibrated_values`, `residual_stats`, correlation matrix | `scenario_pnl.parquet`, `risk_metrics.json`, `mcvar_by_path.json` |
| 6. Portfolio Optimizer | `ranked_paths`, `risk_metrics`, `mcvar`, `capital_config.yaml` | `optimal_positions.parquet` |
| 7. Bid Construction | `optimal_positions`, confidence scores | `auction_bids.csv` (MISO upload format) |

### Versioning Protocol

Every auction run is fully reproducible via an immutable run manifest:

```yaml
# run_manifest.yaml
run_id:          "annual-2026-v3"
timestamp:       2026-03-15T14:30:00Z
git_commit:      <sha256 of pipeline code>
data_snapshot:   <sha256 hash of all Stage 1 outputs>
config:          <sha256 of all .yaml config files>
parameters:
  lambda:        0.15
  cvar_alpha:    0.95
  scenario_count: 2000
  capital_limit: <dollar amount>
```

All intermediate artifacts are written to immutable storage keyed by `run_id`. Any historical run can be replayed exactly by restoring the data snapshot and config.

### Production Monitoring KPIs

Tracked monthly during the delivery year:

| KPI | Computation | Alert Threshold |
|-----|-------------|-----------------|
| Realized vs predicted path value | Vₚ_realized(MTD) vs Vₚ_calibrated · (months elapsed / 12) | Deviation > 2σ of calibration residual |
| Constraint binding accuracy | Predicted binding hours vs actual binding hours per constraint | < 50% directional accuracy at month 6 |
| Portfolio mark-to-market | MTD realized congestion revenue − auction cost accrual | Cumulative P&L < CVaR₉₅ / 2 |
| Model drift | Rolling 3-month calibration R² on new realized data | R² drops below 0.20 for >2 consecutive months |
| Concentration exposure | Actual P&L contribution from single largest constraint group | > 40% of total portfolio P&L |

### Alert Escalation Protocol

| Level | Trigger | Action |
|-------|---------|--------|
| **Yellow** | 1 KPI breaches threshold | Log and review at next weekly meeting |
| **Orange** | 2 KPIs breach | Convene ad-hoc review, consider hedging overlay |
| **Red** | 3+ KPIs breach OR portfolio MTM exceeds CVaR₉₅ | Freeze new activity, activate risk reduction protocol |
