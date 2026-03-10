# Review of `draft-design.md` Against `input.md`

## Verdict
I disagree with the current draft by **~80%** relative to the stated bar. It is a reasonable architecture sketch, but it is not yet a reconstruction of an annual FTR pricing strategy pipeline at institutional quality.

## What Works
- The six-stage decomposition is directionally sensible.
- Data ingestion, path screening, valuation, risk, optimization, and analytics are the right high-level components.
- Using network fundamentals (PTDF/LODF/constraints) is structurally correct.

## Major Gaps (Why It Misses the Bar)
1. It is architecture prose, not strategy design
- The draft describes components but does not define model behavior, formulas, objectives, constraints, or decision rules.
- `input.md` asks for direct module-level reconstruction with practical details.

2. Valuation core is underspecified
- “Run DC-OPF” is not sufficient as a full pricing method.
- Missing calibration methodology from modeled congestion values to realized/cleared outcomes.
- Missing decomposition from path value into binding-constraint drivers for explainability and robustness checks.

3. No explicit auction execution logic
- Missing bid-curve construction (price/quantity laddering), aggressiveness controls, and capital-aware bidding mechanics.
- Missing how path-level estimates convert into auction-ready orders.

4. Risk is named but not designed
- “VaR/CVaR” appears as a label only; no implementation details, horizon assumptions, confidence levels, stress overlays, or risk budget linkage.
- No marginal risk contribution framework for path admission and sizing decisions.

5. No portfolio interaction framework
- Missing cross-path dependency treatment, concentration limits, and shared-constraint crowding penalties.
- Missing optimization objective for risk-adjusted return under portfolio and capital constraints.

6. No validation and governance layer
- Missing walk-forward backtesting design, leakage controls, benchmark definitions, and acceptance thresholds.
- Missing model/version control and reproducibility requirements for production reliability.

## Why This Matters
The current draft can guide discussion, but it cannot be implemented as a high-confidence, production-grade annual pricing module. It is a scaffold, not a strategy specification.

## Required Upgrade Criteria
A revised design should include all of the following before it can be considered close to the requested standard:
- Formal objective function for annual pricing and bidding.
- Constraint-level valuation decomposition and calibration loop.
- Concrete bid construction rules from fair value to executable auction orders.
- Portfolio optimization with explicit capital, concentration, and risk-budget constraints.
- Quantitative validation framework (walk-forward, benchmarks, go/no-go thresholds).
- Production contracts: inputs/outputs, versioning, reproducibility, and monitoring KPIs.

## Bottom Line
`draft-design.md` is acceptable as a **Section 1 architecture overview** only. It does not yet meet the “best-in-class, no-compromise” requirement in `input.md`.
