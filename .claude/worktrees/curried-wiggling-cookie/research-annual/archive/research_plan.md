# Annual Pricing Research Plan (Execution Version)

## 1. Purpose
Validate and finalize annual MCP baseline + bid-price bands for the v1/f0p architecture.

Target pipeline (unchanged):
`baseline (mcp_pred) -> 10 bid-price bands -> clearing_prob -> optimizer`

Companion design doc: `stage2_design.md`.

---

## 2. MISO Annual Structure (Corrected)

**3 rounds × 4 quarters = 12 clearing events per PY.**

|       | aq1 (Jun-Aug) | aq2 (Sep-Nov) | aq3 (Dec-Feb) | aq4 (Mar-May) |
|-------|:---:|:---:|:---:|:---:|
| **R1** | no MTM | no MTM | no MTM | no MTM |
| **R2** | R1's MCP | R1's MCP | R1's MCP | R1's MCP |
| **R3** | R2's MCP | R2's MCP | R2's MCP | R2's MCP |

- Rounds are sequential time events (~Apr 8, ~Apr 22, ~May 5).
- Quarters are delivery periods. Each round auctions ALL 4 quarters simultaneously.
- R1 is the hard case: no market signal for any quarter → baseline from historical data only.

---

## 3. Status Snapshot (as of 2026-02-11)

### Completed
- Code-path audit for annual pricing in v1.
- Data source inventory for annual cleared trades, MCP, and DA congestion.
- Evaluation methodology selected (pooled LOYPO CV).
- Key architecture risks identified:
  - annual `aq1..aq4` not wired in v1 band generator
  - class_type routing risk in training-data lookup
  - annual scaling must be validated to occur exactly once in active path
  - `aq4` month coverage weakness in current `_fill_mtm` month filtering
- Baseline formula research: H/C/R signals defined, R1/R2/R3 formulas proposed (`baseline_formula/report.md`).
- Round count confirmed: MISO = 3 rounds (verified from data PY 2019-2025 + code).
- Repo cleaned for consistent 3-round terminology.

### Not Completed
- No quantitative backtest yet for hypotheses H1-H8 in `stage2_design.md`.
- No annual training parquet generated yet for band calibration.
- No production-ready implementation patch yet.

---

## 4. Locked Decisions

1. Keep f0p/v1 structure unchanged.
2. For annual initial rollout, use rule-based band widths (f2p-style), not LightGBM residual path.
3. Treat current `_fill_mtm` (1-year + 0.85 shrinkage) as control baseline.
4. Use pooled evaluation by `(round, period_type, class_type)` with path-tier stratification.
5. Use leave-one-planning-year-out CV for temporal validity.

---

## 5. Remaining Work Plan

## Phase A: Data Build (blocker for modeling)

### A1. Annual training dataset generation
- Build training parquet for `aq1..aq4` with required columns:
  - `source_id`, `sink_id`, `path`, `period_type`, `class_type`
  - `mtm_1st_mean`, `mtm_2nd_mean`, `mtm_3rd_mean`
  - `mcp_mean`
  - optional revenue proxy columns
- Save to the same hive layout used by v1 band generator.

### A2. `C` proxy table build
- Compute prior-PY clearing proxy with explicit key hierarchy:
  1. `(source_id, sink_id, period_type, class_type, round)`
  2. fallback drop `round`
  3. fallback drop `class_type`
  4. no-`C`
- Output coverage report by key level.

---

## Phase B: Baseline Evaluation

### B1. Baseline variants
- Evaluate variants defined in `stage2_design.md` (H-only, H+C, M+H by round).
- Include proposed round-aware formulas as candidates, not assumptions.

### B2. Evaluation protocol
- LOYPO CV across available planning years.
- Metrics:
  - MAE, median AE, directional accuracy, bias
- Reporting dimensions:
  - `round x period_type x class_type`
  - liquidity tier split

### B3. Acceptance gate
- Promote only if it beats control (`Current_1y`) with stable gains across folds.

---

## Phase C: Band Calibration

### C1. Rule-based width calibration
- Fit residual-width quantiles by `|mtm|` bins.
- Tune round multipliers around prior values.

### C2. Coverage diagnostics
- Check empirical coverage at 50/70/90/95.
- Check asymmetry/skew and whether symmetric bands remain adequate.

### C3. Safety checks
- No double-scaling for annual products.
- Correct class_type partition usage during training-data load.

---

## Phase D: End-to-End Trading Validation

- Simulated clear rate, PnL, credit requirement, concentration metrics.
- Out-of-time holdout (latest PY).
- Sensitivity to shrinkage, H weights, and width multipliers.

---

## 6. Deliverables

1. `annual_baseline_eval.csv` (all folds, all variants, all strata)
2. `annual_c_proxy_coverage.csv`
3. `annual_band_coverage_report.md`
4. `annual_e2e_backtest_report.md`
5. Implementation diff proposal (files + exact config updates)

---

## 7. Appendix: Completed Findings (Condensed)

### Code architecture findings
- `band_generator` currently supports `f0,f1,f2,f3,q2,q3,q4`; annual `aq1..aq4` not routed.
- `BASELINE_CONFIG` missing annual period types.
- Active v1 path requires annual scaling in band-generator branch to avoid missing scaling.
- class_type must be passed correctly to avoid offpeak using onpeak training data.

### Data findings
- Existing training parquet tree currently has non-annual period types only.
- Raw annual cleared/MCP data exists across many planning years, sufficient for pooled CV.
- DA monthly congestion source exists and is already used by `_fill_mtm` logic.

### Modeling findings
- Current annual R1 proxy logic is deterministic DA-based fill with directional shrinkage.
- `aq4` month filtering in current logic can under-cover delivery window and needs explicit validation/handling.

