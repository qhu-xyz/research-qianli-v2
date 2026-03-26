# Annual FTR Bid Pricing — Stage 1 (Reset)

## Why We Scrapped the Previous Analysis

The previous round of research (archived in `archive/`) cataloged the codebase well but jumped to formula design prematurely:

1. **Wrong MISO round count.** Early analysis assumed 4 rounds. MISO annual has **3 rounds** (verified from `autotuning.py:54`, `annual_round_day_map`, and actual cleared trade data PY 2019-2025).

2. **Jumped to formulas before understanding the problem empirically.** Proposed weight blends (0.80*H + 0.20*C etc.) without measuring actual residual distributions, volume characteristics, or how the current production system performs. Priors without data.

3. **Didn't study f0p pricing first.** Understanding the contrast is essential to knowing what annual-specific changes are needed and which parts of v1 can be reused as-is.

4. **Didn't cleanly separate the round × quarter grid.** Mixed round-level questions (R1 has no MTM) with quarter-level questions (aq4 has data sparsity) without a unified framework.

**What was worth keeping:** The archived `learning.md` has verified code-line references for the entire v1 pipeline and remains a valid reference.

---

## MISO Annual Structure (Ground Truth)

**3 rounds × 4 quarters = 12 clearing events per PY.**

|       | aq1 (Jun-Aug) | aq2 (Sep-Nov) | aq3 (Dec-Feb) | aq4 (Mar-May) |
|-------|:---:|:---:|:---:|:---:|
| **R1** (~Apr 8) | no MTM | no MTM | no MTM | no MTM |
| **R2** (~Apr 22) | R1's MCP | R1's MCP | R1's MCP | R1's MCP |
| **R3** (~May 5) | R2's MCP | R2's MCP | R2's MCP | R2's MCP |

- Each round auctions ALL 4 quarters simultaneously.
- R1 is the hard case: no market clearing price for any quarter.
- R2 and R3 have `mtm_1st_mean` = previous round's MCP for that quarter.
- PJM has 4 rounds and has MTMs for all rounds including R1.

---

## f0p Pricing — What We're Contrasting Against

### f0p baseline formula
```
baseline = mtm_weight * mtm_1st_mean + rev_weight * 1(rev)
```

Weights by period type (from `band_generator.py:121-129`):
| Period | mtm_weight | rev_weight |
|--------|-----------|-----------|
| f0 | 0.77 | 0.23 |
| f1 | 0.85 | 0.15 |
| f2 | 0.94 | 0.06 |
| f3 | 0.93 | 0.07 |
| q2-q4 | 0.91-0.93 | 0.07-0.09 |

Key: as delivery gets further from submission, revenue weight drops and MTM dominates.

### f0p band generation
- **f0/f1:** LightGBM residual model + conformal calibration. Trains on prior months' data, predicts |residual|, then scales by conformal scalar per coverage target.
- **f2/f3/q2/q3/q4:** Rule-based binning. Bins by |mtm| (tiny<50, small<250, medium<1000, large). Computes width quantiles per bin from training residuals. Symmetric bands: `baseline ± width`.
- **10 bid levels** from 5 band pairs: upper/lower at 99, 70, 50, 30, 10.
- **Clearing probabilities:** Hybrid empirical + rule-based (v6). Empirical from 3-month lookback; rule-based fallback where empirical is unstable.
- **Width caps:** Low segment p90 cap (hard 1000), high segment p99 cap (hard 3000).
- **Quarterly scaling:** q2/q3/q4 bid_price × 3.

### Annual vs f0p — honest comparison

| Dimension | f0p | Annual | Risk of wrong assumption |
|-----------|-----|--------|--------------------------|
| Data volume | Monthly auctions → thousands of training rows | 3 rounds/year → very few per path | Annual can't use ML; but also can't robustly estimate per-bin quantiles |
| Baseline signal strength | `mtm_1st_mean` = recent clearing (days/weeks old, strong signal) | R1: H = year-old DA congestion (weak signal). R2/R3: previous round's MCP (decent signal) | **R1 residuals could be LARGER than f0p, not smaller.** Weaker signal → worse prediction → wider bands needed, not narrower |
| Revenue signal | Same month → strong for f0 | April revenue for Jun-May delivery → wrong season | R is likely useless for annual |
| Volume per trade | Smaller MW per period | Believed much higher MW | **UNVERIFIED — need to check data** |
| Band width needed | p95 ~500, p99 wider (user claim, width caps at 1000/3000) | Expected narrower (user belief) | **UNVERIFIED.** Market may be "stable" but prediction error may be large due to weak signals |

**[CRITICAL]** The claim "annual bands should be narrower" conflates two things:
- **Market stability** (how much MCPs move between rounds/years) — may be true
- **Prediction difficulty** (how large is |MCP - baseline|) — may actually be *worse* for annual R1 because the baseline signal (H) is much weaker than f0p's MTM

Whether bands should be narrower or wider is an **empirical question** that Phase 2 must answer.

---

## Known Structural Issues (from archived analysis, still valid)

### aq4 data sparsity
`_fill_mtm` cutoff is ~April of PY year. For aq4 (Mar-May Y+1), looking back 1 year: Mar Y, Apr Y, May Y. Only Mar Y passes cutoff. Apr Y and May Y are filtered out. So aq4 H is built from **1 month instead of 3** — structurally weaker proxy than aq1-aq3.

### `aq1-aq4` not wired in band_generator
- `BASELINE_CONFIG` (line 121) doesn't include aq1-aq4 → `KeyError`
- `QUARTERLY_PTYPES` (line 274) only has q2/q3/q4 → annual won't get ×3 scaling
- `generate_bands_for_group` (line 1212) raises `ValueError` for unknown ptypes
- These need code changes before annual can use the v1 band pipeline

### Production system for annual
Previous analysis found PY 2025 annual was run through **ftr23 legacy**, not ftr24/v1. This means:
- The "current production performance" to beat may be ftr23's approach, not `_fill_mtm`
- Need to verify what system will be used for upcoming PY
- **[TODO: verify current status]**

---

## Open Questions for This Stage

### Q1. Does R1 need separate baselines per quarter?

The formula structure can be the same (all use H), but H values are quarter-specific (different delivery months). The real question is: **should the formula _weights_ differ by quarter?**

Arguments for same weights:
- Only 3 data points/year per path per quarter. Estimating separate weights per quarter from this is noisy.
- Simpler is more stable with limited data.

Arguments for different weights:
- aq1 (summer peak) has very different congestion patterns than aq3 (winter) or aq4 (spring).
- aq4 has structural data sparsity (1 month vs 3) — may need heavier multi-year averaging or different fallback.
- Prediction difficulty may vary by quarter.

**This is an empirical question.** Phase 2 should report residual stats by quarter to determine if quarters need distinct treatment.

### Q2. What about R2 and R3?

R2/R3 have `mtm_1st_mean` = previous round's MCP. Tempting to say "same as f2p."

**But the training data problem is different:**
- f2p uses monthly training data pooled across f2/f3/q2/q3/q4 — thousands of rows per auction month per class_type.
- Annual R2/R3 would need to train on prior PY annual clearings. That's ~paths × (1 year × 1 round) = much fewer rows.
- The `compute_bin_widths` function splits into 4 bins by |mtm| size. With sparse annual data, bins could be nearly empty.
- The empirical clearing probability needs `MIN_ROWS_FOR_EMPIRICAL = 100` per flow_type — annual might not meet this.

**R2/R3 cannot naively reuse f2p.** Options:
1. Pool R2/R3 annual training data across quarters (aq1-aq4) and years to get more rows
2. Pool annual + f0p quarterly training data (risky: different market dynamics)
3. Use rule-based fallback clearing probs instead of empirical
4. Use a simpler band approach entirely (e.g., fixed width percentages of |baseline|)

### Q3. How narrow should annual bands be?

**Cannot assume narrower a priori.** Steps:
1. Compute residuals for historical annual trades: `residual = mcp_mean - baseline`
2. Report distribution of |residual| by (round × quarter × class_type)
3. Compare to f0p |residual| distributions
4. If annual |residual| < f0p → narrower bands justified
5. If annual |residual| ≥ f0p → wider bands needed (opposite of intuition)

### Q4. What is the actual volume difference between annual and f0p?

**UNVERIFIED.** Steps:
1. Load cleared trades for PY covering both annual and f0p
2. Compare total MW, per-path MW, number of paths by period_type
3. This determines how important the annual pricing problem is relative to f0p

### Q5. What is the current production approach and its performance?

Need to determine:
- Is the upcoming PY using ftr24/v1 or ftr23 legacy for annual?
- What was the realized PnL/performance of last year's annual positions?
- What is the baseline to beat?

---

## Research Plan

### Phase 1: Data exploration — COMPLETED
**Goal:** Get facts, not assumptions. Answer Q4 and Q5.
**Status:** Done. See `findings_data_profile.md` for full results.

Key findings:
- 11.5M annual rows across PY 2019-2025, all 3 rounds x 4 quarters well-populated
- Annual avg MW/row = 4.24 vs f0p = 2.95 (1.44x ratio, moderate not dramatic)
- Annual MCPs are ~3-4x larger than f0 MCPs (median ~100+ vs ~20-35)
- Market participation growing: PY 2025 has ~2x the rows of PY 2019
- Submitted bid stacks not yet extracted (requires separate notebook analysis)

### Phase 2: Residual analysis — COMPLETED
**Goal:** Answer Q1 and Q3 — how hard is the prediction problem?
**Status:** Done. See `findings_residuals.md` for full results.

Key findings:
- **R1 residuals are MASSIVE:** mean |res| = 849, p95 = 3,300, p99 = 7,346. H is a very weak baseline.
- **R2/R3 residuals are moderate:** mean |res| = 157-191, p95 = 568-693. Previous round MCP is a decent baseline.
- **R1 is 6.7x worse than f0p** (p95: 3,300 vs 492). R2/R3 are 15-40% worse than f0p.
- **Direction accuracy:** R1 = 62-67% (near random for aq4), R2/R3 = 90-93%.
- **aq4 is NOT harder by magnitude** (actually slightly easier than aq2). But aq4 has worse direction accuracy (62% vs 67%).
- **R2/R3 bin feasibility confirmed:** 35k-92k rows per bin per PY. f2p-style binning works.
- **Positive bias throughout:** MCPs tend to clear above baseline in all rounds.

### Phase 3: Baseline formula design — REVISED PLAN
**Goal:** Design baseline given Phase 2 findings.

The data fundamentally changes Phase 3 approach:

**R1 is the critical problem.** H (historical DA congestion) alone gives p95 errors of 3,300 — far too large for useful bidding. Options:
1. Accept wide R1 bands and focus on getting the center right (improve H with multi-year averaging, Dayzer signals, or constraint exposure adjustments)
2. Test whether adding C (constraint exposure signal) or Dayzer reduces R1 residuals
3. Consider that R1 may not be meaningfully improvable — the prediction problem is inherently hard when the baseline is a year-old proxy
4. Focus pricing effort on R2/R3 where baseline is strong

**R2/R3 are tractable.** Previous round's MCP provides p95 ~600-700 residuals — similar magnitude to f0p.
- Use M (previous round MCP) as primary baseline signal, similar to f2p
- Test whether blending M with H improves (hypothesis: probably not, M dominates)
- f2p-style binning by |M| is confirmed feasible

**Quarter-level treatment:**
- Same formula structure for all 4 quarters is justified — residual magnitudes are similar
- aq4 may deserve a direction-correction adjustment or wider symmetric bands to compensate for its lower direction accuracy
- aq2 has the largest residuals — consider slightly wider bands

Concrete steps:
1. For R1: test H-only baseline vs multi-year-averaged H vs H + additional signals
2. For R2/R3: test M-only vs M + H blend. Expect M-only wins.
3. Report residual improvement (if any) from each variant
4. Decide on per-quarter vs unified formula

### Phase 4: Band calibration — REVISED PLAN
**Goal:** Set bid price bands and clearing probabilities for annual.

The data enables a clearer strategy:

**R1 bands:** Must be very wide. p95 residual ~3,300 means the 95% band needs to be ~3,300 around baseline. This is 6-7x wider than f0p. Options:
1. Rule-based: fixed width as function of |H| (percentage-based scaling)
2. Quantile-based: compute historical p90/p95/p99 residual widths from pooled R1 data
3. Hybrid: bin by |H| like f2p, but with annual-specific width quantiles

**R2/R3 bands:** f2p-style binning confirmed viable.
1. Bin by |M| into tiny/small/medium/large
2. Compute width quantiles per bin from pooled R2 or R3 historical residuals
3. 35k+ rows per bin per PY — robust quantile estimates
4. Pool 3-5 years of data for stability

**Clearing probabilities:**
- R2/R3: empirical approach may work (35k+ rows per bin)
- R1: likely need rule-based fallback due to wider spreads
- Consider MIN_ROWS threshold per flow_type within bins

**Round-specific multipliers:**
- R1 needs ~5-7x wider bands than R2/R3
- R3 needs ~0.85x the width of R2 (residuals are 15-20% smaller)
- These can be explicit round multipliers on the base band width

Concrete steps:
1. Implement f2p-style binning for R2/R3 using annual training data
2. Design R1 band approach (likely rule-based given high uncertainty)
3. Set width caps appropriate for annual (higher than f0p's 1000/3000)
4. Determine clearing probability approach per round
5. Backtest on PY 2024 or 2025 (hold out 1 PY)

---

## Key Files Reference

| File | What it contains |
|------|-----------------|
| `pmodel/base/ftr24/v1/band_generator.py` | f0p band generation, BASELINE_CONFIG, clearing probs, bin widths |
| `pmodel/base/ftr24/v1/miso_base.py` | `_fill_mtm` — historical DA congestion proxy computation |
| `pmodel/base/ftr24/v1/params/prod/auc2603/miso_a_offpeak.py` | MISO annual offpeak production params |
| `pmodel/base/ftr24/v1/autotuning.py` | Round/period definitions; MISO annual = 3 rounds × aq1-aq4 |
| `pbase/analysis/tools/miso.py` | `annual_round_day_map`, `fill_mtm_1st_period_with_hist_revenue` |
| `archive/learning.md` | Previous Stage 1 code study (1054 lines, still valid code reference) |
| `archive/stage2_design.md` | Previous design doc with H/C/R formulas (treat as starting hypotheses, not truth) |

---

## Self-Review Flags

Assumptions resolved from Phase 1-2 data (2026-02-11). Full evidence in `findings_residuals.md`.

| # | Assumption | Status | Evidence |
|---|-----------|--------|----------|
| A1 | Annual bands should be narrower than f0p | **REJECTED (R1) / NUANCED (R2/R3)** | R1 p95=3,300 vs f0p p95=492 (6.7x wider needed). R2/R3 p95=568-693 (15-40% wider than f0p). |
| A2 | Annual per-period volume >> f0p per-period volume | **NUANCED** | Annual avg MW/row=4.24 vs f0p=2.95 (1.44x). Moderate, not dramatic. |
| A3 | f0p p95 widths average ~500 | **CONFIRMED** | f0p p95 |residual| = 481-503. |
| A4 | R2/R3 can be treated similarly to f2p | **CONFIRMED** | 35k-92k rows per bin per PY. Residual scaling follows same |mtm| pattern. |
| A5 | Same formula for all 4 quarters / aq4 is harder | **NUANCED** | aq4 residual magnitudes are comparable. But aq4 direction accuracy is notably lower (62% vs 67%). aq2 is hardest by magnitude. |
| A6 | Current production uses ftr24/v1 for annual | **UNVERIFIED** | Not addressable from cleared trade data. Requires production config check. |

### Remaining open items
- A6: Check production system for annual (ftr23 vs ftr24)
- Q5: Current production performance / PnL (not data-addressable from cleared trades alone)
- Submitted bid stack analysis (notebooks exist but not yet extracted)
