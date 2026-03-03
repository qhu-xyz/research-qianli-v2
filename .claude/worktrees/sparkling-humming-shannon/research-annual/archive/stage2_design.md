# Stage 2: Annual MCP Baseline + Bid Price Design (Reset)

## 1. Objective
Design annual pricing that **keeps the f0p v1 structure unchanged**:

`baseline (mcp_pred) -> 10 bid-price bands -> clearing_prob -> optimizer`

Only annual-specific pieces should change:
- how baseline is computed (especially MISO R1)
- how band widths are calibrated under sparse annual data

---

## 2. Ground Truth from Current Code

### 2.1 Annual R1 historical proxy (`H`) in code
For MISO annual R1, historical fill logic is effectively:
- use delivery-quarter months from prior year(s)
- load DA monthly congestion by node
- directional shrinkage (`source<0`, `sink>0`) with 0.85
- path proxy = sink - source
- quarter scaling = `*3`

Implemented in:
- `pmodel/base/ftr24/v1/miso_base.py:_fill_mtm`
- `pbase/analysis/tools/miso.py:fill_mtm_1st_period_with_hist_revenue`
- historical same pattern in `ftr23/v1` and later variants

### 2.2 v1 f0p band generation pattern
- baseline = weighted `mtm_1st_mean` + `1(rev)` by period type
- f0/f1 use LightGBM residual + conformal bands
- f2/f3/q2/q3/q4 use rule-based bin widths
- map 10 bands to `bid_price_1..10` + `clearing_prob_1..10`

Implemented in:
- `pmodel/base/ftr24/v1/band_generator.py`

### 2.3 Current annual gap
`aq1..aq4` are not routed in band_generator period-type branches and not in baseline config.

---

## 3. Recommendation (What to Implement)

## 3.1 Define `H` explicitly (annual-compatible, low-risk)

Use `H` as **historical DA-congestion path proxy**, preserving legacy logic but strengthening robustness:

1. Compute per-year quarter proxy:
- `H_yk = sink_da_cong_yk - source_da_cong_yk`
- keep directional shrinkage (parameterized, default 0.85)

2. Multi-year weighted aggregate:
- `H = 0.60*H_y1 + 0.30*H_y2 + 0.10*H_y3`
- if fewer years exist, renormalize available weights

3. Scale for annual quarter products:
- `H_quarter = H * 3`

This keeps continuity with prior production behavior and reduces one-year noise.

---

## 3.2 Baseline formula by round

Notation:
- `M`: live `mtm_1st_mean` from market (available MISO R2+, PJM all rounds)
- `H`: historical proxy above
- `C`: previous planning-year clearing proxy for same path/quarter (if available)
- `R`: weak delivery-aligned revenue trend proxy (small weight; optional)

### MISO
- **R1** (no live MTM):
  - `baseline = 0.80*H + 0.20*C`
  - fallback if `C` missing: `baseline = H`
- **R2**:
  - `baseline = 0.90*M + 0.10*H`
- **R3**:
  - `baseline = 0.92*M + 0.08*H`

### PJM
- **R1-R4**:
  - `baseline = 0.90*M + 0.10*H`

Rationale:
- MTM dominates once available.
- `H` provides seasonal anchor and drift control.
- `R` is intentionally excluded at first due to annual timing mismatch; add later only if backtest shows value.

---

## 3.3 Annual bid bands using f0p pattern

Route annual (`aq1..aq4`) to **rule-based width path** initially (same family as f2p):

1. Compute residuals on training data:
- `residual = mcp_mean - baseline`
- `abs_residual = |residual|`

2. Compute width quantiles by `|mtm|` bins (same machinery as f2p).

3. Build symmetric bands around baseline:
- `upper_x = baseline + width_x`
- `lower_x = baseline - width_x`

4. Map to 10 bid levels + clearing probs using existing mapping.

5. Round-aware uncertainty multiplier:
- R1 width multiplier: `1.25`
- R2: `1.00`
- R3: `0.90`

This mirrors f0p architecture exactly while respecting annual sample size limits.

---

## 3.4 Minimal code-shape changes

1. `band_generator.py`
- add `aq1..aq4` to baseline config
- add annual period routing to rule-based branch (same as f2p route)
- include annual in quarterly scaling list (x3)
- pass actual group class type into training data lookup (do not rely on default `class_type="onpeak"`)

2. annual model params
- add explicit annual baseline knobs:
  - `annual_h_years=3`
  - `annual_h_weights=[0.6,0.3,0.1]`
  - `annual_h_shrink_source_neg=0.85`
  - `annual_h_shrink_sink_pos=0.85`
  - round multipliers for widths

3. `miso_base._fill_mtm`
- parameterize lookback years and shrinkage
- keep default behavior backward-compatible
- load params from model config (e.g., `set_bid_price_params.prepare_data_params`) and apply inside `_fill_mtm`

4. `trade_finalizer.py`
- define and compute `C` before baseline:
  - default key: same `(source_id, sink_id, period_type, class_type, round)` from previous planning year
  - fallback key order if missing: drop `round`, then drop `class_type`, then no `C`
  - log coverage and fallback rates per round

No optimizer changes are required.

---

## 4. Explicit Hypotheses + Verification Comments for Next AI

> The items below are intentionally written as **verification TODOs**. Do not assume they are true until tested.

### H1. Multi-year `H` improves MISO R1 MCP error vs current 1-year proxy
- Hypothesis: `3y weighted H` lowers MAE by >=5% vs current `_fill_mtm`.
- Verify by backtest:
  - train/eval on historical annual R1 only
  - compare current vs proposed baseline
  - report MAE, median AE, directional hit rate
- Required output artifact:
  - table by period_type (`aq1..aq4`) and class_type

### H2. Adding `C` (prev planning-year clearing proxy) helps R1 enough to justify complexity
- Hypothesis: `0.8*H + 0.2*C` beats `H` alone.
- Verify:
  - coverage of `C` availability by path
  - coverage by exact key and each fallback key level
  - performance split (paths with C vs without C)
  - fallback behavior impact

### H3. Annual should use rule-based bands before LightGBM bands
- Hypothesis: sparse annual sample causes unstable ML residual models.
- Verify:
  - compare rule-based vs LightGBM residual on rolling annual windows
  - include calibration drift diagnostics (coverage error at 90/95)

### H4. R1/R2/R3 width multipliers are directionally correct
- Hypothesis: uncertainty shrinks by round; proposed multipliers improve PnL/credit tradeoff.
- Verify:
  - grid search multipliers around proposed values
  - evaluate cleared MW, expected PnL, realized PnL, credit requirement

### H5. `R` (revenue-like term) is weak for annual and can be omitted initially
- Hypothesis: adding `R` gives negligible lift in annual.
- Verify:
  - ablation: with and without `R`
  - significance by round and quarter

### H6. aq4 needs special handling due to data coverage/cutoff behavior
- Hypothesis: aq4 has structurally worse proxy quality under current month filtering.
- Verify:
  - check month list generated for aq4 under current logic
  - compare proxy error aq4 vs aq1-3
  - test aq4-specific fallback extension

### H7. Annual scaling is applied exactly once in active v1 path
- Hypothesis: adding `aq*` to `band_generator` quarterly scaling is sufficient and no extra `*3` path is active.
- Verify:
  - trace call graph from `base.generate_trades_one_auc_period_round()` to final submission
  - confirm `_adjust_quarter_annual_bid_price` is not invoked in active path
  - assert no double-scaling in sampled outputs

### H8. Class-type-specific training data is used correctly
- Hypothesis: offpeak groups are not accidentally using onpeak training partitions.
- Verify:
  - assert each `(auction_month, period_type, class_type)` group loads matching class_type partition
  - add a unit/integration check for class_type routing

---

## 5. Validation Protocol (Required Before Production)

1. MCP prediction metrics
- MAE / MAPE / directional accuracy
- by `round x period_type x class_type`

2. Band diagnostics
- empirical coverage at 50/70/90/95
- over/under-coverage decomposition

3. Trading impact
- simulated clear rate
- PnL
- credit requirement
- risk concentration metrics

4. Robustness
- out-of-time split (latest PY holdout)
- sensitivity to shrinkage and width multipliers
- no-double-scaling check
- class_type training-partition correctness check

---

## 6. Recommended Initial Parameter Set (Starting Point Only)

- `H years`: 3
- `H weights`: `[0.60, 0.30, 0.10]`
- `shrink source_neg`: `0.85`
- `shrink sink_pos`: `0.85`
- `MISO R1 baseline`: `0.80*H + 0.20*C` (fallback `H`)
- `MISO R2 baseline`: `0.90*M + 0.10*H`
- `MISO R3 baseline`: `0.92*M + 0.08*H`
- `PJM baseline`: `0.90*M + 0.10*H`
- `width multiplier`: `R1=1.25, R2=1.00, R3=0.90`

> These are priors, not truths. Next AI must validate and update based on measured outcomes.
