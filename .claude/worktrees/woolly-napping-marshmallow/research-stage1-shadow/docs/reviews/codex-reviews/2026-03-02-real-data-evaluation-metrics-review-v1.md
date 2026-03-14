# Review: real-data evaluation metrics (2026-03-02)

## Scope
- `docs/plans/2026-03-01-real-data-evaluation-redesign.md`
- `docs/plans/2026-03-01-real-data-evaluation-implementation.md`

## Summary
The proposed metric set is broadly reasonable for a “rank then threshold” binary classifier: the **hard gates** are threshold-independent (AUC, AP, VCAP@100, NDCG) and the **monitor gates** capture calibration (BRIER) and threshold-dependent behavior (REC, CAP@K). The main risks are (a) value-based metrics being ill-defined if `actual_shadow_price` can be ≤ 0 and (b) redundancy / over-constraint from gating on both VCAP and NDCG without a clear interpretation and stability story.

## Findings

### 1. Hard-gate selection is directionally correct (threshold-independent + business alignment)
- **S1-AUC / S1-AP** are reasonable “base discriminative quality” checks and are standard for imbalanced classification.
- **S1-VCAP@100** and **S1-NDCG** both push the model toward ranking high-value events early, which matches the stated “shadow price value capture” objective better than AUC/AP alone.

### 2. VCAP + NDCG are plausible, but likely redundant without sharper definitions
- Both **S1-VCAP@100** and **S1-NDCG** are essentially “value-weighted ranking quality” metrics.
- If both are hard gates, you’re implicitly requiring the model to satisfy two highly correlated constraints; this can make promotion brittle without adding much safety beyond one well-chosen value metric.
- If you keep both, the docs would benefit from a clear “what failure mode does each catch that the other does not?” (e.g., VCAP@100 is ultra top-heavy; NDCG is smoother across ranks).

### 3. Value-based metrics need an explicit policy for non-positive shadow prices
The redesign/implementation docs treat `actual_shadow_price` as a “value” signal for:
- **S1-VCAP@K** (fraction of total value captured in top-K)
- **S1-NDCG** (relevance = shadow price)
- **S1-CAP@K** (top-K by actual shadow price)

If `actual_shadow_price` can be **negative or frequently zero**, these metrics can become hard to interpret or even mathematically awkward:
- “Total value” denominators (VCAP) can be near 0 or negative.
- NDCG typically assumes non-negative relevance; negative relevance flips meaning (“higher is better” becomes unclear).

Recommendation: make the metric definition explicit in the docs (and gates) as one of:
- “Value = `max(actual_shadow_price, 0)`” (most consistent with binding label = `> 0`), or
- “Value = `abs(actual_shadow_price)`” (if large-magnitude negatives are also important), or
- “Value = `log1p(max(actual_shadow_price, 0))`” (if extreme outliers make metrics too volatile).

### 4. Monitor-gate choices are sensible, but REC as a fixed floor deserves justification
- **S1-BRIER** as **monitor-only** is reasonable (calibration matters, but calibration fixes are often post-hoc and can be decoupled from ranking quality).
- **S1-REC** and **S1-CAP@K** as monitors makes sense because they depend on the thresholding policy.
- However, setting **S1-REC floor = 0.10** as a fixed policy number should be tied to an explicit operational constraint (e.g., “we must catch at least X% of binding events”) or to the chosen `threshold_beta` objective; otherwise it risks being either meaningless (always passed) or silently restrictive (blocking useful high-precision models later if it ever becomes hard-gated).

### 5. Metrics can be undefined in some months; docs should specify “data-quality guardrails”
AUC/AP/BRIER generally require both classes present. On a per-month split, it’s plausible to hit months (or ptypes) with extreme imbalance.
The plan’s aggregation approach should specify how to treat months where a metric is undefined:
- If a metric is NaN/None for a month, does that month count as a tail failure, get dropped from aggregation, or fail the run?
- For hard gates, a conservative policy is to require minimum `n_positive` and `n_negative` per eval month (or per stage) and fail fast if violated.

## Suggested edits (docs-only)
1. Add short definitions for **VCAP@K**, **CAP@K**, and **NDCG** (one-liners + the “value transform” rule for non-positive shadow prices).
2. Decide whether hard gates really need **both** VCAP@100 and NDCG; if yes, document the distinct failure modes each is meant to catch.
3. Add an explicit “undefined metric” policy (min class counts, or explicit handling) so multi-month aggregation doesn’t accidentally mask missing/degenerate months.
4. Add a sentence justifying the fixed **S1-REC = 0.10** monitor floor (what operational behavior it’s meant to enforce).

