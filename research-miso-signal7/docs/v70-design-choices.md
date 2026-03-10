# V7.0 Design Choices

Decisions made during implementation and validation. Each section covers a choice,
the alternatives considered, and why we landed where we did.

---

## 1. Tiering: Row-Percentile with V6.2B Tie-Breaking

**Choice:** Sort by ML score descending, break ties with V6.2B rank_ori ascending,
then original index. Assign `rank = row_position / n`, `tier = ceil(rank * 5) - 1`.
This gives exact ~20% per tier.

**Rejected alternative:** Dense-rank / K (matching V6.2B formula).

**Why:** V6.2B's formula produces nearly all unique scores, so `dense_rank / K`
gives even quintiles. But ML with tiered labels (4 levels, ~88% label=0) produces
~55% unique scores — LightGBM assigns identical leaf paths to non-binding
constraints. With dense_rank / K where K ≈ 267 and n ≈ 489:

- tier 0: 10.8%
- tier 4: 45.3%

This is not a bug — the model correctly says "I can't distinguish these." But it
makes tiers operationally unstable and hard to use downstream.

**Row-percentile fixes this** by imposing a deterministic total order even over ties.
The tie-break policy:

1. **ML score** (primary): the real model signal
2. **V6.2B rank_ori** (secondary): stable, informative fallback — if the ML model
   can't distinguish two constraints, defer to the formula
3. **Original index** (tertiary): deterministic, avoids random splitting

**What we do NOT do:**
- Random tie-breaking (non-deterministic, different outputs each run)
- Arbitrary row order as tie-break (unstable across data refreshes)
- Keep dense-rank/K just because V6.2B used it (cargo culting)

**Scope:** Only f0/f1 (ML slices). f2+ passthrough keeps V6.2B's original
dense-rank/K tiering unchanged.

---

## 2. E2/E3 Validation Gates

**E3 (tier distribution):** Mandatory. With row-percentile tiering, each tier is
guaranteed ~20% (±1 constraint from rounding). Gate checks 18-22% per tier.

**E2 (score uniqueness):** Diagnostic only. ~55% unique ML scores is expected with
tiered labels. Ranks are all unique (row-percentile guarantees this). Score
degeneracy (NaN, Inf, constant) is already caught by F1 (mandatory).

---

## 3. Score Polarity

**Choice:** `rank_ori` in V7.0 ML slices contains raw LightGBM scores.
Higher = more binding. This is **opposite** to V6.2B, where lower rank_ori = more
binding.

**Why this is safe:** Downstream code (pmodel ftr22/ftr23) uses `tier`, `rank`,
`shadow_price`, and `equipment`. It does not appear to read `rank_ori` directly.
The `rank` column preserves V6.2B convention (lower = more binding) regardless of
the score polarity, because the rank computation inverts scores.

**Risk:** If any consumer reads `rank_ori` directly, the sign flip silently produces
wrong results. The handoff doc flags this for pre-deployment confirmation.

---

## 4. Binding Frequency Lag: BF_LAG=1 Always

**Choice:** For both f0 and f1, binding frequency features use realized DA strictly
before `prev_month(auction_month)`. The lag is keyed on **auction month**, not
delivery month.

**Why:** The signal for auction month M is submitted ~mid of M-1. At that point,
realized DA through M-2 is complete. This cutoff is the same regardless of whether
we're scoring f0 (delivery = M) or f1 (delivery = M+1).

The training row selection (`collect_usable_months`) handles per-ptype lag separately —
it ensures we only train on months whose delivery-month ground truth is available.
These are two independent lag mechanisms that must not be conflated.

---

## 5. Inference-Only Path (load_v62b_features_only)

**Choice:** Separate `load_v62b_features_only()` function that loads V6.2B + spice6
without realized DA ground truth.

**Rejected alternative:** Add `require_gt=False` parameter to existing `load_v62b_month()`.

**Why:** `load_v62b_month()` lives in research-stage5-tier and is used by evaluation
scripts that always need GT. Adding optional GT would risk accidentally running
evaluations without GT and getting silently wrong results. A separate function makes
the intent explicit and keeps the research code untouched.

---

## 6. Join on constraint_id, Not Positional

**Choice:** ML inference returns `(constraint_ids, scores)` as paired lists. The
assembly step joins V6.2B DataFrame to scores on `constraint_id`, not by position.

**Why:** The ML pipeline loads V6.2B via `load_v62b_features_only()` (polars) while
the signal assembly loads V6.2B via `ConstraintsSignal.load_data()` (pandas with
composite string index). Row order is not guaranteed to match between the two.
Positional assignment would silently misalign scores if orders differ.

The composite index in V6.2B is `{constraint_id}|{flow_direction}|spice`. We
extract constraint_id via `.index.str.split("|").str[0]` and join on that.

---

## 7. SO_MW_Transfer Exception

**Choice:** After ML scoring and tier assignment, force `tier = 1` for any row where
`branch_name == "SO_MW_Transfer"`.

**Why:** V6.2B has this as a hardcoded upstream exception. SO_MW_Transfer gets a
near-zero rank_ori (most extreme) but tier 1 (not 0). This exception is preserved
in V7.0 for compatibility. The validation plan (E4, E6) explicitly excludes
SO_MW_Transfer rows from tier-formula parity and monotonicity checks.

---

## 8. Package Naming: v70/ Not ml/

**Choice:** V7.0 code lives in `research-miso-signal7/v70/` (with `__init__.py`).

**Rejected:** `research-miso-signal7/ml/` (matching stage5 convention).

**Why:** Both research-miso-signal7 and research-stage5-tier need to be on sys.path
simultaneously. If both have `ml/`, Python's import system sees only the first one.
Naming our package `v70` avoids the namespace collision entirely.

---

## 9. V6.2B Formula Score as Feature (v7_formula_score)

**Choice:** The V6.2B formula score is included as a feature in the ML model, but
with **optimized blend weights** that differ from V6.2B's 60/30/10 split.

| Slice | da_rank | density_mix | density_ori |
|-------|:-------:|:-----------:|:-----------:|
| V6.2B formula | 0.60 | 0.30 | 0.10 |
| f0/onpeak | 0.85 | 0.00 | 0.15 |
| f0/offpeak | 0.85 | 0.00 | 0.15 |
| f1/onpeak | 0.70 | 0.00 | 0.30 |
| f1/offpeak | 0.80 | 0.00 | 0.20 |

density_mix gets zero weight in all ML slices — it adds noise, not signal. The
formula score is just one of 9 features; the model can learn to weight it
appropriately. This was determined by grid search over blend weights in stage5.

---

## 10. Passthrough: Exact V6.2B Copy

**Choice:** For f2, f3, q2-q4: load V6.2B DataFrame, save unchanged to V7.0 path.

**Why:** ML training requires realized DA history for the delivery month. f2+
period types have shorter history and sparser data, making ML gains uncertain.
Rather than risk degradation, we pass through V6.2B unchanged. The validation plan
(Gate C) verifies bit-identity for all passthrough slices.

Shift factors for ALL period types (including f0/f1) are also passed through —
SF depends on the constraint universe, not on scoring.
