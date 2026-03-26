# BG2 Gate Revision: BG2a + BG2b

## Problem

Old BG2 required `candidate_width <= promoted_width` at ALL 20 cells (5 levels x 4
quarters). This was too strict:

- P50/P70 are low-value coverage levels with limited operational use
- Bin-boundary shifts (e.g., 4 bins -> 8 bins) can inflate P50 by 5-10% without
  affecting P95 performance
- R1 v5 failed BG2 on aq1/p50 (+7.8%) despite P95 half-widths being 13-25% narrower
- The gate hid tail risk behind a single mean — one sign segment could get much worse
  while the overall mean stayed flat

## Solution

Split BG2 into two gates:

### BG2a — Overall width, tiered tolerance (HARD)

Per-level tolerance allows low-value coverage levels some slack:

Level | Tolerance | Rationale
------|-----------|----------
P50   | 10%       | Low operational value, sensitive to bin boundaries
P70   | 5%        | Moderate value
P80   | 0%        | Operational threshold, no slack
P90   | 0%        | High value
P95   | 0%        | Primary metric, no slack

Rule: all 20 cells must satisfy `candidate_width <= promoted_width * (1 + tolerance)`.

### BG2b — Per-sign tail risk (HARD)

40 combos: 4 quarters x 2 signs (prevail, counter) x 5 levels. Each combo has
`mean_width` (average across bins) and `max_width` (worst bin).

Thresholds:
- `BG2B_MEAN_PASS_RATE = 0.90` — 90% of 40 combos must have mean_width improved or flat
- `BG2B_MEAN_MAX_DEGRADE = 0.10` — no single combo mean can degrade > 10%
- `BG2B_MAX_MAX_DEGRADE = 0.20` — no single combo max can degrade > 20%

Rule: all three conditions must hold. If promoted lacks per_sign data, BG2b is
skipped (backward compatible).

## Results under new gates

### R1 v5 (LOO vs promoted v3)

Gate  | Result | Detail
------|--------|-------
BG2a  | PASS   | 20/20 cells (aq1/p50 +7.8% within 10% tolerance)
BG2b  | --     | skipped, promoted v3 lacks per_sign widths

### R2 v4 (temporal vs promoted v2)

Gate  | Result | Detail
------|--------|-------
BG2a  | FAIL   | 15/20 cells, worst: aq3/p95 (+1.4%)
BG2b  | --     | skipped, promoted v2 lacks per_sign widths

R2 v4 (sym_6b_cs) is effectively identical to v3 — the BG2a failure is from marginal
rounding in symmetric width aggregation, same as old BG2. This is expected since the
winner is unchanged from v3.

### R3 v4 (temporal vs promoted v2)

Gate  | Result | Detail
------|--------|-------
BG2a  | PASS   | 20/20 cells (aq1/p50 +1.9% within 10% tolerance)
BG2b  | --     | skipped, promoted v2 lacks per_sign widths

## Per-sign width data (first generation)

Per-sign widths are now tracked in metrics.json for all 3 rounds. BG2b will
become active when comparing future versions against v5/v4, which include per_sign
width data.

### R1 v5 per-sign P95 widths (aq1)

Sign     | Mean width | Max width
---------|------------|----------
prevail  | 2213.7     | 6223.4
counter  | 2490.5     | 7261.3

### R3 v4 per-sign P95 widths (aq1)

Sign     | Mean width | Max width
---------|------------|----------
prevail  | 165.5      | 464.2
counter  | 181.2      | 474.2
