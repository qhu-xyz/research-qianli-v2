# MISO Annual: DA Revenue Features (1(rev))

## Definition

`1(rev)` = realized DA congestion revenue for a path over a quarter, from the most recent
fully-settled period available at auction time.

**Path revenue = sink_node_DA_congestion - source_node_DA_congestion**, summed over the
relevant settlement months. (Same sign convention as MCP and nodal_f0: sink - source.)

**SIGN BUG FOUND (2026-03-20):** The original computation used source - sink (wrong).
Fixed by negating the saved values. Backup at `r1_1rev_option_b.parquet.wrong_sign_bak`.

## Auction Timing

PY N annual auction clears **~April of year N**. DA settlement data through **March of year N**
is available. Any data from April onward is leak.

## 1(rev) Calendar by Quarter

For PY N auction (~April N):

| Quarter | Delivery | 1(rev) months | Source | Fully settled? |
|---------|----------|--------------|--------|---------------|
| aq1 (Jun-Aug) | Jun-Aug N | Jun-Aug N-1 | PY N-1 | Yes (8+ months ago) |
| aq2 (Sep-Nov) | Sep-Nov N | Sep-Nov N-1 | PY N-1 | Yes (5+ months ago) |
| aq3 (Dec-Feb) | Dec N to Feb N+1 | Dec N-1 to Feb N | PY N-1 | Yes (1+ months ago) |
| **aq4 (Mar-May)** | **Mar-May N** | **Mar-May N-1** | **PY N-1** | Yes (conservative) |

### aq4: conservative implementation

The frozen implementation uses **all 3 months from year N-1** (March/April/May of year N-1).
This is the conservative approach — all months are from PY N-1, fully settled by auction time.

A more aggressive version could use March of year N (1 month old at auction), but this
depends on DA pipeline lag and was not verified. The conservative version was used for all
results in this research.

## Loading Approaches

### Option A: Self-join (TESTED — 25% coverage, NOT viable)

Join PY N paths with PY N-1 same-quarter own-PY DA revenue by (source, sink, class).
Only 25% of paths have a match — path universe changes too much between years.

### Option B: Per-node DA loading (IMPLEMENTED — champion method)

Load `MisoDaLmpMonthlyAgg` per month, get per-node congestion, compute path revenue
as **sink_congestion - source_congestion**. Works for ANY path regardless of trading history.

**Implementation:** `all_rounds_1rev.parquet` covers all 3 rounds at 90.1% coverage via
nodal DA + `MisoNodalReplacement` for retired nodes. R1-specific file `r1_1rev_option_b.parquet`
has 99.2% coverage (R1 paths only, built earlier with separate process).
Import: `from pbase.data.dataset.da.miso import MisoDaLmpMonthlyAgg`

**Recommended outlier treatment:** Winsorize per-node DA congestion at P1/P99 before
computing path differences, to prevent extreme node values from dominating.

## Findings

- **Sign bug fixed 2026-03-20.** All results below use corrected sign (sink - source).
- With correct sign, f0 and 1(rev) have correlation +0.76 and 77% sign agreement.
- Corrected blend weights: q5 prevail w=0.50 (50% revenue), q5 counter w=0.60 (40% revenue).
  Much heavier revenue weight than wrong-sign blend (which was w=0.85-0.90).
- Overall MAE with corrected blend: 827 quarterly (vs 855 for raw f0). Modest improvement.
- Counter q5 bias reduced but worst cells (PY2023) remain structural.
- Full coverage via Option B (per-node loading): 99.2%.
