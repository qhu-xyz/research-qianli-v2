## Gate Calibration

### Current State: Gates Set at v0 Exact Mean (DYSFUNCTIONAL)

Current `registry/gates.json` has Group A floors at v0's exact mean values. This creates a paradox where v0 itself fails L1 on EV-VC@100 and EV-NDCG (floating-point precision differences between metrics.json and gates.json).

| Gate | Floor | v0 Mean | Ratio | Status |
|------|-------|---------|-------|--------|
| EV-VC@100 | 0.069032 | 0.069032 | 1.000 | v0 fails L1 |
| EV-VC@500 | 0.21595 | 0.21595 | 1.000 | Borderline |
| EV-NDCG | 0.747213 | 0.74721 | 1.000 | v0 fails L1 |
| Spearman | 0.392798 | 0.39280 | 1.000 | v0005 fails L1 by 0.0008 |

**Impact**: v0005 (EV-VC@100 +6.5%, EV-VC@500 +5.9%) cannot be promoted because Spearman dropped 0.0008 (0.2%). No version can be promoted unless ALL metrics simultaneously improve or exactly match v0 — this is functionally a "beat champion on every metric" gate, not a quality floor.

### Recommended Fix (HUMAN_SYNC)

Restore floors to ~0.87-0.90x v0 mean (matching what gate_calibration.md previously documented as "correct" committed gates):

| Gate | Recommended Floor | Ratio to v0 Mean | v0005 L1 |
|------|------------------|-------------------|----------|
| EV-VC@100 | 0.0600 | 0.87x | PASS |
| EV-VC@500 | 0.1900 | 0.88x | PASS |
| EV-NDCG | 0.6900 | 0.92x | PASS |
| Spearman | 0.3420 | 0.87x | PASS |

Alternative: Use 25th percentile of v0 per-month distribution as floor. This anchors floors to the champion's own variance.

### Noise Tolerance (L3) Issue

Codex review noted that `noise_tolerance=0.02` is not scale-aware:
- For EV-VC@100 (mean ~0.07, bot2 ~0.007): 0.02 is generous — threshold goes to -0.013 (effectively no floor)
- For C-RMSE (mean ~3000, bot2 ~5300): 0.02 is meaningless at this scale — effectively requires exact match
- For Spearman (mean ~0.39, bot2 ~0.27): 0.02 is reasonable

**Consider**: metric-scaled tolerance (e.g., 5-10% of champion's bottom_2_mean per metric)

### v0005 Gate Check (against current dysfunctional gates)

| Gate | v0005 Mean | Floor | L1 | Tail Fails | L2 | v0005 Bot-2 | Threshold | L3 | Overall |
|------|-----------|-------|----|-----------|----|-----------|-----------|----|----|
| EV-VC@100 | 0.0735 | 0.0690 | P | 0 | P | 0.0084 | -0.013 | P | **P** |
| EV-VC@500 | 0.2287 | 0.2160 | P | 0 | P | 0.0689 | 0.036 | P | **P** |
| EV-NDCG | 0.7501 | 0.7472 | P | 1 | P | 0.6458 | 0.628 | P | **P** |
| Spearman | 0.3920 | 0.3928 | **F** | 1 | P | 0.2669 | 0.249 | P | **F** |

**Result**: NOT PROMOTABLE (Spearman L1 fails by 0.0008)
