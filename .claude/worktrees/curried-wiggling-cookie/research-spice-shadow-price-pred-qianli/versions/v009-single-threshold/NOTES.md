# v009 — Single Threshold 0.7

**Hypothesis**: v008's threshold behavior is inconsistent — f0 default model optimizes to
~0.91, f1 and all branch models fall back to 0.50. A single fixed threshold (0.7) should
produce more predictable behavior and potentially improve the precision-recall trade-off.

**Changes vs v008**:
- `threshold_override=0.7` — bypass all per-group threshold optimization
- All other config identical to v008

## Threshold Analysis (v008 actual behavior)

Before running, we analyzed v008's actual thresholds from worker logs:

| Group | v008 default threshold | v008 branch thresholds | v009 (all) |
|-------|----------------------|----------------------|------------|
| f0 | 0.847–0.938 (mean 0.907) | 0.50 (fallback) | 0.70 |
| f1 | 0.500 | 0.50 (fallback) | 0.70 |
| long | 0.500 (fallback) | 0.50 (fallback) | 0.70 |

Key discovery: v008's threshold_manifest.json was unreliable due to BUG-6 (f0/f1 worker
race condition overwrites the file). Actual thresholds had to be extracted from worker logs.

## Results (32-run benchmark, 2026-02-25)

| Gate | v008 | v009 | Delta | Floor | Pass |
|------|------|------|-------|-------|------|
| S1-AUC | 0.6894 | 0.6895 | +0.000 | 0.65 | PASS |
| S1-REC | 0.2894 | 0.2515 | **-0.038** | 0.25 | **FAIL** |
| S1-PREC | 0.3412 | 0.3633 | +0.022 | 0.25 | PASS |
| S2-SPR | 0.4117 | 0.4110 | -0.001 | 0.30 | PASS |
| C-VC@1000 | 0.8428 | 0.8428 | 0.000 | 0.50 | PASS |
| C-RMSE | 1108.91 | 1103.30 | -5.6 | 2000 | PASS |

**Gates passed**: 10/11 (S1-REC FAILS in onpeak/f1)
**Failed segment**: onpeak/f1 recall = 0.2277 < 0.25 floor

### Per-period breakdown

| Segment | v008 REC | v009 REC | v008 PREC | v009 PREC |
|---------|----------|----------|-----------|-----------|
| onpeak/f0 | 0.3145 | 0.2636 | 0.3654 | 0.3886 |
| onpeak/f1 | 0.2577 | **0.2277** | 0.3171 | 0.3303 |
| offpeak/f0 | 0.3062 | 0.2633 | 0.3646 | 0.3868 |
| offpeak/f1 | 0.2792 | 0.2515 | 0.3347 | 0.3476 |

### Key observations

1. **AUC and VC@1000 unchanged**: These are threshold-independent (ranking metrics).

2. **Recall dropped 3-5pp across ALL segments**: Setting threshold=0.7 RAISED the threshold
   for hundreds of branch models (from their 0.50 fallback to 0.70). Since branch models
   handle the majority of predictions, this dominated the aggregate and reduced recall.

3. **Precision improved 1-2pp**: Fewer false positives from higher branch thresholds.

4. **onpeak/f1 breached the floor**: Already the weakest segment (hardest to predict),
   raising the threshold pushed it below the 0.25 recall gate.

**Conclusion**: threshold=0.7 is too aggressive for a single uniform threshold. The branch
models' 0.50 fallback was actually providing valuable recall. Abandoning per-group optimization
loses the natural balance between conservative defaults and liberal branches.
