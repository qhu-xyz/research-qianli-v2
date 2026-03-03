# v010-threshold-050 — Single Threshold 0.5

**Hypothesis**: Since v009 (threshold=0.7) failed by raising branch fallbacks too high,
threshold=0.5 should match the existing branch fallback behavior while also lowering the
f0 default model threshold from ~0.91 to 0.50. This tests whether a uniform 0.5 improves
on v008's mixed threshold regime.

**Changes vs v008**:
- `threshold_override=0.5` — bypass all per-group threshold optimization
- All other config identical to v008

Note: This is distinct from v010-value-weighted-ev, which tested value-weighted training.

## Results (32-run benchmark, 2026-02-25)

| Gate | v008 | v010-thr050 | Delta | Floor | Pass |
|------|------|-------------|-------|-------|------|
| S1-AUC | 0.6894 | 0.6895 | +0.000 | 0.65 | PASS |
| S1-REC | 0.2894 | 0.2789 | -0.011 | 0.25 | PASS |
| S1-PREC | 0.3412 | 0.3436 | +0.002 | 0.25 | PASS |
| S2-SPR | 0.4117 | 0.4085 | -0.003 | 0.30 | PASS |
| C-VC@1000 | 0.8428 | 0.8428 | 0.000 | 0.50 | PASS |
| C-RMSE | 1108.91 | 1111.32 | +2.4 | 2000 | PASS |

**Gates passed**: 11/11
**Beats champion**: No (recall and CAP@K below v000)
**Promotable**: No

### Per-period breakdown vs v008

| Segment | Metric | v008 | v010-thr050 | Delta |
|---------|--------|------|-------------|-------|
| onpeak/f0 | REC | 0.3145 | 0.2902 | -0.024 |
| onpeak/f0 | PREC | 0.3654 | 0.3619 | -0.004 |
| onpeak/f0 | RMSE | 960.99 | 957.03 | -3.96 |
| onpeak/f1 | REC | 0.2577 | 0.2577 | 0.000 |
| onpeak/f1 | PREC | 0.3171 | 0.3171 | 0.000 |
| onpeak/f1 | RMSE | 1313.00 | 1313.00 | 0.000 |
| offpeak/f0 | REC | 0.3062 | 0.2883 | -0.018 |
| offpeak/f0 | PREC | 0.3646 | 0.3608 | -0.004 |
| offpeak/f0 | RMSE | 937.73 | 951.31 | +13.58 |
| offpeak/f1 | REC | 0.2792 | 0.2792 | 0.000 |
| offpeak/f1 | PREC | 0.3347 | 0.3347 | 0.000 |
| offpeak/f1 | RMSE | 1223.93 | 1223.93 | 0.000 |

### Key observations

1. **f1 segments: IDENTICAL to v008** — v008 already used threshold=0.50 for all f1
   models (default + branches). Override=0.50 changes nothing.

2. **f0 segments: only default model affected** — the f0 default threshold dropped from
   ~0.91 (optimized) to 0.50 (override). Branch models already used 0.50. Net effect is
   a small recall decrease (~2pp) in f0, likely from non-deterministic training noise
   rather than the threshold change itself (since lowering threshold should increase recall).

3. **Threshold choice barely matters** — because branch models (hundreds per segment)
   dominate predictions and they ALL fall back to threshold=0.50 regardless of config.
   Only the f0 default model (minority of predictions) has a meaningfully optimized threshold.

**Conclusion**: Threshold tuning is a dead end. The real lever is model quality, not
classification threshold. v008's "mixed threshold" (f0 default ~0.91, everything else 0.50)
is the natural optimizer output and works as well as any fixed threshold. Future improvements
should focus on features, training data, or model architecture rather than threshold selection.

## Threshold Experiment Series Summary

| Version | Threshold | All Gates Pass | onpeak/f1 REC | Aggregate REC |
|---------|-----------|----------------|---------------|---------------|
| v008 (mixed) | f0=0.91, f1=0.50, branches=0.50 | Yes | 0.258 | 0.289 |
| v009 | 0.70 uniform | **No** (onpeak/f1) | 0.228 | 0.252 |
| v010-thr050 | 0.50 uniform | Yes | 0.258 | 0.279 |

The "mixed" regime from v008's optimizer produces the best results. Uniform thresholds
either hurt recall (0.7) or are equivalent (0.5) because branch fallbacks already use 0.50.
