# v001-xgb-20260220-001 — F2.0 Threshold

## Hypothesis

Switching the threshold optimization from F0.5 (precision-weighted) to F2.0 (recall-weighted) should increase recall past the 0.30 floor. F2.0 accepts more false positives in exchange for catching more true bindings.

## Changes

| Parameter | Baseline (v000) | This version |
|-----------|-----------------|--------------|
| `ThresholdConfig.threshold_beta` | 0.5 | **2.0** |

Single parameter change. All features, models, and training config identical to baseline.

## Results

| Gate | Onpeak | Offpeak | Mean | Floor | vs v000 |
|------|-------:|--------:|-----:|------:|--------:|
| S1-AUC | 0.6596 | 0.6672 | 0.6634 | 0.80 | -3.2 pp |
| S1-REC | 0.3333 | 0.3448 | 0.3391 | 0.30 | **+6.6 pp** |
| S2-SPR | 0.3813 | 0.4228 | 0.4021 | 0.30 | -1.0 pp |
| C-VC@1000 | 0.7804 | 0.8099 | 0.7952 | 0.50 | -5.6 pp |
| C-RMSE | $1,223 | $1,369 | $1,296 | $2,000 | **-$260** |

Promotable: **No** — S1-AUC fails floor; C-VC@1000 regresses vs champion.

## Conclusion

F2.0 threshold works as expected: recall jumps from 0.27 to 0.34 (passes the 0.30 floor). But the tradeoff is costly — AUC drops 3 pp and value capture drops 5.6 pp. The lower threshold admits more false positives, which dilutes ranking quality. RMSE improves because more true positives are now predicted with non-zero values instead of being zeroed out.

**Lesson**: Threshold tuning alone cannot fix the AUC gap. The classifier itself needs better discrimination before adjusting the threshold.
