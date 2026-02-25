# v000-legacy-20260220 — Legacy Baseline

## Hypothesis

No hypothesis — this is the unmodified legacy pipeline, force-promoted as the initial champion to establish a comparison baseline.

## Changes

None. This is the original codebase from commit `b32bf6b` (merge of `research-spice-shadow-price-pred`).

## Results

| Gate | Onpeak | Offpeak | Mean | Floor | Status |
|------|-------:|--------:|-----:|------:|--------|
| S1-AUC | 0.6905 | 0.7002 | 0.6954 | 0.80 | BELOW |
| S1-REC | 0.2702 | 0.2766 | 0.2734 | 0.30 | BELOW |
| S2-SPR | 0.3774 | 0.4466 | 0.4120 | 0.30 | PASS |
| C-VC@1000 | 0.8379 | 0.8637 | 0.8508 | 0.50 | PASS |
| C-RMSE | $1,255 | $1,857 | $1,556 | $2,000 | PASS |

## Conclusion

Baseline fails 2 of 5 absolute floors (S1-AUC and S1-REC). The classifier has weak discrimination (AUC 0.69) and low recall (0.27). Regressor ranking and value capture are strong. The primary improvement target is the classifier.
