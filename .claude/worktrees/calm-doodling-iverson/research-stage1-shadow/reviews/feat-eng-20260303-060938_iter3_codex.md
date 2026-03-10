## Review: feat-eng-20260303-060938 Iteration 3

### Summary
v0006 is a mixed result with a clear metric split: top-of-stack ranking metrics improved materially (S1-NDCG +0.0227, S1-VCAP@100 +0.0121 vs v0), but positive-class ranking quality regressed (S1-AP -0.0044 mean, AP bottom_2 -0.0094 vs v0). Given the business objective (precision over recall), the AP degradation is the primary concern even though all Group A gates pass.

Across 12 eval months, improvements are not uniformly distributed: AUC is only 5W/7L vs v0 and appears outlier-assisted by 2022-12, while AP is 3W/9L (broad regression). NDCG (10W/2L) and VCAP@100 (9W/3L) are much more consistent. No tail-floor breaches occurred, but tail safety is weakly informative because tail floors are very loose. `registry/champion.json` is null, so Layer 3 non-regression is effectively disabled in production gating.

### Gate Analysis
Champion column uses v0 mean as reference because no promoted champion is set (`registry/champion.json` is null).

| Gate | Value | Floor | Champion | Pass | Delta |
|------|-------|-------|----------|------|-------|
| S1-AUC | 0.8354 | 0.7848 | 0.8348 | PASS | +0.0006 |
| S1-AP | 0.3892 | 0.3436 | 0.3936 | PASS | -0.0044 |
| S1-VCAP@100 | 0.0270 | -0.0351 | 0.0149 | PASS | +0.0121 |
| S1-NDCG | 0.7560 | 0.6833 | 0.7333 | PASS | +0.0227 |
| S1-BRIER | 0.1540 | 0.1703 | 0.1503 | PASS | +0.0037 (worse) |
| S1-VCAP@500 | 0.1172 | 0.0408 | 0.0908 | PASS | +0.0264 |
| S1-VCAP@1000 | 0.1821 | 0.1091 | 0.1591 | PASS | +0.0230 |
| S1-REC | 0.4179 | 0.1000 | 0.4192 | PASS | -0.0013 |
| S1-CAP@100 | 0.7892 | 0.7325 | 0.7825 | PASS | +0.0067 |
| S1-CAP@500 | 0.7770 | 0.7240 | 0.7740 | PASS | +0.0030 |

Three-layer Group A details:
- Mean layer: all pass; closest to floor is S1-AP (margin +0.0456), then S1-AUC (+0.0506).
- Tail safety: 0 months below tail_floor for all Group A gates; weakest months are AUC (2021-12, 2022-12), AP (2022-09, 2021-06), VCAP@100 (2022-12, 2021-04), NDCG (2021-04, 2022-12).
- Tail non-regression: disabled by null champion. Diagnostic vs v0 bottom_2_mean: AUC +0.0050, AP -0.0094, VCAP@100 +0.0038, NDCG +0.0108 (all still inside ±0.02 tolerance).

### Code Findings
1. HIGH: Threshold leakage persists in both single-month and benchmark paths: threshold is tuned on validation labels and then evaluated on the same split, biasing threshold-dependent metrics (BRIER/REC/CAP/precision). See [benchmark.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:76), [benchmark.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:81), [pipeline.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:105), [pipeline.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:123).
2. MEDIUM: Threshold boundary mismatch remains: PR-curve thresholds are inclusive, but inference uses strict `>`; borderline scores are forced negative, which can depress recall/capture inconsistently month-to-month. See [threshold.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:33), [threshold.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:64).
3. MEDIUM: Silent fallback in period parsing can corrupt train/val windowing: malformed `ptype` falls back to `horizon=3` instead of failing fast. See [data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/data_loader.py:104), [data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/data_loader.py:105).
4. LOW: v0006 feature-pruning implementation itself is internally consistent (13 features + monotone vector + updated tests) and I did not find a new iteration-specific shape/type bug in the changed files. See [config.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/config.py:16), [test_config.py](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/tests/test_config.py:4).

### Recommendations
1. Do not promote v0006 over v0004 for precision-centric objectives; AP mean and AP tail both regressed while AUC is effectively flat.
2. Set a real champion (at minimum v0, preferably explicit best candidate) so Layer 3 becomes operational; current null champion removes the intended tail non-regression protection.
3. Treat v0006’s NDCG/VCAP gains as evidence that top-K ordering improved, but AP deterioration indicates broader positive-class ranking worsened; preserve this as a tradeoff result, not a net win.
4. Prioritize fixing threshold methodology debt before interpreting Group B trend lines (holdout threshold tuning and threshold boundary alignment).
5. Keep focus on ranking-signal feature work for next cycle; AP weakness remains concentrated in 2022-09 and secondarily 2021-06.

### Gate Calibration
- Keep Group A mean floors unchanged for now.
- Activate champion-based Layer 3 immediately (operational change, not floor change).
- Noise tolerance 0.02 is loose relative to observed bottom_2 shifts (AUC/AP/NDCG typically within about 0.01); consider metric-specific Layer 3 tolerances once champion is active: AUC ~0.01, AP ~0.015, NDCG ~0.01, keep VCAP@100 looser (~0.02).
- S1-VCAP@100 floor remains non-binding (negative); if intent is to make it informative, discuss raising floor toward 0.0 after champion activation and another round of evidence.
