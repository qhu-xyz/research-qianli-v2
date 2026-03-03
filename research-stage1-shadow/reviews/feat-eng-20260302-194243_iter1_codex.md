## Review: feat-eng-20260302-194243 Iteration 1

### Summary
`v0003` (train window 10→14 months) is a small directional improvement on mean Group A metrics, but the gains are not robust across months and are mostly concentrated in a few outlier months. Mean deltas vs v0 are AUC `+0.0013`, AP `+0.0012`, NDCG `+0.0019`, VCAP@100 `+0.0034`, with all Group A gates passing by large floor margins. However, AP and NDCG tail quality regressed (`bottom_2_mean`: AP `-0.0045`, NDCG `-0.0059` vs v0), and late-2022 remains mixed (2022-12 improves materially; 2022-09 AP degrades).

The implementation mostly matches direction, but there are still correctness risks unrelated to gate pass status: `f2p` parsing can crash in real-data loading, threshold tuning/evaluation still share the same validation split (leakage for threshold-dependent metrics), and threshold application uses `>` while optimization logic is based on PR-curve thresholds that are effectively `>=`. These issues do not invalidate Group A ranking metrics, but they do weaken rigor of recall/CAP/precision monitoring and cascade reliability.

### Gate Analysis
| Gate | Value | Floor | Champion | Pass | Delta |
|------|-------|-------|----------|------|-------|
| S1-AUC | 0.8361 | 0.7848 | 0.8348 | YES | +0.0013 |
| S1-AP | 0.3948 | 0.3436 | 0.3936 | YES | +0.0012 |
| S1-VCAP@100 | 0.0183 | -0.0351 | 0.0149 | YES | +0.0034 |
| S1-NDCG | 0.7352 | 0.6833 | 0.7333 | YES | +0.0019 |
| S1-BRIER | 0.1514 | 0.1703 | 0.1503 | YES | +0.0011 |
| S1-VCAP@500 | 0.0845 | 0.0408 | 0.0908 | YES | -0.0063 |
| S1-VCAP@1000 | 0.1610 | 0.1091 | 0.1591 | YES | +0.0019 |
| S1-REC | 0.4130 | 0.1000 | 0.4192 | YES | -0.0062 |
| S1-CAP@100 | 0.7708 | 0.7325 | 0.7825 | YES | -0.0117 |
| S1-CAP@500 | 0.7633 | 0.7240 | 0.7740 | YES | -0.0107 |

Three-layer check notes:
- Mean quality: all Group A/B gates pass.
- Tail safety: 0 months below tail floor for every gate.
- Tail non-regression: champion is `null` in comparison report, so Layer 3 is effectively auto-pass in code; against v0 reference, Group A bottom_2 deltas are AUC `+0.0057`, AP `-0.0045`, VCAP@100 `+0.0002`, NDCG `-0.0059` (all within 0.02 tolerance).
- Closest to failing mean floor: `S1-AP` (margin `+0.0512`) and `S1-AUC` (`+0.0513`) in Group A; `S1-CAP@100` (`+0.0383`) and `S1-CAP@500` (`+0.0393`) in Group B.

Consistency and seasonality notes (12 months):
- AUC 7W/4L/1T is weak evidence; removing 2022-12 (`+0.0098`) leaves mean AUC delta ~`+0.0005`.
- AP 8W/4L but net gain is concentrated: removing 2022-12 (`+0.0142`) nearly eliminates aggregate gain.
- NDCG gain is outlier-driven: 2021-01 contributes `+0.0226`; excluding it leaves near-zero average improvement.
- VCAP@100 gain is concentrated in 2021-01 and 2022-03; without those two months, average delta is approximately flat/slightly negative.

### Code Findings
1. HIGH: `f2p` period type can crash real-data loading. In horizon parsing, `int(ptype[1:])` fails for cascade stage label `"f2p"` (`int("2p")`). This is a hard runtime failure for stage-3 evaluation paths. [ml/data_loader.py:103](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/data_loader.py:103)
2. HIGH: Threshold is tuned and then evaluated on the same validation split, which leaks threshold selection into reported threshold-dependent metrics (`S1-REC`, `CAP@K`, precision, F1). This biases monitoring metrics upward and complicates iteration-to-iteration inference. [ml/benchmark.py:72](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:72) [ml/benchmark.py:77](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:77) [ml/pipeline.py:105](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:105) [ml/pipeline.py:123](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:123)
3. MEDIUM: Threshold optimization/application mismatch remains (`precision_recall_curve` thresholds correspond to inclusive cut behavior, but inference uses strict `>`). Borderline samples at exactly-threshold are dropped at inference, which is most harmful in sparse-positive months and can depress recall/CAP unpredictably. [ml/threshold.py:33](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:33) [ml/threshold.py:64](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:64)
4. LOW: Test fixture drift persists: synthetic feature fixture is 17-wide while production `FeatureConfig` is back to 14 features. This can mask shape/constraint assumptions and weakens test realism around monotone constraints. [ml/tests/conftest.py:8](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/tests/conftest.py:8) [ml/config.py:16](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/config.py:16)

### Recommendations
1. Fix `ptype` parsing to support `f0`, `f1`, `f2`, and `f2p` safely (regex or explicit map), then add a unit test for `f2p` path in `load_data`.
2. Separate threshold tuning from final evaluation (e.g., tune on `val`, report on holdout month/split) so threshold-dependent metrics are unbiased; keep Group A interpretation unchanged.
3. Align threshold semantics (`>=` in application or adjusted threshold selection) and lock this behavior with explicit boundary-value tests.
4. Keep this version as non-promoted exploratory evidence: signal is positive but too outlier-driven for a robust precision-ranking claim.
5. For next feature work, target persistent weak-tail months directly (2022-09 AP, 2021-04/2021-08 NDCG, 2022-12 VCAP@100) and report leave-one-month-out sensitivity, not only mean deltas.

### Gate Calibration
- `noise_tolerance=0.02` is very loose relative to observed Group A bottom_2 deltas (about 0.0002 to 0.0059 this run). A tighter metric-specific Layer-3 tolerance (for example: AUC/AP/NDCG around 0.005-0.01, VCAP@100 looser) would better detect real tail regression.
- `S1-VCAP@100` floors remain effectively non-binding (`floor=-0.0351`, `tail_floor=-0.0995`). Consider recalibrating this gate after more real-data iterations using empirical quantiles rather than fixed `0.1` tail offset.
- Since champion is currently `null`, Layer 3 is effectively disabled in practice. Promote/select a reference champion (even if v0) to activate true non-regression behavior.
