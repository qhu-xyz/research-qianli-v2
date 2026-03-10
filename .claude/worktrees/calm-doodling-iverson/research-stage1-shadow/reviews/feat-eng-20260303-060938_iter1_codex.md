## Review: feat-eng-20260303-060938 Iteration 1

### Summary
v0004 is directionally positive on the business-critical ranking gates: S1-AUC +0.0015 (9W/3L), S1-VCAP@100 +0.0056 (10W/2L), and S1-NDCG +0.0038 (7W/5L) versus the v0 baseline reference. Precision-focused behavior is preserved (`precision` 0.4449 vs v0 0.4423; `pred_binding_rate` 0.0749 vs 0.0750), which is aligned with the stated objective (precision over recall).

That said, the effect size is still small relative to month-level variance and is not uniformly stable across tails: AP is 6W/6L, and bottom-2 non-regression is slightly negative on AP (-0.0040), VCAP@100 (-0.0003), and NDCG (-0.0060) vs v0 bottom-2, though all remain within the current 0.02 tolerance. Weak months persist (2022-09 AP=0.3072, 2022-12 VCAP@100=0.0008). I do not see regressions in the specific v0004 implementation targets (17-feature config, f2p parse fix, benchmark dual-default cleanup), but two pre-existing evaluation-method issues still materially affect trust in threshold-dependent metrics.

### Gate Analysis
| Gate | Value | Floor | Champion | Pass | Delta |
|------|-------|-------|----------|------|-------|
| S1-AUC | 0.8363 | 0.7848 | 0.8348 | YES | +0.0015 |
| S1-AP | 0.3951 | 0.3436 | 0.3936 | YES | +0.0015 |
| S1-VCAP@100 | 0.0205 | -0.0351 | 0.0149 | YES | +0.0056 |
| S1-NDCG | 0.7371 | 0.6833 | 0.7333 | YES | +0.0038 |
| S1-BRIER | 0.1516 | 0.1703 | 0.1503 | YES | -0.0013 |
| S1-VCAP@500 | 0.0843 | 0.0408 | 0.0908 | YES | -0.0065 |
| S1-VCAP@1000 | 0.1578 | 0.1091 | 0.1591 | YES | -0.0013 |
| S1-REC | 0.4174 | 0.1000 | 0.4192 | YES | -0.0018 |
| S1-CAP@100 | 0.7850 | 0.7325 | 0.7825 | YES | +0.0025 |
| S1-CAP@500 | 0.7750 | 0.7240 | 0.7740 | YES | +0.0010 |

Champion reference above is v0 mean (current `registry/champion.json` is null).

Three-layer checks:
- Layer 1 (mean): all Group A and Group B gates pass; closest mean-margin is S1-BRIER (0.1703 - 0.1516 = 0.0187).
- Layer 2 (tail safety): 0 tail-floor breaches for every gate. Weakest safety margin is S1-BRIER tail (0.2086 - max 0.1608 = 0.0478).
- Layer 3 (tail non-regression vs v0 bottom_2): all pass under tolerance 0.02.
  - Group A bottom_2 deltas vs v0: AUC +0.0059, AP -0.0040, VCAP@100 -0.0003, NDCG -0.0060.
  - Closest Group A to flip Layer 3 is S1-NDCG (margin to fail boundary: +0.0140).
  - Closest overall (monitor-only) are S1-VCAP@1000 (+0.0100) and S1-VCAP@500 (+0.0118).

Consistency / seasonality:
- AUC improvement is broad but modest: 9/12 wins, median monthly delta about +0.0009; 2022-12 (+0.0098) is the largest single contributor.
- AP is mixed (6/6), indicating mean AP lift is not robust.
- VCAP@100 gains are the clearest precision-aligned signal (10/2), not just one-month noise (still positive if 2021-01 outlier is removed).
- Persistent weak months remain late-2022, especially 2022-09 on AP.

### Code Findings
1. HIGH: threshold boundary mismatch between optimization semantics and inference semantics remains. `find_optimal_threshold()` optimizes thresholds from `precision_recall_curve` (inclusive boundary), but `apply_threshold()` uses strict `>` ([`ml/threshold.py:33`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:33 ), [`ml/threshold.py:64`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/threshold.py:64 )). This can drop boundary positives and distort REC/CAP/precision in low-granularity tails.
2. HIGH: threshold tuning and final scoring are done on the same validation split, creating optimistic bias for threshold-dependent metrics (REC, CAP@K, precision, F1, BRIER): [`ml/benchmark.py:75`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:75 )- [`ml/benchmark.py:81`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:81 ), and same pattern in pipeline [`ml/pipeline.py:104`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:104 )- [`ml/pipeline.py:125`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/pipeline.py:125 ).
3. MEDIUM: `_load_real()` silently falls back to `horizon=3` when `period_type` parsing fails, which can hide malformed ptypes and produce unintended train windows instead of failing fast ([`ml/data_loader.py:104`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/data_loader.py:104 )- [`ml/data_loader.py:106`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/data_loader.py:106 )).
4. LOW: feature prep materializes full dense numpy arrays for both train and validation and also creates an intermediate frame for interactions (`with_columns`), which is a potential memory pressure point on larger cascades ([`ml/features.py:40`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/features.py:40 )- [`ml/features.py:55`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/features.py:55 ), [`ml/benchmark.py:66`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:66 )- [`ml/benchmark.py:69`](/home/xyz/workspace/research-qianli-v2/research-stage1-shadow/ml/benchmark.py:69 )).

### Recommendations
1. Fix threshold boundary consistency first (`>=` in `apply_threshold`, with regression tests around exact-threshold probabilities).
2. Separate threshold selection from evaluation (nested split or rolling train/val/test) before trusting absolute threshold-dependent metric changes across iterations.
3. Change ptype parse fallback to explicit validation (`raise ValueError` on unknown format) to avoid silent lookback misconfiguration.
4. Keep current feature set for one more iteration if exploring ranking only, but require improvement claims to emphasize Group A + bottom_2 behavior rather than AP mean alone.

### Gate Calibration
- No floor changes recommended yet; current floors are clearly non-binding for Group A and still not blocking candidates.
- Recommend tightening Layer-3 tolerance by metric after 1-2 more real-data iterations: 0.02 is loose relative to observed deltas (most bottom_2 shifts are within ±0.01).
  - Candidate starting point for discussion: AUC/AP/NDCG tolerance in 0.005-0.01 range, keep VCAP@100 looser due higher variance.
