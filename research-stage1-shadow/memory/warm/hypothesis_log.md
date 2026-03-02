# Hypothesis Log

> Hypotheses from smoke runs (n=20 synthetic data) are archived below.
> Real-data hypotheses start from batch 1.

## H1 (smoke): Infrastructure determinism — CONFIRMED
**Result**: All 10 metrics bit-for-bit identical across runs with seed=42.

## H2 (smoke): Threshold-beta reduction fixes S1-REC — FAILED
**Result**: Beta < 1 weights precision, not recall. Direction had formula inverted. No effect on metrics.
**Lesson**: beta=0.7 is precision-favoring, which aligns with business objective. Do not change.

---

## H3 (real data, iter1): HP tuning improves ranking quality — REFUTED
**Batch**: hp-tune-20260302-134412, **Version**: v0003
**Changes**: max_depth 4→6, n_estimators 200→400, lr 0.1→0.05, min_child_weight 10→5
**Expected**: AUC +0.005–0.015, AP +0.01–0.03, NDCG +0.005–0.015
**Actual**:
- AUC: -0.0025 (0W/11L/1T, p≈0.003) — **statistically significant degradation**
- AP: -0.0015 (4W/8L)
- NDCG: -0.0010 (4W/8L)
- VCAP@100: +0.0015 (7W/5L, mean up but tail down)
- BRIER: -0.0041 (12W/0L, p≈0.0002) — **statistically significant improvement** (calibration only, Group B)

**Key Insight**: v0 defaults were already near-optimal. The model is **feature-limited, not complexity-limited**. Deeper trees improve probability calibration but slightly worsen discrimination. The 14 features have reached their informational ceiling for ranking quality at AUC ~0.835.

**Lesson**: Standard XGBoost HP tuning (deeper + slower + finer) is not a viable path to improvement for this dataset. Next lever: feature engineering to provide new discriminative signal.
