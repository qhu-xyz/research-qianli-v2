# Hypothesis Log

> Hypotheses from smoke runs (n=20 synthetic data) are archived below.
> Real-data hypotheses start from batch 1.

## H1 (smoke): Infrastructure determinism — CONFIRMED
**Result**: All 10 metrics bit-for-bit identical across runs with seed=42.

## H2 (smoke): Threshold-beta reduction fixes S1-REC — FAILED
**Result**: Beta < 1 weights precision, not recall. Direction had formula inverted. No effect on metrics.
**Lesson**: beta=0.7 is precision-favoring, which aligns with business objective. Do not change.
