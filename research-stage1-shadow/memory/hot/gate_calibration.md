# Gate Calibration — Real v0 Baseline (12 months, f0, onpeak)

Calibrated from real-data v0 benchmark (commit d167090). Gate floors set as:
- `floor = v0_mean - v0_offset`
- `tail_floor = v0_extreme - v0_tail_offset`

## Group A (blocking) — Must Pass All 3 Layers for Promotion

| Gate | v0 Mean | v0 Min | Floor | Tail Floor | Headroom (mean) | Headroom (worst) |
|------|---------|--------|-------|------------|-----------------|------------------|
| S1-AUC | 0.8348 | 0.8088 | 0.7848 | 0.7088 | +0.0500 | +0.1000 |
| S1-AP | 0.3936 | 0.3150 | 0.3436 | 0.2150 | +0.0500 | +0.1000 |
| S1-VCAP@100 | 0.0149 | 0.0005 | -0.0351 | -0.0995 | +0.0500 | +0.1000 |
| S1-NDCG | 0.7333 | 0.6601 | 0.6833 | 0.5601 | +0.0500 | +0.1000 |

## Group B (monitor) — Informational, Does Not Block Promotion

| Gate | v0 Mean | v0 Extreme | Floor | Tail Floor | Headroom (mean) | Notes |
|------|---------|------------|-------|------------|-----------------|-------|
| S1-BRIER | 0.1503 | 0.1586 (max) | 0.1703 | 0.2086 | +0.0200 | Lower is better |
| S1-VCAP@500 | 0.0908 | 0.0433 | 0.0408 | -0.0567 | +0.0500 | |
| S1-VCAP@1000 | 0.1591 | 0.0809 | 0.1091 | -0.0191 | +0.0500 | |
| S1-REC | 0.4192 | 0.3130 | 0.1000 | 0.0000 | +0.3192 | Very loose floor |
| S1-CAP@100 | 0.7825 | 0.3600 | 0.7325 | 0.2600 | +0.0500 | High variance (std=0.25) |
| S1-CAP@500 | 0.7740 | 0.4180 | 0.7240 | 0.3180 | +0.0500 | High variance (std=0.19) |

## Three-Layer Check Reference
- **Layer 1**: mean(metric) >= floor (or <= for BRIER)
- **Layer 2**: count(months below tail_floor) <= 1
- **Layer 3**: bottom_2_mean >= champion_bottom_2_mean - 0.02

## Iteration 1 Observations (v0007, feat-eng-2-20260303-092848) — PROMOTED

**v0007 gate headroom (Group A)**:

| Gate | v0007 Mean | Floor | Headroom | v0007 Bot2 | v0 Bot2 | Δ Bot2 | L3 Margin |
|------|-----------|-------|----------|-----------|---------|--------|-----------|
| S1-AUC | 0.8485 | 0.7848 | +0.064 | 0.8188 | 0.8105 | +0.0083 | +0.028 |
| S1-AP | 0.4391 | 0.3436 | +0.096 | 0.3685 | 0.3322 | +0.0363 | +0.056 |
| S1-VCAP@100 | 0.0247 | -0.0351 | +0.060 | 0.0094 | 0.0014 | +0.0080 | +0.028 |
| S1-NDCG | 0.7333 | 0.6833 | +0.050 | 0.6562 | 0.6716 | **-0.0154** | **+0.005** |

**v0007 gate headroom (Group B — concerns)**:

| Gate | v0007 Mean | Floor | Headroom | Notes |
|------|-----------|-------|----------|-------|
| S1-BRIER | 0.1395 | 0.1703 | +0.031 | REVERSED — 6-experiment narrowing ended |
| S1-CAP@100 | 0.7342 | 0.7325 | **+0.002** | CRITICAL — essentially at floor |
| S1-CAP@500 | 0.7280 | 0.7240 | **+0.004** | HIGH — very near floor |
| S1-VCAP@1000 | 0.1401 | 0.1091 | +0.031 | Moderate decline (-0.019 vs v0) |

### Key Observations

1. **NDCG L3 margin is dangerously thin**: At 0.0046, any future version building on v0007 (now champion) must not regress NDCG bot2 by more than 0.02 from 0.6562 (i.e., must stay above 0.6362). This is the tightest constraint for iter 2+.

2. **CAP@100/500 effectively at Group B floors**: v0007's model profile shifted from threshold-dependent capture to ranking quality. The CAP floors (designed around v0's profile) no longer match the model's operating point. **Recommend relaxing both floors by 0.02 at HUMAN_SYNC** — this doesn't lower standards, it acknowledges the model trades CAP for AUC/AP (a good trade per business objective).

3. **BRIER headroom recovered**: From the 6-experiment low of 0.0163 (v0006) to 0.031 (v0007). The shift factor features improved calibration — no longer a concern.

4. **VCAP@100 floor still non-binding**: v0007 at 0.0247, floor at -0.0351. Tighten to 0.0 at HUMAN_SYNC.

5. **Layer 3 now critical**: With v0007 as champion, Layer 3 is activated. Future versions will be checked against:
   - AUC bot2 ≥ 0.8188 - 0.02 = 0.7988
   - AP bot2 ≥ 0.3685 - 0.02 = 0.3485
   - VCAP@100 bot2 ≥ 0.0094 - 0.02 = -0.0106
   - NDCG bot2 ≥ 0.6562 - 0.02 = **0.6362** (tightest constraint)

### HUMAN_SYNC Gate Recommendations (cumulative, 7 real-data experiments)
1. **Set champion to v0007** — activates Layer 3
2. **VCAP@100 floor**: Tighten from -0.035 to 0.0
3. **CAP@100/500 floors**: Relax by 0.02 (to 0.7125 and 0.7040) — model profile has changed
4. **Keep noise_tolerance at 0.02** — uniform, adequate for all metrics
5. **No changes to Group A mean floors** — all pass with 0.05+ headroom

### Cumulative ranges across 7 real-data experiments
| Metric | Min Mean | Max Mean | Range | v0007 |
|--------|----------|----------|-------|-------|
| AUC | 0.8323 | **0.8485** | **0.016** | **New high (broke ceiling)** |
| AP | 0.3892 | **0.4391** | **0.050** | **New high (broke ceiling)** |
| NDCG | 0.7323 | 0.7560 | 0.024 | 0.7333 (mid-range) |
| VCAP@100 | 0.0149 | 0.0270 | 0.012 | 0.0247 (near-high) |

v0007 dramatically widened the AUC and AP operating envelopes while keeping NDCG and VCAP@100 within prior ranges.
