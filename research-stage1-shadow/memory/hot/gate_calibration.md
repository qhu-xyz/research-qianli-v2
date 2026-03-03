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

## Iteration 1 Observations (v0008, feat-eng-3-20260303-104101) — PROMOTED

**v0008 gate headroom (Group A)**:

| Gate | v0008 Mean | Floor | Headroom | v0008 Bot2 | v0007 Bot2 | Δ Bot2 | L3 Margin |
|------|-----------|-------|----------|-----------|-----------|--------|-----------|
| S1-AUC | 0.8498 | 0.7848 | +0.065 | 0.8199 | 0.8188 | +0.0011 | +0.021 |
| S1-AP | 0.4418 | 0.3436 | +0.098 | 0.3726 | 0.3685 | +0.0041 | +0.024 |
| S1-VCAP@100 | 0.0240 | -0.0351 | +0.059 | 0.0061 | 0.0094 | -0.0033 | +0.017 |
| S1-NDCG | 0.7346 | 0.6833 | +0.051 | 0.6663 | 0.6562 | **+0.0101** | **+0.030** |

**v0008 gate headroom (Group B — concerns)**:

| Gate | v0008 Mean | Floor | Headroom | Notes |
|------|-----------|-------|----------|-------|
| S1-BRIER | 0.1383 | 0.1703 | +0.032 | Best calibration ever; continued improvement |
| S1-CAP@100 | 0.7142 | 0.7325 | **-0.018** | FAILED — model profile shifted to ranking quality |
| S1-CAP@500 | 0.7175 | 0.7240 | **-0.007** | FAILED — same cause as CAP@100 |
| S1-VCAP@500 | 0.0955 | 0.0408 | +0.055 | Improved |
| S1-VCAP@1000 | 0.1479 | 0.1091 | +0.039 | Improved |

### Key Observations

1. **NDCG L3 margin now comfortable**: Expanded from 0.0046 (v0007) to 0.0301 (v0008). The tightest constraint from v0007 is no longer the binding concern.

2. **VCAP@100 is now the closest L3 risk**: L3 margin +0.0167 (bot2 0.0061 vs floor -0.0106). With 4W/8L vs champion, this requires monitoring. If v0008 becomes champion, the new L3 floor would be 0.0061 - 0.02 = -0.0139 — still generous but trending toward binding.

3. **CAP@100/500 now fail Group B**: Both crossed their floors. The model profile has definitively moved from threshold-dependent capture to ranking quality. **Urgent**: relax both floors by 0.02-0.03 at HUMAN_SYNC.

4. **BRIER headroom strong**: v0008 at 0.1383 with floor at 0.1703 — +0.032 headroom. No longer a concern.

5. **Layer 3 now based on v0008**: Future versions checked against:
   - AUC bot2 ≥ 0.8199 - 0.02 = 0.7999
   - AP bot2 ≥ 0.3726 - 0.02 = 0.3526
   - VCAP@100 bot2 ≥ 0.0061 - 0.02 = -0.0139
   - NDCG bot2 ≥ 0.6663 - 0.02 = **0.6463** (still tightest, but now with 0.030 vs champion)

### HUMAN_SYNC Gate Recommendations (cumulative, 8 real-data experiments)
1. **Champion is v0008** — Layer 3 active
2. **VCAP@100 floor**: Tighten from -0.035 to 0.0
3. **CAP@100/500 floors**: Relax by 0.03 (to 0.7025 and 0.6940) — model profile has definitively changed; current floors are two versions misaligned
4. **Keep noise_tolerance at 0.02** — adequate for all metrics
5. **No changes to Group A mean floors** — all pass with 0.05+ headroom

### Cumulative ranges across 8 real-data experiments
| Metric | Min Mean | Max Mean | Range | v0008 |
|--------|----------|----------|-------|-------|
| AUC | 0.8323 | **0.8498** | 0.018 | **New high** |
| AP | 0.3892 | **0.4418** | 0.053 | **New high** |
| NDCG | 0.7323 | 0.7560 | 0.024 | 0.7346 (mid-range) |
| VCAP@100 | 0.0149 | 0.0270 | 0.012 | 0.0240 (near-high) |
