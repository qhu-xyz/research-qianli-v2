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

## Key Observations

1. **S1-VCAP@100 has negative floors** — v0 min was only 0.0005, with 0.10 tail_offset the tail_floor is -0.0995. Effectively non-binding unless a version produces strongly negative VCAP.

2. **S1-CAP@100 and S1-CAP@500 have HIGH variance** (std 0.25 and 0.19). Worst months (0.36 and 0.42) are far below mean. The tail_floor is loose but the mean floor is tight — a version needs consistently high CAP to pass Layer 1.

3. **S1-REC floor is very loose** (0.10 vs v0 mean of 0.42). Any model that predicts some positives will pass.

4. **S1-BRIER is the tightest Group B gate** — only 0.02 headroom from v0 mean to floor. Threshold changes that increase positive predictions tend to increase Brier score.

5. **S1-AUC is the most stable metric** (std=0.015) — hardest to improve but also hardest to accidentally break.

## Three-Layer Check Reference
- **Layer 1**: mean(metric) >= floor (or <= for BRIER)
- **Layer 2**: count(months below tail_floor) <= 1
- **Layer 3**: bottom_2_mean >= champion_bottom_2_mean - 0.02

## Iteration 1 Observations (v0003, 2026-03-02)

- All floors remain appropriate — v0003 passed all 3 layers with ~0.048 headroom on mean
- No gate calibration changes recommended (only 1 real-data iteration beyond v0)
- Codex suggested metric-specific noise_tolerance (tighter for AUC/AP/NDCG, looser for VCAP@100) — good idea but premature; revisit after 3-4 iterations
- Layer 3 is effectively disabled when champion=null (defaults to pass) — acceptable for now
- BRIER headroom actually increased (v0003 BRIER=0.146 vs floor=0.170, headroom now +0.024 vs v0's +0.020)
