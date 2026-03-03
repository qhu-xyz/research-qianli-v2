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

## Iteration 1 Observations (v0002, hp-tune-20260302-144146)

- All floors remain appropriate — v0002 passed all 3 layers with ~0.05 headroom on mean (identical to v0)
- No gate calibration changes recommended (2 real-data iterations beyond v0 — still premature)
- Layer 3 still effectively disabled (champion=null) — bottom_2 deltas vs v0 are very small: AUC +0.000, AP -0.002, VCAP@100 -0.001, NDCG -0.001. All within 0.02 tolerance.
- Codex notes Layer 3 tolerance (0.02) is very loose relative to observed deltas (0.001-0.002) — valid, but need more data points to set appropriate metric-specific tolerances
- VCAP@100 floor (-0.035) remains non-binding. v0002 VCAP@100 min=0.0000, still far above tail_floor (-0.0995). High variance metric (std=0.013, min=0.000, max=0.046).
- BRIER headroom back to +0.020 (v0002 BRIER=0.1505 vs floor=0.170) — interaction features had negligible calibration effect
- **Cumulative**: After 2 real-data experiments, no version has improved beyond v0 on Group A metrics. Gates are not too tight (not blocking good candidates) — there simply haven't been improvements to promote yet.

## Iteration 1 Observations (v0003, feat-eng-20260302-194243)

- All floors remain appropriate — v0003 passed all 3 layers with ~0.05 headroom on mean
- No gate calibration changes recommended (3 real-data iterations now)
- v0003 gate headroom: AUC +0.0513, AP +0.0512, VCAP@100 +0.0534, NDCG +0.0519 — all stable and consistent with v0
- **BRIER headroom narrowing**: v0003 BRIER=0.1514 vs floor=0.1703, headroom=0.0189 (was 0.0200 for v0). Not critical but if trend continues, BRIER could become binding.
- **CAP@100 headroom narrowing**: v0003 mean=0.7708 vs floor=0.7325, headroom=0.0383 (was 0.0500 for v0). CAP has very high variance (std=0.25), so floor is loose relative to variability but tightening.
- **CAP@500 headroom narrowing**: v0003 mean=0.7633 vs floor=0.7240, headroom=0.0393 (was 0.0500 for v0). Similar trajectory as CAP@100.
- Layer 3 still disabled (champion=null). Against v0 reference: AUC bot2 +0.0057, AP bot2 -0.0045, NDCG bot2 -0.0059, VCAP@100 bot2 +0.0002 — all within 0.02 tolerance.
- Codex suggestion for metric-specific Layer 3 tolerances (AUC/AP/NDCG ~0.005-0.01, VCAP@100 looser) remains valid. Need 2+ more iterations to calibrate.
- **Cumulative**: After 3 real-data experiments, deltas are consistently small (±0.003 AUC, ±0.003 AP, ±0.005 NDCG, ±0.005 VCAP@100). This gives a natural range for future tolerance calibration.

## Iteration 1 Observations (v0004, feat-eng-20260303-060938)

- All floors remain appropriate — v0004 passed all 3 layers with ~0.05 headroom on Group A means
- No gate calibration changes recommended (4 real-data iterations now)
- v0004 gate headroom: AUC +0.0515, AP +0.0515, VCAP@100 +0.0556, NDCG +0.0538 — stable, consistent with all prior versions
- **BRIER headroom continues narrowing**: v0004 BRIER=0.1516 vs floor=0.1703, headroom=0.0187. Trend: v0(0.0200) → v0003-HP(0.0241) → v0002(0.0198) → v0003-win(0.0189) → v0004(0.0187). Monotonically declining except for v0003-HP. Not yet critical but the pattern is clear: adding features or expanding window slightly degrades calibration.
- **VCAP@500 bot2 approaching floor**: v0004 bot2=0.0387 vs floor=0.0408. Margin only 0.0021. 3rd consecutive VCAP@500 regression: v0002(-0.0043), v0003(-0.0063), v0004(-0.0065). If next iteration has similar bot2, it may breach the Group B floor. This appears to be an inherent tradeoff: improving top-100 ranking systematically degrades the 100-500 range.
- **CAP@100/500 recovered slightly**: v0004 CAP@100=0.785 (vs v0003-win 0.771, back toward v0's 0.783). CAP@500=0.775 (vs v0003-win 0.763). The interaction features helped recover some broad-ranking ability lost in window expansion.
- Layer 3 still disabled (champion=null). Against v0 reference: AUC bot2 +0.0059, AP bot2 -0.0040, NDCG bot2 -0.0060, VCAP@100 bot2 -0.0003 — all well within 0.02 tolerance. Closest to boundary: NDCG at -0.0060 (margin 0.0140 to fail).
- **Codex recommendation**: Tighten Layer 3 to metric-specific tolerances: AUC/AP/NDCG ~0.005-0.01, keep VCAP@100 looser. With 4 iterations of data, the observed bot2 shifts range ±0.006 for AUC, ±0.005 for AP, ±0.006 for NDCG, ±0.001 for VCAP@100. A tolerance of 0.01 for AUC/AP/NDCG would still pass all versions seen so far. Worth discussing at HUMAN_SYNC.
- **Cumulative ranges across 4 real-data experiments**: AUC mean ∈ [0.8323, 0.8363], AP ∈ [0.3921, 0.3951], NDCG ∈ [0.7323, 0.7371], VCAP@100 ∈ [0.0149, 0.0205]. The operating range is narrow — ~0.004 AUC, ~0.003 AP, ~0.005 NDCG, ~0.006 VCAP@100.
