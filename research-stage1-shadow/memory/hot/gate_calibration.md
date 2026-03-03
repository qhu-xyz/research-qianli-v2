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

## Iteration 2 Observations (v0009, feat-eng-3-20260303-104101) — PROMOTED

**v0009 gate headroom (Group A)**:

| Gate | v0009 Mean | Floor | Headroom | v0009 Bot2 | v0008 Bot2 | Δ Bot2 | L3 Margin |
|------|-----------|-------|----------|-----------|-----------|--------|-----------|
| S1-AUC | 0.8495 | 0.7848 | +0.065 | 0.8189 | 0.8199 | -0.0010 | +0.019 |
| S1-AP | 0.4445 | 0.3436 | +0.101 | 0.3712 | 0.3726 | -0.0014 | +0.019 |
| S1-VCAP@100 | 0.0266 | -0.0351 | +0.062 | **0.0089** | 0.0061 | **+0.0028** | **+0.023** |
| S1-NDCG | 0.7359 | 0.6833 | +0.053 | 0.6648 | 0.6663 | -0.0015 | +0.019 |

**v0009 gate headroom (Group B)**:

| Gate | v0009 Mean | Floor | Headroom | Notes |
|------|-----------|-------|----------|-------|
| S1-BRIER | 0.1376 | 0.1703 | +0.033 | Best calibration ever |
| S1-CAP@100 | 0.7158 | 0.7325 | **-0.017** | FAILED — 3rd consecutive version |
| S1-CAP@500 | 0.7188 | 0.7240 | **-0.005** | FAILED — 3rd consecutive version |
| S1-VCAP@500 | 0.0955 | 0.0408 | +0.055 | Unchanged |
| S1-VCAP@1000 | 0.1483 | 0.1091 | +0.039 | Unchanged |

### Key Observations

1. **VCAP@100 recovered**: Bot2 improved +0.0028 vs champion — the only Group A metric where bot2 moved in the right direction. L3 margin expanded from +0.017 to +0.023.

2. **L3 margins uniformly comfortable**: All ≥ +0.019. No single metric is near the boundary.

3. **CAP@100/500 failures are structural**: Now failing for v0007, v0008, AND v0009. The model profile has shifted permanently to ranking quality. **Floor relaxation is essential at HUMAN_SYNC.**

4. **BRIER continues to improve**: 0.1376 (best ever) with +0.033 headroom. No overfitting despite 29 features.

5. **Layer 3 now based on v0009**: Future versions checked against:
   - AUC bot2 ≥ 0.8189 - 0.02 = **0.7989**
   - AP bot2 ≥ 0.3712 - 0.02 = **0.3512**
   - VCAP@100 bot2 ≥ 0.0089 - 0.02 = **-0.0111**
   - NDCG bot2 ≥ 0.6648 - 0.02 = **0.6448** (tightest)

## Iteration 3 Observations (v0010, feat-eng-3-20260303-104101) — NOT PROMOTED (null)

**v0010 gate headroom (Group A)**:

| Gate | v0010 Mean | Floor | Headroom | v0010 Bot2 | v0009 Bot2 | Δ Bot2 | L3 Margin |
|------|-----------|-------|----------|-----------|-----------|--------|-----------|
| S1-AUC | 0.8496 | 0.7848 | +0.065 | 0.8172 | 0.8189 | -0.0017 | +0.018 |
| S1-AP | 0.4424 | 0.3436 | +0.099 | 0.3748 | 0.3712 | +0.0036 | +0.024 |
| S1-VCAP@100 | 0.0254 | -0.0351 | +0.060 | 0.0070 | 0.0089 | -0.0019 | +0.018 |
| S1-NDCG | 0.7359 | 0.6833 | +0.053 | 0.6685 | 0.6648 | +0.0037 | +0.024 |

**v0010 gate headroom (Group B)**:

| Gate | v0010 Mean | Floor | Headroom | Notes |
|------|-----------|-------|----------|-------|
| S1-BRIER | 0.1374 | 0.1703 | +0.033 | Best calibration ever (marginally) |
| S1-CAP@100 | 0.7083 | 0.7325 | **-0.024** | FAILED — 4th consecutive version, worst yet |
| S1-CAP@500 | 0.7153 | 0.7240 | **-0.009** | FAILED — 4th consecutive version |
| S1-VCAP@500 | 0.0952 | 0.0408 | +0.054 | Stable |
| S1-VCAP@1000 | 0.1472 | 0.1091 | +0.038 | Slight decline |

### Key Observations

1. **Null result confirms capacity ceiling**: v0010 vs v0009 differences are noise. Tree count and learning rate are not the binding constraint.

2. **Bot2 movements mixed**: AP bot2 +0.0036 and NDCG bot2 +0.0037 (slight tail improvement) while AUC bot2 -0.0017 and VCAP@100 bot2 -0.0019 (slight tail degradation). All within 0.02 noise tolerance.

3. **CAP@100 continues to decline**: 0.7825 → 0.7342 → 0.7142 → 0.7158 → 0.7083 across v0→v0007→v0008→v0009→v0010. More trees slightly worsen this — finer probability granularity makes hard-threshold capture less effective.

4. **Champion unchanged**: v0009 remains champion. L3 floors unchanged.

### HUMAN_SYNC Gate Recommendations — FINAL (cumulative, 10 real-data experiments)
1. **Champion is v0009** — Layer 3 active
2. **VCAP@100 floor**: Tighten from -0.035 to **0.0** (recommended 4 consecutive iterations; never observed below 0.0)
3. **CAP@100/500 floors**: Relax by **0.03** (to 0.7025 and 0.6940) — failing 4 consecutive champion versions; model profile is ranking-first by design
4. **Keep noise_tolerance at 0.02** — adequate; tightest L3 margins +0.018 (comfortable)
5. **No changes to Group A mean floors** — all pass with 0.05+ headroom
6. **Consider tightening noise_tolerance to 0.015 in future** — observed bot2 deltas mostly 0.001-0.004; 0.02 is generous
7. **NEW: Add month-coverage assertion** — require `n_months == n_months_requested` before gate evaluation (Codex iter3 finding)

### Cumulative ranges across 10 real-data experiments
| Metric | Min Mean | Max Mean | Range | Champion (v0009) |
|--------|----------|----------|-------|-----------------|
| AUC | 0.8323 | 0.8498 | 0.018 | 0.8495 (near-high) |
| AP | 0.3892 | **0.4445** | 0.055 | **Pipeline high** |
| NDCG | 0.7323 | 0.7560 | 0.024 | 0.7359 (mid-range) |
| VCAP@100 | 0.0149 | **0.0270** | 0.012 | **Pipeline high** (0.0266) |
| BRIER | 0.1374 | 0.1540 | 0.017 | **Pipeline best** (0.1376) |
