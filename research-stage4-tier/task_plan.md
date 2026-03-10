# Task Plan: Clean Leakage, Rebuild Baseline + ML Models

## Goal
Remove all leakage from V6.2B pipeline, establish a fair baseline, build ML v1 with same features, then improve.

## Key Finding
V6.2B formula is EXACTLY: `rank_ori = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value`
- 60% of the signal is `da_rank_value` (realized DA shadow price percentile rank) = LEAKAGE
- Fair V6.2B features (no leakage): `density_mix_rank_value`, `density_ori_rank_value`, `ori_mean`, `mix_mean`, `mean_branch_max`
- These fair features have Spearman ~ -0.05 with realized shadow prices (near zero)

## Phases

### Phase 1: Clean baseline (v0_fair) `status: not_started`
- [ ] Rewrite `baseline_v62b.py` to produce a fair v0 using only non-leaky V6.2B columns
- [ ] Score = weighted combo of density_mix_rank_value + density_ori_rank_value (the 40% forecast part)
- [ ] Run 12-month eval, save to registry/v0_fair
- [ ] Delete registry/v1, v2, v3, v1_noleak (all tainted)
- [ ] Update gates.json calibrated from v0_fair
- [ ] Update champion.json to v0_fair

### Phase 2: ML v1_fair — same features as v0_fair `status: not_started`
- [ ] Config: features = [density_mix_rank_value, density_ori_rank_value, ori_mean, mix_mean, mean_branch_max]
- [ ] Use LightGBM backend (fast, has early stopping)
- [ ] Strip ALL zero-filled features
- [ ] Run 12-month eval, save to registry/v1
- [ ] Compare against v0_fair

### Phase 3: ML v2 — add spice6 features `status: not_started`
- [ ] Add spice6 density features: prob_exceed_110, prob_exceed_100, prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit
- [ ] 11 features total (5 V6.2B + 6 spice6)
- [ ] Run 12-month eval, save to registry/v2
- [ ] Compare against v0_fair and v1

### Phase 4: Clean up docs `status: not_started`
- [ ] Update mem.md with findings
- [ ] Update config.py leakage guard and feature lists
- [ ] Clean up any stale references
