# Tier Pipeline — Results Summary

## CRITICAL CORRECTION (2026-03-08)
**`da_rank_value` is NOT leakage.** It's a historical DA shadow price rank (Spearman ~0.81 with V6.4B `hist_shadow`, only ~0.36 with `actual_shadow_price`). Adjacent months share ~65-70% identical values. Stage 4's v0-v5 results were built on a weakened baseline missing 60% of signal.

## V6.2B Formula (Verified Exact)
```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```
- `da_rank_value` = percentile rank of historical DA shadow prices (LEGITIMATE)
- `density_mix_rank_value` = within-month percentile rank of mixed flow forecast
- `density_ori_rank_value` = within-month percentile rank of original flow forecast

## True Leaky Columns (Blocked)
rank, rank_ori, tier, shadow_sign, shadow_price

## True V6.2B Formula Baseline (v0)
| Metric | v0 (formula) |
|--------|-------------|
| VC@20 | 0.5169 |
| VC@100 | 0.8240 |
| Recall@20 | 0.6917 |
| Recall@50 | 0.7500 |
| Recall@100 | 0.7833 |
| NDCG | 0.9370 |
| Spearman | 0.3809 |
| Tier0-AP | 0.7543 |

## v6: ML with V6.2B features (6 features incl. da_rank_value)
Setup: LightGBM lambdarank, 100 trees, lr=0.05, 31 leaves, 8mo train / 0 val

| Metric | v0 | v6 | Delta |
|--------|------|------|-------|
| VC@20 | 0.5169 | 0.6089 | +17.8% |
| VC@100 | 0.8240 | 0.8485 | +3.0% |
| Recall@20 | 0.6917 | 0.9125 | +31.9% |
| Recall@50 | 0.7500 | 0.8983 | +19.8% |
| Recall@100 | 0.7833 | 0.7617 | -2.8% |
| NDCG | 0.9370 | 0.9666 | +3.2% |
| Spearman | 0.3809 | 0.4512 | +18.4% |
| Tier0-AP | 0.7543 | 0.8087 | +7.2% |

## v7: ML with V6.2B + spice6 (12 features)
Same setup + prob_exceed_110/100/90/85/80, constraint_limit

| Metric | v0 | v7 | Delta |
|--------|------|------|-------|
| VC@20 | 0.5169 | 0.5854 | +13.3% |
| VC@100 | 0.8240 | 0.8800 | +6.8% |
| Recall@20 | 0.6917 | 0.8125 | +17.5% |
| Recall@50 | 0.7500 | 0.9250 | +23.3% |
| Recall@100 | 0.7833 | 0.8925 | +13.9% |
| NDCG | 0.9370 | 0.9465 | +1.0% |
| Spearman | 0.3809 | 0.5150 | +35.2% |
| Tier0-AP | 0.7543 | 0.9239 | +22.5% |

## Key Findings (Updated)
1. **v6 beats v0 formula on 7/8 metrics** — ML with same features outperforms fixed-weight formula
2. **v7 beats v0 formula on ALL 8 metrics** — spice6 features add significant value
3. v6 excels at top-of-list (VC@20 +17.8%, Recall@20 +31.9%) but loses Recall@100 (-2.8%)
4. v7 is more balanced — wins everywhere including Recall@100 (+13.9%) and Spearman (+35.2%)
5. v7 Tier0-AP = 0.9239 vs v0's 0.7543 → massive improvement in tier-0 identification
6. Folding val into train (8/0 vs 6/2) = important setting from stage 4
7. Previous v0-v5 results were invalid (da_rank_value wrongly removed)

## Feature Importance (v6, gain)
da_rank_value dominates (as expected — it's 60% of the formula signal)

## Registry
- v0: true V6.2B formula baseline (reference)
- v6: ML with same 6 V6.2B features (beats formula)
- v7: ML with 12 features (V6.2B + spice6) — **new champion candidate**
- v1-v5: INVALIDATED (built on wrong baseline without da_rank_value)
