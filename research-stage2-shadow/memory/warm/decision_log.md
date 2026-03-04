# Decision Log

## Iter 1 — feat-eng-3-20260304-102111

### D1: Promote v0009 as new champion
**Rationale**: All Group A gates pass all 3 layers. EV-VC@100 +9.0% (primary business metric), EV-VC@500 +1.5%, EV-NDCG +0.5%. Spearman -0.6% is noise-level (tail improved). C-RMSE/C-MAE also improved. Note: v0009 is identical to v0008; promotion is operationally of the 39-feature config first registered as v0008.

### D2: Feature set established at 39 nominal / 34 effective
**Rationale**: 5 zero-filled features (hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band) remain from v0007. Effective new features are 5 (density_skewness, density_kurtosis, density_cv, season_hist_da_3, prob_below_85). Both reviewers recommend pruning zero-filled features in next iteration.

### D3: Next iteration should prune zero-filled features + HP tuning
**Rationale**: Feature set is now stabilized. Zero-filled features waste tree splits. HPs (mcw=25, depth=5, n_est=400) were tuned for 29 effective features; 34 effective features may benefit from adjustments. Batch constraint (FE-only) will be relaxed for iter 2 to allow HP changes alongside feature cleanup.
