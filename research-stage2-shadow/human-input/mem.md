## purpose of this repo
- fork from https://github.com/xyzpower/research-spice-shadow-price-pred. There are two pipelines. /home/xyz/workspace/research-qianli-v2/research-stage1-shadow is focusing on the 1st pipeline, classification. this repo focuses on the next, shadow price prediction
- structural organization:
    - I want to port over the 3-iter-per-report engineering structure from the 1st pipeline
    - scan the repo andthe import pieces
    - some notable mentions: memory system dsign, functionalities of each agents with roles and accesses to files, and registries of promotions and gates

## dataset & metrics
- metrics is what makes or breaks this repo. 
- notice the business incentive file: /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/human-input/business_context.md. align our metric choices and dataset to that goal.
- previous repo chooses stage 2 to only include data from stage 1. how can we replicate this? does it mean that we need to port over stage 1's logic before using stage 2? what would you suggest?

- now before planning, dive deep into the repos and let's first brainstorm.


## check if the two new features are valuable:
- main cause for our trades cp instability the optimizer have **picked different** trades when compared to our big pool. the mcps for the picked trades have higher raised mcp. 
- MCP won't be known unless trades are cleared, but we wishes to know how the differences are first introduced, and to separate pool into 2: one is normal trades, one has higher probability to be optimizer picked candidate
some theories (focusing on f0 trades):
1. perhaps the trades paths have higher series_rev_mapped than usual or higher dz_raw_all
2. perhaps trade paths have higher volatility abs(mtm_1st_mean - mtm 2nd) or abs(1(rev) - 2(rev)). 
3. perhaps some combo such as high series_rev_mapped + high mtm_1st_mean or high 1(rev) are more closely akin to the trades picked.

you need to do research and see.

- now do similar analysis but using the dz_raw_all column, for this trades: /opt/temp/shiyi/trash/pjm_onpeak.parquet

And some more thoughts:
- We might need to update our lightgbm module to use more features. (the fact that v3 has better MAE but worse cp calibration might point to this)
- when using blend, make sure that we are handling nan for cols

now create v4 and v5:

- v4 uses your suggestions but we need to answer key questions:
1. we are talking about training ONLY in the big pool right?
2. which baseline you intend to use? i think overall, the formula in v3 hurts the other path's performance of the baseline
2. and why is your stratified conformal banding idea different from mine? you essentially are suggesting not segmented training, but only validation. but I think: a. the bigger issue might be the baseline calculation. without this, the lightgbm part is not that useful.
3. but i agree with lightgbm feature expansion. you might even add more like mtm1st - mtm2nd etcetc

- v5 uses my suggestion. but let's get to that later. let's do v4.
now gimme ur detailed design choice of v4.

1. are we reusing v2's baseline + lightgbm training with extra features, redo banding and rest? so we dont need to change dataset right? 
2. are we training on big pool and only use trades as evaluation?
3. are you going to report comprehenisvely the v4's cp on f0 trades for prevail and counter?


### baseline setup

1. i wish to use 6 month train + 2 month val for ALL future models (of course i just mean f0, for f1 and f2p, adjust accordingly but still use 6 + 2 setup). and the report should report on **target month only** -> do u agree?
2. the original pipeline: https://github.com/xyzpower/research-spice-shadow-price-pred's stage 1 uses fewer features than the v000 in /home/xyz/workspace/research-qianli-v2/research-stage1-shadow right?
3. aside from this, if we align those 2 above, is this apple to apple?

## features used:
1. which features? the original pipeline: https://github.com/xyzpower/research-spice-shadow-price-pred's stage 1 uses fewer features than the v000 in /home/xyz/workspace/research-qianli-v2/research-stage1-shadow right? 
2. the best version (when comparing everything) i think would be v0011 right? what features does it use?

## v0 vs v1 analysis (2026-03-03)

v1 changed 3 things at once vs v0: features (10→22), n_estimators (200→300), learning_rate (0.1→0.07).
Results: v1 improved EV-VC@100 (+0.0014) and EV-VC@500 (+0.0048) but regressed Spearman (-0.047, -12%) and EV-NDCG (-0.002). Neither is promotable. The kitchen-sink change makes attribution impossible.

**Lesson**: isolate changes — one lever per hypothesis.

## iteration efficiency protocol (agreed 2026-03-04)

**Problem**: each hypothesis takes ~35 min (12-month benchmark). One idea per iteration wastes time.

**Solution**: 2-hypothesis screening per iteration.
1. Orchestrator generates 2 hypotheses with `--overrides` JSON + picks 2 screen months (1 weak, 1 strong)
2. Worker runs both on 2 months using `--overrides` flag (~6 min each, no code changes needed)
3. Worker picks winner, implements in code, runs full 12-month benchmark (~35 min)
4. Total: ~47 min vs ~35 min for 1 idea — doubles idea throughput per iteration

**Screen month selection**: 1 weak month (worst metric performance) + 1 strong month (best performance). Catches "does it help?" + "does it regress?"

**Key**: `benchmark.py --overrides '{"regressor": {...}}' --eval-months M1 M2` already works. No code changes to ML infra needed.

## baseline spec (agreed 2026-03-03)

### evaluation setup
- **train**: 6 months, **val**: 2 months (threshold optimization only), **test**: target month
- report on **target month only** — val set is NOT used for reporting
- 12 eval months: 2020-09, 2020-11, 2021-01, 2021-03, 2021-05, 2021-07, 2021-09, 2021-11, 2022-03, 2022-06, 2022-09, 2022-12
- class_type=onpeak, ptype=f0

### versions
| | Stage 1 (classifier) | Stage 2 (regressor) |
|---|---|---|
| **v0** | 14 features (original pipeline set) | ALL available features |
| **v1** | 29 features (v0011 set) | ALL available features |

The classifier feature set is the variable under test. The regressor always gets maximum information.

### v0 classifier features (14)
prob_exceed_110, prob_exceed_105, prob_exceed_100, prob_exceed_95, prob_exceed_90,
prob_below_100, prob_below_95, prob_below_90,
expected_overload, density_skewness, density_kurtosis, density_cv,
hist_da, hist_da_trend

### v1 classifier features (29) — from v0011
prob_exceed_110, prob_exceed_105, prob_exceed_100, prob_exceed_95, prob_exceed_90,
prob_below_100, prob_below_95, prob_below_90,
expected_overload, hist_da, hist_da_trend,
hist_physical_interaction, overload_exceedance_product,
sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac,
is_interface, constraint_limit,
density_mean, density_variance, density_entropy,
tail_concentration, prob_band_95_100, prob_band_100_105,
hist_da_max_season,
band_severity, sf_exceed_interaction, hist_seasonal_band

### regressor features (ALL available)
all 29 classifier features + prob_exceed_85, prob_exceed_80,
recent_hist_da, season_hist_da_1, season_hist_da_2
(total: 34 features for both v0 and v1 regressor)

### metrics — all threshold-independent

**stage 1 (classifier quality):**
| Metric | Direction | Purpose |
|---|---|---|
| AUC | higher | overall discrimination |
| AP | higher | positive-class ranking quality (handles 7% imbalance) |
| Brier | lower | probability calibration (needed for downstream EV) |
| S1-VCAP@100 | higher | value capture by probability ranking, top-100 |
| S1-VCAP@500 | higher | value capture, top-500 |
| S1-NDCG | higher | ranking quality weighted by position |

**stage 2 (full pipeline — capital allocation quality):**
| Metric | Direction | Purpose |
|---|---|---|
| EV-VC@100 | higher | **the money metric** — value captured in top-100 by EV |
| EV-VC@500 | higher | value capture, broader allocation |
| EV-NDCG | higher | ranking quality by expected value |
| Spearman | higher | rank correlation on binding samples |
| C-RMSE | lower | price prediction error on binding samples |
| C-MAE | lower | same, less outlier-sensitive |
| EV-VC@1000 | higher | monitor — value capture at scale |
| R-REC@500 | higher | monitor — are we finding binders in top-500? |

**blocking gates (for promotion):**
EV-VC@100, EV-VC@500, EV-NDCG, Spearman

**monitoring only (reported, not blocking):**
everything else

### what was dropped
- precision, recall, F1, pred_binding_rate — all threshold-dependent
- prior registry/v0 metrics — computed on val split, not target month, invalid