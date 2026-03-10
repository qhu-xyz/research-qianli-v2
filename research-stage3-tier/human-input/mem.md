## purpose of this repo
- Stage 3: **tier classification** — replaces the two-stage (classifier + regressor) pipeline with a single multi-class XGBoost model predicting 5 shadow price tiers directly
- Ported from research-stage2-shadow, inheriting the 3-iter-per-batch autonomous pipeline structure (orchestrator plan → worker → dual review → orchestrator synthesize)
- Key innovation: `tier_ev_score = sum(P(tier=t) * midpoint[t])` as continuous ranking signal for capital allocation

## tier definitions
| Tier | Shadow Price Range | Midpoint |
|------|-------------------|----------|
| 0 | [3000, +inf) | $4000 |
| 1 | [1000, 3000) | $2000 |
| 2 | [100, 1000) | $550 |
| 3 | [0, 100) | $50 |
| 4 | (-inf, 0) | $0 |

Bins: `[-inf, 0, 100, 1000, 3000, inf]`, labels: `[4, 3, 2, 1, 0]`

## model architecture
- Single XGBoost `multi:softprob` with `num_class=5`
- Class weights handle imbalance: `{0: 10, 1: 5, 2: 2, 3: 1, 4: 0.5}`
- All TierConfig parameters are mutable (no frozen classifier)
- 34 features from stage 2's regressor feature set

## evaluation setup
- **train**: 6 months, **val**: 2 months, **test**: target month
- report on **target month only**
- 12 eval months: 2020-09, 2020-11, 2021-01, 2021-03, 2021-05, 2021-07, 2021-09, 2021-11, 2022-03, 2022-06, 2022-09, 2022-12
- class_type=onpeak, ptype=f0
- All metrics are higher-is-better (no direction inversions)

## metrics
**Blocking (Group A):** Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK
**Monitor (Group B):** Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1

## iteration efficiency protocol (inherited from stage 2)
- 2-hypothesis screening per iteration
- `benchmark.py --overrides '{"tier": {...}}' --eval-months M1 M2` for screening
- Winner gets full 12-month benchmark

## v0 baseline
- v0 uses default TierConfig: 34 features, default class weights, default XGB hyperparams
- Gate floors will be calibrated from v0 results using `populate_v0_gates.py`


## 
- check signal generation notebook: /home/xyz/workspace/psignal/notebook/hz/2025-planning-year/feb/miso/submission/spice6
- check how signal there is defined. we are defining bins of tiers based on shadow prices absolute value in this repo; check this location to see how original tiers are defined. 
- also, i think this repo contains a mechanism of not putting constraint with high correlations all into the same tiers; for instance, if the same monitor line happens to have to be living in multiple constraints, we only include the "strongest" binding constraints related to that montitor line --> is my understanding correct? where does that code live?

## on re-estabilishing the ranks and re-building the constraints
- compare our tier definition with the notebook's logic. which one do you prefer and why? do all of their components in the ranks originate from the shadow prices or some kind of flow's overflow probability?
- Flow deviation distance + DA shadow price vs Historical shadow prices directly  which is better
- or more essentially, if we carry on with our method -> how we can compare results of the two?
- is it possible to port over dedup by equipment 100%?

## questions on comparison between older spice and our model
- why did you say that our model is backward-looking? are we not doing prediction?

## my intention: a directional shift but still using multi-class ML modeling
- let's carry on with our approach.
two important things:
1. we lack this dedup structure, which is a must. so we need to port it.
2. we need to use their pipeline as baseline and try to beat them. 
- can we produce, for each constraint in our setting, their tier prediction putting this constraint to which tier?
But here is the important part: do ALL of our metrics produce apple to apple comparisons?
- for example, tier-vc@100. assume we have 200 tier0, they have 120 tier0. how can we measure tier-vc@100 ?
- let's comb thru the metrics one by one.

- think about our 3-iter-per-report pipeline. given our new requirement, is it now too heavy?
- can we build on our recent discoveries while taking this directional shift?

1. why don't we also incorporate direction into our constraint?
2. can we "map" our constraints to their universe, then compare? 
 i haven't read your approach, but just let me know if we can do this comparison in a fair manner. you don't have to follow my advice.

**one very important thing** => we are producing too much tier0 constraints. “or a typical month, they have ~1,700 constraints.” but you just told me our tier0 have 1k. 
- goal: need to shrink down tier0 to around 50-200, overall none-zero tier constraints to be around the same 2k, (the last tier should be sp = 0)


eseentially, are you suggesting:
- let's use their set up, such as outage data aggregation and flow preservation, so we can more or less generate the same dataset?
 - then we use ML, which is a different method, to assign tiers

 check carefully. if that is the case -> are you 100pct sure we can replicate their pre-processing? by that i mena let's say that  
  for one fixed month, for one period type, they have 1700 constraints. 1. do we have ground truth shadow prices for those 1700      
  constraints? 2. can our pipeline reproduce those same 1700 constraints to feed into lightgbm? 3. can we then compare tir           
  performance using metrics that is not affected by the number of constraints in tier? 4. can we control the number of               
  constraints in tier0 to be around the same level?                                  

- are you saying that out of 1700 constraints in this example, 1684 has DA shadow for that month?
    - then if using their pipeline, how can we measure if their model is good or not?
- take a look at our results: 
1. basically, our model collapses to a binary classifier
- so can we:
1. port over their preprocessing pipeline to a tee, 
2. and find methods to improve upon that?

tbh i'm thinking about completely rebuilding or pipeline.


-  Accept the 97% overlap (pragmatic)
- if we do this:
    - do we have ground truth sp using our own method for EACH constraint?
    - will we do dedup?
    - can we calculate their old score and see their bsaeline performance for each metric?

- when you say GT tier 0, you mean tier0 using their definition?
- then how many tier0 would they predict each month? gimme a similar table as above but with more accurate and comprehensive info


1. divin into VC@100 details; their method produces 224 tier-0 constraints: how is vc@100 calculated
2. if we collapse into 1712 constraints, do we get the same feature we've been using for ML pipeline？
3. so do u think there is REAL opportunity of ML given such imbalanced dataset?

- if they have 200ish tier0, how do they define 100? 
- if our model only produces tiers, how can we define top100?

